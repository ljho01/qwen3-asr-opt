import mlx.core as mx
import pytest
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from test_compiled_prefill import prompt, setup_model

from qwen_asr_opt import continuous
from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.bucketed_prefill import BucketedPrefill


@pytest.mark.parametrize("quantum", [32, 64])
def test_dynamic_valid_length_and_zero_suffix_match_reference(quantum):
    model = setup_model()
    helper = BucketedPrefill(model, quantum=quantum)
    # Several different lengths deliberately share each compiled bucket.
    for length in [1, 7, 31, 32, 33, 63, 64, 65, 127, 129]:
        value = prompt(length, min(3, length - 1), offset=9)
        expected = prefill_serial(model, [value], 137)
        actual = helper(model, [value], 137)
        assert mx.array_equal(actual[0], expected[0]).item()
        for left, right in zip(actual[1] + actual[2], expected[1] + expected[2], strict=True):
            assert left.shape == right.shape
            assert mx.allclose(left[:, :, :length], right[:, :, :length], atol=.004, rtol=.004).item()
            assert mx.all(left[:, :, length:] == 0).item()
    assert helper.cache_hits > 0 and helper.graphs_created < helper.calls
    assert helper.padded_tokens >= helper.valid_tokens


@pytest.mark.parametrize("quantum", [32, 64])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_refill_and_generation_preserve_independent_streams(monkeypatch, quantum, batch_size):
    model = setup_model()
    jobs = [(prompt(length, count, offset=i), GenerationConfig(max_new_tokens=cap, eos_token_ids=[]))
            for i, (length, count, cap) in enumerate([(7, 1, 5), (19, 3, 2), (131, 7, 6), (69, 0, 3), (17, 2, 4)])]
    expected = [generate_with_info(model, *value, config) for value, config in jobs]
    helper = BucketedPrefill(model, quantum=quantum, max_graphs=2)
    monkeypatch.setattr(continuous, "prefill_serial", helper)
    actual, stats = continuous.generate_continuous(model, jobs, batch_size, kv_policy="growing128")
    assert actual == expected
    assert stats["refills"] == max(0, len(jobs) - batch_size)
    assert helper.calls == len(jobs) and len(helper.graphs) <= 2


def test_multiple_prompts_and_lru_eviction():
    model = setup_model()
    helper = BucketedPrefill(model, quantum=32, max_graphs=2)
    for lengths in [(7, 17), (69, 129), (7, 17)]:
        values = [prompt(length, i + 1) for i, length in enumerate(lengths)]
        expected = prefill_serial(model, values, 160)
        actual = helper(model, values, 160)
        assert mx.array_equal(actual[0], expected[0]).item()
        for left, right in zip(actual[1] + actual[2], expected[1] + expected[2], strict=True):
            assert mx.allclose(left, right, atol=.004, rtol=.004).item()
    assert helper.graphs_created == 4 and helper.cache_hits == 2 and len(helper.graphs) == 2


def test_input_validation_precedes_padding_and_compilation():
    model = setup_model()
    helper = BucketedPrefill(model)
    ids, features, positions = prompt(7, 2)
    for invalid in [(ids, features[:, :1], positions), (ids, features[:, :, :64], positions),
                    (ids.astype(mx.float32), features, positions), (ids * 99999, features, positions)]:
        with pytest.raises((ValueError, TypeError)):
            helper(model, [invalid], 32)
    with pytest.raises(ValueError):
        helper(model, [(ids, features, positions)], 6)
    with pytest.raises(ValueError):
        helper(model, [(ids, features, positions[:, :, :6])], 32)
    with pytest.raises(ValueError):
        helper(object(), [(ids, features, positions)], 32)
    assert helper.calls == helper.graphs_created == 0


@pytest.mark.parametrize("kwargs", [{"quantum": 0}, {"quantum": True}, {"max_graphs": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BucketedPrefill(object(), **kwargs)
