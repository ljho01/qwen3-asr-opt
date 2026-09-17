import mlx.core as mx
import pytest
from mlx import nn
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from mlx_qwen3_asr.model import Qwen3ASRModel
from test_speculative import small_model_config

from qwen_asr_opt import continuous
from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.compiled_prefill import PrefixPrefill, prefix_outputs


def setup_model():
    mx.random.seed(227)
    model = Qwen3ASRModel(small_model_config())
    model.set_dtype(mx.float16)
    nn.quantize(model.model, group_size=64, bits=8)
    model.eval()
    return model


def prompt(length, audio_tokens=1, offset=0):
    ids = mx.array([[1] + [5] * audio_tokens + [2] * (length - audio_tokens - 1)])
    features = mx.random.normal((1, audio_tokens, 128)).astype(mx.float16)
    positions = mx.broadcast_to((mx.arange(length) + offset)[None, None, :], (1, 3, length))
    return ids, features, positions


@pytest.mark.parametrize("compiled", [False, True])
def test_prefill_tokens_cache_padding_and_eviction_across_shapes(compiled):
    model = setup_model()
    helper = PrefixPrefill(model, compiled=compiled, max_graphs=2)
    for lengths in [(7, 17), (69, 129), (7, 17), (7, 17)]:
        prompts = [prompt(length, i + 1, offset=3 * i) for i, length in enumerate(lengths)]
        reference = prefill_serial(model, prompts, 256)
        actual = helper(model, prompts, 256)
        assert mx.array_equal(actual[0], reference[0]).item()
        for left, right in zip(actual[1] + actual[2], reference[1] + reference[2], strict=True):
            assert mx.allclose(left, right, atol=0.004, rtol=0.004).item()
            for row, length in enumerate(lengths):
                assert mx.all(left[row, :, length:, :] == 0).item()
        assert len(helper.graphs) <= 2
    if compiled:
        assert helper.graphs_created == 6 and helper.cache_hits == 2
    else:
        assert helper.graphs_created == 0


def test_fresh_prefix_logits_and_all_kv_match_preallocated_cache():
    model = setup_model()
    for length, count in [(1, 0), (31, 2), (129, 5)]:
        ids, features, positions = prompt(length, count)
        old = model.create_cache(max_seq_len=length)
        expected = model.prefill(ids, features, positions, old)
        embeds = model._inject_audio_features(model._embed_tokens(ids, validate_input_ids=True),
                                               features, ids == model.audio_token_id)
        logits, keys, values = prefix_outputs(model, embeds, positions)
        assert mx.allclose(logits, expected, atol=0.001, rtol=0.001).item()
        for left, right in zip(keys + values, old.keys + old.values, strict=True):
            assert mx.allclose(left, right, atol=0.001, rtol=0.001).item()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_continuous_refill_matches_independent_generation(monkeypatch, compiled, batch_size):
    model = setup_model()
    jobs = [(prompt(length, count), GenerationConfig(max_new_tokens=cap, eos_token_ids=[]))
            for length, count, cap in [(7, 1, 5), (19, 3, 2), (131, 7, 6), (69, 0, 3), (17, 2, 4)]]
    expected = [generate_with_info(model, *value, config) for value, config in jobs]
    helper = PrefixPrefill(model, compiled=compiled, max_graphs=2)
    monkeypatch.setattr(continuous, "prefill_serial", helper)
    actual, stats = continuous.generate_continuous(model, jobs, batch_size, kv_policy="growing128")
    assert actual == expected
    assert stats["refills"] == max(0, len(jobs) - batch_size)
    assert helper.calls == len(jobs)


@pytest.mark.parametrize("compiled", [False, True])
def test_upstream_validation_is_retained_before_compile(compiled):
    model = setup_model()
    helper = PrefixPrefill(model, compiled=compiled)
    ids, features, positions = prompt(7, 2)
    with pytest.raises(ValueError, match="out of bounds"):
        helper(model, [(ids, features[:, :1], positions)], 128)
    with pytest.raises(ValueError, match="channel mismatch"):
        helper(model, [(ids, features[:, :, :64], positions)], 128)
    with pytest.raises((ValueError, TypeError)):
        helper(model, [(ids.astype(mx.float32), features, positions)], 128)
    with pytest.raises(ValueError):
        helper(model, [(ids * 99999, features, positions)], 128)
    assert helper.calls == helper.graphs_created == 0
    # A zero-audio prompt is valid, including an empty feature sequence.
    value = prompt(7, 0)
    expected = prefill_serial(model, [value], 128)
    assert mx.array_equal(helper(model, [value], 128)[0], expected[0]).item()
