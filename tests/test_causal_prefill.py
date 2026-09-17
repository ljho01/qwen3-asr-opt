import mlx.core as mx
import pytest
from mlx_qwen3_asr import decoder
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from test_compiled_prefill import prompt, setup_model

from qwen_asr_opt import continuous
from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.causal_prefill import prefill_causal


@pytest.mark.parametrize("length", [1, 7, 31, 64, 129, 399])
def test_fresh_prefix_cache_and_token_match_additive_mask(length):
    model = setup_model()
    value = prompt(length, min(3, length - 1), offset=7)
    expected = prefill_serial(model, [value], 512)
    actual = prefill_causal(model, [value], 512)
    assert mx.array_equal(actual[0], expected[0]).item()
    for left, right in zip(actual[1] + actual[2], expected[1] + expected[2], strict=True):
        assert mx.allclose(left, right, atol=.004, rtol=.004).item()
        assert mx.all(left[:, :, length:] == 0).item()


def test_each_prompt_uses_square_attention_before_capacity_padding(monkeypatch):
    model = setup_model()
    original = decoder._scaled_dot_product_attention
    observed = []

    def spy(q, k, v, mask=None, scale=None):
        observed.append((q.shape[2], k.shape[2], v.shape[2], mask))
        return original(q, k, v, mask=mask, scale=scale)

    monkeypatch.setattr(decoder, "_scaled_dot_product_attention", spy)
    inputs = [prompt(7, 1), prompt(69, 2), prompt(129, 3)]
    actual = prefill_causal(model, inputs, 512)
    assert observed and {q for q, _, _, _ in observed} == {7, 69, 129}
    assert all(q == k == v and mask == "causal" for q, k, v, mask in observed)
    assert all(array.shape[0] == 3 and array.shape[2] == 512 for array in actual[1] + actual[2])


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_continuous_refill_preserves_independent_generation(monkeypatch, batch_size):
    model = setup_model()
    jobs = [(prompt(length, count), GenerationConfig(max_new_tokens=cap, eos_token_ids=[]))
            for length, count, cap in [(7, 1, 5), (19, 3, 2), (131, 7, 6), (69, 0, 3), (17, 2, 4)]]
    expected = [generate_with_info(model, *value, config) for value, config in jobs]
    monkeypatch.setattr(continuous, "prefill_serial", prefill_causal)
    actual, stats = continuous.generate_continuous(model, jobs, batch_size, kv_policy="growing128")
    assert actual == expected
    assert stats["refills"] == max(0, len(jobs) - batch_size)


def test_validation_remains_upstream_and_precedes_attention(monkeypatch):
    model = setup_model()
    ids, features, positions = prompt(7, 2)

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid input reached attention")

    monkeypatch.setattr(decoder, "_scaled_dot_product_attention", unexpected)
    for invalid in [(ids, features[:, :1], positions), (ids, features[:, :, :64], positions),
                    (ids.astype(mx.float32), features, positions), (ids * 99999, features, positions)]:
        with pytest.raises((ValueError, TypeError)):
            prefill_causal(model, [invalid], 128)
    with pytest.raises(ValueError):
        prefill_causal(model, [(ids, features, positions)], 6)
    with pytest.raises(ValueError):
        prefill_causal(model, [(ids, features, positions[:, :, :6])], 128)
    with pytest.raises(ValueError):
        prefill_causal(model, [], 128)
