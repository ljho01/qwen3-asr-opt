import mlx.core as mx
import pytest
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from mlx_qwen3_asr.model import Qwen3ASRModel
from test_speculative import small_model_config

from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.continuous import generate_continuous
from qwen_asr_opt.kv_window import attention_width, make_compact_step


def test_current_write_and_capacity_edges_are_included():
    assert [attention_width(i, 300) for i in (0, 127, 128, 255, 256, 299)] == [128, 128, 256, 256, 300, 300]
    for position in (-1, 300):
        with pytest.raises(ValueError):
            attention_width(position, 300)


def test_bucket_transition_masks_future_and_matches_independent_cache():
    mx.random.seed(20260918)
    model = Qwen3ASRModel(small_model_config())
    model.eval()
    prompts = []
    stock = []
    lengths = [127, 9]
    for length in lengths:
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        rope = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        prompt = (ids, mx.random.normal((1, 1, 128)), rope)
        cache = model.create_cache(max_seq_len=256)
        model.prefill(*prompt, cache)
        mx.eval(cache.keys, cache.values)
        prompts.append(prompt)
        stock.append(cache)
    _, keys, values = prefill_serial(model, prompts, 256)
    # Poison unused cache slots: future entries must remain invisible before AND
    # after widening attention from128 to256, and be overwritten at each write.
    for layer in range(len(keys)):
        for row, length in enumerate(lengths):
            keys[layer][row, :, length:] = 20
            values[layer][row, :, length:] = -20
    mx.eval(keys, values)
    for index in range(4):
        positions = mx.array([n + index for n in lengths], dtype=mx.int32)
        token = mx.array([[10 + index], [20 + index]])
        width = attention_width(max(lengths) + index, 256)
        actual, keys, values = make_compact_step(model, width)(token, positions, keys, values)
        mx.eval(actual, keys, values)
        expected = []
        for row, cache in enumerate(stock):
            logits = model.step(token[row:row + 1],
                                mx.full((1, 3, 1), lengths[row] + index, dtype=mx.int32), cache)
            expected.append(mx.argmax(logits, axis=-1).item())
            for layer in range(len(keys)):
                end = lengths[row] + index + 1
                assert mx.allclose(keys[layer][row:row + 1, :, :end], cache.keys[layer][:, :, :end],
                                   atol=1e-5, rtol=1e-5).item()
                assert mx.allclose(values[layer][row:row + 1, :, :end], cache.values[layer][:, :, :end],
                                   atol=1e-5, rtol=1e-5).item()
        assert actual.reshape(-1).tolist() == expected
        assert all(array.shape[2] == 256 for array in keys + values)


@pytest.mark.parametrize("policy", ["compact128", "growing128", "adaptive128"])
def test_refill_across_attention_widths_matches_full_independent_generation(policy):
    mx.random.seed(25)
    model = Qwen3ASRModel(small_model_config())
    model.eval()
    jobs = []
    for length, cap in [(119, 15), (110, 1), (120, 4), (4, 3), (128, 1), (12, 2)]:
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        jobs.append(((ids, mx.random.normal((1, 1, 128)), positions),
                     GenerationConfig(max_new_tokens=cap, eos_token_ids=[])))
    expected = [generate_with_info(model, *prompt, config) for prompt, config in jobs]
    actual, stats = generate_continuous(model, jobs, 2, shared_prefix=128, token_budget=16,
                                        kv_policy=policy)
    assert actual == expected
    assert set(stats["attention_width_calls"]) == {"128", "256"}
    assert stats["refills"] == 4
    if policy in ("growing128", "adaptive128"):
        assert stats["cache_capacity_history"] == [128, 256]


@pytest.mark.parametrize("policy", ["adaptive128", "adaptive64"])
def test_prompt_growth_during_initialization_and_refill_preserves_other_rows(policy):
    mx.random.seed(925)
    model = Qwen3ASRModel(small_model_config())
    model.eval()
    jobs = []
    # Initial row63 requires a decode write in its existing64 slots; row127
    # raises the initial pool. The earliest refill introduces a much longer
    # prompt while another original row is still decoding. A511 prompt then
    # crosses512 during decoding while the original row is still active.
    for length, cap in [(63, 1), (127, 20), (300, 1), (511, 3)]:
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        rope = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        jobs.append(((ids, mx.random.normal((1, 1, 128)), rope),
                     GenerationConfig(max_new_tokens=cap, eos_token_ids=[])))
    expected = [generate_with_info(model, *prompt, config) for prompt, config in jobs]
    actual, stats = generate_continuous(model, jobs, 2, shared_prefix=512, token_budget=32,
                                       kv_policy=policy)
    assert actual == expected
    # The first decode advances row127 to128 before the next iteration; refill
    # to300 has already expanded the pool, so no separate192 stage is required.
    expected_history = [128, 384, 512, 640] if policy == "adaptive128" else [64, 128, 320, 512, 576]
    assert stats["cache_capacity_history"] == expected_history
    assert stats["prompt_lengths"] == [63, 127, 300, 511]
    assert stats["refills"] == 2


@pytest.mark.parametrize("policy", ["adaptive128", "adaptive64"])
def test_empty_adaptive_pool_allocates_no_cache(policy):
    actual, stats = generate_continuous(Qwen3ASRModel(small_model_config()), [], kv_policy=policy)
    assert actual == []
    assert stats["cache_capacity_history"] == []
    assert stats["cache_capacity"] == 0


@pytest.mark.parametrize("policy", ["adaptive128", "adaptive64"])
@pytest.mark.parametrize("job_count", [1, 3])
def test_initial_partial_pool_keeps_unfilled_rows_inactive(policy, job_count):
    mx.random.seed(34)
    model = Qwen3ASRModel(small_model_config())
    model.eval()
    jobs = []
    for length in [7, 69, 129][:job_count]:
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        rope = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        jobs.append(((ids, mx.random.normal((1, 1, 128)), rope),
                     GenerationConfig(max_new_tokens=3, eos_token_ids=[])))
    expected = [generate_with_info(model, *prompt, config) for prompt, config in jobs]
    completions = []
    actual, stats = generate_continuous(model, jobs, 8, kv_policy=policy,
                                       on_complete=lambda index, result: completions.append(index))
    assert actual == expected
    assert sorted(completions) == list(range(job_count))
    assert stats["peak_active_rows"] == job_count
    assert stats["prefills"] == job_count and stats["refills"] == 0
