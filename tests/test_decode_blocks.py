import mlx.core as mx
import pytest
from mlx import nn
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from mlx_qwen3_asr.model import Qwen3ASRModel
from test_speculative import small_model_config

from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.decode_blocks import generate_continuous_blocks, make_decode_block
from qwen_asr_opt.kv_window import make_compact_step


def model_and_prompts(lengths):
    mx.random.seed(229)
    model = Qwen3ASRModel(small_model_config())
    model.set_dtype(mx.float16)
    nn.quantize(model.model, bits=8, group_size=64)
    model.eval()
    prompts = []
    for length in lengths:
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        audio = mx.random.normal((1, 1, 128)).astype(mx.float16)
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        prompts.append((ids, audio, positions))
    return model, prompts


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_block_tokens_and_full_kv_match_individual_compiled_steps(steps):
    model, prompts = model_and_prompts([7, 31, 511])
    token, keys, values = prefill_serial(model, prompts, 640)
    positions = mx.array([7, 31, 511])
    actual = make_decode_block(model, 640, steps)(token, positions, keys, values)
    consumed = []
    step = make_compact_step(model, 640)
    for index in range(steps):
        consumed.append(token)
        token, keys, values = step(token, positions + index, keys, values)
    assert mx.array_equal(actual[0], mx.concatenate(consumed, axis=1)).item()
    assert mx.array_equal(actual[1], token).item()
    for left, right in zip(actual[2] + actual[3], keys + values, strict=True):
        assert mx.allclose(left, right, atol=0.004, rtol=0.004).item()


@pytest.mark.parametrize("block_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_refill_growth_budget_and_callback_preserve_independent_results(block_size, batch_size):
    model, prompts = model_and_prompts([511, 7, 127, 512, 31, 255, 131, 9, 17])
    caps = [8, 2, 7, 3, 10, 5, 6, 5, 1]
    jobs = [(prompt, GenerationConfig(max_new_tokens=cap, eos_token_ids=[]))
            for prompt, cap in zip(prompts, caps, strict=True)]
    expected = [generate_with_info(model, *prompt, config) for prompt, config in jobs]
    callbacks = []
    actual, stats = generate_continuous_blocks(model, jobs, batch_size, block_size=block_size,
                                               on_complete=lambda i, r: callbacks.append((i, r)))
    assert actual == expected
    assert sorted(callbacks) == list(enumerate(expected))
    assert stats["refills"] == len(jobs) - batch_size
    assert stats["cache_capacity_history"] == [512, 640]
    assert stats["decode_calls"] == sum(int(size) * calls for size, calls in stats["actual_block_sizes"].items())
    assert stats["allocated_row_steps"] == batch_size * stats["decode_calls"]
    assert stats["actual_block_sizes"].get("1", 0) > 0
    assert stats["deferred_finish_row_steps"] == 0  # Known budgets end at a host boundary.


@pytest.mark.parametrize("block_size", [2, 4])
def test_eos_inside_block_never_emits_surplus_and_refills_finished_rows(block_size):
    model, prompts = model_and_prompts([7, 31, 17, 127, 9, 69])
    jobs = []
    for index, prompt in enumerate(prompts):
        generated = generate_with_info(model, *prompt, GenerationConfig(max_new_tokens=8, eos_token_ids=[]))
        eos = generated.tokens[0 if index % 2 == 0 else 2]
        jobs.append((prompt, GenerationConfig(max_new_tokens=8, eos_token_ids=[eos])))
    expected = [generate_with_info(model, *prompt, config) for prompt, config in jobs]
    actual, stats = generate_continuous_blocks(model, jobs, 2, block_size=block_size)
    assert actual == expected and all(r.finish_reason == "eos" for r in actual)
    assert stats["refills"] == 4 and stats["deferred_finish_row_steps"] > 0


def test_repetition_inside_block_and_partial_initial_pool():
    model, prompts = model_and_prompts([7, 31, 17])

    class ConstantHead(nn.Module):
        def __call__(self, hidden):
            return mx.broadcast_to(mx.array([0., 0., 0., 100., *([0.] * 60)]),
                                   (*hidden.shape[:-1], 64))

    model.lm_head = ConstantHead()
    jobs = [(prompt, GenerationConfig(max_new_tokens=128, eos_token_ids=[])) for prompt in prompts]
    expected = [generate_with_info(model, *prompt, config) for prompt, config in jobs]
    actual, stats = generate_continuous_blocks(model, jobs, 8, block_size=4)
    assert actual == expected and all(r.finish_reason == "repetition" for r in actual)
    assert stats["refills"] == 0 and stats["peak_active_rows"] == 3


def test_empty_input_and_invalid_configuration():
    model, _ = model_and_prompts([])
    results, stats = generate_continuous_blocks(model, [], 4)
    assert results == [] and stats["decode_calls"] == 0
    assert not hasattr(model, "_autoregressive_blocks")
    with pytest.raises(ValueError):
        generate_continuous_blocks(model, [], block_size=5)
    with pytest.raises(ValueError):
        generate_continuous_blocks(model, [], kv_policy="padded")
