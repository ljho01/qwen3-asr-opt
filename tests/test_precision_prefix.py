from copy import deepcopy

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from test_mlp_precision import pair

from qwen_asr_opt import continuous
from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.mlp_precision import is_mlp, quantize_mlp
from qwen_asr_opt.precision_prefix import HighPrecisionPrefix


def models():
    decode, prefix = pair()
    quantize_mlp(decode, 6)
    return prefix, decode


def prompt(length):
    ids = mx.array([[1, 5]+[2]*(length-2)])
    positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
    return ids, mx.random.normal((1, 1, 128)).astype(mx.float16), positions


def test_protected_storage_shared_but_low_bit_mlp_and_q8_prefix_are_preserved():
    prefix, decode = models()
    original = dict(tree_flatten(decode.parameters()))
    helper = HighPrecisionPrefix(prefix, decode)
    before, after = dict(tree_flatten(prefix.parameters())), dict(tree_flatten(decode.parameters()))
    for name, value in after.items():
        assert value is (original[name] if is_mlp(name.rsplit(".", 1)[0]) else before[name])
    prompts = [prompt(7), prompt(31)]
    reference, actual = prefill_serial(prefix, prompts, 128), helper(decode, prompts, 128)
    assert mx.array_equal(reference[0], actual[0]).item()
    assert all(mx.array_equal(a, b).item() for a, b in zip(reference[1]+reference[2], actual[1]+actual[2], strict=True))
    assert helper.statistics()["prefix_calls"] == 2 and helper.additional_mlp_payload_bytes > 0
    with pytest.raises(ValueError, match="different decode"):
        helper(prefix, prompts, 128)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_hybrid_refills_match_independent_q8_prefill_then_mlp6_steps(monkeypatch, batch_size):
    prefix, decode = models()
    helper = HighPrecisionPrefix(prefix, decode)
    jobs = [(prompt(length), GenerationConfig(max_new_tokens=cap, eos_token_ids=[]))
            for length, cap in zip([7, 31, 127]*3, [2, 5, 7]*3, strict=True)]
    # Independent upstream single-stream generator: only its model.prefill is replaced.
    with monkeypatch.context() as patch:
        patch.setattr(decode, "prefill", prefix.prefill)
        expected = [generate_with_info(decode, *p, c) for p, c in jobs]
    monkeypatch.setattr(continuous, "prefill_serial", helper)
    actual, stats = continuous.generate_continuous(decode, jobs, batch_size, shared_prefix=128,
                                                 token_budget=16, kv_policy="growing128")
    assert actual == expected and helper.calls == len(jobs)
    assert stats["refills"] == len(jobs)-batch_size and max(stats["cache_capacity_history"]) == 256


def test_reject_incompatible_or_precompiled_models_before_sharing():
    prefix, decode = models()
    decode.config = deepcopy(decode.config)
    decode.config.text_config.rope_theta += 1
    with pytest.raises(ValueError, match="configuration"):
        HighPrecisionPrefix(prefix, decode)
    prefix, decode = models()
    decode._continuous_steps = {}
    with pytest.raises(ValueError, match="before compiling"):
        HighPrecisionPrefix(prefix, decode)
    prefix, decode = models()
    decode.model.norm.weight = decode.model.norm.weight+1
    expected = dict(tree_flatten(decode.parameters()))
    with pytest.raises(ValueError, match="Protected tensor"):
        HighPrecisionPrefix(prefix, decode)
    assert all(value is expected[name] for name, value in tree_flatten(decode.parameters()))
