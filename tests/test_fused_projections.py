import mlx.core as mx
import pytest
from mlx import nn
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from mlx_qwen3_asr.model import Qwen3ASRModel
from test_speculative import small_model_config

from qwen_asr_opt.continuous import generate_continuous
from qwen_asr_opt.fused_projections import PackedQuantizedLinear, fuse_decoder_projections


@pytest.mark.parametrize("bits,bias", [(4, False), (8, False), (8, True)])
def test_packed_weight_bytes_and_projection_outputs_preserved(bits, bias):
    mx.random.seed(71)
    linears = []
    for width in (256, 128, 128):
        source = nn.Linear(128, width, bias=bias)
        source.set_dtype(mx.float16)
        linears.append(nn.QuantizedLinear.from_linear(source, group_size=64, bits=bits))
    packed = PackedQuantizedLinear(linears)
    offset = 0
    for layer in linears:
        width = layer.weight.shape[0]
        for field in ("weight", "scales", "biases", *(('bias',) if bias else ())):
            assert mx.array_equal(packed[field][offset:offset + width], layer[field]).item()
        offset += width
    for shape in ((1, 1, 128), (4, 1, 128), (8, 1, 128), (1, 31, 128), (2, 129, 128)):
        x = mx.random.normal(shape).astype(mx.float16)
        expected = mx.concatenate([layer(x) for layer in linears], axis=-1)
        assert mx.allclose(packed(x), expected, atol=0.002, rtol=0.002).item()


@pytest.mark.parametrize("mode", ["qkv", "gate_up", "both"])
def test_fused_prefill_kv_and_refilling_generation_match_independent_original(mode):
    mx.random.seed(43)
    model = Qwen3ASRModel(small_model_config())
    model.set_dtype(mx.float16)
    nn.quantize(model.model, group_size=64, bits=8)
    model.eval()
    jobs, caches, logits = [], [], []
    for length, cap in [(7, 8), (17, 2), (69, 6), (4, 3), (129, 4)]:
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        prompt = (ids, mx.random.normal((1, 1, 128)).astype(mx.float16), positions)
        jobs.append((prompt, GenerationConfig(max_new_tokens=cap, eos_token_ids=[])))
        cache = model.create_cache(max_seq_len=length)
        logit = model.prefill(*prompt, cache)
        mx.eval(logit, cache.keys, cache.values)
        caches.append(cache)
        logits.append(logit)
    expected = [generate_with_info(model, *prompt, config) for prompt, config in jobs]
    # Populate the real compiled graph before replacing its projection modules.
    previous, _ = generate_continuous(model, jobs, 2, kv_policy="growing128")
    assert previous == expected
    assert model._continuous_steps
    meta = fuse_decoder_projections(model, mode)
    assert not hasattr(model, "_continuous_steps") and meta["requantized"] is False
    for (prompt, _), old_cache, old_logits in zip(jobs, caches, logits, strict=True):
        cache = model.create_cache(max_seq_len=prompt[0].shape[1])
        actual = model.prefill(*prompt, cache)
        assert mx.allclose(actual, old_logits, atol=0.004, rtol=0.004).item()
        for left, right in zip(cache.keys + cache.values, old_cache.keys + old_cache.values, strict=True):
            assert mx.allclose(left, right, atol=0.004, rtol=0.004).item()
    actual, stats = generate_continuous(model, jobs, 2, kv_policy="growing128")
    assert actual == expected and stats["refills"] == 3
    with pytest.raises(ValueError, match="Reload"):
        fuse_decoder_projections(model, mode)


def test_incompatible_quantization_is_rejected_before_model_mutation():
    model = Qwen3ASRModel(small_model_config())
    nn.quantize(model.model, group_size=64, bits=8)
    original = model.model.layers[0].self_attn
    model.model.layers[-1].self_attn.k_proj = nn.QuantizedLinear(128, 128, bits=4, group_size=64)
    with pytest.raises(ValueError, match="must match"):
        fuse_decoder_projections(model, "both")
    assert model.model.layers[0].self_attn is original
