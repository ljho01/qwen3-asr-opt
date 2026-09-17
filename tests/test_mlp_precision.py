import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from mlx_qwen3_asr.model import Qwen3ASRModel
from test_speculative import small_model_config

from qwen_asr_opt.mlp_precision import checkpoint_arrays, is_mlp, quantize_mlp


def pair():
    mx.random.seed(233)
    config = small_model_config()
    config.text_config.tie_word_embeddings = True
    original, baseline = Qwen3ASRModel(config), Qwen3ASRModel(config)
    original.set_dtype(mx.float16)
    original.eval()
    baseline.load_weights(tree_flatten(original.parameters()), strict=True)
    baseline.eval()
    nn.quantize(baseline, bits=8, group_size=64)
    mx.eval(original.parameters(), baseline.parameters())
    return original, baseline


@pytest.mark.parametrize("bits", [6])
def test_only_mlp_linears_change_and_protected_tensors_stay_exact(bits):
    model, baseline = pair()
    configs = quantize_mlp(model, bits)
    assert sum(c["bits"] == bits for c in configs.values()) == 6
    assert all(c["bits"] == 8 for p, c in configs.items() if not is_mlp(p))
    assert configs["lm_head"] == configs["model.embed_tokens"]
    reference = dict(tree_flatten(baseline.parameters()))
    for name, tensor in tree_flatten(model.parameters()):
        if not is_mlp(name.rsplit(".", 1)[0]):
            assert mx.array_equal(tensor, reference[name]).item()


@pytest.mark.parametrize("bits", [6])
def test_checkpoint_roundtrip_preserves_tied_storage_logits_and_greedy_tokens(tmp_path, bits):
    model, _ = pair()
    configs = quantize_mlp(model, bits)
    arrays, aliases = checkpoint_arrays(model)
    assert len(aliases) == 3 and all(not name.startswith("lm_head.") for name in arrays)
    path = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(path), arrays)
    restored = Qwen3ASRModel(model.config)
    nn.quantize(restored, class_predicate=lambda p, m: configs.get(p, False))
    weights = mx.load(str(path))
    for target, source in aliases.items():
        weights[target] = weights[source]
    restored.load_weights(list(weights.items()), strict=True)
    restored.eval()
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    assert restored.lm_head.scales is restored.model.embed_tokens.scales
    for batch, length in [(1, 1), (4, 1), (1, 31), (2, 129)]:
        hidden = mx.random.normal((batch, length, 128)).astype(mx.float16)
        assert mx.array_equal(restored.lm_head(hidden), model.lm_head(hidden)).item()
    for length in (7, 31, 129):
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        features = mx.random.normal((1, 1, 128)).astype(mx.float16)
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        logits = [m.prefill(ids, features, positions, m.create_cache(max_seq_len=length)) for m in (model, restored)]
        assert mx.array_equal(*logits).item()
        assert mx.array_equal(mx.argmax(logits[0], axis=-1), mx.argmax(logits[1], axis=-1)).item()


def test_requantizing_q8_and_unsupported_precision_are_rejected():
    model, baseline = pair()
    with pytest.raises(ValueError, match="original unquantized"):
        quantize_mlp(baseline, 6)
    with pytest.raises(ValueError, match="6bit"):
        quantize_mlp(model, 4)


def test_invalid_tied_head_is_not_silently_aliased():
    model, _ = pair()
    quantize_mlp(model, 6)
    model.lm_head.weight = model.lm_head.weight + mx.array(1, dtype=mx.uint32)
    with pytest.raises(ValueError, match="differ"):
        checkpoint_arrays(model)
