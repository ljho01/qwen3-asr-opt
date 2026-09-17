import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx_qwen3_asr.model import Qwen3ASRModel
from test_speculative import small_model_config

from qwen_asr_opt.scaled_mlp import ROLES, ScaledMLPLinear, activation_scales, install_scaled_mlp


@pytest.mark.parametrize("alpha", [0, .25, .5, .75, 1])
def test_scaled_quantized_linear_matches_its_reconstructed_matrix(alpha):
    mx.random.seed(717)
    original = nn.Linear(128, 64, bias=False)
    original.set_dtype(mx.float16)
    original = nn.QuantizedLinear.from_linear(original, group_size=64, bits=8)
    x = mx.random.normal((2, 7, 128)).astype(mx.float16)
    scale = activation_scales(np.asarray(x).reshape(-1, 128), alpha)
    candidate = ScaledMLPLinear.from_q8(original, scale)
    dense = mx.dequantize(candidate.weight, candidate.scales, candidate.biases, group_size=64, bits=6)
    expected = (x / candidate.input_scale) @ dense.T
    assert mx.allclose(candidate(x), expected, atol=.004, rtol=.004).item()
    assert mx.allclose(mx.compile(candidate)(x), candidate(x), atol=.004, rtol=.004).item()
    assert candidate(x).dtype == mx.float16
    assert np.isfinite(scale).all() and (scale > 0).all()
    if alpha == 0:
        assert np.array_equal(scale, np.ones(128, dtype=np.float16))
    assert "original" not in candidate


@pytest.mark.parametrize("samples,alpha", [(np.empty((0, 64)), .5), (np.array([[np.nan]]), .5),
    (np.ones(64), .5), (np.ones((1, 64)), .3)])
def test_scale_calibration_validation(samples, alpha):
    with pytest.raises(ValueError):
        activation_scales(samples, alpha)


def test_zero_channels_and_extreme_activations_remain_finite():
    data = np.array([[0, 1e-30, 1, 1e30]], dtype=np.float64)
    for alpha in (0, .5, 1):
        scale = activation_scales(data, alpha)
        assert scale.dtype == np.float16 and (scale >= 1/256).all() and (scale <= 256).all()
        assert np.isfinite(1/scale).all()


def test_installer_protects_other_modules_and_rejects_partial_and_stale_replacements():
    model = Qwen3ASRModel(small_model_config())
    model.set_dtype(mx.float16)
    nn.quantize(model.model, group_size=64, bits=8)
    saved = model.model.layers[0].mlp.gate_proj
    embedding, attention, encoder = model.model.embed_tokens, model.model.layers[0].self_attn, model.audio_tower
    arrays = {}
    for index, layer in enumerate(model.model.layers):
        for role in ROLES:
            module = getattr(layer.mlp, role)
            candidate = ScaledMLPLinear.from_q8(module, np.ones(module.scales.shape[1]*64, np.float16))
            arrays.update({f"{index}.{role}.{key}": value for key, value in candidate.parameters().items()})
    bad = dict(arrays)
    bad.pop(next(reversed(bad)))
    with pytest.raises(KeyError):
        install_scaled_mlp(model, bad)
    assert model.model.layers[0].mlp.gate_proj is saved
    install_scaled_mlp(model, arrays)
    assert model.model.embed_tokens is embedding and model.audio_tower is encoder
    assert model.model.layers[0].self_attn is attention
    model._continuous_steps = {}
    with pytest.raises(ValueError, match="compilation"):
        install_scaled_mlp(model, arrays)


def test_invalid_input_scale_rejected_before_quantization():
    source = nn.QuantizedLinear.from_linear(nn.Linear(128, 64, bias=False), group_size=64, bits=8)
    for scale in (np.zeros(128, np.float16), np.ones(64, np.float16), np.ones(128, np.float32)):
        with pytest.raises(ValueError):
            ScaledMLPLinear.from_q8(source, scale)
