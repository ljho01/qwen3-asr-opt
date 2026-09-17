import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from test_compiled_prefill import setup_model

from qwen_asr_opt.approximate_head import ApproximateHead, install_fresh_head


@pytest.mark.parametrize("bits,columns", [(4, [3, 7, 65]), (4, []), (6, [])])
@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 128)])
def test_head_matches_qmm_plus_explicit_residual_without_changing_q8(bits, columns, shape):
    mx.random.seed(81)
    source = nn.QuantizedLinear.from_linear(nn.Linear(128, 256, bias=False), group_size=64, bits=8)
    before = [np.array(source[k]) for k in ("weight", "scales", "biases")]
    head = ApproximateHead.from_q8(source, bits=bits, columns=columns)
    x = mx.random.normal(shape).astype(source.scales.dtype)
    qlow = mx.quantized_matmul(x, head.weight, head.scales, head.biases, group_size=64, bits=bits)
    w8 = np.array(mx.dequantize(source.weight, source.scales, source.biases, group_size=64, bits=8))
    wlow = np.array(mx.dequantize(head.weight, head.scales, head.biases, group_size=64, bits=bits))
    if columns:
        residual = w8[:, columns].astype(np.float32) - wlow[:, columns].astype(np.float32)
        np.testing.assert_array_equal(np.array(head.correction), residual)
        expected = np.array(qlow).astype(np.float32) + np.array(x)[..., columns].astype(np.float32) @ residual.T
    else:
        expected = np.array(qlow)
    for function in (head, mx.compile(head)):
        np.testing.assert_allclose(np.array(function(x)), expected, atol=2e-5, rtol=2e-5)
    for name, old in zip(("weight", "scales", "biases"), before, strict=True):
        np.testing.assert_array_equal(np.array(source[name]), old)


def test_install_keeps_input_embedding_and_rejects_compiled_model():
    model = setup_model()
    embedding = model.model.embed_tokens
    original = embedding(mx.array([[1, 2, 3]]))
    head = ApproximateHead.from_q8(embedding, bits=4, columns=[3, 7])
    install_fresh_head(model, head)
    assert model.model.embed_tokens is embedding
    assert mx.array_equal(original, embedding(mx.array([[1, 2, 3]]))).item()
    model._continuous_steps = {}
    with pytest.raises(ValueError, match="before decoder compilation"):
        install_fresh_head(model, head)


@pytest.mark.parametrize("columns", [[1, 1], [-1], [128], [1.5]])
def test_invalid_correction_columns_are_rejected(columns):
    original = nn.QuantizedLinear.from_linear(nn.Linear(128, 256, bias=False), group_size=64, bits=8)
    with pytest.raises(ValueError, match="columns"):
        ApproximateHead.from_q8(original, bits=4, columns=columns)
