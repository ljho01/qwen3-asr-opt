import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from qwen_asr_opt.approximate_head import ApproximateHead


@pytest.mark.parametrize("bits,columns", [(4, [1, 7, 22, 67]), (6, [])])
def test_fp16_checkpoint_head_and_fp32_residual_match_explicit_math(bits, columns):
    mx.random.seed(247)
    linear = nn.Linear(128, 256, bias=False)
    linear.set_dtype(mx.float16)
    original = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
    assert original.scales.dtype == mx.float16
    head = ApproximateHead.from_q8(original, bits=bits, columns=columns)
    x = mx.random.normal((4, 1, 128)).astype(mx.float16)
    coarse = np.array(mx.quantized_matmul(x, head.weight, head.scales, head.biases, group_size=64, bits=bits))
    expected = coarse
    if columns:
        w8 = np.array(mx.dequantize(original.weight, original.scales, original.biases, group_size=64, bits=8)).astype(np.float32)
        wlow = np.array(mx.dequantize(head.weight, head.scales, head.biases, group_size=64, bits=bits)).astype(np.float32)
        residual = w8[:, columns]-wlow[:, columns]
        np.testing.assert_array_equal(np.array(head.correction), residual)
        expected = coarse.astype(np.float32)+np.array(x)[..., columns].astype(np.float32) @ residual.T
    for function in (head, mx.compile(head)):
        np.testing.assert_allclose(np.array(function(x)), expected, atol=2e-5, rtol=2e-5)
