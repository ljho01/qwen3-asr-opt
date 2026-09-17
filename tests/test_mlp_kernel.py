import mlx.core as mx
import pytest
from mlx import nn

from qwen_asr_opt.mlp_kernel import TILES, q8_tiled_matmul


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("shape", [(1, 64), (31, 128), (65, 256), (2, 137, 128)])
def test_quantized_matrix_tiles_cover_tail_and_batch(shape, tile):
    mx.random.seed(20260917)
    linear = nn.Linear(shape[-1], 128, bias=True)
    linear.set_dtype(mx.float16)
    q = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
    x = mx.random.normal(shape).astype(mx.float16)
    actual = q8_tiled_matmul(x, q, tile)
    expected = q(x)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, expected, atol=.002, rtol=.002).item()


def test_strided_input_matches_contiguous_input():
    mx.random.seed(89)
    q = nn.QuantizedLinear(128, 256, group_size=64, bits=8, bias=False)
    q.set_dtype(mx.float16)
    x = mx.random.normal((129, 256)).astype(mx.float16)[:, ::2]
    assert mx.allclose(q8_tiled_matmul(x, q), q(x), atol=.004, rtol=.004).item()


def test_wrong_precision_and_empty_input_fail_before_gpu_dispatch():
    q = nn.QuantizedLinear(128, 128, group_size=64, bits=8, bias=False)
    q.set_dtype(mx.float16)
    with pytest.raises(ValueError, match="fp16"):
        q8_tiled_matmul(mx.zeros((32, 128)), q)
    with pytest.raises(ValueError, match="positive"):
        q8_tiled_matmul(mx.zeros((0, 128), dtype=mx.float16), q)
    with pytest.raises(ValueError, match="Unsupported"):
        q8_tiled_matmul(mx.zeros((32, 128), dtype=mx.float16), q, (256, 64, 64))
