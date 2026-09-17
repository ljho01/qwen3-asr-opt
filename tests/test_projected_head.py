import mlx.core as mx
import numpy as np
import pytest

from qwen_asr_opt.projected_head import ProjectedHead


@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 128)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_projected_head_matches_low_rank_residual_and_compiled_path(shape, dtype):
    mx.random.seed(278)
    weights = mx.random.normal((256, 128)).astype(dtype)
    packed = mx.quantize(weights, group_size=64, bits=4)
    basis, _ = np.linalg.qr(np.random.default_rng(81).normal(size=(128, 8)))
    basis = basis.astype(np.float32)
    low = np.array(mx.dequantize(*packed, group_size=64, bits=4)).astype(np.float64)
    correction = ((np.array(weights).astype(np.float64)-low) @ basis.astype(np.float64)).astype(np.float32)
    head = ProjectedHead(*packed, mx.array(basis), mx.array(correction))
    x = mx.random.normal(shape).astype(dtype)
    coarse = mx.quantized_matmul(x, *packed, group_size=64, bits=4)
    expected = np.array(coarse).astype(np.float32)+(np.array(x).astype(np.float32) @ basis) @ correction.T
    for fn in (head, mx.compile(head)):
        np.testing.assert_allclose(np.array(fn(x)), expected, atol=3e-5, rtol=3e-5)


def test_axis_basis_equals_explicit_column_correction():
    mx.random.seed(379)
    weights = mx.random.normal((256, 128)).astype(mx.float16)
    packed = mx.quantize(weights, group_size=64, bits=4)
    columns = [7, 3, 91]
    basis = mx.eye(128)[:, mx.array(columns)]
    residual = weights.astype(mx.float32)-mx.dequantize(*packed, group_size=64, bits=4).astype(mx.float32)
    correction = residual[:, mx.array(columns)]
    head = ProjectedHead(*packed, basis, correction)
    x = mx.random.normal((4, 1, 128)).astype(mx.float16)
    expected = mx.quantized_matmul(x, *packed, group_size=64, bits=4).astype(mx.float32)+x[..., mx.array(columns)].astype(mx.float32) @ correction.T
    np.testing.assert_allclose(np.array(head(x)), np.array(expected), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("basis_shape,correction_shape", [((64, 8), (256, 8)), ((128, 0), (256, 0)), ((128, 8), (255, 8))])
def test_projection_shapes_reject_incompatible_tensors(basis_shape, correction_shape):
    weights = mx.zeros((256, 128), dtype=mx.float16)
    packed = mx.quantize(weights, group_size=64, bits=4)
    with pytest.raises(ValueError):
        ProjectedHead(*packed, mx.zeros(basis_shape), mx.zeros(correction_shape))
