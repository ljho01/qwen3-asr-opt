"""Experimental q8/group64 MLP matrix tiles using MLX's MIT Metal helpers.

Not installed into the CLI or default model. Weights and quantization are unchanged.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

TILES = ((32, 32, 32), (32, 64, 32), (64, 32, 32), (32, 32, 64),
         (64, 64, 64), (64, 32, 128), (64, 64, 128), (128, 64, 64))


@lru_cache(maxsize=1)
def _kernel():
    if mx.__version__ != "0.32.2":
        raise RuntimeError("Experimental helper requires MLX 0.32.2")
    header = (Path(__file__).with_name("native") / "mlx_qmm_v0322.h").read_text()
    source = """
        constexpr int BK_padded = BK + 8;
        threadgroup half Xs[BM * BK_padded];
        threadgroup half Ws[BN * BK_padded];
        qmm_t_impl<half, 64, 8, true, BM, BK, BN>(
            W, scales, biases, X, Y, Xs, Ws,
            X_shape[1], W_shape[0], X_shape[0], X_shape[1],
            threadgroup_position_in_grid, thread_index_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    """
    return mx.fast.metal_kernel("q8_mlp_tiles", ["X", "W", "scales", "biases"],
                                ["Y"], source, header=header,
                                compile_options={"math_mode": "safe"})


def q8_tiled_matmul(x, quantized, tile=(64, 64, 64)):
    """Use row-contiguous fp16 inputs and affine q8 weights; return original shape.

    Tile order is (BM, BK, BN). Only aligned N/K and positive matrix dimensions
    are supported. The upstream safe store handles partial M tiles.
    """
    if tile not in TILES:
        raise ValueError("Unsupported experimental tile")
    q = quantized
    if (q.bits, q.group_size, q.mode) != (8, 64, "affine"):
        raise ValueError("Require affine q8/group64")
    if x.ndim < 2 or x.dtype != mx.float16 or q.scales.dtype != mx.float16 or q.biases.dtype != mx.float16:
        raise ValueError("Require fp16 matrix input and affine parameters")
    k, n = x.shape[-1], q.weight.shape[0]
    bm, bk, bn = tile
    if k < 64 or k % 64 or n < bn or n % bn or q.weight.shape[1] * 4 != k:
        raise ValueError("Require aligned and matching matrix dimensions")
    flat = x.reshape(-1, k)
    m = flat.shape[0]
    if m == 0:
        raise ValueError("Require positive input length")
    result = _kernel()(
        inputs=[flat, q.weight, q.scales, q.biases],
        template=[("BM", bm), ("BK", bk), ("BN", bn)],
        grid=(n // bn * 32, (m + bm - 1) // bm * 2, 2), threadgroup=(32, 2, 2),
        output_shapes=[(m, n)], output_dtypes=[mx.float16],
    )[0].reshape(*x.shape[:-1], n)
    return result + q.bias if "bias" in q else result
