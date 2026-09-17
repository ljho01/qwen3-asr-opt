"""Experimental q4 vocabulary head with an activation-subspace residual.

Scores are approximate q4(x)+(x@basis)@correction.T. The q8 input embedding
remains unchanged; this module has no default/CLI installation or argmax guarantee.
"""
from __future__ import annotations

import mlx.core as mx
from mlx import nn


class ProjectedHead(nn.Module):
    def __init__(self, weight, scales, biases, basis, correction):
        super().__init__()
        if weight.ndim != 2 or scales.ndim != 2 or biases.shape != scales.shape:
            raise ValueError("Require 2D affine q4 weights and matching scales/biases")
        rows, width = weight.shape[0], scales.shape[1]*64
        if weight.shape[1]*8 != width or scales.shape[0] != rows:
            raise ValueError("Packed q4/group64 dimensions disagree")
        if basis.ndim != 2 or basis.shape[0] != width or not 1 <= basis.shape[1] <= width:
            raise ValueError("Projection basis must have input-width rows and a positive rank")
        if correction.shape != (rows, basis.shape[1]) or basis.dtype != mx.float32 or correction.dtype != mx.float32:
            raise ValueError("Require matching FP32 basis and residual projection")
        self.weight, self.scales, self.biases = weight, scales, biases
        self.basis, self.correction = basis, correction

    def __call__(self, x):
        coarse = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                    group_size=64, bits=4, mode="affine", transpose=True)
        coordinates = x.astype(mx.float32) @ self.basis
        return coarse.astype(mx.float32) + coordinates @ self.correction.T
