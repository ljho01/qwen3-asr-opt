"""Experimental q4 shortlist followed by original q8 weight rescoring.

Candidates outside the shortlist receive -inf. This is approximate: q4 can omit
the true q8 winner, and different q8 matmul shapes can round differently. No
default/CLI loads this module. Install only before compiling a fresh decoder.
"""
from __future__ import annotations

import mlx.core as mx
from mlx import nn


class ShortlistHead(nn.Module):
    def __init__(self, q8, q4, *, candidates=64, method="shared"):
        super().__init__()
        if method not in ("take", "gather", "shared"):
            raise ValueError("Unknown q8 rescore method")
        if len(q8) != 3 or len(q4) != 3:
            raise ValueError("Require affine weight/scale/bias triples")
        n, width = q8[0].shape[0], q8[0].shape[1] * 4
        if type(candidates) is not int or not 1 <= candidates <= n:
            raise ValueError("Candidate count must be between 1 and vocabulary size")
        for values, bits in ((q8, 8), (q4, 4)):
            weight, scales, biases = values
            if (weight.shape != (n, width * bits // 32)
                    or scales.shape != (n, width // 64) or biases.shape != scales.shape
                    or weight.dtype != mx.uint32):
                raise ValueError("Incompatible affine group64 parameters")
        self.weight8, self.scales8, self.biases8 = q8
        self.weight4, self.scales4, self.biases4 = q4
        self.candidates, self.method = candidates, method

    def select(self, x):
        flat = x.reshape(-1, x.shape[-1])
        coarse = mx.quantized_matmul(flat, self.weight4, self.scales4, self.biases4,
                                    bits=4, group_size=64, mode="affine")
        n = coarse.shape[-1]
        return mx.argpartition(coarse, n - self.candidates, axis=-1)[:, -self.candidates:]

    def rescore(self, x, indices):
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        if self.method == "take":
            scores = mx.quantized_matmul(
                flat[:, None, :], self.weight8[indices], self.scales8[indices],
                self.biases8[indices], bits=8, group_size=64, mode="affine",
            ).squeeze(-2)
        elif self.method == "gather":
            scores = mx.gather_qmm(
                flat[:, None, :], self.weight8[:, None, :], self.scales8[:, None, :],
                self.biases8[:, None, :], lhs_indices=mx.arange(rows, dtype=mx.uint32)[:, None],
                rhs_indices=indices, bits=8, group_size=64, mode="affine",
            ).reshape(indices.shape)
        else:
            # A fixed-size pool preserves the baseline's M dimension. Repeated
            # vocabulary rows are harmless; no data-dependent unique/host sync.
            pool = indices.reshape(-1)
            scores = mx.quantized_matmul(
                flat, self.weight8[pool], self.scales8[pool], self.biases8[pool],
                bits=8, group_size=64, mode="affine",
            )
            indices = mx.broadcast_to(pool[None, :], scores.shape)
        return indices, scores

    def __call__(self, x):
        indices, scores = self.rescore(x, self.select(x))
        full = mx.full((indices.shape[0], self.weight8.shape[0]), -mx.inf, dtype=scores.dtype)
        full = mx.put_along_axis(full, indices, scores, axis=-1)
        return full.reshape(*x.shape[:-1], self.weight8.shape[0])
