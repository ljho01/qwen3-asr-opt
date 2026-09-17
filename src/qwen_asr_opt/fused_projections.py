"""Experimental packed QKV and gate/up projections; no requantization.

Attention follows the Apache-2.0 mlx-qwen3-asr 0.4.0 decoder contract. Only
independent linear projections of the same input are combined into one matmul.
"""
from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx_qwen3_asr.attention import _scaled_dot_product_attention
from mlx_qwen3_asr.mrope import apply_rotary_pos_emb


class PackedQuantizedLinear(nn.Module):
    """Concatenate output rows of already-quantized affine matrices exactly."""

    def __init__(self, linears):
        super().__init__()
        if not linears or not all(isinstance(layer, nn.QuantizedLinear) for layer in linears):
            raise ValueError("Require existing quantized linears")
        first = linears[0]
        expected = (first.group_size, first.bits, first.mode, first.weight.shape[1],
                    first.weight.dtype, first.scales.dtype, "bias" in first)
        if first.mode != "affine" or any(
            (layer.group_size, layer.bits, layer.mode, layer.weight.shape[1],
             layer.weight.dtype, layer.scales.dtype, "bias" in layer) != expected
            or layer.get("biases") is None for layer in linears
        ):
            raise ValueError("Projection quantization, dtype, input size and bias presence must match")
        self.group_size, self.bits, self.mode = first.group_size, first.bits, first.mode
        self.output_sizes = tuple(layer.weight.shape[0] for layer in linears)
        self.weight = mx.concatenate([layer.weight for layer in linears], axis=0)
        self.scales = mx.concatenate([layer.scales for layer in linears], axis=0)
        self.biases = mx.concatenate([layer.biases for layer in linears], axis=0)
        if "bias" in first:
            self.bias = mx.concatenate([layer.bias for layer in linears], axis=0)
        self.freeze()

    def __call__(self, x):
        result = mx.quantized_matmul(x, self.weight, scales=self.scales, biases=self.biases,
                                     transpose=True, group_size=self.group_size,
                                     bits=self.bits, mode=self.mode)
        if "bias" in self:
            result = result + self.bias
        return result


class FusedTextAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            source.num_heads, source.num_kv_heads, source.head_dim)
        self.qkv_proj = PackedQuantizedLinear([source.q_proj, source.k_proj, source.v_proj])
        self.o_proj, self.q_norm, self.k_norm = source.o_proj, source.q_norm, source.k_norm

    def __call__(self, x, cos, sin, mask=None, cache=None, layer_idx=0):
        batch, length, _ = x.shape
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        q, k, v = mx.split(self.qkv_proj(x), [q_width, q_width + kv_width], axis=-1)
        q = self.q_norm(q.reshape(batch, length, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(batch, length, self.num_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)
        output = _scaled_dot_product_attention(q, k, v, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class FusedSwiGLU(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.gate_up_proj = PackedQuantizedLinear([source.gate_proj, source.up_proj])
        self.down_proj = source.down_proj

    def __call__(self, x):
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


def fuse_decoder_projections(model, mode="both"):
    """One-time inference-only repacking; original checkpoint files stay unchanged.

    Construct every replacement before publishing any, then invalidate compiled
    callables that may have captured the old module graph. Reload to change modes.
    """
    if mode not in ("off", "qkv", "gate_up", "both"):
        raise ValueError("Fusion mode must be off, qkv, gate_up or both")
    if hasattr(model, "_fused_projection_mode"):
        raise ValueError("Reload the checkpoint before changing projection fusion")
    if mode == "off":
        return {"mode": mode, "qkv_layers": 0, "gate_up_layers": 0, "repacked_bytes": 0}
    if hasattr(model, "_dense_prefill_modules"):
        raise ValueError("Projection fusion is not combined with experimental dense prefill")
    replacements, arrays = [], []
    for layer in model.model.layers:
        attn = FusedTextAttention(layer.self_attn) if mode in ("qkv", "both") else None
        mlp = FusedSwiGLU(layer.mlp) if mode in ("gate_up", "both") else None
        replacements.append((layer, attn, mlp))
        for module in (attn.qkv_proj if attn is not None else None,
                       mlp.gate_up_proj if mlp is not None else None):
            if module is not None:
                arrays.extend([module.weight, module.scales, module.biases])
                if "bias" in module:
                    arrays.append(module.bias)
    mx.eval(arrays)
    mx.synchronize()
    for layer, attn, mlp in replacements:
        if attn is not None:
            layer.self_attn = attn
        if mlp is not None:
            layer.mlp = mlp
    for name in ("_continuous_steps", "_optimized_step", "_optimized_batch_step", "_speculative_verifiers"):
        if hasattr(model, name):
            delattr(model, name)
    model._fused_projection_mode = mode
    model.eval()
    return {"mode": mode, "qkv_layers": sum(a is not None for _, a, _ in replacements),
            "gate_up_layers": sum(m is not None for _, _, m in replacements),
            "repacked_bytes": sum(array.nbytes for array in arrays), "requantized": False}
