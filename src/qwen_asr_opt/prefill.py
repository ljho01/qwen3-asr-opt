"""Use dense GEMM for many-token prefill and quantized GEMV for single-token decode."""
from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten


class DensePrefillLinear(nn.Module):
    def __init__(self, quantized):
        super().__init__()
        self.quantized = quantized
        self.mode = "off"
        self._dense_weight = None

    def unpack(self):
        q = self.quantized
        return mx.dequantize(q.weight, q.scales, q.get("biases"),
                             group_size=q.group_size, bits=q.bits, mode=q.mode)

    def __call__(self, x):
        if self.mode == "off" or x.ndim < 2 or x.shape[-2] == 1:
            return self.quantized(x)
        weight = self._dense_weight if self.mode == "cached" else self.unpack()
        y = x @ weight.T
        if "bias" in self.quantized:
            y = y + self.quantized.bias
        return y


def configure_dense_prefill(model, mode):
    """Only wrap decoder layer linears; embedding and vocabulary head stay quantized.

    FP16 weights are reconstructed from the SAME quantized checkpoint, not loaded
    from the original FP16 checkpoint. Rounding may still affect prefill activations.
    Transient mode reconstructs on demand, cached mode retains an additional copy.
    """
    if mode not in ("off", "transient", "cached"):
        raise ValueError("Dense prefill mode must be off, transient, or cached")
    if mode == "off" and not hasattr(model, "_dense_prefill_modules"):
        return {"mode": mode, "modules": 0, "retained_dense_bytes": 0}
    if not hasattr(model, "_dense_prefill_modules"):
        replacements = []
        for path, module in tree_flatten(model.model.leaf_modules(), is_leaf=nn.Module.is_module):
            if path.startswith("layers.") and isinstance(module, nn.QuantizedLinear):
                replacements.append((path, DensePrefillLinear(module)))
        model.model.update_modules(tree_unflatten(replacements))
        model._dense_prefill_modules = [module for _, module in replacements]
    for module in model._dense_prefill_modules:
        module.mode = mode
        if mode == "cached" and module._dense_weight is None:
            module._dense_weight = module.unpack()
        elif mode != "cached":
            module._dense_weight = None
    dense = [module._dense_weight for module in model._dense_prefill_modules
             if module._dense_weight is not None]
    mx.eval(dense)
    return {"mode": mode, "modules": len(model._dense_prefill_modules),
            "retained_dense_bytes": sum(weight.nbytes for weight in dense)}
