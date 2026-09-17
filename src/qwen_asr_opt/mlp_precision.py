"""Experimental MLP-only6bit decoder; MLX encoder, attention and vocabulary remainq8."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_qwen3_asr import load_model


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def is_mlp(path):
    return path.startswith("model.layers.") and path.rsplit(".", 2)[-2:] in (
        ["mlp", "gate_proj"], ["mlp", "up_proj"], ["mlp", "down_proj"])


def quantize_mlp(model, layer_bits):
    """Quantize only an unquantized source; never lower an existing q8 checkpoint."""
    if type(layer_bits) is not int or layer_bits != 6:
        raise ValueError("Experimental MLP precision must be6bit")
    modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    if any(isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)) for _, module in modules):
        raise ValueError("Require original unquantized source weights")
    configs = {}

    def predicate(path, module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        config = {"bits": layer_bits if is_mlp(path) else 8,
                  "group_size": 64, "mode": "affine"}
        configs[path] = config
        return config

    nn.quantize(model, class_predicate=predicate)
    model.eval()
    mx.eval(model.parameters())
    expected_layers = len(model.model.layers) * 3
    if sum(is_mlp(path) for path in configs) != expected_layers:
        raise ValueError("Require exactly three MLP projections per decoder layer")
    return configs


def checkpoint_arrays(model):
    """Preserve the upstream tied embedding/head storage alias after quantization."""
    weights, aliases = dict(tree_flatten(model.parameters())), {}
    if model.config.text_config.tie_word_embeddings:
        for suffix in ("weight", "scales", "biases"):
            head, embedding = f"lm_head.{suffix}", f"model.embed_tokens.{suffix}"
            if head not in weights or embedding not in weights:
                raise ValueError(f"Missing tied quantized tensor: {suffix}")
            if not mx.array_equal(weights[head], weights[embedding]).item():
                raise ValueError(f"Tied quantized tensors differ: {suffix}")
            aliases[head] = embedding
            del weights[head]
    return weights, aliases


def convert_mlp(source, destination, layer_bits, source_hashes, q8_weights):
    """Write a new compatible checkpoint and verify all protected tensors vsq8."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name + ".partial")
    if destination.exists() or temporary.exists():
        raise FileExistsError("Destination or preserved partial checkpoint already exists")
    for name, sha in source_hashes.items():
        if digest(source / name) != sha:
            raise ValueError(f"Original source file changed: {name}")
    model, _ = load_model(str(source), dtype=mx.float16)
    configs = quantize_mlp(model, layer_bits)
    arrays, aliases = checkpoint_arrays(model)
    baseline = mx.load(str(q8_weights))
    checked = []
    for name, value in arrays.items():
        if is_mlp(name.rsplit(".", 1)[0]):
            continue
        if name not in baseline or not mx.array_equal(value, baseline[name]).item():
            raise ValueError(f"Protected non-MLP tensor differs fromq8: {name}")
        checked.append(name)
    del baseline
    source_info = json.loads((source / "source.json").read_text())
    temporary.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(str(temporary / "weights.safetensors"), arrays)
    for name in ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copyfile(source / name, temporary / name)
    metadata = {"format": "qwen-asr-opt-v1", "profile": f"mlp{layer_bits}-rest8",
                "source": source_info["model"], "source_revision": source_info["revision"],
                "source_file_sha256": source_hashes, "float_dtype": "float16",
                "module_quantization": configs, "shared_tensors": aliases,
                "weights_bytes": (temporary / "weights.safetensors").stat().st_size,
                "weights_sha256": digest(temporary / "weights.safetensors"),
                "protected_tensors_exact_q8": checked,
                "policy": "Only decoder MLP gate/up/down projections use6bits; encoder, attention and head protected; all other linears/embedding8bit,group64,affine. Norms and other parameters float16. Quantized from originalFP16, notq8."}
    with (temporary / "optimization.json").open("x") as file:
        json.dump(metadata, file, indent=2)
    for name, sha in source_hashes.items():
        if digest(source / name) != sha:
            raise ValueError(f"Original source changed during conversion: {name}")
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    return metadata
