"""Pinned MLX backend with explicit mixed-precision checkpoint metadata."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_qwen3_asr import Session, load_model
from mlx_qwen3_asr.config import Qwen3ASRConfig
from mlx_qwen3_asr.model import Qwen3ASRModel

PROFILES = {
    "q8": (8, 8),
    "q4": (4, 4),
    "q5": (5, 5),
    "mixed8_4": (8, 4),
    "mixed8_5": (8, 5),
}


def convert(source: Path, dest: Path, profile: str, group_size: int = 64) -> dict:
    if dest.exists():
        raise FileExistsError(f"Checkpoint already exists: {dest}")
    model, _ = load_model(str(source), dtype=mx.float16)
    encoder_bits, decoder_bits = PROFILES[profile]
    module_quantization = {}

    def predicate(path, module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        bits = encoder_bits if path.startswith("audio_tower.") else decoder_bits
        cfg = {"bits": bits, "group_size": group_size, "mode": "affine"}
        module_quantization[path] = cfg
        return cfg

    nn.quantize(model, class_predicate=predicate)
    mx.eval(model.parameters())
    # Write into a temporary sibling and atomically expose the completed checkpoint.
    temp = dest.with_name(dest.name + ".partial")
    temp.mkdir(parents=True, exist_ok=False)
    weights = dict(tree_flatten(model.parameters()))
    aliases = {}
    # Qwen ties input embeddings and the vocabulary projection. Preserve that
    # sharing after quantization instead of storing/loading a second large matrix.
    if model.config.text_config.tie_word_embeddings:
        for suffix in ("weight", "scales", "biases"):
            head = f"lm_head.{suffix}"
            embedding = f"model.embed_tokens.{suffix}"
            if head in weights and embedding in weights:
                if not bool(mx.array_equal(weights[head], weights[embedding]).item()):
                    raise ValueError(f"Tied quantized tensors differ: {head}")
                aliases[head] = embedding
                del weights[head]
    mx.save_safetensors(str(temp / "weights.safetensors"), weights)
    for name in ["config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        shutil.copyfile(source / name, temp / name)
    source_metadata = source / "source.json"
    source_info = json.loads(source_metadata.read_text()) if source_metadata.exists() else {}
    revision = source_info.get("revision", source.resolve().name)
    with (temp / "weights.safetensors").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    metadata = {
        "format": "qwen-asr-opt-v1", "profile": profile,
        "source": source_info.get("model", "Qwen/Qwen3-ASR-1.7B"),
        "source_revision": revision,
        "float_dtype": "float16", "module_quantization": module_quantization,
        "weights_bytes": (temp / "weights.safetensors").stat().st_size,
        "weights_sha256": digest,
        "shared_tensors": aliases,
    }
    (temp / "optimization.json").write_text(json.dumps(metadata, indent=2))
    temp.rename(dest)
    return metadata


def load_session(path: str, dtype: str = "float16") -> Session:
    root = Path(path)
    meta_path = root / "optimization.json"
    if not meta_path.exists():
        return Session(model=path, dtype=getattr(mx, dtype))
    metadata = json.loads(meta_path.read_text())
    config = Qwen3ASRConfig.from_dict(json.loads((root / "config.json").read_text()))
    model = Qwen3ASRModel(config)
    nn.quantize(model, class_predicate=lambda p, m: metadata["module_quantization"].get(p, False))
    weights = mx.load(str(root / "weights.safetensors"))
    for target, source in metadata.get("shared_tensors", {}).items():
        weights[target] = weights[source]
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    # Upstream tail refinement resolves the tokenizer through model origin
    # metadata. Preserve the local converted checkpoint path so finalization
    # never falls back to the default remote model.
    model._resolved_model_path = str(root.resolve())
    model._source_model_id = metadata.get("source")
    # Quantized scales and non-quantized weights were saved as float16.
    if dtype != metadata["float_dtype"]:
        raise ValueError("Converted checkpoint requires dtype=float16")
    return Session(model=model, tokenizer_model=path, dtype=mx.float16)
