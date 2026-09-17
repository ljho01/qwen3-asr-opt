"""Opt-in experimental full Core ML audio encoder, with MLX text decoding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn


class CoreMLAudioEncoder(nn.Module):
    def __init__(self, path, config, compute_units="CPU_AND_NE"):
        super().__init__()
        from .coreml_cache import load_cached_model
        root = Path(path)
        metadata = json.loads(root.with_suffix(".json").read_text())
        if not metadata.get("full_encoder") or metadata["window_tokens"] != 104:
            raise ValueError("Requires a full Qwen3-ASR encoder exported by this project")
        weights = root / "Data/com.apple.CoreML/weights/weight.bin"
        with weights.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != metadata["weights_sha256"]:
            raise ValueError("Core ML encoder weights failed integrity validation")
        self.config = config
        self.metadata = metadata
        self._coreml, self.load_info = load_cached_model(root, digest, compute_units)
        self.compute_units = compute_units

    def __call__(self, mel, feature_lens):
        mx.eval(mel, feature_lens)
        batch = self.metadata["batch"]
        capacity = batch * 800
        outputs, output_lengths = [], []
        for index, value in enumerate(feature_lens.tolist()):
            frames = int(value)
            if not 0 < frames <= capacity or frames > mel.shape[-1]:
                raise ValueError("Audio exceeds the Core ML static input capacity; use --long")
            padded = np.zeros((128, capacity), np.float16)
            padded[:, :frames] = np.array(mel[index, :, :frames])
            chunks = padded.reshape(128, batch * 8, 100).transpose(1, 0, 2)[:, None]
            lengths = np.clip(frames - np.arange(batch * 8) * 100, 0, 100).astype(np.float16)
            count = (frames // 100) * 13 + ((frames % 100) + 7) // 8
            encoded = self._coreml.predict({"mel_chunks": chunks, "frame_lengths": lengths})["encoded"]
            outputs.append(mx.array(encoded.reshape(-1, 2048)[:count]).astype(mx.float16))
            output_lengths.append(count)
        maximum = max(output_lengths)
        return mx.stack([mx.pad(x, [(0, maximum - length), (0, 0)])
                         for x, length in zip(outputs, output_lengths, strict=True)]), mx.array(output_lengths)


def install_coreml_encoder(session, path, compute_units="CPU_AND_NE"):
    session.model.audio_tower = CoreMLAudioEncoder(path, session.model.audio_tower.config,
                                                  compute_units=compute_units)
    mx.clear_cache()
    return {**session.model.audio_tower.metadata, "load_info": session.model.audio_tower.load_info}
