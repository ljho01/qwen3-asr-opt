"""Local speech-pause segmentation that preserves every original audio sample.

The ONNX state/input contract follows Silero VAD v6.2 (MIT). Its license is retained
in third_party/SILERO_LICENSE.txt and with the pinned model. No Torch dependency.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .longform import SAMPLE_RATE, audio_chunks, split_buffer


class SileroVad16k:
    """One CPU thread, 512 new samples plus 64 samples of recurrent context."""

    frame_samples = 512

    def __init__(self, directory):
        import onnxruntime as ort

        directory = Path(directory)
        provenance = json.loads((directory / "source.json").read_text())
        metadata = next(row for row in provenance["files"] if row["local_name"] == "silero_vad.onnx")
        model = directory / "silero_vad.onnx"
        if hashlib.sha256(model.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Silero model integrity mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.provenance = provenance
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.calls = 0

    def probability(self, frame):
        if frame.shape != (512,):
            raise ValueError("VAD requires one 512-sample mono frame at16kHz")
        inputs = np.concatenate((self.context, frame[None, :]), axis=1).astype(np.float32)
        output, self.state = self.session.run(None, {"input": inputs, "state": self.state,
                                                     "sr": np.array(16000, dtype=np.int64)})
        self.context = inputs[:, -64:]
        self.calls += 1
        value = float(output[0, 0])
        if not math.isfinite(value):
            raise ValueError("VAD returned a non-finite speech probability")
        return value


class PauseDetector:
    """Emit one midpoint after a sufficiently long low-probability speech pause."""

    def __init__(self, vad, silence_ms=250):
        if not 32 <= silence_ms <= 2000:
            raise ValueError("silence_ms must be32..2000")
        self.vad = vad
        self.vad.reset()
        self.required = math.ceil(silence_ms / 32) * 512
        self.carry = np.empty(0, dtype=np.float32)
        self.position = 0
        self.heard_speech = False
        self.quiet_start = None

    def push(self, wave, final=False):
        joined = np.concatenate((self.carry, wave))
        count = len(joined) // 512
        if final and len(joined) % 512:
            count += 1
        pauses = []
        for index in range(count):
            frame = joined[index * 512:(index + 1) * 512]
            if len(frame) < 512:
                frame = np.pad(frame, (0, 512 - len(frame)))
            probability = self.vad.probability(frame)
            if probability >= 0.5:
                self.heard_speech = True
                self.quiet_start = None
            elif probability >= 0.35:
                self.quiet_start = None
            elif self.heard_speech:
                if self.quiet_start is None:
                    self.quiet_start = self.position
                if self.position + 512 - self.quiet_start >= self.required:
                    pauses.append(self.quiet_start + self.required // 2)
                    self.heard_speech = False
                    self.quiet_start = None
            self.position += 512
        self.carry = joined[min(count * 512, len(joined)):].copy()
        return pauses


def speech_chunks(path, vad, *, silence_ms=250, min_chunk_s=3.0, max_chunk_s=30.0):
    """Stream all audio through pause cuts, falling back to the existing low RMS cut.

    Speech probabilities choose boundaries only: no silence or low-volume audio is
    dropped. At most two coarse30s PCM blocks plus one pending output are retained.
    The VAD state spans ASR chunk boundaries and is reset for each source file.
    """
    if not 1 <= min_chunk_s < max_chunk_s <= 30:
        raise ValueError("Require1 <= min_chunk_s < max_chunk_s <=30")
    detector = PauseDetector(vad, silence_ms)
    minimum, maximum = round(min_chunk_s * SAMPLE_RATE), round(max_chunk_s * SAMPLE_RATE)
    source = iter(audio_chunks(path))
    buffer = np.empty(0, dtype=np.float32)
    base, total = 0, 0
    pauses = []
    pending = None

    def ready(final=False):
        nonlocal base, buffer, pauses
        while len(buffer):
            pauses = [p for p in pauses if p >= base + minimum and p < total]
            within = [p for p in pauses if p <= base + maximum]
            if within:
                cut = within[0] - base
            elif len(buffer) > maximum and (final or detector.position >= base + maximum + detector.required):
                prefix, _ = split_buffer(buffer, chunk_s=max_chunk_s,
                                          search_s=min(2.0, max_chunk_s - 1))
                cut = len(prefix)
            elif final:
                cut = len(buffer)
            else:
                break
            if cut > maximum:
                raise ValueError("Segmentation exceeded model duration limit")
            wave, offset = buffer[:cut], base / SAMPLE_RATE
            buffer = buffer[cut:]
            base += cut
            yield wave, offset

    try:
        for wave, offset in source:
            if round(offset * SAMPLE_RATE) != total:
                raise ValueError("Noncontiguous decoded source")
            total += len(wave)
            buffer = np.concatenate((buffer, wave))
            pauses.extend(detector.push(wave))
            for current in ready():
                if pending is not None:
                    yield pending
                pending = current
        pauses.extend(detector.push(np.empty(0, dtype=np.float32), final=True))
        for current in ready(final=True):
            if pending is not None:
                if len(current[0]) < SAMPLE_RATE and len(pending[0]) + len(current[0]) <= maximum:
                    pending = np.concatenate((pending[0], current[0])), pending[1]
                    continue
                yield pending
            pending = current
        if pending is not None:
            yield pending
    finally:
        source.close()
