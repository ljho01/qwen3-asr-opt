"""Bounded-memory decoding of long recordings into contiguous, energy-aware chunks."""
from __future__ import annotations

import itertools
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from mlx_qwen3_asr.tokenizer import join_text_parts

SAMPLE_RATE = 16000


def split_buffer(samples, *, final=False, chunk_s=30.0, search_s=2.0):
    """Return (ready prefix, remaining suffix); every sample belongs to exactly one chunk."""
    capacity = round(chunk_s * SAMPLE_RATE)
    if capacity < SAMPLE_RATE or search_s < 0 or search_s >= chunk_s:
        raise ValueError("chunk_s >= 1 and 0 <= search_s < chunk_s required")
    if len(samples) < capacity:
        return (samples, samples[:0]) if final else (samples[:0], samples)
    # Search at most the final two seconds, with 20ms energy windows and 10ms stride.
    # No silence skipping: low-volume speech is still transcribed.
    lo = max(0, capacity - round(search_s * SAMPLE_RATE))
    window = 320
    candidates = np.arange(lo, capacity - window + 1, 160)
    if len(candidates):
        energy = np.array([np.mean(np.square(samples[i:i + window])) for i in candidates])
        # Prefer the latest equal-energy candidate to keep silence chunks near maximum size.
        index = len(energy) - 1 - int(np.argmin(energy[::-1]))
        cut = int(candidates[index] + window // 2)
    else:
        cut = capacity
    return samples[:cut], samples[cut:]


def audio_chunks(path: str, chunk_s=30.0):
    """Decode arbitrary FFmpeg-supported audio without loading the full recording."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(Path(path).resolve()),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=errors,
        )
        buffer = np.empty(0, dtype=np.float32)
        sample_offset = 0
        remainder = b""
        try:
            while raw := process.stdout.read(65536):
                raw = remainder + raw
                aligned = len(raw) // 4 * 4
                remainder = raw[aligned:]
                buffer = np.concatenate([buffer, np.frombuffer(raw[:aligned], dtype="<f4")])
                while len(buffer) >= round(chunk_s * SAMPLE_RATE):
                    ready, buffer = split_buffer(buffer, chunk_s=chunk_s)
                    yield ready, sample_offset / SAMPLE_RATE
                    sample_offset += len(ready)
            code = process.wait()
            if code or remainder:
                errors.seek(0)
                raise RuntimeError(f"FFmpeg decode failed ({code}): {errors.read().decode()}")
            if len(buffer):
                yield buffer, sample_offset / SAMPLE_RATE
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def transcribe_long(session, audio: str, output: Path, language=None, chunk_s=30.0, batch_size=1,
                    scheduler="fixed", segmenter="energy", vad_model=None, audio_prefetch=0,
                    kv_policy="padded"):
    """Persist completed chunks immediately; return a final transcript and exact chunk spans."""
    if any(p.exists() for p in [output, output.with_suffix(".chunks.jsonl"),
                               output.with_suffix(".txt")]):
        raise FileExistsError(f"Refusing to overwrite transcription: {output}")
    if not 1 <= batch_size <= 8:
        raise ValueError("batch_size must be 1..8")
    if scheduler not in ("fixed", "continuous"):
        raise ValueError("scheduler must be fixed or continuous")
    if kv_policy not in ("padded", "growing128"):
        raise ValueError("kv_policy must be padded or growing128")
    if kv_policy != "padded" and scheduler != "continuous":
        raise ValueError("Growing KV cache requires continuous scheduling")
    if segmenter not in ("energy", "vad"):
        raise ValueError("segmenter must be energy or vad")
    if segmenter == "vad" and not 8 < chunk_s <= 30:
        raise ValueError("The validated VAD profile requires8 < chunk_s <=30")
    if not 0 <= audio_prefetch <= 8:
        raise ValueError("audio_prefetch must be0..8")
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    started = time.perf_counter()
    segmenter_metadata = {"name": segmenter}
    if segmenter == "vad":
        import onnxruntime as ort

        from .vad import SileroVad16k, speech_chunks
        directory = vad_model or Path(__file__).resolve().parents[2] / "models/vad-silero-v6.2"
        detector = SileroVad16k(directory)
        stream = iter(speech_chunks(audio, detector, silence_ms=500, min_chunk_s=8,
                                    max_chunk_s=chunk_s))
        model_file = next(row for row in detector.provenance["files"] if row["local_name"] == "silero_vad.onnx")
        segmenter_metadata.update(silence_ms=500, effective_silence_ms=512, min_chunk_s=8,
                                  model_revision=detector.provenance["revision"],
                                  model_sha256=model_file["sha256"], onnxruntime=ort.__version__,
                                  provider="CPUExecutionProvider", threads=1,
                                  audio_samples_dropped=0)
    else:
        stream = iter(audio_chunks(audio, chunk_s))
    if audio_prefetch:
        from .pipeline import PrefetchChunks
        stream = PrefetchChunks(stream, audio_prefetch)

    def persist(row):
        chunks.append(row)
        with output.with_suffix(".chunks.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    scheduler_stats = None
    try:
        if scheduler == "continuous":
            from .continuous import transcribe_continuous_chunks
            scheduler_stats = transcribe_continuous_chunks(session, stream, language,
                                                           batch_size, persist, kv_policy=kv_policy)
        else:
            while group := list(itertools.islice(stream, batch_size)):
                begin = time.perf_counter()
                if batch_size == 1:
                    results = [session.transcribe(group[0][0], language=language, return_chunks=True)]
                else:
                    from .batch import transcribe_batch
                    results = transcribe_batch(session, [wave for wave, _ in group],
                                               [language] * len(group))
                elapsed = time.perf_counter() - begin
                group_samples = sum(len(wave) for wave, _ in group)
                for (wave, offset), result in zip(group, results, strict=True):
                    persist({"index": len(chunks), "start": offset,
                             "end": offset + len(wave) / SAMPLE_RATE,
                             "text": result.text, "language": result.language,
                             "elapsed_s": elapsed * len(wave) / group_samples,
                             "elapsed_scope": "single_chunk" if batch_size == 1 else "batch_allocated",
                             "truncated": result.truncated, "finish_reason": result.finish_reason})
    finally:
        stream.close()
    if not chunks:
        raise ValueError("Input contains no decodable audio")
    chunks.sort(key=lambda row: row["index"])
    final = {
        "text": join_text_parts([r["text"] for r in chunks], language or "unknown"),
        "chunks": chunks, "audio_s": chunks[-1]["end"],
        "elapsed_s": time.perf_counter() - started,
        "truncated": any(r["truncated"] for r in chunks),
        "language_hint": language, "chunk_s": chunk_s, "batch_size": batch_size,
        "scheduler": scheduler, "scheduler_stats": scheduler_stats,
        "kv_policy": kv_policy if scheduler == "continuous" else None,
        "segmenter": segmenter_metadata,
        "audio_prefetch": audio_prefetch,
    }
    output.write_text(json.dumps(final, ensure_ascii=False, indent=2))
    output.with_suffix(".txt").write_text(final["text"] + "\n")
    return final
