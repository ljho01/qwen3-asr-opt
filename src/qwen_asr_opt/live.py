"""Native, bounded-memory live transcription for files and microphone audio."""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import BinaryIO, TextIO

import mlx.core as mx
import numpy as np
from mlx_qwen3_asr.streaming import (
    feed_audio,
    finish_streaming,
    init_streaming,
    streaming_metrics,
)

from .longform import SAMPLE_RATE

DEFAULT_CHUNK_SECONDS = {"Korean": 2.5, "English": 2.4}


def output_paths(output: Path) -> tuple[Path, Path, Path]:
    """Return the summary, text, and append-only event paths for a live run."""
    return output, output.with_suffix(".txt"), output.with_suffix(".events.jsonl")


def validate_output(output: Path | None) -> None:
    if output is None:
        return
    existing = [path for path in output_paths(output) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite transcription: {existing[0]}")


def _packet_reader(stream: BinaryIO, tick_samples: int, sample_width: int):
    target_bytes = tick_samples * sample_width
    buffered = bytearray()
    while True:
        raw = stream.read(target_bytes - len(buffered))
        if raw:
            buffered.extend(raw)
        if len(buffered) == target_bytes or (not raw and buffered):
            if len(buffered) % sample_width:
                raise RuntimeError("Audio stream ended with an incomplete PCM sample")
            packet = bytes(buffered)
            buffered.clear()
            if sample_width == 2:
                yield np.frombuffer(packet, dtype="<i2").astype(np.float32) / 32768.0
            else:
                yield np.frombuffer(packet, dtype="<f4").copy()
        if not raw:
            return


def _ffmpeg_packets(source: str | None, microphone: str | None, tick_samples: int):
    with tempfile.TemporaryFile() as errors:
        if microphone is not None:
            input_args = ["-f", "avfoundation", "-i", f":{microphone}"]
        else:
            input_args = ["-i", str(Path(source).resolve())]
        process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", *input_args, "-vn", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        assert process.stdout is not None
        try:
            yield from _packet_reader(process.stdout, tick_samples, 4)
            code = process.wait()
            if code:
                errors.seek(0)
                message = errors.read().decode(errors="replace").strip()
                raise RuntimeError(f"FFmpeg audio capture failed ({code}): {message}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def audio_packets(
    source: str | None,
    *,
    microphone: str | None,
    tick_seconds: float,
    stdin: BinaryIO | None = None,
):
    """Yield normalized float32 packets from a file, a microphone, or PCM16 stdin."""
    tick_samples = round(tick_seconds * SAMPLE_RATE)
    if tick_samples < 1:
        raise ValueError("tick_seconds is too small")
    if source == "-":
        reader = stdin if stdin is not None else sys.stdin.buffer
        yield from _packet_reader(reader, tick_samples, 2)
    else:
        yield from _ffmpeg_packets(source, microphone, tick_samples)


def _new_state(model: Path, language: str, chunk_seconds: float, max_context_seconds: float):
    return init_streaming(
        model=str(model),
        language=language,
        chunk_size_sec=chunk_seconds,
        max_context_sec=max_context_seconds,
        unfixed_chunk_num=0,
        unfixed_token_num=2,
        finalization_mode="accuracy",
        enable_tail_refine=True,
        endpointing_mode="fixed",
    )


def warmup_stream(
    session,
    model: Path,
    language: str,
    chunk_seconds: float,
    max_context_seconds: float,
) -> float:
    """Compile the two steady-state streaming turns before live audio starts."""
    state = _new_state(model, language, chunk_seconds, max_context_seconds)
    silence = np.zeros(round(chunk_seconds * SAMPLE_RATE), dtype=np.float32)
    started = time.perf_counter()
    for _ in range(2):
        feed_audio(silence, state, session.model)
    mx.synchronize()
    return time.perf_counter() - started


def _stable_delta(previous: str, current: str) -> tuple[str, bool]:
    if current.startswith(previous):
        return current[len(previous):], False
    return current, True


def _provisional_tail(stable: str, current: str) -> str:
    return current.removeprefix(stable)


def _write_event(row: dict, stream: TextIO, event_file: TextIO | None, output_format: str):
    encoded = json.dumps(row, ensure_ascii=False)
    if event_file is not None:
        event_file.write(encoded + "\n")
        event_file.flush()
    if output_format == "jsonl":
        print(encoded, file=stream, flush=True)
        return
    if row["kind"] == "final":
        print(f"[final {row['audio_s']:.1f}s] {row['text']}", file=stream, flush=True)
        return
    if row["stable_delta"]:
        label = "stable-reset" if row["stable_reset"] else "stable"
        print(f"[{label} {row['audio_s']:.1f}s] {row['stable_delta'].strip()}",
              file=stream, flush=True)
    if row["provisional"] is not None and row["provisional"].strip():
        print(f"[draft {row['audio_s']:.1f}s] {row['provisional'].strip()}",
              file=stream, flush=True)


def _model_metadata(model: Path) -> dict:
    metadata_path = model / "optimization.json"
    if not metadata_path.is_file():
        return {"path": str(model)}
    metadata = json.loads(metadata_path.read_text())
    return {
        "path": str(model),
        "profile": metadata.get("profile"),
        "source": metadata.get("source"),
        "source_revision": metadata.get("source_revision"),
        "weights_sha256": metadata.get("weights_sha256"),
    }


def transcribe_live(
    session,
    model: Path,
    source: str | None,
    *,
    language: str,
    microphone: str | None = None,
    output: Path | None = None,
    output_format: str = "text",
    tick_seconds: float = 0.2,
    chunk_seconds: float | None = None,
    max_context_seconds: float = 30.0,
    paced: bool = True,
    warmup: bool = True,
    stdin: BinaryIO | None = None,
    stdout: TextIO | None = None,
) -> dict:
    """Run the fixed native-streaming profile and optionally persist bounded events."""
    validate_output(output)
    if language not in DEFAULT_CHUNK_SECONDS:
        raise ValueError("language must be Korean or English")
    if output_format not in {"text", "jsonl"}:
        raise ValueError("output_format must be text or jsonl")
    if tick_seconds <= 0:
        raise ValueError("tick_seconds must be positive")
    chunk_seconds = chunk_seconds or DEFAULT_CHUNK_SECONDS[language]
    if chunk_seconds <= 0 or max_context_seconds < chunk_seconds:
        raise ValueError("chunk_seconds must be positive and no larger than max_context_seconds")
    stream = stdout if stdout is not None else sys.stdout
    model = model.resolve()

    warmup_s = (
        warmup_stream(session, model, language, chunk_seconds, max_context_seconds)
        if warmup else 0.0
    )
    mx.reset_peak_memory()
    state = _new_state(model, language, chunk_seconds, max_context_seconds)
    event_path = output.with_suffix(".events.jsonl") if output is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    event_file = event_path.open("x") if event_path is not None else None
    packets = audio_packets(
        source,
        microphone=microphone,
        tick_seconds=tick_seconds,
        stdin=stdin,
    )
    input_is_live = microphone is not None or source == "-"
    paced = bool(paced and not input_is_live)
    started = time.perf_counter()
    samples_seen = 0
    active_compute_s = 0.0
    event_count = 0
    stable_resets = 0
    previous_text = ""
    previous_stable = ""
    previous_provisional = ""
    first_provisional_s = None
    first_stable_s = None
    interrupted = False

    def emit(*, final: bool, compute_s: float) -> None:
        nonlocal event_count, stable_resets, previous_text, previous_stable
        nonlocal previous_provisional, first_provisional_s, first_stable_s
        elapsed_s = time.perf_counter() - started
        stable_delta, stable_reset = _stable_delta(previous_stable, state.stable_text)
        provisional = _provisional_tail(state.stable_text, state.text)
        stable_changed = state.stable_text != previous_stable
        provisional_changed = provisional != previous_provisional
        text_changed = state.text != previous_text
        if not final and not (stable_changed or provisional_changed or text_changed):
            return
        if stable_reset:
            stable_resets += 1
        if first_provisional_s is None and state.text:
            first_provisional_s = elapsed_s
        if first_stable_s is None and state.stable_text:
            first_stable_s = elapsed_s
        row = {
            "schema": "qwen-asr-live-event-v1",
            "index": event_count,
            "kind": "final" if final else "update",
            "audio_s": samples_seen / SAMPLE_RATE,
            "elapsed_s": elapsed_s,
            "compute_s": compute_s,
            "chunk_id": state.chunk_id,
            "stable_delta": stable_delta if stable_changed else "",
            "stable_reset": stable_reset,
            "provisional": provisional if provisional_changed and not final else None,
        }
        if final:
            row["text"] = state.text
        _write_event(row, stream, event_file, output_format)
        event_count += 1
        previous_text = state.text
        previous_stable = state.stable_text
        previous_provisional = provisional

    try:
        try:
            for packet in packets:
                samples_seen += len(packet)
                if paced:
                    wait_s = started + samples_seen / SAMPLE_RATE - time.perf_counter()
                    if wait_s > 0:
                        time.sleep(wait_s)
                compute_started = time.perf_counter()
                feed_audio(packet, state, session.model)
                mx.synchronize()
                compute_s = time.perf_counter() - compute_started
                active_compute_s += compute_s
                emit(final=False, compute_s=compute_s)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            close_packets = getattr(packets, "close", None)
            if close_packets is not None:
                close_packets()
        if samples_seen == 0:
            raise ValueError("Input contains no audio")
        finish_started = time.perf_counter()
        finish_streaming(state, session.model)
        mx.synchronize()
        finish_compute_s = time.perf_counter() - finish_started
        active_compute_s += finish_compute_s
        emit(final=True, compute_s=finish_compute_s)
    finally:
        if event_file is not None:
            event_file.close()

    elapsed_s = time.perf_counter() - started
    result = {
        "schema": "qwen-asr-live-v1",
        "mode": "live",
        "source": f"microphone:{microphone}" if microphone is not None else source,
        "audio_s": samples_seen / SAMPLE_RATE,
        "language": language,
        "model": _model_metadata(model),
        "configuration": {
            "input_tick_s": tick_seconds,
            "chunk_size_s": chunk_seconds,
            "max_context_s": max_context_seconds,
            "unfixed_chunk_num": 0,
            "unfixed_token_num": 2,
            "endpointing_mode": "fixed",
            "finalization_mode": "accuracy",
            "tail_refine": True,
            "paced_file_input": paced,
            "warmup": warmup,
        },
        "timing": {
            "warmup_s": warmup_s,
            "stream_elapsed_s": elapsed_s,
            "active_compute_s": active_compute_s,
            "first_provisional_s": first_provisional_s,
            "first_stable_s": first_stable_s,
        },
        "interrupted": interrupted,
        "event_count": event_count,
        "stable_resets": stable_resets,
        "text": state.text,
        "streaming_metrics": streaming_metrics(state),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if output is not None:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        output.with_suffix(".txt").write_text(state.text + "\n")
    return result
