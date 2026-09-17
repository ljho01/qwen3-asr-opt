from __future__ import annotations

import dataclasses
import hashlib
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

from .benchmark_conditions import validate_conditions
from .environment import capture
from .metrics import score
from .optimizations import configure
from .runtime import load_session


def run(args) -> dict:
    manifest_path = Path(args.manifest)
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if args.limit:
        # Apply the same cap to each language rather than silently measuring only Korean.
        counts = {}
        selected = []
        for record in records:
            lang = record["language"]
            counts[lang] = counts.get(lang, 0) + 1
            if counts[lang] <= args.limit:
                selected.append(record)
        records = selected
    if not records:
        raise ValueError("Empty benchmark manifest")
    for r in records:
        path = Path(r["audio"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"Audio integrity mismatch: {path}")
    # Interleave languages to reduce sensitivity to heating/order.
    languages = sorted({r["language"] for r in records})
    groups = [[r for r in records if r["language"] == lang] for lang in languages]
    records = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".jsonl").exists():
        raise FileExistsError(f"Preserve existing evidence: {output}")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    env = capture()
    checkpoint = Path(args.model)
    metadata_file = checkpoint / "optimization.json"
    if not metadata_file.exists():
        metadata_file = checkpoint / "source.json"
    checkpoint_metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else None
    if checkpoint_metadata and "module_quantization" in checkpoint_metadata:
        checkpoint_metadata.pop("module_quantization")
    configure(args.decoder, args.cache_mb)
    mx.reset_peak_memory()
    start = time.perf_counter()
    session = load_session(args.model, args.dtype)
    session.model._batch_prefill_mode = getattr(args, "batch_prefill", "serial")
    from .prefill import configure_dense_prefill
    dense_prefill = configure_dense_prefill(session.model, getattr(args, "dense_prefill", "off"))
    coreml_metadata = None
    if getattr(args, "coreml_encoder", None):
        from .coreml_encoder import install_coreml_encoder
        coreml_metadata = install_coreml_encoder(session, args.coreml_encoder)
    mx.synchronize()
    load_s = time.perf_counter() - start
    load_peak_bytes = mx.get_peak_memory()
    warmups = []
    batch_size = getattr(args, "batch_size", 1)

    def transcribe(group):
        hints = [r["language"] if args.language_hint else None for r in group]
        if batch_size == 1:
            return [session.transcribe(group[0]["audio"], language=hints[0], return_chunks=True)]
        from .batch import transcribe_batch
        return transcribe_batch(session, [r["audio"] for r in group], hints)

    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    for group in batches[:args.warmup]:
        start = time.perf_counter()
        transcribe(group)
        mx.synchronize()
        warmups.append(time.perf_counter() - start)
    mx.reset_peak_memory()
    results = []
    timing = []
    timing_windows = []
    timed_environment_before = capture()
    start_epoch = time.time()
    for repeat in range(args.repeats):
        for batch_index, group in enumerate(batches):
            mx.synchronize()
            batch_start_epoch = time.time()
            started = time.perf_counter()
            outputs = transcribe(group)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            batch_end_epoch = time.time()
            timing.append(elapsed)
            timing_windows.append({"start_epoch": batch_start_epoch,
                                   "end_epoch": batch_end_epoch, "elapsed_s": elapsed})
            group_audio = sum(r["duration_s"] for r in group)
            with output.with_suffix(".jsonl").open("a") as f:
                for record, result in zip(group, outputs, strict=True):
                    row = {**record, **dataclasses.asdict(result),
                           "expected_language": record["language"], "repeat": repeat,
                           "elapsed_s": elapsed * record["duration_s"] / group_audio,
                           "rtf": elapsed / group_audio, "batch_index": batch_index,
                           "batch_elapsed_s": elapsed, "batch_start_epoch": batch_start_epoch,
                           "batch_end_epoch": batch_end_epoch,
                           "elapsed_scope": "single_input" if batch_size == 1
                               else "batch_time_allocated_by_audio_duration"}
                    results.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"batch": batch_index + 1, "total_batches": len(batches),
                              "repeat": repeat, "seconds": round(elapsed, 3),
                              "rtf": round(elapsed / group_audio, 4),
                              "truncated": any(r.truncated for r in outputs)}), flush=True)
    end_epoch = time.time()
    unique = [r for r in results if r["repeat"] == 0]
    total_audio = sum(r["duration_s"] for r in results)
    total_compute = sum(timing)
    environment_after = capture()
    timing_validity = validate_conditions(timing_windows,
                                         [timed_environment_before, environment_after])
    observed_timing = {
        "rtf": total_compute / total_audio, "throughput_x": total_audio / total_compute,
        "latency_p50_s": float(np.quantile(timing, 0.5)),
        "latency_p95_s": float(np.quantile(timing, 0.95)),
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model, "dtype": args.dtype,
        "backend": "mlx-qwen3-asr+coreml" if coreml_metadata else "mlx-qwen3-asr",
        "decoder": args.decoder, "cache_mb": args.cache_mb,
        "batch_size": batch_size,
        "batch_prefill": getattr(args, "batch_prefill", "serial"),
        "dense_prefill": dense_prefill,
        "latency_scope": "single_input" if batch_size == 1 else "batch",
        "checkpoint_metadata": checkpoint_metadata,
        "coreml_encoder": coreml_metadata,
        "source_sha256": source_hashes,
        "source_changed_during_run": [p.name for p in Path(__file__).parent.glob("*.py")
                                      if source_hashes.get(p.name) !=
                                      hashlib.sha256(p.read_bytes()).hexdigest()],
        "environment": env, "timed_environment_before": timed_environment_before,
        "environment_after": environment_after, "device": mx.device_info(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "normalization": "NFKC, casefold, delete Unicode punctuation, collapse whitespace",
        "language_hint": args.language_hint, "repeats": args.repeats,
        "model_load_s": load_s, "warmup_s": warmups,
        "model_load_mlx_peak_bytes": load_peak_bytes,
        "timed_start_epoch": start_epoch, "timed_end_epoch": end_epoch,
        "audio_s": total_audio, "compute_s": total_compute,
        "timing_validity": timing_validity,
        "observed_timing": observed_timing,
        "observed_timing_scope": "Unfiltered performance-counter diagnostics; inspect timing_validity before comparison. Per-input elapsed/rtf rows are raw allocations, not validated performance claims.",
        **{key: value if timing_validity["valid_for_performance_comparison"] else None
           for key, value in observed_timing.items()},
        "mlx_peak_bytes": mx.get_peak_memory(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "energy": None, "energy_status": "not_measured",
        "accuracy": {lang: score([r for r in unique if r["expected_language"] == lang])
                     for lang in languages},
        "truncated_samples": sum(r["truncated"] for r in unique),
        "repeat_text_mismatches": sum(r["text"] != unique[i % len(unique)]["text"]
                                      for i, r in enumerate(results)),
        "results": results,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ["rtf", "throughput_x", "timing_validity", "accuracy",
                                            "truncated_samples", "mlx_peak_bytes"]},
                     ensure_ascii=False, indent=2), flush=True)
    return summary
