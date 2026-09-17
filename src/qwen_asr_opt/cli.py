from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 1.7B for Apple Silicon")
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="Create a local quantized checkpoint")
    convert.add_argument("--source", default="models/original", type=Path)
    convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--profile", choices=["q8", "q4", "q5", "mixed8_4", "mixed8_5"], required=True)
    convert.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    bench = sub.add_parser("bench", help="Run a reproducible paired quality/performance evaluation")
    bench.add_argument("--manifest", required=True)
    bench.add_argument("--model", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    bench.add_argument("--limit", type=int)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--language-hint", action=argparse.BooleanOptionalAction, default=True)
    bench.add_argument("--decoder", choices=["stock", "pipelined", "compiled"], default="stock")
    bench.add_argument("--cache-mb", type=int)
    bench.add_argument("--batch-size", type=int, choices=range(1, 9), default=1)
    bench.add_argument("--coreml-encoder", help="Experimental full Core ML encoder package")
    bench.add_argument("--batch-prefill", choices=["serial", "batched"], default="serial",
                       help="Experimental prefill grouping; default serial")
    bench.add_argument("--dense-prefill", choices=["off", "transient", "cached"], default="off",
                       help="Experimental dequantized GEMM; cached retains extra weights")
    transcribe = sub.add_parser("transcribe", help="Transcribe a local audio file")
    transcribe.add_argument("audio")
    choice = transcribe.add_mutually_exclusive_group()
    choice.add_argument("--model", help="Explicit checkpoint path; defaults to stock decoder")
    choice.add_argument("--preset", choices=["balanced", "fast", "quality", "reference"],
                        help="Default: balanced for short files; quality for --long")
    transcribe.add_argument("--language", choices=["Korean", "English"])
    transcribe.add_argument("--output", type=Path)
    transcribe.add_argument("--decoder", choices=["stock", "pipelined", "compiled"])
    transcribe.add_argument("--cache-mb", type=int)
    transcribe.add_argument("--coreml-encoder", help="Experimental full Core ML encoder package")
    transcribe.add_argument("--batch-prefill", choices=["serial", "batched"], default="serial",
                            help="Experimental prefill grouping; default serial")
    transcribe.add_argument("--dense-prefill", choices=["off", "transient", "cached"], default="off",
                            help="Experimental dequantized GEMM; cached retains extra weights")
    transcribe.add_argument("--long", action="store_true", help="Stream-decode long files in bounded memory")
    transcribe.add_argument("--chunk-seconds", type=float, default=30.0)
    transcribe.add_argument("--batch-size", type=int, choices=range(1, 9),
                            help="Default: 4 for optimized --long presets; otherwise 1")
    transcribe.add_argument("--batch-scheduler", choices=["fixed", "continuous"],
                            help="Default: continuous for quality --long compiled batch4 with standard encoder/prefill; otherwise fixed")
    transcribe.add_argument("--kv-cache", choices=["growing", "padded"],
                            help="Continuous cache: default growing with energy cuts, padded with VAD")
    transcribe.add_argument("--segmenter", choices=["energy", "vad"], default="energy",
                            help="Long-file boundaries: existing energy cuts or validated local speech-pause VAD")
    transcribe.add_argument("--vad-model", type=Path, help="Pinned Silero model directory for --segmenter vad")
    transcribe.add_argument("--audio-prefetch", type=int, choices=range(9), default=0,
                            help="Prepare this many --long audio chunks ahead on one CPU worker (measured:2)")
    args = parser.parse_args()
    if args.command == "convert":
        from .runtime import convert as convert_model
        meta = convert_model(args.source, args.output, args.profile, args.group_size)
        print(json.dumps({k: v for k, v in meta.items() if k != "module_quantization"}, indent=2))
    elif args.command == "bench":
        if args.batch_size > 1 and args.decoder != "compiled":
            parser.error("--batch-size > 1 requires --decoder compiled")
        if args.batch_size == 1 and args.batch_prefill != "serial":
            parser.error("--batch-prefill batched requires --batch-size > 1")
        if args.repeats < 1 or args.warmup < 0 or (args.limit is not None and args.limit < 1):
            parser.error("repeats/limit must be positive; warmup must be nonnegative")
        from .benchmark import run
        report = run(args)
        if not report["timing_validity"]["valid_for_performance_comparison"]:
            reasons = ", ".join(report["timing_validity"]["reasons"])
            parser.exit(2, f"Performance comparison invalid ({reasons}). "
                           f"Transcripts and raw measurements saved to {args.output}\n")
    else:
        from .optimizations import configure
        from .runtime import load_session
        preset = None
        if args.model is None:
            presets = {"balanced": ("mixed8_5", "compiled"),
                       "fast": ("mixed8_4", "compiled"),
                       "quality": ("q8", "pipelined"),
                       "reference": ("original", "stock")}
            preset = args.preset or ("quality" if args.long else "balanced")
            model_name, decoder = presets[preset]
            args.batch_size = args.batch_size or (4 if args.long and preset != "reference" else 1)
            if args.batch_size > 1 and decoder != "stock":
                decoder = "compiled"
            model_root = Path(os.environ.get("QWEN_ASR_MODEL_DIR",
                              str(Path(__file__).resolve().parents[2] / "models")))
            args.model = str(model_root / model_name)
            args.decoder = args.decoder or decoder
            if args.cache_mb is None and args.decoder != "stock":
                args.cache_mb = 256
        else:
            args.decoder = args.decoder or "stock"
            args.batch_size = args.batch_size or 1
        if args.batch_scheduler is None:
            verified_long_profile = (
                args.long and preset == "quality" and args.decoder == "compiled"
                and args.batch_size == 4 and args.batch_prefill == "serial"
                and args.dense_prefill == "off" and args.coreml_encoder is None
            )
            args.batch_scheduler = "continuous" if verified_long_profile else "fixed"
        if not Path(args.model).is_dir():
            parser.error(f"Local checkpoint not found: {args.model}; see README setup/conversion")
        if args.output and args.output.exists():
            parser.error(f"Refusing to overwrite transcription: {args.output}")
        if args.long and (not args.output or not 3 <= args.chunk_seconds <= 30):
            parser.error("--long requires --output and --chunk-seconds between 3 and 30")
        if args.batch_size > 1 and (not args.long or args.decoder != "compiled"):
            parser.error("--batch-size > 1 requires --long --decoder compiled")
        if args.batch_size == 1 and args.batch_prefill != "serial":
            parser.error("--batch-prefill batched requires --batch-size > 1")
        if args.batch_scheduler == "continuous":
            if not args.long or args.decoder != "compiled":
                parser.error("--batch-scheduler continuous requires --long --decoder compiled")
            if args.batch_prefill != "serial":
                parser.error("--batch-scheduler continuous requires --batch-prefill serial")
        elif args.kv_cache is not None:
            parser.error("--kv-cache requires --batch-scheduler continuous")
        if args.segmenter == "vad":
            import importlib.util
            if not args.long or not 8 < args.chunk_seconds <= 30:
                parser.error("--segmenter vad requires --long and --chunk-seconds greater than8")
            if importlib.util.find_spec("onnxruntime") is None:
                parser.error("VAD requires ONNX Runtime; install with uv sync --extra vad")
            args.vad_model = args.vad_model or Path(__file__).resolve().parents[2] / "models/vad-silero-v6.2"
            if not (args.vad_model / "source.json").is_file():
                parser.error("Pinned VAD model missing; run scripts/prepare_vad.py")
        elif args.vad_model:
            parser.error("--vad-model requires --segmenter vad")
        if args.audio_prefetch and not args.long:
            parser.error("--audio-prefetch requires --long")
        configure(args.decoder, args.cache_mb)
        session = load_session(args.model)
        session.model._batch_prefill_mode = args.batch_prefill
        if args.dense_prefill != "off":
            from .prefill import configure_dense_prefill
            configure_dense_prefill(session.model, args.dense_prefill)
        if args.coreml_encoder:
            from .coreml_encoder import install_coreml_encoder
            install_coreml_encoder(session, args.coreml_encoder)
        if args.long:
            from .longform import transcribe_long
            result = transcribe_long(session, args.audio, args.output,
                                     args.language, args.chunk_seconds, args.batch_size,
                                     args.batch_scheduler, args.segmenter, args.vad_model,
                                     args.audio_prefetch,
                                     kv_policy=("growing128" if args.batch_scheduler == "continuous"
                                                and (args.kv_cache == "growing" or
                                                     (args.kv_cache is None and args.segmenter == "energy"))
                                                else "padded"))
            if result["truncated"]:
                raise SystemExit("WARNING: one or more chunks hit a token limit")
            return
        result = session.transcribe(args.audio, language=args.language, return_chunks=True)
        print(result.text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2))
        if result.truncated:
            raise SystemExit("WARNING: transcript reached a token limit; inspect output chunks")


if __name__ == "__main__":
    main()
