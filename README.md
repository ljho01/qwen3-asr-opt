# qwen3-asr-opt

Fast local Korean and English transcription with Qwen3-ASR 1.7B on Apple Silicon.
The default long-file profile uses an MLX encoder, an 8-bit model, compiled greedy
decoding, continuous batch scheduling, and a growing KV cache.

The project builds on
[`mlx-qwen3-asr`](https://github.com/moona3k/mlx-qwen3-asr) 0.4.0 and adds a bounded-memory
long-recording pipeline, independent batched streams, compiled decoding, incremental
output persistence, and measured defaults for an M4 Pro.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- FFmpeg available on `PATH`

Homebrew users can install the system tools with:

```sh
brew install uv ffmpeg
```

## Install

```sh
git clone https://github.com/ljho01/qwen3-asr-opt.git
cd qwen3-asr-opt
uv sync
```

Download the official Qwen3-ASR 1.7B checkpoint and convert it to the measured q8
profile. The model files stay under `models/` and are ignored by Git.

```sh
uvx --from huggingface-hub hf download Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --local-dir models/original

./asr convert --source models/original --output models/q8 \
  --profile q8 --group-size 64
```

The conversion is deterministic for the pinned source revision. The resulting q8
weights used for the measurements below have SHA-256:

```text
eba2bdb1ec74f5df99345f9f81492ba551f6b072eedfef564b353ef0dde90bb8
```

## Transcribe

The fixed long-recording profile is the easiest entry point:

```sh
./transcribe-final "/path/to/recording.m4a" Korean "/path/to/result.json"
```

Use `English` for English audio. The command writes:

- `result.json` — full metadata and ordered chunks
- `result.txt` — plain text
- `result.chunks.jsonl` — chunks persisted as they complete

Output paths are never overwritten. Audio decoding and transcription run locally.

The launcher expands to the following explicit configuration:

```sh
./asr transcribe "/path/to/recording.m4a" \
  --language Korean --output "/path/to/result.json" \
  --preset quality --long --decoder compiled --batch-size 4 \
  --batch-scheduler continuous --kv-cache growing --segmenter energy \
  --chunk-seconds 30 --cache-mb 256 --batch-prefill serial \
  --dense-prefill off --audio-prefetch 0
```

## Measured M4 Pro result

| Item | Result |
|---|---:|
| Hardware | M4 Pro, 12-core CPU, 16-core GPU, 24 GB |
| Audio | 785.92 seconds, Korean + English |
| Warm runs | 5 |
| Median elapsed time | 23.789 seconds |
| Throughput | 33.04× real time |
| Run-to-run population CV | 0.44% |

This is a warm end-to-end file transcription measurement: audio decode, feature
extraction, MLX encoder, prompt prefill, autoregressive decoding, and text assembly are
included; model loading and the first compilation are excluded. The input consists of
validation utterances stitched into two long files, so it is not a natural meeting
benchmark. Other GPU activity was visible during the run. No isolated process-energy
number is reported from that session.

Continuous batch 4 and the growing KV cache preserved the checked token sequences while
reducing elapsed time and peak MLX allocation in their selection runs. Lower-precision
decoder candidates were evaluated but were not adopted because they did not preserve the
predefined Korean and English accuracy conditions. The published default remains q8.

## Development

```sh
uv run python -m pytest -q
uv run ruff check src tests scripts
```

The release contains 445 passing tests covering decoding, batching, bounded KV growth,
long-file output persistence, model conversion, metrics, optional VAD, and experimental
paths retained for reproducibility.

## License

The original code in this repository is licensed under the Apache License 2.0. The
bundled MLX Metal helper retains Apple's MIT license in
`src/qwen_asr_opt/native/mlx_qmm_LICENSE`. Downloaded models and other third-party
packages remain subject to their own licenses.
