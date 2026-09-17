# qwen3-asr-opt

Fast local Korean and English transcription with Qwen3-ASR 1.7B on Apple Silicon.
It provides two fixed q8/MLX modes: a high-throughput final pass for long recordings and
bounded-memory live captions with stable-prefix updates.

The project builds on
[`mlx-qwen3-asr`](https://github.com/moona3k/mlx-qwen3-asr) 0.4.0 and adds a bounded-memory
long-recording pipeline, independent batched streams, compiled decoding, incremental
output persistence, native live streaming, and measured defaults for an M4 Pro.

## Standard benchmark

Measured on the complete English and Korean test splits of
[Google FLEURS](https://huggingface.co/datasets/google/fleurs)
([paper](https://arxiv.org/abs/2205.12446)), a public 102-language CC-BY benchmark used by the
[official Qwen3-ASR evaluation](https://github.com/QwenLM/Qwen3-ASR#evaluation) and the
[Hugging Face Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard).

| FLEURS test split | Samples | Audio | Primary error rate ↓ |
|---|---:|---:|---:|
| English `en_us` | 647 | 106.5 min | **4.20% WER** |
| Korean `ko_kr` | 382 | 80.1 min | **4.39% CER** |

The q8 model processed all 3.11 hours in 308.23 seconds: **0.0275 RTF / 36.31× real
time**, with no truncated samples. This is a local M4 Pro batch-4 throughput measurement,
using the compiled greedy decoder, forced known-language hints, two excluded warmup
batches, AC power, and the default macOS power mode. WER/CER
uses corpus-level edit counts after NFKC, case folding, punctuation deletion, and whitespace
normalization; Korean CER additionally removes spaces. The dataset revision, source/model
hashes, error counts, runtime versions, memory peaks, and validity limits are in the
[machine-readable result](benchmarks/fleurs-test-q8-m4-pro.json).

Reproduce the full run without committing model or dataset files:

```sh
uv run python scripts/prepare_fleurs.py --output data/fleurs-test

./asr bench --manifest data/fleurs-test/manifest.jsonl \
  --model models/q8 --output outputs/fleurs-test-q8.json \
  --decoder compiled --cache-mb 256 --batch-size 4 \
  --batch-prefill serial --dense-prefill off \
  --warmup 2 --repeats 1 --language-hint
```

## FP16 and stock q8 comparison

The following paired run compares the official FP16 checkpoint, the same q8 checkpoint
with the upstream stock decoder, and this project's compiled batch-4 path. It uses the
first 32 source rows from each FLEURS test locale, fixed before decoding: 64 clips and
11.88 minutes of audio. These smaller quality columns are diagnostic; the full-test
quality result above remains the primary score.

| MLX path | Batch | English WER ↓ | Korean CER ↓ | Throughput ↑ | Mean system W | Gross J / audio min ↓ | MLX peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official FP16 + stock decoder | 1 | 4.70% | 2.67% | 11.89× | 49.55 W | 250.11 J | 4.40 GiB |
| q8 + stock decoder | 1 | 4.56% | 2.67% | 17.62× | 56.83 W | 193.56 J | 2.48 GiB |
| q8 + compiled decoder | 4 | 4.56% | 2.67% | **29.35×** | 66.57 W | **136.11 J** | 2.79 GiB |

The optimized path produced exactly the same transcript as q8 stock for all 64 clips. It
was 1.67× as fast and used 29.7% less gross system energy per audio minute. Relative to
FP16 stock it was 2.47× as fast, used 45.6% less energy per audio minute, and reduced peak
MLX allocation by 36.7%. Batch 4 raises instantaneous power and uses 12.2% more peak MLX
memory than q8 stock, but its shorter runtime lowers total energy.

Power is the raw Apple SMC `PSTR` total-system estimate sampled every 250 ms with 100%
measurement-window coverage. It is neither wall-outlet nor per-process power. All three
paths ran sequentially on AC power in the default macOS power mode; model loading and two
warmup batches were excluded. This is one run per path, so execution order, residual heat,
the display, and background desktop work remain sources of uncertainty. The exact sample
IDs, model and data hashes, idle diagnostics, raw-evidence hashes, and unrounded values are
in the [machine-readable comparison](benchmarks/fleurs-test-variant-comparison-m4-pro.json).

Reproduce the timing and accuracy portion after preparing the full test data above:

```sh
./asr bench --manifest data/fleurs-test/manifest.jsonl --limit 32 \
  --model models/original --output outputs/fp16-stock.json \
  --decoder stock --batch-size 1 --warmup 2 --language-hint

./asr bench --manifest data/fleurs-test/manifest.jsonl --limit 32 \
  --model models/q8 --output outputs/q8-stock.json \
  --decoder stock --batch-size 1 --warmup 2 --language-hint

./asr bench --manifest data/fleurs-test/manifest.jsonl --limit 32 \
  --model models/q8 --output outputs/q8-optimized-b4.json \
  --decoder compiled --cache-mb 256 --batch-size 4 \
  --batch-prefill serial --dense-prefill off --warmup 2 --language-hint
```

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

## Choose a mode

| Mode | Command | Use it for |
|---|---|---|
| Long recording | `./transcribe-final ...` | Best final transcript from a completed file |
| Live captions | `./asr live ...` | Microphone captions or real-time file playback |

Both modes run locally. The live path displays a quickly changing draft and separately
commits stable text. Run the long-recording path on the saved audio when final transcript
quality matters most.

## Long recordings

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

## Live captions

List the macOS AVFoundation capture devices, then start the microphone by index or name:

```sh
ffmpeg -f avfoundation -list_devices true -i ""

./asr live --microphone 0 --language Korean \
  --output "/path/to/live.json"
```

The model is loaded and the compiled streaming path is warmed before the microphone is
opened. Press `Ctrl-C` to stop; the remaining audio is finalized before the command exits.
Use `English` for English speech.

A file can be delivered to the same decoder at real-time speed:

```sh
./asr live "/path/to/recording.m4a" --language English \
  --output "/path/to/live.json"
```

Add `--unpaced` to exercise the streaming decoder as fast as the Mac can run it. For an
existing live source, send raw 16 kHz mono signed PCM16 on standard input:

```sh
audio-producer | ./asr live - --language Korean --format jsonl
```

Text output uses these records:

- `[draft]` is the current two-token tail and may change.
- `[stable]` contains only newly committed text.
- `[final]` is emitted once after accuracy-oriented tail finalization.

With `--output live.json`, the command also writes `live.txt` and the append-only
`live.events.jsonl`. Update events contain stable deltas and only the short provisional
tail, so the event log grows linearly during long sessions. The fixed profile uses a
0.2-second input tick, a 30-second bounded context, and 2.5-second Korean or 2.4-second
English decode chunks. Korean may need the second chunk before the first stable commit;
the first chunk can still produce a draft.

## Long-file M4 Pro result

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

The release contains 452 passing tests covering decoding, batching, bounded KV growth,
long-file and live output persistence, model conversion, metrics, optional VAD, and
experimental paths retained for reproducibility.

## License

The original code in this repository is licensed under the Apache License 2.0. The
bundled MLX Metal helper retains Apple's MIT license in
`src/qwen_asr_opt/native/mlx_qmm_LICENSE`. Downloaded models and other third-party
packages remain subject to their own licenses.
