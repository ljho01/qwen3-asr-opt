"""Prepare the pinned full FLEURS English/Korean test splits for `asr bench`."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

DATASET = "google/fleurs"
REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
SPLIT = "test"
SAMPLE_RATE = 16000
LOCALES = {
    "en_us": {
        "language": "English",
        "parquet_sha256": "6428a4d04d3aac29e16b45e039bb1470a8bd7aa334cf92f7984c9c520d1f234d",
    },
    "ko_kr": {
        "language": "Korean",
        "parquet_sha256": "1a8319fc61c7996e8c15acde633786de97054e28ae1e463eb13901716176a7ec",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def content_sha256(records: list[dict]) -> str:
    """Hash dataset identities, references, and audio independent of local paths."""
    digest = hashlib.sha256()
    for row in records:
        portable = {key: value for key, value in row.items() if key != "audio"}
        encoded = json.dumps(portable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(encoded.encode() + b"\n")
    return digest.hexdigest()


def extract_parquet(parquet: Path, locale: str, audio_dir: Path) -> list[dict]:
    """Extract embedded WAV bytes and return integrity-bound benchmark rows."""
    config = LOCALES[locale]
    table = pq.read_table(
        parquet,
        columns=["id", "num_samples", "audio", "transcription", "raw_transcription"],
    )
    records = []
    for index, row in enumerate(table.to_pylist()):
        audio = row["audio"]["bytes"]
        if not audio:
            raise ValueError(f"Missing embedded audio: {locale} row {index}")
        with sf.SoundFile(io.BytesIO(audio)) as source:
            properties = source.samplerate, source.channels, len(source)
        expected = SAMPLE_RATE, 1, row["num_samples"]
        if properties != expected:
            raise ValueError(
                f"Unexpected audio properties: {locale} row {index}: {properties} != {expected}"
            )
        name = f"{locale}-{index:04d}-{row['id']}.wav"
        audio_path = audio_dir / name
        with audio_path.open("xb") as output:
            output.write(audio)
        records.append({
            "id": f"fleurs-{locale}-{index:04d}-{row['id']}",
            "audio": str(audio_path.resolve()),
            "language": config["language"],
            "duration_s": row["num_samples"] / SAMPLE_RATE,
            "reference": row["transcription"],
            "raw_reference": row["raw_transcription"],
            "sha256": hashlib.sha256(audio).hexdigest(),
            "dataset": DATASET,
            "revision": REVISION,
            "split": SPLIT,
            "locale": locale,
            "row_index": index,
            "source_id": row["id"],
        })
    return records


def prepare(destination: Path) -> dict:
    manifest = destination / "manifest.jsonl"
    provenance = destination / "source.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to reuse dataset directory: {destination}")
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    sources = {}
    for locale, config in LOCALES.items():
        parquet = Path(hf_hub_download(
            DATASET,
            f"{locale}/{SPLIT}/0000.parquet",
            repo_type="dataset",
            revision=REVISION,
        ))
        digest = sha256(parquet)
        if digest != config["parquet_sha256"]:
            raise ValueError(f"FLEURS parquet integrity mismatch: {locale}: {digest}")
        locale_records = extract_parquet(parquet, locale, audio_dir)
        records.extend(locale_records)
        sources[locale] = {
            "file": f"{locale}/{SPLIT}/0000.parquet",
            "sha256": digest,
            "samples": len(locale_records),
            "audio_s": sum(row["duration_s"] for row in locale_records),
        }
    with manifest.open("x") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": SPLIT,
        "license": "CC-BY-4.0",
        "sources": sources,
        "samples": len(records),
        "audio_s": sum(row["duration_s"] for row in records),
        "content_sha256": content_sha256(records),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
    }
    provenance.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
