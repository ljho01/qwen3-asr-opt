import importlib.util
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

spec = importlib.util.spec_from_file_location(
    "prepare_fleurs", Path(__file__).resolve().parents[1] / "scripts/prepare_fleurs.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extracts_integrity_bound_audio_and_reference(tmp_path):
    encoded = io.BytesIO()
    sf.write(encoded, np.zeros(1600, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
    parquet = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "id": 7,
        "num_samples": 1600,
        "audio": {"bytes": encoded.getvalue(), "path": "source.wav"},
        "transcription": "normalized reference",
        "raw_transcription": "Normalized reference.",
    }]), parquet)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rows = module.extract_parquet(parquet, "en_us", audio_dir)

    assert rows[0]["reference"] == "normalized reference"
    assert rows[0]["duration_s"] == 0.1 and rows[0]["source_id"] == 7
    assert Path(rows[0]["audio"]).read_bytes() == encoded.getvalue()


def test_content_hash_does_not_depend_on_local_audio_path():
    row = {"id": "sample", "audio": "/first/audio.wav", "reference": "text", "sha256": "x"}
    moved = {**row, "audio": "/another/audio.wav"}
    assert module.content_sha256([row]) == module.content_sha256([moved])
