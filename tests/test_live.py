import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_asr_opt import cli, live, optimizations, runtime


def test_pcm16_packets_preserve_short_tail_and_reject_partial_sample():
    samples = np.array([-32768, -1, 0, 16384, 32767], dtype="<i2")
    packets = list(live._packet_reader(io.BytesIO(samples.tobytes()), 3, 2))
    assert [len(packet) for packet in packets] == [3, 2]
    np.testing.assert_allclose(np.concatenate(packets), samples.astype(np.float32) / 32768)
    with pytest.raises(RuntimeError, match="incomplete PCM sample"):
        list(live._packet_reader(io.BytesIO(b"\x00"), 3, 2))


def test_live_output_persists_stable_deltas_and_one_final_text(tmp_path, monkeypatch):
    model = tmp_path / "q8"
    model.mkdir()
    (model / "optimization.json").write_text(json.dumps({
        "profile": "q8",
        "source": "Qwen/Qwen3-ASR-1.7B",
        "source_revision": "revision",
        "weights_sha256": "digest",
    }))
    output = tmp_path / "live.json"
    state = SimpleNamespace(text="", stable_text="", chunk_id=0)
    updates = iter([
        ("hello world", "hello", 1),
        ("hello world again", "hello world", 2),
    ])

    def feed(packet, current, model):
        current.text, current.stable_text, current.chunk_id = next(updates)

    def finish(current, model):
        current.text = current.stable_text = "hello world again"

    monkeypatch.setattr(live, "init_streaming", lambda **kwargs: state)
    monkeypatch.setattr(live, "feed_audio", feed)
    monkeypatch.setattr(live, "finish_streaming", finish)
    monkeypatch.setattr(live, "streaming_metrics", lambda current: {"chunks_processed": 2})
    monkeypatch.setattr(live, "audio_packets", lambda *args, **kwargs: iter([
        np.zeros(3200, dtype=np.float32), np.zeros(3200, dtype=np.float32)
    ]))
    monkeypatch.setattr(live.mx, "synchronize", lambda: None)
    monkeypatch.setattr(live.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(live.mx, "get_peak_memory", lambda: 123)
    rendered = io.StringIO()

    result = live.transcribe_live(
        SimpleNamespace(model=object()), model, "input.wav", language="English",
        output=output, output_format="jsonl", paced=False, warmup=False, stdout=rendered,
    )

    events = [json.loads(line) for line in output.with_suffix(".events.jsonl").read_text().splitlines()]
    assert [event["stable_delta"] for event in events] == ["hello", " world", " again"]
    assert [event["provisional"] for event in events] == [" world", " again", None]
    assert "text" not in events[0] and "text" not in events[1]
    assert events[2]["text"] == "hello world again"
    assert len(rendered.getvalue().splitlines()) == 3
    assert result["text"] == "hello world again" and result["stable_resets"] == 0
    assert json.loads(output.read_text())["model"]["source_revision"] == "revision"
    assert output.with_suffix(".txt").read_text() == "hello world again\n"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        live.transcribe_live(
            SimpleNamespace(model=object()), model, "input.wav", language="English",
            output=output, paced=False, warmup=False,
        )


@pytest.mark.parametrize("language,chunk_seconds", [("Korean", 2.5), ("English", 2.4)])
def test_live_cli_uses_frozen_language_profile(tmp_path, monkeypatch, language, chunk_seconds):
    (tmp_path / "q8").mkdir()
    monkeypatch.setenv("QWEN_ASR_MODEL_DIR", str(tmp_path))
    configured = []
    called = []
    monkeypatch.setattr(optimizations, "configure", lambda *args: configured.append(args))
    monkeypatch.setattr(
        runtime, "load_session", lambda path: SimpleNamespace(model=SimpleNamespace())
    )
    monkeypatch.setattr(live, "transcribe_live", lambda *args, **kwargs: called.append((args, kwargs)))
    monkeypatch.setattr(
        sys, "argv", ["asr", "live", "--microphone", "0", "--language", language,
                      "--no-warmup"]
    )

    cli.main()

    assert configured == [("compiled", 256)]
    assert called[0][0][2] is None
    assert called[0][1]["microphone"] == "0"
    assert called[0][1]["chunk_seconds"] == chunk_seconds
    assert called[0][1]["warmup"] is False


def test_live_cli_rejects_ambiguous_input_before_loading(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"unused")
    loaded = []
    monkeypatch.setattr(runtime, "load_session", lambda path: loaded.append(path))
    monkeypatch.setattr(
        sys, "argv", ["asr", "live", str(audio), "--microphone", "0",
                      "--language", "English"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not loaded
