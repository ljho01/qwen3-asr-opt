import importlib.util
import sys
from types import SimpleNamespace

import pytest

from qwen_asr_opt import cli, longform, optimizations, runtime


@pytest.mark.parametrize("scheduler,segmenter,explicit,expected", [
    (None, "energy", None, "growing128"),
    (None, "vad", None, "padded"),
    ("continuous", "energy", None, "growing128"),
    ("continuous", "energy", "padded", "padded"),
    ("continuous", "vad", None, "padded"),
    ("continuous", "vad", "growing", "growing128"),
    ("fixed", "energy", None, "padded"),
])
def test_cli_routes_only_confirmed_default_and_preserves_explicit_choice(
        tmp_path, monkeypatch, scheduler, segmenter, explicit, expected):
    (tmp_path / "q8").mkdir()
    (tmp_path / "vad").mkdir()
    (tmp_path / "vad/source.json").write_text("{}")
    monkeypatch.setenv("QWEN_ASR_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(optimizations, "configure", lambda *args: None)
    monkeypatch.setattr(runtime, "load_session", lambda path: SimpleNamespace(model=SimpleNamespace()))
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True if name == "onnxruntime" else original(name))
    called = []

    def run(*args, **kwargs):
        called.append((args, kwargs))
        return {"truncated": False}

    monkeypatch.setattr(longform, "transcribe_long", run)
    command = ["asr", "transcribe", "input.wav", "--language", "Korean", "--long",
               "--segmenter", segmenter,
               "--output", str(tmp_path / "result.json")]
    if scheduler is not None:
        command += ["--batch-scheduler", scheduler]
    if segmenter == "vad":
        command += ["--vad-model", str(tmp_path / "vad")]
    if explicit is not None:
        command += ["--kv-cache", explicit]
    monkeypatch.setattr(sys, "argv", command)
    cli.main()
    assert len(called) == 1 and called[0][1]["kv_policy"] == expected


def test_explicit_continuous_cache_on_fixed_fails_before_model_load(tmp_path, monkeypatch):
    (tmp_path / "q8").mkdir()
    monkeypatch.setenv("QWEN_ASR_MODEL_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(runtime, "load_session", lambda path: calls.append(path))
    monkeypatch.setattr(sys, "argv", ["asr", "transcribe", "input.wav", "--long", "--kv-cache", "growing",
                                      "--batch-scheduler", "fixed",
                                      "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not calls
