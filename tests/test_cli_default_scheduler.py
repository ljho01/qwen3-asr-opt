import sys
from types import SimpleNamespace

import pytest

from qwen_asr_opt import cli, longform, optimizations, runtime


@pytest.mark.parametrize("options,model,decoder,batch,scheduler,cache", [
    ([], "q8", "compiled", 4, "continuous", "growing128"),
    (["--preset", "quality"], "q8", "compiled", 4, "continuous", "growing128"),
    (["--batch-scheduler", "fixed"], "q8", "compiled", 4, "fixed", "padded"),
    (["--kv-cache", "padded"], "q8", "compiled", 4, "continuous", "padded"),
    (["--preset", "balanced"], "mixed8_5", "compiled", 4, "fixed", "padded"),
    (["--preset", "reference"], "original", "stock", 1, "fixed", "padded"),
    (["--batch-size", "1"], "q8", "pipelined", 1, "fixed", "padded"),
    (["--batch-size", "8"], "q8", "compiled", 8, "fixed", "padded"),
    (["--batch-prefill", "batched"], "q8", "compiled", 4, "fixed", "padded"),
    (["--dense-prefill", "transient"], "q8", "compiled", 4, "fixed", "padded"),
    (["--coreml-encoder", "dummy.mlpackage"], "q8", "compiled", 4, "fixed", "padded"),
    (["--model", "CUSTOM"], "custom", "stock", 1, "fixed", "padded"),
])
def test_auto_scheduler_applies_to_verified_profile_and_respects_overrides(
        tmp_path, monkeypatch, options, model, decoder, batch, scheduler, cache):
    for name in ("q8", "mixed8_5", "original", "custom"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("QWEN_ASR_MODEL_DIR", str(tmp_path))
    configured, loaded, called = [], [], []
    monkeypatch.setattr(optimizations, "configure", lambda *args: configured.append(args))

    def load(path):
        loaded.append(path)
        return SimpleNamespace(model=SimpleNamespace())

    monkeypatch.setattr(runtime, "load_session", load)
    monkeypatch.setitem(sys.modules, "qwen_asr_opt.prefill",
                        SimpleNamespace(configure_dense_prefill=lambda *args: None))
    monkeypatch.setitem(sys.modules, "qwen_asr_opt.coreml_encoder",
                        SimpleNamespace(install_coreml_encoder=lambda *args: None))

    def run(*args, **kwargs):
        called.append((args, kwargs))
        return {"truncated": False}

    monkeypatch.setattr(longform, "transcribe_long", run)
    resolved_options = [str(tmp_path / "custom") if opt == "CUSTOM" else opt for opt in options]
    monkeypatch.setattr(sys, "argv", ["asr", "transcribe", "input.wav", "--long",
                                      "--output", str(tmp_path / "out.json"), *resolved_options])
    cli.main()
    assert loaded == [str(tmp_path / model)]
    assert configured[0][0] == decoder
    assert len(called) == 1
    assert called[0][0][5:7] == (batch, scheduler)
    assert called[0][1]["kv_policy"] == cache


def test_short_file_keeps_balanced_single_file_path(tmp_path, monkeypatch):
    (tmp_path / "mixed8_5").mkdir()
    monkeypatch.setenv("QWEN_ASR_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(optimizations, "configure", lambda *args: None)
    called = []

    def transcribe(*args, **kwargs):
        called.append((args, kwargs))
        return SimpleNamespace(text="ok", truncated=False)

    session = SimpleNamespace(model=SimpleNamespace(), transcribe=transcribe)
    monkeypatch.setattr(runtime, "load_session", lambda path: session)
    monkeypatch.setattr(longform, "transcribe_long", lambda *args, **kwargs: pytest.fail("long path"))
    monkeypatch.setattr(sys, "argv", ["asr", "transcribe", "short.wav"])
    cli.main()
    assert len(called) == 1 and called[0][0] == ("short.wav",)
