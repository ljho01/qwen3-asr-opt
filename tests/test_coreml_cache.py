import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_asr_opt.coreml_cache import load_cached_model


def test_persistent_cache_reuses_path_and_invalidates_changed_graph(tmp_path, monkeypatch):
    package = tmp_path / "source.mlpackage"
    graph = package / "Data/com.apple.CoreML/model.mlmodel"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"graph one")
    calls = []

    def compile_model(source, destination_path):
        calls.append(source)
        Path(destination_path).mkdir()
        (Path(destination_path) / "model.mil").write_bytes(graph.read_bytes())

    fake = SimpleNamespace(__version__="test",
                           ComputeUnit=SimpleNamespace(CPU_AND_NE="ne"),
                           models=SimpleNamespace(utils=SimpleNamespace(compile_model=compile_model),
                                                  CompiledMLModel=lambda path, **_: path))
    monkeypatch.setitem(sys.modules, "coremltools", fake)
    digest = hashlib.sha256(b"verified weights").hexdigest()
    cache = tmp_path / "cache"
    first, a = load_cached_model(package, digest, cache_root=cache)
    second, b = load_cached_model(package, digest, cache_root=cache)
    assert first == second and len(calls) == 1
    assert not a["compiled_artifact_reused"] and b["compiled_artifact_reused"]
    graph.write_bytes(b"graph two")
    third, c = load_cached_model(package, digest, cache_root=cache)
    assert third != first and len(calls) == 2
    assert Path(first).is_dir() and not c["compiled_artifact_reused"]
    fourth, _ = load_cached_model(package, digest + "changed", cache_root=cache)
    assert fourth != third and len(calls) == 3


def test_failed_compilation_does_not_publish_partial_cache(tmp_path, monkeypatch):
    package = tmp_path / "source.mlpackage"
    graph = package / "Data/com.apple.CoreML/model.mlmodel"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"graph")

    def fail_compile(source, destination_path):
        Path(destination_path).mkdir()
        (Path(destination_path) / "incomplete").write_text("partial")
        raise RuntimeError("simulated compiler failure")

    fake = SimpleNamespace(__version__="test",
                           models=SimpleNamespace(utils=SimpleNamespace(compile_model=fail_compile)))
    monkeypatch.setitem(sys.modules, "coremltools", fake)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="simulated compiler failure"):
        load_cached_model(package, "verified digest", cache_root=cache)
    assert not list(cache.glob("*.mlmodelc"))
    assert not list(cache.glob("*.json"))
    assert not any(path.is_dir() for path in cache.iterdir())
