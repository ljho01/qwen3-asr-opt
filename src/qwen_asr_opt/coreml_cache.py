"""Persistent compiled paths let Core ML reuse its device-specialization cache."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path


def package_fingerprint(package, verified_weights_sha256, tool_version, compute_units):
    """Fingerprint graph, manifest, verified weights, runtime and target device."""
    package = Path(package)
    files = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if relative == "Data/com.apple.CoreML/weights/weight.bin":
            files[relative] = verified_weights_sha256
        else:
            with path.open("rb") as stream:
                files[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    if "Data/com.apple.CoreML/model.mlmodel" not in files:
        raise ValueError("Missing Core ML model graph")
    details = {"format": 1, "files": files, "weights_sha256": verified_weights_sha256,
               "coremltools": tool_version,
               "platform": platform.platform(), "compute_units": compute_units}
    key = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    return key, details


def load_cached_model(package, verified_weights_sha256, compute_units="CPU_AND_NE", cache_root=None):
    """Compile atomically once, then always load from the same absolute path.

    The caller verifies source weight integrity first. The fingerprint changes when
    source graph/weights, Core ML Tools, OS, architecture or compute units change.
    A per-key lock prevents concurrent compilation and initial device specialization.
    Existing compiled artifacts are never silently overwritten.
    """
    import coremltools as ct
    started = time.perf_counter()
    key, details = package_fingerprint(package, verified_weights_sha256, ct.__version__, compute_units)
    root = Path(cache_root or Path.home() / ".cache/qwen3-asr-opt/coreml-compiled").resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{key}.mlmodelc"
    compile_s = 0.
    with (root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reused = target.is_dir()
        if not reused:
            begin = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix=f"{key}.", dir=root) as temp:
                temporary = Path(temp) / "model.mlmodelc"
                ct.models.utils.compile_model(str(Path(package).resolve()),
                                              destination_path=str(temporary))
                os.rename(temporary, target)
            compile_s = time.perf_counter() - begin
            (root / f"{key}.json").write_text(json.dumps(details, indent=2))
        begin = time.perf_counter()
        model = ct.models.CompiledMLModel(str(target), compute_units=getattr(ct.ComputeUnit, compute_units))
        load_s = time.perf_counter() - begin
    return model, {"key": key, "compiled_path": str(target), "compiled_artifact_reused": reused,
                   "compile_s": compile_s, "device_load_s": load_s,
                   "cache_total_s": time.perf_counter() - started}
