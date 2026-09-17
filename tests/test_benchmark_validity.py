from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from qwen_asr_opt import benchmark, cli, prefill


def environment(source="AC Power"):
    return {"power_source": f"Now drawing from '{source}'",
            "power_settings": "Battery Power:\n powermode 1\nAC Power:\n powermode 0"}


@dataclasses.dataclass
class Result:
    text: str = "hello"
    truncated: bool = False


def execute(tmp_path, monkeypatch, environments, *, wall_end=102):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fixture; inference is mocked")
    row = {"id": "fixture", "audio": str(audio), "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
           "language": "English", "duration_s": 2, "reference": "hello"}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row)+"\n")
    output = tmp_path / "benchmark.json"
    args = SimpleNamespace(manifest=str(manifest), limit=None, output=str(output),
                           model=str(tmp_path / "model"), decoder="compiled", cache_mb=256,
                           dtype="float16", warmup=0, repeats=1, batch_size=1, language_hint=True)
    captures = iter(environments)
    monkeypatch.setattr(benchmark, "capture", lambda: next(captures))
    monkeypatch.setattr(benchmark, "configure", lambda *a: None)
    monkeypatch.setattr(prefill, "configure_dense_prefill", lambda *a: {"mode": "off"})
    session = SimpleNamespace(model=SimpleNamespace(), transcribe=lambda *a, **k: Result())
    monkeypatch.setattr(benchmark, "load_session", lambda *a: session)
    for name in ("reset_peak_memory", "synchronize"):
        monkeypatch.setattr(benchmark.mx, name, lambda: None)
    monkeypatch.setattr(benchmark.mx, "get_peak_memory", lambda: 100)
    monkeypatch.setattr(benchmark.mx, "device_info", dict)
    counter, wall = iter([0, 1, 2, 3]), iter([100, 101, wall_end, wall_end+1])
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: next(counter), time=lambda: next(wall)))
    result = benchmark.run(args)
    assert json.loads(output.read_text()) == result
    raw = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    assert raw == result["results"] and raw[0]["text"] == "hello"
    assert result["accuracy"]["English"]["word_errors"] == 0
    return result


def test_valid_measurement_keeps_published_performance(tmp_path, monkeypatch):
    report = execute(tmp_path, monkeypatch, [environment()]*3)
    assert report["timing_validity"]["valid_for_performance_comparison"]
    assert report["rtf"] == .5 and report["throughput_x"] == 2
    assert report["observed_timing"]["throughput_x"] == 2


@pytest.mark.parametrize("condition", ["power", "sleep", "unknown"])
def test_invalid_measurement_keeps_transcript_and_raw_values_but_not_performance_claim(tmp_path, monkeypatch, condition):
    captures = [environment()]*3
    end = 102
    if condition == "power":
        captures[-1] = environment("Battery Power")
    elif condition == "sleep":
        end += 24.74
    else:
        captures[-1] = {}
    report = execute(tmp_path, monkeypatch, captures, wall_end=end)
    assert not report["timing_validity"]["valid_for_performance_comparison"]
    assert all(report[k] is None for k in ("rtf", "throughput_x", "latency_p50_s", "latency_p95_s"))
    assert report["observed_timing"]["throughput_x"] == 2 and report["compute_s"] == 1
    assert report["results"][0]["elapsed_s"] == 1


def test_preload_power_change_does_not_taint_stable_timed_section(tmp_path, monkeypatch):
    report = execute(tmp_path, monkeypatch, [environment(), environment("Battery Power"), environment("Battery Power")])
    assert report["timing_validity"]["valid_for_performance_comparison"]
    assert report["throughput_x"] == 2


@pytest.mark.parametrize("valid", [True, False])
def test_cli_exit_status_exposes_invalid_measurements(monkeypatch, capsys, valid):
    monkeypatch.setattr(sys, "argv", ["asr", "bench", "--manifest", "input.jsonl", "--model", "q8", "--output", "saved.json"])
    monkeypatch.setattr(benchmark, "run", lambda args: {"timing_validity": {
        "valid_for_performance_comparison": valid, "reasons": [] if valid else ["clock_windows_invalid"]}})
    if valid:
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        message = capsys.readouterr().err
        assert "clock_windows_invalid" in message and "saved.json" in message
