import subprocess
import time

import pytest

from qwen_asr_opt.energy import deadline_command, integrate, parse_sample


def test_milliwatts_seconds_and_audio_minute_units():
    text = "*** Sampled system activity (1000.00ms elapsed) ***\nCPU Power: 1000 mW\nGPU Power: 2000 mW\nANE Power: 0 mW\n"
    sample = parse_sample(text, 11)
    result = integrate([sample], 10, 11, 60, idle_watts=1)
    assert result["gross_joules"] == 3
    assert result["joules_per_audio_minute"] == 3
    assert result["idle_subtracted_joules"] == 2


def test_missing_ane_is_not_silently_zero():
    assert parse_sample("(1000ms elapsed)\nCPU Power: 1 mW\nGPU Power: 2 mW", 1) is None
    result = integrate([], 1, 2, 10)
    assert result["gross_joules"] is None
    assert not result["valid"]


def test_gpu_diagnostic_does_not_replace_combined_power_component():
    text = ("(200ms elapsed)\nCPU Power: 8050 mW\nGPU Power: 6304 mW\n"
            "ANE Power: 0 mW\nCombined Power (CPU + GPU + ANE): 14354 mW\n"
            "**** GPU usage ****\nGPU Power: 6932 mW\n")
    assert parse_sample(text, 1)["watts"] == pytest.approx(14.354)


def test_partial_overlap_and_no_double_counting():
    samples = [{"start_epoch": 0, "end_epoch": 1, "watts": 4},
               {"start_epoch": 0.9, "end_epoch": 2, "watts": 2}]
    result = integrate(samples, 0.5, 1.5, 30)
    assert result["gross_joules"] == pytest.approx(3)
    assert result["coverage"] == 1


def test_watchdog_stops_child_that_cancels_and_ignores_alarms():
    command = ["/usr/bin/perl", "-e", '$SIG{ALRM} = "IGNORE"; alarm 0; sleep 10;']
    start = time.monotonic()
    result = subprocess.run(deadline_command(command, seconds=0.2), check=False)
    assert result.returncode == 124
    assert time.monotonic() - start < 2


def test_watchdog_preserves_early_exit_code():
    result = subprocess.run(deadline_command(["/usr/bin/perl", "-e", "exit 7;"], seconds=1),
                            check=False)
    assert result.returncode == 7
