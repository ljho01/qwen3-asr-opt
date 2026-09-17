from datetime import UTC, datetime

import pytest

from qwen_asr_opt.telemetry import intervals, measured_energy, smc_energy


def row(t, gpu, system, total=None):
    return {"timestamp": datetime.fromtimestamp(t, UTC).isoformat(), "gpu_power": gpu,
            "cpu_power": 0, "sys_power": system, "all_power": gpu if total is None else total}


def test_distinct_energy_scopes_and_missing_cpu_are_not_combined():
    rows = [row(100 + i / 4, 2 if i < 5 else 6, 10 if i < 5 else 20) for i in range(13)]
    result = measured_energy(rows, 101.25, 103, 60, (100, 101))
    assert result["cpu_gpu_ane"] is None
    assert result["gpu_power"]["gross_joules"] == pytest.approx(10.5)
    assert result["sys_power"]["gross_joules"] == pytest.approx(35)
    assert result["gpu_power"]["idle_watts"] == 2
    assert result["sys_power"]["idle_watts"] == 10
    assert result["sys_power"]["scope"] == "system_smc_PSTR_estimate"


def test_smc_trapezoid_and_counter_rectangle_have_distinct_semantics():
    samples = [row(10, 2, 10), row(10.5, 6, 20)]
    assert intervals(samples, "gpu_power")[0]["watts"] == 6
    assert intervals(samples, "sys_power")[0]["watts"] == 15


def test_invalid_gaps_nonfinite_values_and_clamped_system_power_are_excluded():
    assert intervals([row(10, 2, 10), row(12, 2, 10)], "gpu_power") == []
    assert intervals([row(10, 2, 10), row(10.25, float("nan"), 10)], "gpu_power") == []
    assert intervals([row(10, 2, 10), row(10.25, 2, 10, total=10)], "sys_power") == []
    with pytest.raises(ValueError, match="separately"):
        intervals([], "all_power")


def smc_row(t, watts):
    return {"start_epoch": t - 0.0001, "end_epoch": t + 0.0001,
            "read_duration_s": 0.0002, "read_monotonic_s": t - 10, "pstr_watts": watts}


def test_raw_smc_linear_clipped_boundaries_and_idle_subtraction():
    rows = [smc_row(100, 10), smc_row(100.5, 30)]
    result = smc_energy(rows, 100.25, 100.5, 30, idle_watts=10)
    assert result["gross_joules"] == pytest.approx(6.25)  # Mean25W over0.25s.
    assert result["joules_per_audio_minute"] == pytest.approx(12.5)
    assert result["idle_subtracted_joules"] == pytest.approx(3.75)
    assert result["coverage"] == 1


@pytest.mark.parametrize("field,value", [("pstr_watts", 0), ("pstr_watts", float("nan")),
                                         ("read_duration_s", 0.1), ("read_monotonic_s", 91),
                                         ("read_monotonic_s", float("nan"))])
def test_raw_smc_rejects_missing_values_slow_reads_and_clock_discontinuity(field, value):
    rows = [smc_row(100, 10), smc_row(100.5, 30)]
    rows[1][field] = value
    assert smc_energy(rows, 100, 100.5, 60)["gross_joules"] is None


def test_raw_smc_gap_is_missing_coverage_not_zero_energy():
    rows = [smc_row(100, 10), smc_row(100.25, 10), smc_row(101.25, 10)]
    result = smc_energy(rows, 100, 101.25, 60)
    assert result["coverage"] == pytest.approx(0.2)
    assert result["gross_joules"] is None
