from copy import deepcopy

import pytest

from qwen_asr_opt.benchmark_conditions import power_signature, validate_conditions


def env(source="AC Power", ac=0, battery=1):
    return {"power_source": f"Now drawing from '{source}'\nBattery details omitted",
            "power_settings": f"Battery Power:\n powermode {battery}\n sleep 1\nAC Power:\n powermode {ac}\n sleep 1"}


def test_only_active_section_is_compared():
    out = validate_conditions([{"start_epoch": 10, "end_epoch": 11, "elapsed_s": 1}], [env(), env(battery=2)])
    assert out["valid_for_performance_comparison"]
    assert power_signature(env())["policy"] == {"powermode": 0}


def test_ac_battery_change_and_policy_are_reported_without_mutating_input():
    rows = [{"start_epoch": 0, "end_epoch": 1, "elapsed_s": 1}]
    original = deepcopy(rows)
    out = validate_conditions(rows, [env(), env("Battery Power")])
    assert out["reasons"] == ["power_source_changed", "active_power_policy_changed"]
    assert not out["valid_for_performance_comparison"] and rows == original


def test_power_source_change_even_when_policy_values_match():
    out = validate_conditions([{"start_epoch": 0, "end_epoch": 1, "elapsed_s": 1}], [env(battery=0), env("Battery Power", battery=0)])
    assert out["reasons"] == ["power_source_changed"]


def test_policy_change_on_same_source():
    out = validate_conditions([{"start_epoch": 0, "end_epoch": 1, "elapsed_s": 1}], [env(), env(ac=2)])
    assert out["reasons"] == ["active_power_policy_changed"]


@pytest.mark.parametrize("wall,elapsed", [(80.930274, 56.187624), (.9, 1), (0, 1), (1, 0), (1, float("nan"))])
def test_invalid_clock_windows(wall, elapsed):
    out = validate_conditions([{"start_epoch": 10, "end_epoch": 10+wall, "elapsed_s": elapsed}], [env()])
    assert out["reasons"] == ["clock_windows_invalid"]


def test_minor_adjacent_read_delay_is_allowed():
    assert validate_conditions([{"start_epoch": 10, "end_epoch": 11.0003, "elapsed_s": 1}], [env()])["valid_for_performance_comparison"]


@pytest.mark.parametrize("environment", [{}, {"power_source": "AC Power"}, {"power_source": "Now drawing from 'AC Power'", "power_settings": "Battery Power:\n powermode 1"}])
def test_unavailable_power_is_not_treated_as_normal(environment):
    out = validate_conditions([{"start_epoch": 0, "end_epoch": 1, "elapsed_s": 1}], [environment])
    assert out["reasons"] == ["power_state_unavailable"]


def test_legacy_low_power_mode_key_is_preserved():
    e = env()
    e["power_settings"] = e["power_settings"].replace("powermode", "lowpowermode")
    assert power_signature(e)["policy"] == {"lowpowermode": 0}


def test_empty_inputs_cannot_pass():
    out = validate_conditions([], [])
    assert out["reasons"] == ["no_measurement_windows", "power_state_unavailable"]


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        validate_conditions([], [], clock_tolerance_s=tolerance)
