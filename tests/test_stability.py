from copy import deepcopy

import pytest

from qwen_asr_opt.stability import cpu_intervals, window_mean


def sample(t, counters):
    return {"epoch": 100+t, "monotonic": 10+t, "read_s": .003,
            "system_cpu": {"user": 2*t, "nice": 0, "system": t, "idle": 9*t},
            "processes": {str(pid): {"cpu_s": cpu, "created": 1} for pid, cpu in counters.items()}}


def test_cpu_seconds_and_partial_window():
    rows = [sample(0, {1: 0, 2: 0, 3: 5}), sample(.5, {1: 1, 2: .01, 3: 5.3}), sample(1, {1: 2, 2: .02, 3: 5.6})]
    intervals = cpu_intervals(rows, asr_pid=1, sampler_pid=2)
    out = window_mean(intervals, 100.25, 100.75)
    assert out["coverage"] == 1
    assert out["means"] == pytest.approx({"asr_cores": 2, "sampler_cores": .02, "observed_other_cores": .6, "system_busy_cores": 3})


def test_reused_pids_births_missing_and_decreasing_counters():
    a = sample(0, {1: 0, 2: 1, 3: 8, 4: 9, 6: 3})
    b = sample(.5, {1: .5, 2: 1.01, 3: 99, 4: 2, 5: 20})
    b["processes"]["3"]["created"] = 2
    out = cpu_intervals([a, b], asr_pid=1, sampler_pid=2)[0]
    assert out["asr_cores"] == 1 and out["observed_other_cores"] == 0
    assert out["matched_processes"] == 2 and out["skipped_current_processes"] == 3


def test_missing_asr_or_sampler_cannot_imply_zero_cpu():
    a, b = sample(0, {1: 0}), sample(.5, {1: .5})
    assert cpu_intervals([a, b], asr_pid=1, sampler_pid=2) == []


@pytest.mark.parametrize("change", ["clock", "gap", "read", "nan", "system", "negative_read"])
def test_invalid_counter_intervals_are_missing(change):
    a, b = sample(0, {1: 0, 2: 0}), sample(.5, {1: .5, 2: .01})
    if change == "clock":
        b["epoch"] += 1
    elif change == "gap":
        b["epoch"] += 2
        b["monotonic"] += 2
    elif change == "read":
        b["read_s"] = .051
    elif change == "nan":
        b["epoch"] = float("nan")
    elif change == "system":
        b["system_cpu"]["system"] = -1
    else:
        b["read_s"] = -.1
    assert cpu_intervals([a, b], asr_pid=1, sampler_pid=2) == []


def test_missing_coverage_is_not_zero_utilization():
    row = {"start": 0, "end": .5, "value": 2}
    assert window_mean([row], 0, 1, fields=("value",))["means"]["value"] is None
    assert not window_mean([], 0, 1)["valid"]


def test_overlapping_intervals_are_not_double_counted():
    row = {"start": 0, "end": 1, "value": 2}
    b = deepcopy(row)
    b.update(start=.5, end=1.5, value=4)
    out = window_mean([row, b], 0, 1.5, fields=("value",))
    assert out["coverage"] == 1 and out["means"]["value"] == pytest.approx(8/3)


@pytest.mark.parametrize("start,end", [(0, 0), (1, 0), (float("nan"), 1)])
def test_invalid_window(start, end):
    with pytest.raises(ValueError):
        window_mean([], start, end)
