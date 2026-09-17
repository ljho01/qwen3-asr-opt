"""CPU-only accounting for read-only runtime-stability diagnostics.

One core equivalent is one CPU second per wall second. Process coverage is
partial: only readable processes present with the same birth time in both
snapshots contribute. This is not attribution of GPU use or causal interference.
"""
from __future__ import annotations

import math
from itertools import pairwise

FIELDS = ("asr_cores", "sampler_cores", "observed_other_cores", "system_busy_cores")


def cpu_intervals(samples, *, asr_pid, sampler_pid, max_gap_s=1.5, max_read_s=.05):
    rows = []
    for left, right in pairwise(samples):
        start, end = left["epoch"], right["epoch"]
        dt = right["monotonic"]-left["monotonic"]
        numbers = (start, end, dt, left["read_s"], right["read_s"])
        if (not all(math.isfinite(n) for n in numbers) or not 0 < dt <= max_gap_s
                or abs((end-start)-dt) > .01 or min(left["read_s"], right["read_s"]) < 0
                or max(left["read_s"], right["read_s"]) > max_read_s):
            continue
        system = [right["system_cpu"][k]-left["system_cpu"][k] for k in ("user", "nice", "system")]
        if any(not math.isfinite(n) or n < 0 for n in system):
            continue
        values = dict.fromkeys(FIELDS, 0.0)
        values["system_busy_cores"] = sum(system)/dt
        matched = skipped = 0
        matched_ids = set()
        for pid, current in right["processes"].items():
            previous = left["processes"].get(pid)
            if previous is None or current["created"] != previous["created"]:
                skipped += 1
                continue
            delta = current["cpu_s"]-previous["cpu_s"]
            if not math.isfinite(delta) or delta < 0:
                skipped += 1
                continue
            field = "asr_cores" if int(pid) == asr_pid else "sampler_cores" if int(pid) == sampler_pid else "observed_other_cores"
            values[field] += delta/dt
            matched += 1
            matched_ids.add(int(pid))
        if not {asr_pid, sampler_pid}.issubset(matched_ids):
            continue
        rows.append({"start": start, "end": end, **values, "matched_processes": matched,
                     "skipped_current_processes": skipped})
    return rows


def window_mean(intervals, start, end, fields=FIELDS):
    """Clip partial intervals; gaps are missing, never zero or interpolated."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Require a finite positive measurement window")
    sums = dict.fromkeys(fields, 0.0)
    covered, last_end = 0.0, start
    for row in intervals:
        lo, hi = max(start, row["start"], last_end), min(end, row["end"])
        if hi <= lo:
            continue
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in fields):
            continue
        duration = hi-lo
        covered += duration
        last_end = hi
        for key in fields:
            sums[key] += row[key]*duration
    coverage = covered/(end-start)
    valid = coverage >= .95
    return {"coverage": coverage, "valid": valid, "covered_s": covered,
            "means": {key: total/covered if valid else None for key, total in sums.items()}}
