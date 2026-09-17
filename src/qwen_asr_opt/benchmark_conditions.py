"""Conservative boundary/clock validity checks without changing stored results."""
from __future__ import annotations

import math
import re


def power_signature(environment):
    source = environment.get("power_source", "")
    match = re.search(r"^Now drawing from '(AC Power|Battery Power)'", source)
    if match is None:
        return {"valid": False, "source": None, "policy": None}
    name = match[1]
    sections = re.split(r"(?m)^(AC Power|Battery Power):\s*\n", environment.get("power_settings", ""))
    policies = dict(zip(sections[1::2], sections[2::2], strict=True))
    text = policies.get(name, "")
    policy = {key: int(value) for key, value in re.findall(r"(?m)^\s*(powermode|lowpowermode)\s+(\d+)\s*$", text)}
    return {"valid": bool(policy), "source": name, "policy": policy or None}


def validate_conditions(windows, environments, *, clock_tolerance_s=.01):
    """Report tainted comparisons, not a guarantee that every condition was stable.

    Boundary captures cannot detect an unobserved change that reverses between
    snapshots. A clock mismatch does not by itself distinguish sleep, wall-clock
    adjustment, or delayed clock reads. Keep all raw measurements unchanged.
    """
    if not math.isfinite(clock_tolerance_s) or clock_tolerance_s < 0:
        raise ValueError("Require a finite nonnegative clock tolerance")
    checks = []
    for row in windows:
        start, end, elapsed = (row[k] for k in ("start_epoch", "end_epoch", "elapsed_s"))
        finite = all(math.isfinite(x) for x in (start, end, elapsed))
        wall = end-start if finite else None
        delta = wall-elapsed if finite else None
        checks.append({"wall_elapsed_s": wall, "counter_elapsed_s": elapsed if finite else None,
                       "clock_difference_s": delta,
                       "valid": finite and wall > 0 and elapsed > 0 and abs(delta) <= clock_tolerance_s})
    signatures = [power_signature(e) for e in environments]
    reasons = []
    if not checks:
        reasons.append("no_measurement_windows")
    elif not all(c["valid"] for c in checks):
        reasons.append("clock_windows_invalid")
    if not signatures or not all(s["valid"] for s in signatures):
        reasons.append("power_state_unavailable")
    if len({s["source"] for s in signatures if s["source"] is not None}) > 1:
        reasons.append("power_source_changed")
    policies = {tuple(sorted(s["policy"].items())) for s in signatures if s["policy"] is not None}
    if len(policies) > 1:
        reasons.append("active_power_policy_changed")
    return {"valid_for_performance_comparison": not reasons, "reasons": reasons,
            "clock_tolerance_s": clock_tolerance_s, "clock_windows": checks,
            "power_signatures": signatures,
            "scope": "Observed power boundaries and clock agreement only; not proof of quiet CPU/GPU, stable thermals, or constant power between snapshots"}
