"""Bounded unprivileged macmon sampling; keep unsupported CPU energy unavailable."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from .energy import deadline_command, integrate


def intervals(rows, field, max_gap_s=0.75):
    """Approximate counter intervals/SMC trapezoids from emitted JSON timestamps.

    macmon emits timestamps after sensor reads, so these are not exact IOReport
    sample boundaries. The initial sample has no preceding boundary and is omitted.
    `sys_power` is accepted only when PSTR dominates macmon's max(PSTR, all_power).
    CPU+GPU+ANE aggregate is deliberately unsupported while the CPU counter is zero.
    """
    if field not in ("gpu_power", "sys_power"):
        raise ValueError("Only separately labelled GPU or SMC system energy is supported")
    samples = []
    for previous, current in pairwise(rows):
        start = datetime.fromisoformat(previous["timestamp"]).timestamp()
        end = datetime.fromisoformat(current["timestamp"]).timestamp()
        if not 0 < end - start <= max_gap_s:
            continue
        values = [float(previous[field]), float(current[field])]
        if any(not math.isfinite(v) or v < 0 for v in values):
            continue
        if field == "sys_power":
            if any(float(row[field]) <= float(row["all_power"]) for row in (previous, current)):
                continue  # Cannot distinguish PSTR from macmon's lower-bound fallback.
            watts = sum(values) / 2
        else:
            watts = values[1]  # IOReport energy delta / actual sample duration.
        samples.append({"start_epoch": start, "end_epoch": end, "watts": watts})
    return samples


def measured_energy(rows, start, end, audio_s, idle_window):
    result = {"cpu_gpu_ane": None, "cpu_counter_status": "unusable_zero_under_load",
              "backend": "macmon0.8.2_unprivileged", "timestamp_scope": "post_sensor_read_json_timestamp"}
    for field, scope in (("gpu_power", "system_gpu_ioreport_estimate"),
                         ("sys_power", "system_smc_PSTR_estimate")):
        samples = intervals(rows, field)
        idle = integrate(samples, *idle_window, 60)
        value = integrate(samples, start, end, audio_s, idle_watts=idle["mean_watts"])
        value.update(scope=scope, idle_coverage=idle["coverage"],
                     timing_note="Emission timestamps approximate counter boundaries; PSTR is a sampled system estimate, not a wall meter or per-process reading.")
        result[field] = value
    return result


class JsonlCapture:
    """Own only this sampler's process group, with an independent299s watchdog."""

    def __init__(self, command, output, *, seconds=299, sample_kind=None):
        if os.geteuid() == 0:
            raise ValueError("This capture must run as an ordinary user")
        output = Path(output)
        if output.exists() or output.with_suffix(".stderr.txt").exists():
            raise FileExistsError(output)
        self.command = deadline_command(command, seconds)
        self.sample_kind = sample_kind
        self.log = output.open("x")
        self.errors = output.with_suffix(".stderr.txt").open("x")
        self.rows = []
        self.metadata = []
        self.parse_errors = []
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.started = time.time()
        self.closed = False
        try:
            self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=self.errors,
                                            text=True, start_new_session=True)
        except BaseException:
            self.log.close()
            self.errors.close()
            raise
        self.reader = threading.Thread(target=self._read, name="power-sampler-reader", daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            received = time.time()
            self.log.write(line)
            self.log.flush()
            try:
                row = json.loads(line)
                row["received_epoch"] = received
                with self.lock:
                    if self.sample_kind is None or row.get("kind") == self.sample_kind:
                        self.rows.append(row)
                    else:
                        self.metadata.append(row)
                    if len(self.rows) >= 3:
                        self.ready.set()
            except (ValueError, TypeError) as error:
                self.parse_errors.append(str(error))

    def snapshot(self):
        with self.lock:
            return list(self.rows)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        self.finished_epoch = time.time()
        self.process.stdout.close()
        self.log.close()
        self.errors.close()
        if self.reader.is_alive():
            raise RuntimeError("Sampler reader failed to close")


class MacmonCapture(JsonlCapture):
    def __init__(self, executable, output, *, seconds=299):
        super().__init__([str(executable), "-i", "250", "pipe", "-s", "1200"], output, seconds=seconds)


class SmcCapture(JsonlCapture):
    def __init__(self, executable, output, *, seconds=299, interval_ms=250):
        if not 100 <= interval_ms <= 1000:
            raise ValueError("Require100 <= interval_ms <=1000")
        super().__init__([str(executable), str(round(seconds * 1000)), str(interval_ms)],
                         output, seconds=seconds, sample_kind="sample")


def smc_energy(rows, start, end, audio_s, *, idle_watts=None, max_gap_s=0.75):
    """Integrate raw PSTR linearly, including exact clipping at window boundaries.

    Values are observed at the midpoint of the client read; hardware sensor update
    timing and calibration are unknown. Reject zero/invalid readings, clock jumps,
    reads longer than50ms and gaps longer than750ms. Require95% time coverage.
    """
    if end <= start or audio_s <= 0:
        raise ValueError("Positive window and audio duration required")

    def point(row):
        a, b, duration = row["start_epoch"], row["end_epoch"], row["read_duration_s"]
        watts = row["pstr_watts"]
        if not all(math.isfinite(x) for x in (a, b, duration, watts, row["read_monotonic_s"])):
            return None
        if not 0 <= duration <= 0.05 or b < a or abs(b - a - duration) > 0.005 or watts <= 0:
            return None
        return (a + b) / 2, watts

    covered, joules, previous_end = 0.0, 0.0, start
    for a, b in pairwise(rows):
        left, right = point(a), point(b)
        if left is None or right is None:
            continue
        t0, w0 = left
        t1, w1 = right
        if not 0 < t1 - t0 <= max_gap_s:
            continue
        # A discontinuity between monotonic and wall time invalidates this interval.
        mono_delta = b["read_monotonic_s"] - a["read_monotonic_s"]
        if abs(mono_delta - (t1 - t0)) > 0.005:
            continue
        lo, hi = max(start, t0, previous_end), min(end, t1)
        if hi <= lo:
            continue
        slope = (w1 - w0) / (t1 - t0)
        watts_lo, watts_hi = w0 + slope * (lo - t0), w0 + slope * (hi - t0)
        joules += (watts_lo + watts_hi) / 2 * (hi - lo)
        covered += hi - lo
        previous_end = hi
    coverage = covered / (end - start)
    valid = coverage >= 0.95
    return {"scope": "system_smc_PSTR_raw_estimate", "coverage": coverage, "covered_s": covered,
            "valid": valid, "sample_count": len(rows), "gross_joules": joules if valid else None,
            "joules_per_audio_minute": joules / (audio_s / 60) if valid else None,
            "mean_watts": joules / covered if valid else None, "idle_watts": idle_watts,
            "idle_subtracted_joules": joules - idle_watts * covered if valid and idle_watts is not None else None,
            "status": "measured_estimate" if valid else "insufficient_sample_coverage",
            "timing_scope": "client_read_midpoint; hardware sensor update timing unknown"}
