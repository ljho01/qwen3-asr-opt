"""Integrate measured system-wide SoC power; never infer watts from runtime."""
from __future__ import annotations

import re

POWER = re.compile(r"^(CPU|GPU|ANE) Power:\s*([0-9.]+)\s*mW", re.MULTILINE)
ELAPSED = re.compile(r"\(([0-9.]+)ms elapsed\)")

# Separate parent process: the sampled program can reset/ignore its own timers.
# WNOHANG does not reap a live child, so its PID cannot be reused before timeout kill.
WATCHDOG = r'''
use strict;
use warnings;
use POSIX qw(WNOHANG);
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC sleep);
my $seconds = shift @ARGV;
my $deadline = clock_gettime(CLOCK_MONOTONIC) + $seconds;
my $child = fork();
die "fork: $!" unless defined $child;
if ($child == 0) { exec @ARGV; die "exec: $!"; }
while (1) {
    my $result = waitpid($child, WNOHANG);
    if ($result == $child) {
        my $status = $?;
        exit(($status & 127) ? 128 + ($status & 127) : $status >> 8);
    }
    die "waitpid: $!" if $result == -1;
    if (clock_gettime(CLOCK_MONOTONIC) >= $deadline) {
        kill 9, $child;
        waitpid($child, 0);
        exit 124;
    }
    sleep(0.02);
}
'''


def deadline_command(command: list[str], seconds: float = 299) -> list[str]:
    """Run only the supplied child under an independent monotonic watchdog."""
    if not command or not 0 < seconds <= 299:
        raise ValueError("Nonempty command and 0 < seconds <= 299 required")
    return ["/usr/bin/perl", "-e", WATCHDOG, "--", str(seconds), *command]


def parse_sample(text: str, end_epoch: float) -> dict | None:
    elapsed = ELAPSED.search(text)
    parts = {}
    # cpu_power reports the three components and their combined SoC estimate.
    # gpu_power can print a second GPU value from a different sampling counter;
    # preserve the first component rather than mixing those counters.
    for kind, value in POWER.findall(text):
        parts.setdefault(kind.lower(), float(value) / 1000)
    if not elapsed or set(parts) != {"cpu", "gpu", "ane"}:
        return None
    duration = float(elapsed[1]) / 1000
    if duration <= 0:
        return None
    return {"start_epoch": end_epoch - duration, "end_epoch": end_epoch,
            "watts": sum(parts.values()), "components_w": parts}


def integrate(samples: list[dict], start: float, end: float, audio_s: float,
              idle_watts: float | None = None) -> dict:
    if end <= start or audio_s <= 0:
        raise ValueError("Positive measurement window and audio duration required")
    joules = 0.0
    covered = 0.0
    previous_end = start
    for sample in sorted(samples, key=lambda s: s["start_epoch"]):
        # Receipt timestamp jitter can overlap adjacent samples: count each instant once.
        lo = max(start, sample["start_epoch"], previous_end)
        hi = min(end, sample["end_epoch"])
        if hi <= lo:
            continue
        overlap = hi - lo
        joules += sample["watts"] * overlap
        covered += overlap
        previous_end = hi
    coverage = covered / (end - start)
    valid = coverage >= 0.95
    return {
        "scope": "system_soc_cpu_gpu_ane_estimate", "coverage": coverage,
        "valid": valid, "sample_count": len(samples), "covered_s": covered,
        "gross_joules": joules if valid else None,
        "joules_per_audio_minute": joules / (audio_s / 60) if valid else None,
        "mean_watts": joules / covered if valid else None,
        "idle_watts": idle_watts,
        # Negative differences are kept: background variability must stay visible.
        "idle_subtracted_joules": joules - idle_watts * covered
            if valid and idle_watts is not None else None,
        "status": "measured" if valid else "insufficient_sample_coverage",
    }
