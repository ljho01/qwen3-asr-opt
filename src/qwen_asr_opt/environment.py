from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return (result.stdout or result.stderr).strip()


def capture() -> dict:
    # Deliberately omit serial number, UUID, hostname and account details.
    packages = {}
    for name in ["mlx", "mlx-metal", "mlx-qwen3-asr", "mlx-whisper", "numpy", "jiwer"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "os": platform.platform(), "python": platform.python_version(),
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_cores": command("sysctl", "-n", "hw.ncpu"),
        "memory_bytes": command("sysctl", "-n", "hw.memsize"),
        "power_source": command("pmset", "-g", "batt"),
        "power_settings": command("pmset", "-g", "custom"),
        "thermal": command("pmset", "-g", "therm"),
        "system_load_average": os.getloadavg(),
        "swap": command("sysctl", "vm.swapusage"),
        "memory_pressure": command("memory_pressure", "-Q"),
        "packages": packages,
    }
