"""Launcher resilient to macOS hidden-file flags on editable-install .pth files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_asr_opt.cli import main

main()
