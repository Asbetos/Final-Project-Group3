"""Thin CLI wrapper for the standard speculative sweep."""

from pathlib import Path
import sys

_MODULE_DIR = Path(__file__).resolve().parents[1]
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from core.sweep import main


if __name__ == "__main__":
    main()
