"""Thin CLI wrapper for validation and smoke tests."""

from pathlib import Path
import sys

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.test_correctness import main


if __name__ == "__main__":
    main()
