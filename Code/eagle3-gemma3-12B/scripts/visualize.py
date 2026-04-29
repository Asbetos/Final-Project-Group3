"""CLI wrapper for plot generation."""

import argparse

from pathlib import Path
import sys

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.visualize import generate_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from a summary CSV")
    parser.add_argument(
        "--csv-path",
        default="artifacts/results/eagle3_gemma3_full/summary.csv",
        help="Path to the input summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/figures/generated",
        help="Directory to write generated plots",
    )
    args = parser.parse_args()
    generate_all_plots(csv_path=args.csv_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
