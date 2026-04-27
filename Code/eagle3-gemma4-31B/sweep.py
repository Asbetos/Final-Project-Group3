"""CLI entry point for the full experiment grid sweep."""

import argparse
import logging
import os
import sys

from config import (
    ALL_GAMMAS,
    ALL_TASKS,
    ALL_TEMPERATURES,
    ALL_TREE_BUDGETS,
    PAIR_MAP,
    ALL_PAIRS,
    EAGLE3_PAIR_MAP,
    ALL_EAGLE3_PAIRS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma speculative decoding experiment sweep"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["F", "G"],
        choices=["F", "G"],
        help=(
            "Standard speculative pair IDs to run "
            "(default: F G; F=Gemma3-12B+1B, G=Gemma4-31B+1B)"
        ),
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=int,
        default=ALL_GAMMAS,
        help=f"Speculation lengths (default: {ALL_GAMMAS})",
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=ALL_TEMPERATURES,
        help=f"Temperatures (default: {ALL_TEMPERATURES})",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        choices=ALL_TASKS,
        help=f"Task names (default: {ALL_TASKS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens per prompt (default: 128)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts per task (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment grid without running",
    )
    # EAGLE-3 arguments
    parser.add_argument(
        "--eagle3-only",
        action="store_true",
        help="Run only EAGLE-3 experiments (skip standard pairs)",
    )
    parser.add_argument(
        "--eagle3",
        action="store_true",
        help="Run EAGLE-3 experiments",
    )
    parser.add_argument(
        "--eagle3-pairs",
        nargs="+",
        default=["I"],
        choices=["H", "I"],
        help="EAGLE-3 pair IDs to run (default: I; H=Gemma4-31B EAGLE-3, I=Gemma3-12B EAGLE-3)",
    )
    parser.add_argument(
        "--tree-budgets",
        nargs="+",
        type=int,
        default=ALL_TREE_BUDGETS,
        help=f"Tree budgets for EAGLE-3 (default: {ALL_TREE_BUDGETS})",
    )
    return parser.parse_args()


def main():
    import torch

    # A100 GPU optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_cap = torch.cuda.get_device_capability(0)
        logger.info("GPU: %s (compute capability %d.%d)", gpu_name, *gpu_cap)

    args = parse_args()
    pairs = [PAIR_MAP[pid] for pid in args.pairs]

    total_baseline = len(args.tasks) * len(args.temps) * len(pairs)
    total_spec = len(pairs) * len(args.gammas) * len(args.temps) * len(args.tasks)
    total = total_baseline + total_spec

    logger.info(
        "Experiment grid: %d pairs x %d gammas x %d temps x %d tasks = "
        "%d speculative + %d baseline = %d total configs",
        len(pairs),
        len(args.gammas),
        len(args.temps),
        len(args.tasks),
        total_spec,
        total_baseline,
        total,
    )

    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN — would execute the following configurations:")
        print(f"{'='*70}\n")
        for pair in pairs:
            print(f"Pair {pair.pair_id}: {pair.target_model_id} + {pair.draft_model_id}")
            print(f"  4-bit target: {pair.target_quantize_4bit}")
            print(f"  VRAM estimate: {pair.total_vram_estimate_gb:.1f} GB\n")
            for task in args.tasks:
                for temp in args.temps:
                    print(f"  [BASELINE] {task} t={temp}")
            for gamma in args.gammas:
                for temp in args.temps:
                    for task in args.tasks:
                        print(f"  [SPEC] {task} gamma={gamma} t={temp}")
            print()
        if args.eagle3:
            eagle3_pairs = [EAGLE3_PAIR_MAP[pid] for pid in args.eagle3_pairs]
            for pair in eagle3_pairs:
                print(f"EAGLE-3 Pair {pair.pair_id}: {pair.target_model_id} + draft head")
                print(f"  4-bit target: {pair.target_quantize_4bit}")
                print(f"  Checkpoint: {pair.checkpoint_path}\n")
                for task in args.tasks:
                    for temp in args.temps:
                        print(f"  [BASELINE] {task} t={temp}")
                for tb in args.tree_budgets:
                    for temp in args.temps:
                        for task in args.tasks:
                            print(f"  [EAGLE3] {task} tree_budget={tb} t={temp}")
                print()
        print(f"Total: {total} standard configs, {args.num_prompts} prompts each")
        return

    # Defer heavy imports until actually needed (allows --dry-run without GPU deps)
    from runner import run_pair_sweep, run_eagle3_pair_sweep

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "baseline"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "speculative"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "eagle3"), exist_ok=True)

    # Run standard speculative decoding pair by pair (skip if --eagle3-only)
    if not getattr(args, 'eagle3_only', False):
        for pair in pairs:
            logger.info("=" * 70)
            logger.info("Starting pair %s", pair.pair_id)
            logger.info("=" * 70)
            run_pair_sweep(
                pair,
                gammas=args.gammas,
                temperatures=args.temps,
                tasks=args.tasks,
                max_new_tokens=args.max_tokens,
                num_prompts=args.num_prompts,
                seed=args.seed,
                output_dir=args.output_dir,
            )

    if getattr(args, 'eagle3_only', False) and not args.eagle3:
        args.eagle3 = True

    if args.eagle3:
        eagle3_pairs = [EAGLE3_PAIR_MAP[pid] for pid in args.eagle3_pairs]
        for pair in eagle3_pairs:
            logger.info("=" * 70)
            logger.info("Starting EAGLE-3 pair %s", pair.pair_id)
            logger.info("=" * 70)
            run_eagle3_pair_sweep(
                pair,
                tree_budgets=args.tree_budgets,
                temperatures=args.temps,
                tasks=args.tasks,
                max_new_tokens=args.max_tokens,
                num_prompts=args.num_prompts,
                seed=args.seed,
                output_dir=args.output_dir,
            )

    logger.info("All experiments complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
