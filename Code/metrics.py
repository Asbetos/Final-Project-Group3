"""Metric dataclasses, CUDA timing helpers, and result serialization."""

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-round and per-generation metric containers
# ---------------------------------------------------------------------------


@dataclass
class RoundMetrics:
    """Metrics for a single draft-verify round."""

    round_index: int
    draft_tokens_proposed: int
    tokens_accepted: int
    bonus_token_generated: bool
    total_tokens_produced: int
    per_token_accepted: List[bool]
    draft_time_ms: float
    verify_time_ms: float
    round_time_ms: float


@dataclass
class GenerationMetrics:
    """Aggregated metrics for one full generation (one prompt)."""

    prompt_index: int
    total_tokens_generated: int
    total_rounds: int
    wall_clock_ms: float
    ttft_ms: float
    tokens_per_second: float
    acceptance_rate: float
    acceptance_length: float
    draft_overhead_ratio: float
    peak_vram_bytes: int
    rounds: List[RoundMetrics] = field(default_factory=list)

    @staticmethod
    def aggregate(metrics_list: List["GenerationMetrics"]) -> Dict:
        """
        Aggregate across all prompts for one configuration.

        Returns dict with mean, std, p95 for each scalar metric.
        """
        if not metrics_list:
            return {}

        def _stats(values):
            arr = np.array(values, dtype=np.float64)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

        return {
            "tokens_per_second": _stats([m.tokens_per_second for m in metrics_list]),
            "acceptance_rate": _stats([m.acceptance_rate for m in metrics_list]),
            "acceptance_length": _stats([m.acceptance_length for m in metrics_list]),
            "ttft_ms": _stats([m.ttft_ms for m in metrics_list]),
            "wall_clock_ms": _stats([m.wall_clock_ms for m in metrics_list]),
            "draft_overhead_ratio": _stats(
                [m.draft_overhead_ratio for m in metrics_list]
            ),
            "peak_vram_gb": _stats(
                [m.peak_vram_bytes / (1024**3) for m in metrics_list]
            ),
            "total_tokens": _stats(
                [m.total_tokens_generated for m in metrics_list]
            ),
        }


# ---------------------------------------------------------------------------
# CUDA timing helpers
# ---------------------------------------------------------------------------


class CudaTimer:
    """Context manager for GPU-accurate timing via CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)


class WallTimer:
    """Context manager for wall-clock timing (user-perceived latency)."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def record_peak_vram() -> int:
    """Return peak VRAM allocated in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def reset_peak_vram() -> None:
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def _round_to_json(r: RoundMetrics) -> dict:
    return {
        "round": r.round_index,
        "proposed": r.draft_tokens_proposed,
        "accepted": r.tokens_accepted,
        "bonus": r.bonus_token_generated,
        "produced": r.total_tokens_produced,
        "draft_ms": round(r.draft_time_ms, 2),
        "verify_ms": round(r.verify_time_ms, 2),
    }


def _gen_to_json(g: GenerationMetrics) -> dict:
    return {
        "prompt_index": g.prompt_index,
        "total_tokens": g.total_tokens_generated,
        "rounds": g.total_rounds,
        "wall_clock_ms": round(g.wall_clock_ms, 2),
        "ttft_ms": round(g.ttft_ms, 2),
        "tps": round(g.tokens_per_second, 2),
        "acceptance_rate": round(g.acceptance_rate, 4),
        "acceptance_length": round(g.acceptance_length, 2),
        "draft_overhead": round(g.draft_overhead_ratio, 4),
        "peak_vram_gb": round(g.peak_vram_bytes / (1024**3), 2),
        "round_details": [_round_to_json(r) for r in g.rounds],
    }


def save_results_json(
    config_dict: dict,
    summary: dict,
    per_prompt: List[GenerationMetrics],
    path: str,
) -> None:
    """Write full results (config + summary + per-prompt metrics) as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": config_dict,
        "summary": summary,
        "per_prompt": [_gen_to_json(g) for g in per_prompt],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_summary_csv(
    row: dict,
    path: str = "results/summary.csv",
) -> None:
    """Append one summary row to the master CSV. Creates the file and header if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path)

    if file_exists:
        # Read existing header to merge with new row's keys
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_fields = next(reader, [])
        # Merge: keep existing order, append any new columns
        all_fields = list(existing_fields)
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)
        # If new columns were added, rewrite the file with the updated header
        if len(all_fields) > len(existing_fields):
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)
                writer.writerow(row)
        else:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
                writer.writerow(row)
    else:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
