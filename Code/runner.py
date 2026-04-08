"""Single-config experiment executor and pair-level sweep orchestrator."""

import logging
import os
import time
from typing import Dict, List, Optional

import torch

from baseline import autoregressive_decode
from config import (
    ALL_GAMMAS,
    ALL_TASKS,
    ALL_TEMPERATURES,
    ExperimentConfig,
    ModelPairConfig,
)
from data import tokenize_prompts
from metrics import GenerationMetrics, save_results_json, save_summary_csv
from models import load_model_pair, unload_models
from speculative import speculative_decode

logger = logging.getLogger(__name__)


def _make_generator(seed: int, device: str = "cuda") -> torch.Generator:
    """Create a seeded torch generator on the given device."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def run_single_config(
    config: ExperimentConfig,
    target_model,
    draft_model,
    tokenizer,
    output_dir: str = "results",
) -> Dict:
    """
    Run one full experiment configuration (one cell in the 180-grid).

    Steps:
        1. Tokenize prompts for config.task.
        2. Run warm-up passes (discard metrics).
        3. Run speculative decoding on each prompt, collecting metrics.
        4. Aggregate and save results.

    Returns:
        Summary dict with mean/std/p95 for all metrics.
    """
    run_id = config.run_id()
    logger.info("Running config: %s", run_id)

    prompts = tokenize_prompts(
        config.task, tokenizer, config.num_prompts, config.seed
    )

    # Warm-up passes (discard results)
    for i in range(min(config.num_warmup, len(prompts))):
        gen = _make_generator(config.seed + i)
        speculative_decode(
            target_model,
            draft_model,
            prompts[i]["input_ids"],
            prompts[i]["attention_mask"],
            gamma=config.gamma,
            temperature=config.temperature,
            max_new_tokens=min(32, config.max_new_tokens),
            tokenizer=tokenizer,
            generator=gen,
        )

    # Main measurement loop
    all_metrics: List[GenerationMetrics] = []
    for idx, prompt_data in enumerate(prompts):
        gen = _make_generator(config.seed + 1000 + idx)
        result = speculative_decode(
            target_model,
            draft_model,
            prompt_data["input_ids"],
            prompt_data["attention_mask"],
            gamma=config.gamma,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            tokenizer=tokenizer,
            generator=gen,
        )
        m = result["metrics"]
        m.prompt_index = idx
        all_metrics.append(m)

        if (idx + 1) % 10 == 0:
            logger.info("  %s: %d/%d prompts done", run_id, idx + 1, len(prompts))

    # Aggregate
    summary = GenerationMetrics.aggregate(all_metrics)

    # Save per-prompt JSON
    json_path = os.path.join(output_dir, "speculative", f"{run_id}.json")
    save_results_json(config.to_dict(), summary, all_metrics, json_path)

    logger.info("Config %s complete. Mean TPS: %.1f", run_id, summary["tokens_per_second"]["mean"])
    return summary


def run_baseline_for_pair(
    pair: ModelPairConfig,
    target_model,
    tokenizer,
    tasks: List[str],
    temperatures: List[float],
    max_new_tokens: int = 128,
    num_prompts: int = 50,
    num_warmup: int = 3,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict:
    """
    Run baseline autoregressive decoding for all (task, temperature) combos.

    Baseline is independent of gamma, so we only run it once per
    (pair, task, temperature) combination.

    Returns:
        Dict mapping (task, temperature) -> summary dict.
    """
    baselines = {}

    for task in tasks:
        prompts = tokenize_prompts(task, tokenizer, num_prompts, seed)

        for temp in temperatures:
            run_id = f"{pair.pair_id}_{task}_baseline_t{temp}"
            logger.info("Running baseline: %s", run_id)

            # Warm-up
            for i in range(min(num_warmup, len(prompts))):
                gen = _make_generator(seed + i)
                autoregressive_decode(
                    target_model,
                    prompts[i]["input_ids"],
                    prompts[i]["attention_mask"],
                    temperature=temp,
                    max_new_tokens=min(32, max_new_tokens),
                    tokenizer=tokenizer,
                    generator=gen,
                )

            # Measure
            all_metrics: List[GenerationMetrics] = []
            for idx, prompt_data in enumerate(prompts):
                gen = _make_generator(seed + 1000 + idx)
                result = autoregressive_decode(
                    target_model,
                    prompt_data["input_ids"],
                    prompt_data["attention_mask"],
                    temperature=temp,
                    max_new_tokens=max_new_tokens,
                    tokenizer=tokenizer,
                    generator=gen,
                )
                m = result["metrics"]
                m.prompt_index = idx
                all_metrics.append(m)

            summary = GenerationMetrics.aggregate(all_metrics)
            baselines[(task, temp)] = summary

            # Save
            json_path = os.path.join(output_dir, "baseline", f"{run_id}.json")
            config_dict = {
                "pair_id": pair.pair_id,
                "task": task,
                "temperature": temp,
                "is_baseline": True,
            }
            save_results_json(config_dict, summary, all_metrics, json_path)

            # Append to master CSV
            csv_row = {
                "pair_id": pair.pair_id,
                "task": task,
                "gamma": 0,
                "temperature": temp,
                "is_baseline": True,
                "mean_tps": round(summary["tokens_per_second"]["mean"], 2),
                "std_tps": round(summary["tokens_per_second"]["std"], 2),
                "p95_tps": round(summary["tokens_per_second"]["p95"], 2),
                "mean_acceptance_rate": 1.0,
                "mean_acceptance_length": 1.0,
                "mean_ttft_ms": round(summary["ttft_ms"]["mean"], 2),
                "mean_peak_vram_gb": round(summary["peak_vram_gb"]["mean"], 2),
                "mean_draft_overhead": 0.0,
                "speedup": 1.0,
            }
            save_summary_csv(csv_row, os.path.join(output_dir, "summary.csv"))

            logger.info(
                "Baseline %s done. Mean TPS: %.1f",
                run_id,
                summary["tokens_per_second"]["mean"],
            )

    return baselines


def run_pair_sweep(
    pair: ModelPairConfig,
    gammas: Optional[List[int]] = None,
    temperatures: Optional[List[float]] = None,
    tasks: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    num_prompts: int = 50,
    seed: int = 42,
    output_dir: str = "results",
) -> None:
    """
    Load models for one pair, then sweep all (gamma, temperature, task) combos.

    Runs baselines first, then speculative configs.
    Unloads models when done.
    """
    gammas = gammas or ALL_GAMMAS
    temperatures = temperatures or ALL_TEMPERATURES
    tasks = tasks or ALL_TASKS

    total_configs = len(gammas) * len(temperatures) * len(tasks)
    logger.info(
        "Pair %s: %d speculative configs + baselines",
        pair.pair_id,
        total_configs,
    )

    # Load models
    target_model, draft_model, tokenizer = load_model_pair(pair)

    try:
        # Run baselines first
        baselines = run_baseline_for_pair(
            pair,
            target_model,
            tokenizer,
            tasks,
            temperatures,
            max_new_tokens=max_new_tokens,
            num_prompts=num_prompts,
            seed=seed,
            output_dir=output_dir,
        )

        # Run speculative configs
        done = 0
        for gamma in gammas:
            for temp in temperatures:
                for task in tasks:
                    config = ExperimentConfig(
                        pair=pair,
                        gamma=gamma,
                        temperature=temp,
                        task=task,
                        max_new_tokens=max_new_tokens,
                        num_prompts=num_prompts,
                        seed=seed,
                    )
                    summary = run_single_config(
                        config, target_model, draft_model, tokenizer, output_dir
                    )

                    # Compute speedup vs baseline
                    baseline_key = (task, temp)
                    baseline_tps = baselines[baseline_key]["tokens_per_second"]["mean"]
                    spec_tps = summary["tokens_per_second"]["mean"]
                    speedup = spec_tps / baseline_tps if baseline_tps > 0 else 0.0

                    # Append to master CSV
                    csv_row = {
                        "pair_id": pair.pair_id,
                        "task": task,
                        "gamma": gamma,
                        "temperature": temp,
                        "is_baseline": False,
                        "mean_tps": round(spec_tps, 2),
                        "std_tps": round(summary["tokens_per_second"]["std"], 2),
                        "p95_tps": round(summary["tokens_per_second"]["p95"], 2),
                        "mean_acceptance_rate": round(
                            summary["acceptance_rate"]["mean"], 4
                        ),
                        "mean_acceptance_length": round(
                            summary["acceptance_length"]["mean"], 2
                        ),
                        "mean_ttft_ms": round(summary["ttft_ms"]["mean"], 2),
                        "mean_peak_vram_gb": round(
                            summary["peak_vram_gb"]["mean"], 2
                        ),
                        "mean_draft_overhead": round(
                            summary["draft_overhead_ratio"]["mean"], 4
                        ),
                        "speedup": round(speedup, 3),
                    }
                    save_summary_csv(
                        csv_row, os.path.join(output_dir, "summary.csv")
                    )

                    done += 1
                    logger.info(
                        "Pair %s progress: %d/%d (speedup=%.2fx)",
                        pair.pair_id,
                        done,
                        total_configs,
                        speedup,
                    )
    finally:
        unload_models(target_model, draft_model)
