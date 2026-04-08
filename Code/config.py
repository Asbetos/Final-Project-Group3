"""Experiment configuration dataclasses and grid constants."""

from dataclasses import dataclass, asdict
from typing import Optional
import json
import os


@dataclass
class ModelPairConfig:
    """Defines one of the 3 model pair configurations (A, B, C)."""

    pair_id: str
    target_model_id: str
    draft_model_id: str
    target_quantize_4bit: bool
    target_vram_estimate_gb: float
    draft_vram_estimate_gb: float

    @property
    def total_vram_estimate_gb(self) -> float:
        return self.target_vram_estimate_gb + self.draft_vram_estimate_gb


@dataclass
class ExperimentConfig:
    """A single point in the 180-configuration grid."""

    pair: ModelPairConfig
    gamma: int                  # speculation length: 1, 3, 5, 7, 10
    temperature: float          # 0.0, 0.6, 1.0
    task: str                   # humaneval, triviaqa, cnn_dailymail, writingprompts
    max_new_tokens: int = 128
    num_prompts: int = 50
    num_warmup: int = 3
    seed: int = 42

    def run_id(self) -> str:
        """Unique string identifier for this config, used as filename stem."""
        return f"{self.pair.pair_id}_{self.task}_g{self.gamma}_t{self.temperature}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pair"] = asdict(self.pair)
        return d

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Pre-built model pair definitions
# ---------------------------------------------------------------------------

PAIR_A = ModelPairConfig(
    pair_id="A",
    target_model_id="Qwen/Qwen3-8B",
    draft_model_id="Qwen/Qwen3-0.6B",
    target_quantize_4bit=False,
    target_vram_estimate_gb=16.7,
    draft_vram_estimate_gb=3.9,
)

PAIR_B = ModelPairConfig(
    pair_id="B",
    target_model_id="Qwen/Qwen3-8B",
    draft_model_id="Qwen/Qwen3-1.7B",
    target_quantize_4bit=False,
    target_vram_estimate_gb=16.7,
    draft_vram_estimate_gb=6.1,
)

PAIR_C = ModelPairConfig(
    pair_id="C",
    target_model_id="Qwen/Qwen3-8B",
    draft_model_id="Qwen/Qwen3-0.6B",
    target_quantize_4bit=True,
    target_vram_estimate_gb=4.3,
    draft_vram_estimate_gb=3.9,
)

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

ALL_PAIRS = [PAIR_A, PAIR_B, PAIR_C]
PAIR_MAP = {p.pair_id: p for p in ALL_PAIRS}

ALL_GAMMAS = [1, 3, 5, 7, 10]
ALL_TEMPERATURES = [0.0, 0.6, 1.0]
ALL_TASKS = ["humaneval", "triviaqa", "cnn_dailymail", "writingprompts"]


def build_grid(
    pairs=None,
    gammas=None,
    temperatures=None,
    tasks=None,
    **kwargs,
):
    """Build a list of ExperimentConfig for every combination in the grid."""
    pairs = pairs or ALL_PAIRS
    gammas = gammas or ALL_GAMMAS
    temperatures = temperatures or ALL_TEMPERATURES
    tasks = tasks or ALL_TASKS

    configs = []
    for pair in pairs:
        for gamma in gammas:
            for temp in temperatures:
                for task in tasks:
                    configs.append(
                        ExperimentConfig(
                            pair=pair,
                            gamma=gamma,
                            temperature=temp,
                            task=task,
                            **kwargs,
                        )
                    )
    return configs
