"""Experiment configuration dataclasses and grid constants."""

from dataclasses import asdict, dataclass
from typing import List
import json
import os

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelPairConfig:
    """Defines one model pair configuration (target + draft model)."""

    pair_id: str
    target_model_id: str
    draft_model_id: str
    target_quantize_4bit: bool
    target_vram_estimate_gb: float
    draft_vram_estimate_gb: float
    draft_quantize_4bit: bool = False  # set True to load draft in 4-bit NF4

    @property
    def total_vram_estimate_gb(self) -> float:
        return self.target_vram_estimate_gb + self.draft_vram_estimate_gb


@dataclass
class ExperimentConfig:
    """A single point in the active Gemma standard-speculation grid."""

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
# Active Gemma model pair definitions
# ---------------------------------------------------------------------------

# Pair F — Gemma 3 12B (4-bit target) + Gemma 3 1B (bf16 draft)
# Inference VRAM: ~6.6 + 2.0 = ~8.6 GB.
PAIR_F = ModelPairConfig(
    pair_id="F",
    target_model_id="google/gemma-3-12b-it",
    draft_model_id="google/gemma-3-1b-it",
    target_quantize_4bit=True,
    target_vram_estimate_gb=6.6,
    draft_vram_estimate_gb=2.0,
)

# Pair G — Gemma 4 31B (4-bit target) + Gemma 3 1B (bf16 draft)
# Inference VRAM: ~15.5 + 2.0 = ~17.5 GB.
PAIR_G = ModelPairConfig(
    pair_id="G",
    target_model_id="google/gemma-4-31B",
    draft_model_id="google/gemma-3-1b-it",
    target_quantize_4bit=True,
    draft_quantize_4bit=False,
    target_vram_estimate_gb=15.5,
    draft_vram_estimate_gb=2.0,
)

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

ALL_PAIRS = [PAIR_F, PAIR_G]
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

