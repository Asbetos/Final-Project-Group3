"""Experiment configuration dataclasses and grid constants."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
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
    target_vram_estimate_gb=17.5,
    draft_vram_estimate_gb=2.0
)

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

ALL_PAIRS = [PAIR_F, PAIR_G]
PAIR_MAP = {p.pair_id: p for p in ALL_PAIRS}

ALL_GAMMAS = [1, 5, 10]
ALL_TEMPERATURES = [0.0]  # greedy-only evaluation
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


# ---------------------------------------------------------------------------
# EAGLE-3 configuration
# ---------------------------------------------------------------------------


@dataclass
class Eagle3PairConfig:
    """Defines an EAGLE-3 model pair (target + draft head, no separate draft model)."""

    pair_id: str
    target_model_id: str
    target_quantize_4bit: bool
    checkpoint_path: str
    tree_budget: int = 60
    max_depth: int = 6
    top_k: int = 10
    target_vram_estimate_gb: float = 16.7


@dataclass
class Eagle3ExperimentConfig:
    """A single EAGLE-3 experiment configuration for active Gemma runs."""

    pair: Eagle3PairConfig
    tree_budget: int
    temperature: float
    task: str
    max_new_tokens: int = 128
    num_prompts: int = 50
    num_warmup: int = 3
    seed: int = 42

    def run_id(self) -> str:
        return f"{self.pair.pair_id}_{self.task}_tb{self.tree_budget}_t{self.temperature}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pair"] = asdict(self.pair)
        return d

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Active EAGLE-3 pair definition

_GEMMA4_CKPT = os.environ.get(
    "EAGLE3_GEMMA4_CHECKPOINT",
    os.path.join(
        _CODE_DIR, "checkpoints", "eagle3", "gemma4_31b", "eagle3_gemma4_31b_final.pt"
    ),
)

# Pair H — EAGLE-3 with Gemma 4 31B (4-bit) target.
# Inference VRAM: ~15.5 GB (target) + ~0.8 GB (draft head) = ~16.3 GB.
# Training VRAM: ~19–20 GB (target frozen + head + optimizer + activations).
EAGLE3_PAIR_H = Eagle3PairConfig(
    pair_id="H",
    target_model_id="google/gemma-4-31B",
    target_quantize_4bit=True,
    checkpoint_path=_GEMMA4_CKPT,
    tree_budget=60,
    max_depth=6,
    top_k=10,
    target_vram_estimate_gb=15.5,
)

ALL_EAGLE3_PAIRS = [EAGLE3_PAIR_H]
EAGLE3_PAIR_MAP = {p.pair_id: p for p in ALL_EAGLE3_PAIRS}

ALL_TREE_BUDGETS = [20, 60]


def build_eagle3_grid(
    pairs: Optional[List[Eagle3PairConfig]] = None,
    tree_budgets: Optional[List[int]] = None,
    temperatures: Optional[List[float]] = None,
    tasks: Optional[List[str]] = None,
    **kwargs,
):
    """Build a list of Eagle3ExperimentConfig for every combination in the grid."""
    pairs = pairs or ALL_EAGLE3_PAIRS
    tree_budgets = tree_budgets or ALL_TREE_BUDGETS
    temperatures = temperatures or ALL_TEMPERATURES
    tasks = tasks or ALL_TASKS

    configs = []
    for pair in pairs:
        for tb in tree_budgets:
            for temp in temperatures:
                for task in tasks:
                    configs.append(
                        Eagle3ExperimentConfig(
                            pair=pair,
                            tree_budget=tb,
                            temperature=temp,
                            task=task,
                            **kwargs,
                        )
                    )
    return configs
