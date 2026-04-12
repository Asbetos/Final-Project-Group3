"""Experiment configuration dataclasses and grid constants."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import json
import os

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


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
    """A single EAGLE-3 experiment configuration."""

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


# Pre-built EAGLE-3 pair definitions

_DEFAULT_CKPT = os.environ.get(
    "EAGLE3_CHECKPOINT",
    os.path.join(_CODE_DIR, "checkpoints", "eagle3", "eagle3_final.pt"),
)

EAGLE3_PAIR_D = Eagle3PairConfig(
    pair_id="D",
    target_model_id="Qwen/Qwen3-8B",
    target_quantize_4bit=False,
    checkpoint_path=_DEFAULT_CKPT,
    target_vram_estimate_gb=16.7,
)

EAGLE3_PAIR_E = Eagle3PairConfig(
    pair_id="E",
    target_model_id="Qwen/Qwen3-8B",
    target_quantize_4bit=True,
    checkpoint_path=_DEFAULT_CKPT,
    target_vram_estimate_gb=4.3,
)

ALL_EAGLE3_PAIRS = [EAGLE3_PAIR_D, EAGLE3_PAIR_E]
EAGLE3_PAIR_MAP = {p.pair_id: p for p in ALL_EAGLE3_PAIRS}

ALL_TREE_BUDGETS = [20, 40, 60]


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
