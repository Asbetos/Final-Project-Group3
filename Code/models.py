"""Model loading, tokenizer setup, and VRAM management."""

import gc
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import Eagle3PairConfig, ModelPairConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A100 GPU optimizations — enable TF32 for any stray FP32 matmuls
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_device() -> torch.device:
    """Return the CUDA device, raising if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available")
    return torch.device("cuda:0")


def load_tokenizer(model_id: str = "Qwen/Qwen3-0.6B") -> AutoTokenizer:
    """Load the shared Qwen3 tokenizer (vocab_size=151936, identical across sizes)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str,
    quantize_4bit: bool = False,
    device: torch.device = None,
    compile_model: bool = True,
) -> AutoModelForCausalLM:
    """
    Load a single Qwen3 model for inference.

    Args:
        model_id: HuggingFace model identifier (e.g. "Qwen/Qwen3-8B").
        quantize_4bit: If True, apply NF4 4-bit quantization via bitsandbytes.
        device: Target device (defaults to cuda:0).

    Returns:
        The loaded model in eval mode with gradients disabled.
    """
    device = device or get_device()

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = {"": device}

    logger.info("Loading model %s (4-bit=%s) ...", model_id, quantize_4bit)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if not quantize_4bit:
        model = model.to(device=device, dtype=torch.bfloat16)

    model.eval()

    # torch.compile disabled — CUDA graphs conflict with dynamic KV cache
    # in transformers 5.x + PyTorch 2.4. SDPA on A100 is already fast.
    if False and compile_model:
        logger.info("Compiling model %s with torch.compile ...", model_id)
        model = torch.compile(model, mode="reduce-overhead")

    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.info("Model %s loaded. VRAM allocated: %.2f GB", model_id, vram_gb)

    return model


def load_model_pair(pair: ModelPairConfig):
    """
    Load target model, draft model, and shared tokenizer for a given pair.

    Returns:
        (target_model, draft_model, tokenizer)
    """
    logger.info("Loading model pair %s ...", pair.pair_id)

    tokenizer = load_tokenizer(pair.draft_model_id)

    # Load target first (larger) — fail fast on OOM
    target_model = load_model(
        pair.target_model_id,
        quantize_4bit=pair.target_quantize_4bit,
    )
    target_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    draft_model = load_model(pair.draft_model_id, quantize_4bit=pair.draft_quantize_4bit)
    total_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    logger.info(
        "Pair %s loaded. Target VRAM: %.2f GB, Total VRAM: %.2f GB",
        pair.pair_id,
        target_vram,
        total_vram,
    )

    return target_model, draft_model, tokenizer


def load_eagle3_pair(pair: Eagle3PairConfig):
    """
    Load target model, EAGLE-3 draft head, config, and tokenizer.

    Returns:
        (target_model, draft_head, eagle3_config, tokenizer)
    """
    from eagle3 import Eagle3Config, Eagle3DraftHead
    from eagle3_train import load_checkpoint

    logger.info("Loading EAGLE-3 pair %s ...", pair.pair_id)

    tokenizer = load_tokenizer(pair.target_model_id)

    # Load target model (compile disabled initially — need hidden states access)
    target_model = load_model(
        pair.target_model_id,
        quantize_4bit=pair.target_quantize_4bit,
        compile_model=False,
    )

    # Derive draft head config from the loaded target model so architecture
    # dimensions (hidden_size, vocab_size, head_dim, etc.) are always correct
    # regardless of which model family is used (Qwen3, Gemma3, Gemma4, ...).
    eagle3_config = Eagle3Config.from_model(
        target_model,
        tree_budget=pair.tree_budget,
        max_depth=pair.max_depth,
        top_k=pair.top_k,
    )

    draft_head = Eagle3DraftHead(eagle3_config, target_model)
    device = get_device()
    draft_head = draft_head.to(device=device, dtype=torch.bfloat16)

    if pair.checkpoint_path and os.path.exists(pair.checkpoint_path):
        load_checkpoint(draft_head, pair.checkpoint_path)
        logger.info("Loaded EAGLE-3 checkpoint: %s", pair.checkpoint_path)
    else:
        raise FileNotFoundError(
            f"EAGLE-3 checkpoint not found at '{pair.checkpoint_path}'. "
            f"Run eagle3_train.py first, or set EAGLE3_CHECKPOINT env var. "
            f"CWD={os.getcwd()}"
        )

    draft_head.eval()

    # torch.compile disabled — see note in load_model()
    # draft_head = torch.compile(draft_head, mode="reduce-overhead")

    total_vram = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.info("EAGLE-3 pair %s loaded. Total VRAM: %.2f GB", pair.pair_id, total_vram)

    return target_model, draft_head, eagle3_config, tokenizer


def unload_models(*models) -> None:
    """Delete models and free GPU memory."""
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    logger.info("Models unloaded. VRAM freed.")
