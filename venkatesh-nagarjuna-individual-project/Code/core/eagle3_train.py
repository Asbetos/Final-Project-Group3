"""
Training pipeline for the EAGLE-3 draft head.

Trains the draft head using the target model's hidden states as supervision,
with a multi-step training loss that teaches the head to recover from its
own prediction errors at inference time.

Optimized for CUDA GPUs with BF16 mixed precision, torch.compile, and a
pinned-memory DataLoader.
"""

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.eagle3 import Eagle3Config, Eagle3DraftHead
from core.models import get_device, load_model, load_tokenizer
from core.data import _fallback_chat_text, _is_qwen_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Avoid tokenizer fork warnings once DataLoader workers start.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# CUDA GPU optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for EAGLE-3 draft head training."""

    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    num_samples: int = -1
    max_seq_len: int = 512
    multi_step_k: int = 5  # number of autoregressive draft steps for loss
    step_decay: float = 0.8  # loss weight decay per step
    warmup_steps: int = 100
    save_every: int = 500
    log_every: int = 10
    checkpoint_dir: str = "checkpoints/eagle3/gemma3_12b"
    final_checkpoint_name: str = "eagle3_gemma3_12b_final.pt"
    target_model_id: str = "google/gemma-3-12b-it"
    target_quantize_4bit: bool = True
    dataset_name: str = "vicgalle/alpaca-gpt4"
    num_workers: int = 0
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    compile_draft_head: bool = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AlpacaDataset(Dataset):
    """Wraps Alpaca dataset formatted with the appropriate chat template."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "vicgalle/alpaca-gpt4",
        num_samples: int = -1,
        max_seq_len: int = 512,
        seed: int = 42,
    ):
        logger.info("Loading dataset %s ...", dataset_name)
        ds = load_dataset(dataset_name)["train"]
        ds = ds.shuffle(seed=seed)
        if num_samples > 0:
            ds = ds.select(range(min(num_samples, len(ds))))

        self.examples = []
        for row in ds:
            # Format with the target tokenizer's chat template.
            instruction = row["instruction"]
            input_text = row.get("input", "")
            output_text = row.get("output", "")

            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output_text},
            ]

            if not getattr(tokenizer, "chat_template", None):
                text = _fallback_chat_text(
                    "You are a helpful assistant.",
                    user_content,
                    output_text,
                )
            else:
                try:
                    if _is_qwen_tokenizer(tokenizer):
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, enable_thinking=False
                        )
                    else:
                        text = tokenizer.apply_chat_template(messages, tokenize=False)
                except Exception:
                    text = _fallback_chat_text(
                        "You are a helpful assistant.",
                        user_content,
                        output_text,
                    )
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            })

        logger.info("Prepared %d training examples", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_training_dataset(
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
) -> Dataset:
    """Create the tokenized training dataset once up front."""
    return AlpacaDataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset_name,
        num_samples=config.num_samples,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
    )


def create_training_dataloader(
    dataset: Dataset,
    config: TrainingConfig,
    epoch: int,
) -> DataLoader:
    """Create one deterministic epoch DataLoader tuned for smaller GPU boxes."""
    num_workers = min(config.num_workers, os.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "generator": torch.Generator().manual_seed(config.seed + epoch),
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs.update(
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2,
        )

    return DataLoader(
        dataset,
        **loader_kwargs,
    )


# ---------------------------------------------------------------------------
# Multi-step training loss
# ---------------------------------------------------------------------------


def compute_multi_step_loss(
    draft_head: Eagle3DraftHead,
    target_logits: torch.Tensor,
    target_features: List[torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: TrainingConfig,
    eagle3_config: Eagle3Config,
) -> torch.Tensor:
    """
    Compute the EAGLE-3 multi-step training loss.

    At each step t:
      - Step 0: use ground-truth target features
      - Steps 1-K: use draft head's own hidden state output (autoregressive)
    Loss at each step: KL divergence vs target logits, weighted by decay^t.

    Args:
        draft_head: the EAGLE-3 draft head (trainable)
        target_logits: (B, S, V) from frozen target model
        target_features: list of 3 tensors each (B, S, H) from feature layers
        input_ids: (B, S) token IDs
        attention_mask: (B, S) mask
        config: training config
        eagle3_config: eagle3 model config

    Returns:
        scalar loss
    """

    def masked_kl_from_logits(
        draft_logits_local: torch.Tensor,
        target_logits_local: torch.Tensor,
        mask_local: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """Compute KL(target || draft) only on valid positions."""
        if not mask_local.bool().any():
            return draft_logits_local.new_zeros(())

        total_kl = draft_logits_local.new_zeros(())
        total_positions = 0
        batch_size = draft_logits_local.shape[0]
        for batch_idx in range(batch_size):
            valid_positions = torch.nonzero(mask_local[batch_idx], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            for start in range(0, valid_positions.numel(), chunk_size):
                pos_chunk = valid_positions[start:start + chunk_size]
                draft_chunk = draft_logits_local[batch_idx, pos_chunk, :]
                target_chunk = target_logits_local[batch_idx, pos_chunk, :]
                draft_log_probs = F.log_softmax(draft_chunk, dim=-1)
                target_log_probs = F.log_softmax(target_chunk, dim=-1)
                total_kl = total_kl + F.kl_div(
                    draft_log_probs,
                    target_log_probs,
                    reduction="sum",
                    log_target=True,
                )
                total_positions += pos_chunk.numel()

        if total_positions == 0:
            return draft_logits_local.new_zeros(())
        return total_kl / total_positions

    B, S, V = target_logits.shape
    device = target_logits.device
    total_loss = torch.tensor(0.0, device=device)

    # Fuse target features
    fused = draft_head.fuse_target_features(target_features)  # (B, S, H)

    # We predict tokens at positions 1..S-1 using features from 0..S-2
    # Target logits at position t predict token at t+1
    # So target_logits[:, t, :] is the distribution for token at position t+1

    # Step 0: ground-truth features → draft logits
    # Use features from positions 0..S-2 and token IDs from 0..S-2
    if S < 2:
        return total_loss

    src_ids = input_ids[:, :-1]  # (B, S-1)
    src_features = fused[:, :-1, :]  # (B, S-1, H)
    position_ids = torch.arange(S - 1, device=device).unsqueeze(0).expand(B, -1)

    draft_logits, draft_hidden, _ = draft_head(
        token_ids=src_ids,
        fused_hidden=src_features,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
    )

    # Target distribution at positions 0..S-2 (predicting tokens 1..S-1)
    mask = attention_mask[:, 1:]  # mask for positions 1..S-1
    kl_step0 = masked_kl_from_logits(
        draft_logits,
        target_logits[:, :-1, :],
        mask,
    )

    total_loss = total_loss + kl_step0

    # Steps 1..K: autoregressive with draft head's own hidden state
    current_hidden = draft_hidden.detach()  # detach to prevent backprop through all steps at once
    weight = config.step_decay
    # Track cumulative offset into the original sequence for target logit alignment
    cumulative_offset = 0

    for step in range(1, config.multi_step_k):
        if current_hidden.shape[1] < 2:
            break

        # Use draft head's own prediction as next input token
        with torch.no_grad():
            pred_tokens = draft_logits.argmax(dim=-1)

        # Always slice at offset 1 from current (already-shrunk) tensors
        step_ids = pred_tokens[:, 1:]
        step_features = current_hidden[:, 1:, :]
        cumulative_offset += 1
        step_positions = torch.arange(
            cumulative_offset, cumulative_offset + step_ids.shape[1],
            device=device,
        ).unsqueeze(0).expand(B, -1)

        draft_logits_k, draft_hidden_k, _ = draft_head(
            token_ids=step_ids,
            fused_hidden=step_features,
            position_ids=step_positions,
            past_key_values=None,
            use_cache=False,
        )

        # Align target logits: predict positions (cumulative_offset+1) .. (S-1)
        target_logits_k = target_logits[:, cumulative_offset:-1, :]
        # Trim draft to match target length (target may be shorter)
        min_len = min(draft_logits_k.shape[1], target_logits_k.shape[1])
        draft_logits_k = draft_logits_k[:, :min_len, :]
        target_logits_k = target_logits_k[:, :min_len, :]

        mask_k = attention_mask[:, cumulative_offset + 1:cumulative_offset + 1 + min_len]
        kl_step_k = masked_kl_from_logits(
            draft_logits_k,
            target_logits_k,
            mask_k,
        )

        total_loss = total_loss + weight * kl_step_k

        current_hidden = draft_hidden_k.detach()
        draft_logits = draft_logits_k
        weight *= config.step_decay

    return total_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_eagle3_head(
    target_model: AutoModelForCausalLM,
    draft_head: Eagle3DraftHead,
    eagle3_config: Eagle3Config,
    train_dataset: Dataset,
    config: TrainingConfig,
    device: torch.device,
) -> Eagle3DraftHead:
    """
    Train the EAGLE-3 draft head.

    Args:
        target_model: frozen target model in BF16
        draft_head: trainable draft head
        eagle3_config: eagle3 config
        train_dataset: tokenized training dataset
        config: training hyperparameters
        device: CUDA device

    Returns:
        trained draft head
    """
    # Ensure target model parameters don't accumulate gradients.
    # The target forward runs under torch.no_grad() so gradients don't flow,
    # but requires_grad_(False) also prevents VRAM being wasted on grad buffers.
    target_model.requires_grad_(False)

    # Prefer 8-bit AdamW (bitsandbytes) when the target is 4-bit quantized:
    # fp32 Adam state on ~623M draft-head params is ~5 GB and pushes A10G
    # (22 GB total) over its limit with a 15.5 GB quantized target. 8-bit
    # state cuts that to ~1.25 GB. Fall back to fp32 AdamW if bnb is missing.
    try:
        import bitsandbytes as _bnb
        optimizer = _bnb.optim.AdamW8bit(
            draft_head.trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        logger.info("Using 8-bit AdamW (bitsandbytes) for draft head")
    except ImportError:
        optimizer = torch.optim.AdamW(
            draft_head.trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        logger.info("Using fp32 AdamW for draft head (bitsandbytes unavailable)")

    # Linear warmup + cosine decay
    batches_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.grad_accum_steps)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = min(config.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    runner_head = draft_head
    compile_enabled = config.compile_draft_head
    if compile_enabled:
        logger.info("Compiling draft head with torch.compile ...")
        runner_head = torch.compile(draft_head, mode="default")

    draft_head.train()
    global_step = 0
    start_epoch = 0
    start_batch_in_epoch = 0
    accum_loss = 0.0
    accum_count = 0

    if config.resume_checkpoint:
        resume_state = load_checkpoint(
            draft_head,
            config.resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        global_step = resume_state["global_step"]
        start_epoch = resume_state["epoch"]
        start_batch_in_epoch = resume_state["batch_in_epoch"]
        if start_epoch >= config.epochs:
            logger.info(
                "Checkpoint already completed %d epochs; nothing left to train.",
                config.epochs,
            )
            return draft_head

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        num_batches = 0
        total_batches = math.ceil(len(train_dataset) / config.batch_size)
        microbatches_since_step = 0
        current_window_target = 0
        resume_offset = start_batch_in_epoch if epoch == start_epoch else 0
        dataloader = create_training_dataloader(train_dataset, config, epoch)

        if resume_offset:
            logger.info(
                "Resuming at epoch %d batch %d",
                epoch + 1,
                resume_offset,
            )

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < resume_offset:
                continue

            if microbatches_since_step == 0:
                remaining_batches = total_batches - batch_idx
                current_window_target = min(config.grad_accum_steps, remaining_batches)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # Forward pass through frozen target
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                target_out = target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                target_logits = target_out.logits.detach()
                target_features = [
                    target_out.hidden_states[li].detach()
                    for li in eagle3_config.feature_layers
                ]

            # Draft head forward + multi-step loss
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                try:
                    raw_loss = compute_multi_step_loss(
                        runner_head,
                        target_logits,
                        target_features,
                        input_ids,
                        attention_mask,
                        config,
                        eagle3_config,
                    )
                except RuntimeError as exc:
                    compile_error = (
                        compile_enabled
                        and (
                            "torch._dynamo" in str(exc)
                            or "fake_tensor" in str(exc)
                            or "Failed running call_function" in str(exc)
                        )
                    )
                    if not compile_error:
                        raise
                    logger.warning(
                        "torch.compile failed for the draft head; falling back to eager: %s",
                        exc,
                    )
                    runner_head = draft_head
                    compile_enabled = False
                    torch.cuda.empty_cache()
                    raw_loss = compute_multi_step_loss(
                        runner_head,
                        target_logits,
                        target_features,
                        input_ids,
                        attention_mask,
                        config,
                        eagle3_config,
                    )
                loss = raw_loss / current_window_target

            loss.backward()
            accum_loss += raw_loss.item()
            accum_count += 1
            microbatches_since_step += 1

            batches_seen = batch_idx + 1
            if microbatches_since_step == current_window_target:
                torch.nn.utils.clip_grad_norm_(
                    draft_head.trainable_parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                microbatches_since_step = 0
                current_window_target = 0

                if global_step % config.log_every == 0:
                    avg_loss = accum_loss / max(1, accum_count)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "Epoch %d, step %d/%d, loss=%.4f, lr=%.2e",
                        epoch + 1,
                        global_step,
                        total_steps,
                        avg_loss,
                        lr,
                    )
                    accum_loss = 0.0
                    accum_count = 0

                if global_step % config.save_every == 0:
                    save_checkpoint(
                        draft_head,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        batches_seen,
                        config.checkpoint_dir,
                        final_checkpoint_name=config.final_checkpoint_name,
                    )

            epoch_loss += raw_loss.item()
            num_batches += 1

        epoch_time = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        logger.info(
            "Epoch %d complete. Avg loss=%.4f, Time=%.1fs",
            epoch + 1,
            avg_epoch_loss,
            epoch_time,
        )

        # Save end-of-epoch checkpoint
        save_checkpoint(
            draft_head,
            optimizer,
            scheduler,
            global_step,
            epoch + 1,
            0,
            config.checkpoint_dir,
            final_checkpoint_name=config.final_checkpoint_name,
        )

    # Save final checkpoint
    save_checkpoint(
        draft_head,
        optimizer,
        scheduler,
        global_step,
        config.epochs,
        0,
        config.checkpoint_dir,
        final=True,
        final_checkpoint_name=config.final_checkpoint_name,
    )

    return draft_head


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    draft_head: Eagle3DraftHead,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    checkpoint_dir: str,
    final: bool = False,
    final_checkpoint_name: str = "eagle3_final.pt",
) -> str:
    """Save draft head weights and optimizer state."""
    checkpoint_stem = os.path.splitext(final_checkpoint_name)[0]
    if checkpoint_stem.endswith("_final"):
        checkpoint_stem = checkpoint_stem[:-6]

    if final:
        path = os.path.join(checkpoint_dir, final_checkpoint_name)
    else:
        path = os.path.join(checkpoint_dir, f"{checkpoint_stem}_step{global_step}.pt")

    # Save only trainable parameters (not frozen shared refs)
    trainable_state = {
        name: param
        for name, param in draft_head.state_dict().items()
        if any(
            name.startswith(prefix)
            for prefix in ["fusion_fc", "input_fc", "decoder_layer"]
        )
    }

    checkpoint = {
        "draft_head_state": trainable_state,
        "global_step": global_step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
    }
    if not final:
        checkpoint["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()

    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    logger.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(
    draft_head: Eagle3DraftHead,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
) -> dict:
    """
    Load draft head weights from checkpoint.

    Returns:
        Dict containing global_step, epoch, and batch_in_epoch.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load trainable parameters
    draft_head.load_state_dict(ckpt["draft_head_state"], strict=False)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    logger.info(
        "Loaded checkpoint: %s (step=%d, epoch=%d, batch=%d)",
        checkpoint_path,
        ckpt.get("global_step", 0),
        ckpt.get("epoch", 0),
        ckpt.get("batch_in_epoch", 0),
    )
    return {
        "global_step": ckpt.get("global_step", 0),
        "epoch": ckpt.get("epoch", 0),
        "batch_in_epoch": ckpt.get("batch_in_epoch", 0),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE-3 draft head")
    parser.add_argument(
        "--target-model",
        type=str,
        default="google/gemma-3-12b-it",
        help="Target model ID (default: google/gemma-3-12b-it)",
    )
    parser.add_argument(
        "--target-4bit",
        dest="target_4bit",
        action="store_true",
        default=True,
        help="Load target in 4-bit quantization (default: enabled).",
    )
    parser.add_argument(
        "--no-target-4bit",
        dest="target_4bit",
        action="store_false",
        help="Disable 4-bit target loading.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--multi-step-k",
        type=int,
        default=5,
        help=(
            "Number of autoregressive multi-step loss iterations. Each step "
            "holds its own forward graph in VRAM until the outer backward call, "
            "so large K × (vocab=262144) blows up memory on A10G. Reduce to 2 "
            "or 3 for memory-constrained targets like Gemma 4 31B."
        ),
    )
    parser.add_argument("--dataset-name", type=str, default="vicgalle/alpaca-gpt4")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile-draft-head",
        action="store_true",
        help="Enable torch.compile for the draft head (off by default).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/eagle3/gemma3_12b",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--final-checkpoint-name",
        type=str,
        default=None,
        help=(
            "Filename for the final saved checkpoint "
            "(default: eagle3_gemma4_31b_final.pt for the active Gemma-4-31B run)"
        ),
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")

    torch.set_float32_matmul_precision("high")

    device = get_device()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Training on: %s", gpu_name)

    # Auto-derive final checkpoint name from model ID if not explicitly given
    final_ckpt_name = args.final_checkpoint_name
    if final_ckpt_name is None:
        model_lower = args.target_model.lower()
        if "gemma-4" in model_lower or "gemma4" in model_lower:
            final_ckpt_name = "eagle3_gemma4_31b_final.pt"
        elif "gemma-3-12b" in model_lower or "gemma3-12b" in model_lower:
            final_ckpt_name = "eagle3_gemma3_12b_final.pt"
        else:
            final_ckpt_name = "eagle3_final.pt"

    config = TrainingConfig(
        target_model_id=args.target_model,
        target_quantize_4bit=args.target_4bit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        multi_step_k=args.multi_step_k,
        dataset_name=args.dataset_name,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        final_checkpoint_name=final_ckpt_name,
        resume_checkpoint=args.resume,
        compile_draft_head=args.compile_draft_head,
    )

    # Load target model (frozen)
    logger.info("Loading target model: %s", config.target_model_id)
    target_model = load_model(
        config.target_model_id,
        quantize_4bit=config.target_quantize_4bit,
        compile_model=False,  # don't compile for training (need hidden states)
    )

    tokenizer = load_tokenizer(config.target_model_id)

    # Create draft head — auto-derive architecture from target model config
    eagle3_config = Eagle3Config.from_model(target_model)
    draft_head = Eagle3DraftHead(eagle3_config, target_model)
    draft_head = draft_head.to(device=device, dtype=torch.bfloat16)

    logger.info(
        "Draft head created: %d trainable params (%.1f M)",
        draft_head.num_trainable_params(),
        draft_head.num_trainable_params() / 1e6,
    )

    # Create training dataset once, then rebuild the DataLoader per epoch.
    train_dataset = create_training_dataset(tokenizer, config)

    # Train
    logger.info("Starting training ...")
    torch.cuda.empty_cache()
    train_eagle3_head(
        target_model, draft_head, eagle3_config, train_dataset, config, device
    )

    logger.info("Training complete. Checkpoints in: %s", config.checkpoint_dir)


if __name__ == "__main__":
    main()
