"""
Training pipeline for the EAGLE-3 draft head.

Trains the draft head using the target model's hidden states as supervision,
with a multi-step training loss that teaches the head to recover from its
own prediction errors at inference time.

Optimized for A100 GPU: BF16 mixed-precision, torch.compile, gradient
checkpointing, pinned memory DataLoader.
"""

import argparse
import logging
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

from eagle3 import Eagle3Config, Eagle3DraftHead
from models import load_model, load_tokenizer, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# A100 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for EAGLE-3 draft head training."""

    batch_size: int = 2
    grad_accum_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 3
    num_samples: int = 5000
    max_seq_len: int = 512
    multi_step_k: int = 5  # number of autoregressive draft steps for loss
    step_decay: float = 0.8  # loss weight decay per step
    warmup_steps: int = 100
    save_every: int = 500
    log_every: int = 10
    checkpoint_dir: str = "checkpoints/eagle3"
    target_model_id: str = "Qwen/Qwen3-8B"
    target_quantize_4bit: bool = False
    dataset_name: str = "tatsu-lab/alpaca"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AlpacaDataset(Dataset):
    """Wraps Alpaca dataset formatted with Qwen3 chat template."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 5000,
        max_seq_len: int = 512,
        seed: int = 42,
    ):
        logger.info("Loading Alpaca dataset ...")
        ds = load_dataset("tatsu-lab/alpaca")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

        self.examples = []
        for row in ds:
            # Format as Qwen3 chat
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

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False
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


def create_training_dataloader(
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
) -> DataLoader:
    """Create DataLoader with A100-optimized settings."""
    dataset = AlpacaDataset(
        tokenizer=tokenizer,
        num_samples=config.num_samples,
        max_seq_len=config.max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
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
    target_dist = F.log_softmax(target_logits[:, :-1, :], dim=-1)  # (B, S-1, V)
    draft_dist = F.log_softmax(draft_logits, dim=-1)  # (B, S-1, V)

    # KL(target || draft) per position, masked
    mask = attention_mask[:, 1:].float()  # mask for positions 1..S-1
    kl_step0 = F.kl_div(draft_dist, target_dist.exp(), reduction="none").sum(-1)  # (B, S-1)
    kl_step0 = (kl_step0 * mask).sum() / mask.sum().clamp(min=1)

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
        target_dist_k = F.log_softmax(
            target_logits[:, cumulative_offset:-1, :], dim=-1
        )
        # Trim draft to match target length (target may be shorter)
        min_len = min(draft_logits_k.shape[1], target_dist_k.shape[1])
        draft_dist_k = F.log_softmax(draft_logits_k[:, :min_len, :], dim=-1)
        target_dist_k = target_dist_k[:, :min_len, :]

        mask_k = attention_mask[:, cumulative_offset + 1:cumulative_offset + 1 + min_len].float()
        kl_step_k = F.kl_div(draft_dist_k, target_dist_k.exp(), reduction="none").sum(-1)
        kl_step_k = (kl_step_k * mask_k).sum() / mask_k.sum().clamp(min=1)

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
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> Eagle3DraftHead:
    """
    Train the EAGLE-3 draft head.

    Args:
        target_model: frozen target model in BF16
        draft_head: trainable draft head
        eagle3_config: eagle3 config
        dataloader: training data
        config: training hyperparameters
        device: CUDA device

    Returns:
        trained draft head
    """
    # Enable gradient checkpointing on target to save VRAM
    target_model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        draft_head.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Linear warmup + cosine decay
    total_steps = len(dataloader) * config.epochs // config.grad_accum_steps
    warmup_steps = min(config.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Compile draft head for training speedup
    compiled_head = torch.compile(draft_head, mode="default")

    draft_head.train()
    global_step = 0
    accum_loss = 0.0

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

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
                loss = compute_multi_step_loss(
                    compiled_head,
                    target_logits,
                    target_features,
                    input_ids,
                    attention_mask,
                    config,
                    eagle3_config,
                )
                loss = loss / config.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    draft_head.trainable_parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every == 0:
                    avg_loss = accum_loss / config.log_every
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

                if global_step % config.save_every == 0:
                    save_checkpoint(
                        draft_head,
                        optimizer,
                        global_step,
                        epoch,
                        config.checkpoint_dir,
                    )

            epoch_loss += loss.item() * config.grad_accum_steps
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
            draft_head, optimizer, global_step, epoch, config.checkpoint_dir
        )

    # Disable gradient checkpointing
    target_model.gradient_checkpointing_disable()

    # Save final checkpoint
    save_checkpoint(
        draft_head, optimizer, global_step, config.epochs - 1,
        config.checkpoint_dir, final=True,
    )

    return draft_head


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    draft_head: Eagle3DraftHead,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    checkpoint_dir: str,
    final: bool = False,
) -> str:
    """Save draft head weights and optimizer state."""
    if final:
        path = os.path.join(checkpoint_dir, "eagle3_final.pt")
    else:
        path = os.path.join(checkpoint_dir, f"eagle3_step{global_step}.pt")

    # Save only trainable parameters (not frozen shared refs)
    trainable_state = {
        name: param
        for name, param in draft_head.state_dict().items()
        if any(
            name.startswith(prefix)
            for prefix in ["fusion_fc", "input_fc", "decoder_layer"]
        )
    }

    torch.save(
        {
            "draft_head_state": trainable_state,
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        },
        path,
    )
    logger.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(
    draft_head: Eagle3DraftHead,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, int]:
    """
    Load draft head weights from checkpoint.

    Returns:
        (global_step, epoch) from the checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load trainable parameters
    draft_head.load_state_dict(ckpt["draft_head_state"], strict=False)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    logger.info(
        "Loaded checkpoint: %s (step=%d, epoch=%d)",
        checkpoint_path,
        ckpt.get("global_step", 0),
        ckpt.get("epoch", 0),
    )
    return ckpt.get("global_step", 0), ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE-3 draft head")
    parser.add_argument(
        "--target-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Target model ID",
    )
    parser.add_argument(
        "--target-4bit",
        action="store_true",
        help="Load target in 4-bit quantization",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/eagle3")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    device = get_device()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Training on: %s", gpu_name)

    config = TrainingConfig(
        target_model_id=args.target_model,
        target_quantize_4bit=args.target_4bit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Load target model (frozen)
    logger.info("Loading target model: %s", config.target_model_id)
    target_model = load_model(
        config.target_model_id,
        quantize_4bit=config.target_quantize_4bit,
        compile_model=False,  # don't compile for training (need hidden states)
    )

    tokenizer = load_tokenizer(config.target_model_id)

    # Create draft head
    eagle3_config = Eagle3Config()
    draft_head = Eagle3DraftHead(eagle3_config, target_model)
    draft_head = draft_head.to(device=device, dtype=torch.bfloat16)

    logger.info(
        "Draft head created: %d trainable params (%.1f M)",
        draft_head.num_trainable_params(),
        draft_head.num_trainable_params() / 1e6,
    )

    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(draft_head, args.resume)

    # Create dataloader
    dataloader = create_training_dataloader(tokenizer, config)

    # Train
    logger.info("Starting training ...")
    torch.cuda.empty_cache()
    train_eagle3_head(
        target_model, draft_head, eagle3_config, dataloader, config, device
    )

    logger.info("Training complete. Checkpoints in: %s", config.checkpoint_dir)


if __name__ == "__main__":
    main()
