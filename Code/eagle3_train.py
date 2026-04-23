"""
Training pipeline for the EAGLE-3 draft head.

v2 improvements targeted at fixing weak task performance (code/creative writing):

  1. Mixed dataset: ShareGPT + The Stack (code) + WritingPrompts (creative)
     Previous version used only ShareGPT → draft head only saw conversational
     text, so it failed on code and creative writing domains at inference time.

  2. Top-K focused KL loss: KL divergence computed only on top-50 target logits
     instead of full 256k vocab. The long tail is dominated by near-zero
     probabilities whose gradients amount to noise.

  3. Longer sequences (512 → 1024): speculative decoding's speedup compounds
     over generation length.

Same pipeline structure as v1 — same imports, same training loop, same
checkpoint format. Only the dataset composition and the loss computation
differ.
"""

import argparse
import logging
import math
import os
import random
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
from data import _fallback_chat_text, _is_qwen_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 3
    num_samples: int = 10000
    # Longer sequences so the head learns deeper generation trajectories
    max_seq_len: int = 1024
    multi_step_k: int = 5
    step_decay: float = 0.8
    ce_loss_weight: float = 0.3
    # Top-K focus for KL loss: only on target's top tokens per position
    topk_kl: int = 50
    warmup_ratio: float = 0.05
    warmup_steps: int = 100
    save_every: int = 500
    log_every: int = 10
    checkpoint_dir: str = "checkpoints/eagle3/gemma3_12b"
    final_checkpoint_name: str = "eagle3_gemma3_12b_final.pt"
    target_model_id: str = "google/gemma-3-12b-it"
    target_quantize_4bit: bool = True
    # Mixed dataset fractions (must sum to ~1.0)
    sharegpt_fraction: float = 0.5
    code_fraction: float = 0.3
    creative_fraction: float = 0.2
    num_workers: int = 0
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    compile_draft_head: bool = False


# ---------------------------------------------------------------------------
# Multi-source dataset formatting
# ---------------------------------------------------------------------------


def _format_sharegpt_row(row, tokenizer) -> Optional[str]:
    conversations = row.get("conversations", [])
    if not conversations:
        return None
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from", ""))
        if role is None:
            continue
        messages.append({"role": role, "content": turn.get("value", "")})
    if not messages or "user" not in [m["role"] for m in messages]:
        return None
    try:
        if _is_qwen_tokenizer(tokenizer):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False
            )
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def _format_code_row(row, tokenizer) -> Optional[str]:
    """Format a code example (The Stack's 'content' field or CodeAlpaca)."""
    if "content" in row:
        code = row.get("content", "")
        if not code or len(code) < 50:
            return None
        messages = [
            {"role": "user", "content": "Complete the following code:"},
            {"role": "assistant", "content": code},
        ]
    else:
        instr = row.get("instruction", "")
        out = row.get("output", "")
        if not instr or not out:
            return None
        messages = [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": out},
        ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return _fallback_chat_text(
            "You are a helpful assistant.",
            messages[0]["content"], messages[1]["content"],
        )


def _format_creative_row(row, tokenizer) -> Optional[str]:
    prompt = row.get("prompt", "").replace("[WP]", "").replace("[SP]", "").strip()
    story = row.get("story", "").strip()
    if not prompt or not story or len(story) < 100:
        return None
    messages = [
        {"role": "user", "content": f"Write a short story: {prompt}"},
        {"role": "assistant", "content": story},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return _fallback_chat_text(
            "You are a helpful assistant.",
            messages[0]["content"], messages[1]["content"],
        )


class Eagle3MixedDataset(Dataset):
    """Multi-source training dataset — ShareGPT + code + creative writing."""

    def __init__(self, tokenizer: AutoTokenizer, config: TrainingConfig):
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len

        n_sharegpt = int(config.num_samples * config.sharegpt_fraction)
        n_code = int(config.num_samples * config.code_fraction)
        n_creative = config.num_samples - n_sharegpt - n_code

        texts: list[str] = []

        # --- ShareGPT ---
        try:
            logger.info("Loading ShareGPT (target: %d samples)...", n_sharegpt)
            ds = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                trust_remote_code=True,
            )
            split = "train" if "train" in ds else list(ds.keys())[0]
            ds = ds[split].shuffle(seed=config.seed).select(
                range(min(n_sharegpt * 3, len(ds[split])))
            )
            added = 0
            for row in ds:
                t = _format_sharegpt_row(row, tokenizer)
                if t:
                    texts.append(t)
                    added += 1
                if added >= n_sharegpt:
                    break
            logger.info("  Added %d ShareGPT examples", added)
        except Exception as e:
            logger.warning("ShareGPT load failed (%s); falling back to Alpaca", e)
            ds = load_dataset("tatsu-lab/alpaca")["train"]
            ds = ds.shuffle(seed=config.seed).select(
                range(min(n_sharegpt, len(ds)))
            )
            for row in ds:
                instr = row["instruction"]
                inp = row.get("input", "")
                outp = row.get("output", "")
                user = f"{instr}\n\n{inp}" if inp else instr
                try:
                    texts.append(tokenizer.apply_chat_template([
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": outp},
                    ], tokenize=False))
                except Exception:
                    texts.append(_fallback_chat_text(
                        "You are a helpful assistant.", user, outp,
                    ))

        # --- Code (The Stack or CodeAlpaca fallback) ---
        try:
            logger.info("Loading code dataset (target: %d samples)...", n_code)
            try:
                ds = load_dataset(
                    "bigcode/the-stack-smol",
                    data_dir="data/python",
                    trust_remote_code=True,
                )
                split = "train" if "train" in ds else list(ds.keys())[0]
                ds = ds[split].shuffle(seed=config.seed).select(
                    range(min(n_code * 2, len(ds[split])))
                )
            except Exception:
                logger.info("  The Stack unavailable; using CodeAlpaca fallback")
                ds = load_dataset("sahil2801/CodeAlpaca-20k")["train"]
                ds = ds.shuffle(seed=config.seed).select(
                    range(min(n_code, len(ds)))
                )
            added = 0
            for row in ds:
                t = _format_code_row(row, tokenizer)
                if t:
                    texts.append(t)
                    added += 1
                if added >= n_code:
                    break
            logger.info("  Added %d code examples", added)
        except Exception as e:
            logger.warning("Code load failed (%s); skipping code", e)

        # --- Creative (WritingPrompts) ---
        try:
            logger.info("Loading WritingPrompts (target: %d samples)...", n_creative)
            ds = load_dataset("euclaise/writingprompts", trust_remote_code=True)
            split = "train" if "train" in ds else list(ds.keys())[0]
            ds = ds[split].shuffle(seed=config.seed).select(
                range(min(n_creative * 2, len(ds[split])))
            )
            added = 0
            for row in ds:
                t = _format_creative_row(row, tokenizer)
                if t:
                    texts.append(t)
                    added += 1
                if added >= n_creative:
                    break
            logger.info("  Added %d creative examples", added)
        except Exception as e:
            logger.warning("Creative load failed (%s); skipping", e)

        # Shuffle so sources interleave
        rng = random.Random(config.seed)
        rng.shuffle(texts)

        logger.info(
            "Tokenizing %d mixed examples (max_len=%d)...",
            len(texts), config.max_seq_len,
        )
        self.examples = []
        for t in texts:
            enc = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_seq_len,
                padding="max_length",
            )
            self.examples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

        logger.info("Dataset ready: %d examples", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_training_dataset(
    tokenizer: AutoTokenizer, config: TrainingConfig,
) -> Dataset:
    return Eagle3MixedDataset(tokenizer, config)


def create_training_dataloader(
    dataset: Dataset, config: TrainingConfig, epoch: int,
) -> DataLoader:
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
    return DataLoader(dataset, **loader_kwargs)


# ---------------------------------------------------------------------------
# Multi-step training loss with top-K focus
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
    """KL_topk(target || draft) + ce_loss_weight * CE(draft, argmax target)."""

    def masked_kl_topk(
        draft_logits_local: torch.Tensor,
        target_logits_local: torch.Tensor,
        mask_local: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        if not mask_local.bool().any():
            return draft_logits_local.new_zeros(())
        total_kl = draft_logits_local.new_zeros(())
        total_positions = 0
        B = draft_logits_local.shape[0]
        for b in range(B):
            valid = torch.nonzero(mask_local[b], as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            for start in range(0, valid.numel(), 32):
                pos = valid[start:start + 32]
                d_chunk = draft_logits_local[b, pos, :]
                t_chunk = target_logits_local[b, pos, :]
                if topk > 0 and topk < t_chunk.shape[-1]:
                    _, topk_idx = torch.topk(t_chunk, topk, dim=-1)
                    t_sel = torch.gather(t_chunk, -1, topk_idx)
                    d_sel = torch.gather(d_chunk, -1, topk_idx)
                    d_log = F.log_softmax(d_sel, dim=-1)
                    t_log = F.log_softmax(t_sel, dim=-1)
                else:
                    d_log = F.log_softmax(d_chunk, dim=-1)
                    t_log = F.log_softmax(t_chunk, dim=-1)
                total_kl = total_kl + F.kl_div(
                    d_log, t_log, reduction="sum", log_target=True,
                )
                total_positions += pos.numel()
        return total_kl / max(total_positions, 1)

    def masked_ce(
        draft_logits_local: torch.Tensor,
        target_logits_local: torch.Tensor,
        mask_local: torch.Tensor,
    ) -> torch.Tensor:
        if not mask_local.bool().any():
            return draft_logits_local.new_zeros(())
        hard = target_logits_local.argmax(dim=-1)
        B, S, V = draft_logits_local.shape
        flat_l = draft_logits_local.reshape(B * S, V)
        flat_y = hard.reshape(B * S)
        flat_m = mask_local.reshape(B * S).bool()
        return F.cross_entropy(flat_l[flat_m], flat_y[flat_m])

    B, S, V = target_logits.shape
    device = target_logits.device
    total_loss = torch.tensor(0.0, device=device)

    fused = draft_head.fuse_target_features(target_features)

    if S < 2:
        return total_loss

    src_ids = input_ids[:, :-1]
    src_features = fused[:, :-1, :]
    position_ids = torch.arange(S - 1, device=device).unsqueeze(0).expand(B, -1)

    draft_logits, draft_hidden, _ = draft_head(
        token_ids=src_ids,
        fused_hidden=src_features,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
    )

    mask = attention_mask[:, 1:]
    kl_step0 = masked_kl_topk(
        draft_logits, target_logits[:, :-1, :], mask, config.topk_kl,
    )
    total_loss = total_loss + kl_step0

    if config.ce_loss_weight > 0:
        ce_step0 = masked_ce(draft_logits, target_logits[:, :-1, :], mask)
        total_loss = total_loss + config.ce_loss_weight * ce_step0

    current_hidden = draft_hidden.detach()
    weight = config.step_decay
    cumulative_offset = 0

    for step in range(1, config.multi_step_k):
        if current_hidden.shape[1] < 2:
            break

        with torch.no_grad():
            pred_tokens = draft_logits.argmax(dim=-1)

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

        target_logits_k = target_logits[:, cumulative_offset:-1, :]
        min_len = min(draft_logits_k.shape[1], target_logits_k.shape[1])
        draft_logits_k = draft_logits_k[:, :min_len, :]
        target_logits_k = target_logits_k[:, :min_len, :]
        mask_k = attention_mask[
            :, cumulative_offset + 1:cumulative_offset + 1 + min_len
        ]

        kl_step_k = masked_kl_topk(
            draft_logits_k, target_logits_k, mask_k, config.topk_kl,
        )
        total_loss = total_loss + weight * kl_step_k

        current_hidden = draft_hidden_k.detach()
        draft_logits = draft_logits_k
        weight *= config.step_decay

    return total_loss


# ---------------------------------------------------------------------------
# Training loop  (same pipeline as v1)
# ---------------------------------------------------------------------------


def train_eagle3_head(
    target_model: AutoModelForCausalLM,
    draft_head: Eagle3DraftHead,
    eagle3_config: Eagle3Config,
    train_dataset: Dataset,
    config: TrainingConfig,
    device: torch.device,
) -> Eagle3DraftHead:
    target_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        draft_head.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    batches_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.grad_accum_steps)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(config.warmup_steps, int(config.warmup_ratio * total_steps))
    warmup_steps = min(warmup_steps, total_steps // 5)

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
            draft_head, config.resume_checkpoint,
            optimizer=optimizer, scheduler=scheduler,
        )
        global_step = resume_state["global_step"]
        start_epoch = resume_state["epoch"]
        start_batch_in_epoch = resume_state["batch_in_epoch"]
        if start_epoch >= config.epochs:
            logger.info("Checkpoint already completed %d epochs.", config.epochs)
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
            logger.info("Resuming epoch %d from batch %d", epoch + 1, resume_offset)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < resume_offset:
                continue

            if microbatches_since_step == 0:
                remaining_batches = total_batches - batch_idx
                current_window_target = min(
                    config.grad_accum_steps, remaining_batches,
                )

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

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

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                try:
                    raw_loss = compute_multi_step_loss(
                        runner_head, target_logits, target_features,
                        input_ids, attention_mask, config, eagle3_config,
                    )
                except RuntimeError as exc:
                    compile_error = compile_enabled and (
                        "torch._dynamo" in str(exc)
                        or "fake_tensor" in str(exc)
                        or "Failed running call_function" in str(exc)
                    )
                    if not compile_error:
                        raise
                    logger.warning("torch.compile failed; falling back to eager: %s", exc)
                    runner_head = draft_head
                    compile_enabled = False
                    torch.cuda.empty_cache()
                    raw_loss = compute_multi_step_loss(
                        runner_head, target_logits, target_features,
                        input_ids, attention_mask, config, eagle3_config,
                    )
                loss = raw_loss / current_window_target

            loss.backward()
            accum_loss += raw_loss.item()
            accum_count += 1
            microbatches_since_step += 1

            batches_seen = batch_idx + 1
            if microbatches_since_step == current_window_target:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    draft_head.trainable_parameters(), max_norm=1.0,
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
                        "Epoch %d, step %d/%d, loss=%.4f, lr=%.2e, grad_norm=%.3f",
                        epoch + 1, global_step, total_steps,
                        avg_loss, lr, grad_norm,
                    )
                    accum_loss = 0.0
                    accum_count = 0

                if global_step % config.save_every == 0:
                    save_checkpoint(
                        draft_head, optimizer, scheduler,
                        global_step, epoch, batches_seen,
                        config.checkpoint_dir,
                        final_checkpoint_name=config.final_checkpoint_name,
                    )

            epoch_loss += raw_loss.item()
            num_batches += 1

        epoch_time = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        logger.info(
            "Epoch %d complete. Avg loss=%.4f, Time=%.1fs",
            epoch + 1, avg_epoch_loss, epoch_time,
        )

        save_checkpoint(
            draft_head, optimizer, scheduler,
            global_step, epoch + 1, 0,
            config.checkpoint_dir,
            final_checkpoint_name=config.final_checkpoint_name,
        )

    save_checkpoint(
        draft_head, optimizer, scheduler,
        global_step, config.epochs, 0,
        config.checkpoint_dir, final=True,
        final_checkpoint_name=config.final_checkpoint_name,
    )

    return draft_head


# ---------------------------------------------------------------------------
# Checkpoint I/O  (unchanged)
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
    stem = os.path.splitext(final_checkpoint_name)[0]
    if stem.endswith("_final"):
        stem = stem[:-6]
    path = (
        os.path.join(checkpoint_dir, final_checkpoint_name)
        if final
        else os.path.join(checkpoint_dir, f"{stem}_step{global_step}.pt")
    )

    trainable_state = {
        name: param
        for name, param in draft_head.state_dict().items()
        if any(name.startswith(p) for p in ["fusion_fc", "input_fc", "decoder_layer"])
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

    tmp = f"{path}.tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)
    logger.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(
    draft_head: Eagle3DraftHead,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train EAGLE-3 draft head (v2 — mixed dataset)"
    )
    parser.add_argument("--target-model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--target-4bit", dest="target_4bit",
                        action="store_true", default=True)
    parser.add_argument("--no-target-4bit", dest="target_4bit", action="store_false")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--ce-loss-weight", type=float, default=0.3)
    parser.add_argument("--topk-kl", type=int, default=50,
                        help="Top-K for focused KL loss (0 = full vocab)")
    parser.add_argument("--sharegpt-fraction", type=float, default=0.5)
    parser.add_argument("--code-fraction", type=float, default=0.3)
    parser.add_argument("--creative-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-draft-head", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/eagle3/gemma3_12b")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--final-checkpoint-name", type=str, default=None)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")

    total = (
        args.sharegpt_fraction + args.code_fraction + args.creative_fraction
    )
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"Dataset fractions must sum to 1.0 (got {total:.3f})")

    torch.set_float32_matmul_precision("high")
    device = get_device()
    if torch.cuda.is_available():
        logger.info("Training on: %s", torch.cuda.get_device_name(0))

    final_ckpt_name = args.final_checkpoint_name
    if final_ckpt_name is None:
        m = args.target_model.lower()
        if "gemma-4" in m or "gemma4" in m:
            final_ckpt_name = "eagle3_gemma4_31b_final.pt"
        elif "gemma-3-12b" in m or "gemma3-12b" in m:
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
        ce_loss_weight=args.ce_loss_weight,
        topk_kl=args.topk_kl,
        sharegpt_fraction=args.sharegpt_fraction,
        code_fraction=args.code_fraction,
        creative_fraction=args.creative_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        final_checkpoint_name=final_ckpt_name,
        resume_checkpoint=args.resume,
        compile_draft_head=args.compile_draft_head,
    )

    logger.info("Loading target model: %s", config.target_model_id)
    target_model = load_model(
        config.target_model_id,
        quantize_4bit=config.target_quantize_4bit,
        compile_model=False,
    )
    tokenizer = load_tokenizer(config.target_model_id)

    eagle3_config = Eagle3Config.from_model(target_model)
    draft_head = Eagle3DraftHead(eagle3_config, target_model)
    draft_head = draft_head.to(device=device, dtype=torch.bfloat16)

    logger.info(
        "Draft head: %d trainable params (%.1f M)",
        draft_head.num_trainable_params(),
        draft_head.num_trainable_params() / 1e6,
    )

    train_dataset = create_training_dataset(tokenizer, config)

    logger.info(
        "Starting training (v2 — mixed dataset, top-K KL, 1024 seq len)..."
    )
    torch.cuda.empty_cache()
    train_eagle3_head(
        target_model, draft_head, eagle3_config, train_dataset, config, device,
    )

    logger.info("Training complete. Checkpoints in: %s", config.checkpoint_dir)


if __name__ == "__main__":
    main() 