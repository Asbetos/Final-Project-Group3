# `core/eagle3_train.py`

## Purpose

`core/eagle3_train.py` trains the EAGLE-3 draft head for the Gemma-3-12B target.

The target model stays frozen. Only the draft head is trained.

## Training Data

The script uses:

1. `vicgalle/alpaca-gpt4`

The helper dataset class:

1. `AlpacaDataset`

formats each example into a chat-style instruction-following sequence and tokenizes it to a fixed maximum length.

## Main Components

1. `TrainingConfig`
   Stores all training hyperparameters.
2. `AlpacaDataset`
   Loads and tokenizes the training set.
3. `create_training_dataloader(...)`
   Builds a deterministic per-epoch DataLoader.
4. `compute_multi_step_loss(...)`
   Implements the multi-step KL training objective.
5. `train_eagle3_head(...)`
   Main training loop.
6. `save_checkpoint(...)`
   Saves trainable draft-head weights and optimizer state.
7. `load_checkpoint(...)`
   Restores a previous checkpoint.
8. `main()`
   CLI entrypoint.

## Important Training Choices

1. target model can be loaded in 4-bit mode
2. BF16 autocast is used
3. 8-bit AdamW is preferred when `bitsandbytes` is available
4. gradient accumulation is supported
5. warmup plus cosine decay is used
6. gradient clipping is applied
7. checkpoints are saved periodically and at epoch end

## Multi-Step Loss

The training loss starts with target-derived fused features, then rolls forward using the draft head's own predictions for later steps. This is how the head learns to recover from its own errors at inference time.

## Active Training Defaults

The defaults in the file are already aligned with the active Gemma-3-12B run:

1. target model: `google/gemma-3-12b-it`
2. checkpoint dir: `checkpoints/eagle3/gemma3_12b`
3. final checkpoint name: `eagle3_gemma3_12b_final.pt`

## Current Scope Notes

1. Some help text still contains older Gemma-4 wording in one CLI description branch.
2. The actual defaults and saved checkpoints are now Gemma-3-12B oriented.
3. The completed training run used this script and produced the active final checkpoint for pair `I`.
