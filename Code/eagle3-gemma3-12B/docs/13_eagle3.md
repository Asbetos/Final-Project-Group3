# `core/eagle3.py`

## Purpose

`core/eagle3.py` contains the core EAGLE-3 implementation used by the active Gemma-3-12B workflow.

This is the most important file in the module after `core/runner.py` and `core/eagle3_train.py`.

## Main Pieces

### Configuration

1. `Eagle3Config`
   Stores hidden size, attention shape, feature layers, tree budget, depth, and vocabulary size.
2. `Eagle3Config.from_model(...)`
   Derives the config directly from the loaded target model.

### Draft head

1. `Eagle3DraftHead`
   The lightweight trainable head that:
   fuses three target hidden-state layers, combines them with token embeddings, runs one copied decoder layer, then reuses the frozen target norm and LM head.

### Tree generation

1. `TreeNode`
   Stores one candidate node in the draft tree.
2. `build_draft_tree(...)`
   Expands candidate continuations using BFS up to a tree budget.
3. `build_tree_attention_mask(...)`
   Builds a true tree mask for verification.
4. `verify_tree(...)`
   Verifies candidate paths against target probabilities.

### Generation loop

1. `_extract_target_features(...)`
   Runs the target model and extracts the configured hidden states.
2. `eagle3_decode(...)`
   Runs the complete EAGLE generation loop for one prompt.

## Important Current Implementation Detail

Although the file contains full tree-mask and path-verification utilities, the active decode loop does not currently use the full 4D masked tree verification path.

Instead, `eagle3_decode(...)` currently:

1. samples a root token from target logits
2. builds a draft tree from the fused hidden state
3. extracts all root-to-leaf paths
4. chooses the best path by cumulative log-probability
5. verifies that path sequentially in one target forward pass

This linearized path was chosen because it is more stable with current `transformers` cache behavior.

## Metric Behavior

The file records:

1. draft time
2. verification time
3. round time
4. TTFT
5. tokens per second
6. acceptance rate
7. acceptance length
8. draft overhead ratio
9. peak VRAM

## Current Scope Notes

1. The active project uses pair `I` with this file.
2. `tree_budget <= 1` becomes a degenerate linear path and is used by the greedy equivalence test.
3. The file contains more tree-related machinery than the currently active decode path uses, because it preserves experimentation hooks for future work.
