"""
EAGLE-3 speculative decoding: lightweight draft head on target hidden states
with tree-structured candidate verification.

Key components:
  - Eagle3Config: hyperparameters for the draft head and tree search
  - Eagle3DraftHead: single-layer decoder head fusing multi-layer target features
  - build_draft_tree: BFS tree expansion with budget cap
  - build_tree_attention_mask: 4D causal mask for tree verification
  - verify_tree: accept longest valid root-to-leaf path
  - eagle3_decode: main generation loop
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import (
    CudaTimer,
    GenerationMetrics,
    RoundMetrics,
    record_peak_vram,
    reset_peak_vram,
)
from sampling import (
    rejection_sample_token,
    sample_bonus_token,
    sample_from_logits,
)
from speculative import _get_cache_seq_len

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _clone_draft_kv(kv):
    """Lightweight KV cache clone via tensor .clone() instead of copy.deepcopy."""
    if kv is None:
        return None
    if hasattr(kv, "key_cache"):
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        for layer_idx in range(len(kv.key_cache)):
            new_cache.update(
                kv.key_cache[layer_idx].clone(),
                kv.value_cache[layer_idx].clone(),
                layer_idx,
            )
        return new_cache
    if isinstance(kv, (list, tuple)):
        return type(kv)(
            (entry[0].clone(), entry[1].clone()) for entry in kv
        )
    return copy.deepcopy(kv)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Eagle3Config:
    """Hyperparameters for the EAGLE-3 draft head and tree search."""

    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288  # Qwen3-8B MLP size
    feature_layers: Tuple[int, ...] = (4, 16, 31)  # low / mid / high
    tree_budget: int = 60
    max_depth: int = 6
    top_k: int = 10
    vocab_size: int = 151936  # Qwen3 tokenizer
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0


# ---------------------------------------------------------------------------
# Draft Head
# ---------------------------------------------------------------------------


class Eagle3DraftHead(nn.Module):
    """
    Lightweight draft head for EAGLE-3 speculative decoding.

    Architecture:
      1. fusion_fc: Linear(3*H, H) — fuse 3 target hidden layers
      2. input_fc: Linear(2*H, H) — combine [token_embedding, fused_features]
      3. One Qwen3-style decoder layer (GQA attention + SiLU MLP)
      4. Shared frozen lm_head + norm from target model

    The draft head reuses the target model's embedding and lm_head (frozen).
    Only fusion_fc, input_fc, and the decoder layer are trainable (~400M params).
    """

    def __init__(self, config: Eagle3Config, target_model: AutoModelForCausalLM):
        super().__init__()
        self.config = config
        H = config.hidden_size

        # Trainable layers
        self.fusion_fc = nn.Linear(3 * H, H, bias=False)
        self.input_fc = nn.Linear(2 * H, H, bias=False)

        # Copy one decoder layer from the target as the draft decoder
        # Use the first layer as template, then it's trained from scratch
        self.decoder_layer = copy.deepcopy(target_model.model.layers[0])
        # Re-enable gradients on the decoder layer
        for p in self.decoder_layer.parameters():
            p.requires_grad = True

        # Shared frozen references from target
        self.embed_tokens = target_model.model.embed_tokens
        self.norm = target_model.model.norm
        self.lm_head = target_model.lm_head
        self.rotary_emb = target_model.model.rotary_emb

        # Freeze shared components
        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False

    def forward(
        self,
        token_ids: torch.Tensor,
        fused_hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        """
        Forward pass of the draft head.

        Args:
            token_ids: (B, S) token IDs for embedding lookup
            fused_hidden: (B, S, H) fused multi-layer features from target
            position_ids: (B, S) position IDs for RoPE
            past_key_values: KV cache for the single decoder layer
            use_cache: whether to return updated KV cache

        Returns:
            (logits, hidden_state, new_kv_cache)
            - logits: (B, S, V) draft logits over vocabulary
            - hidden_state: (B, S, H) draft hidden state (fed back for tree expansion)
            - new_kv_cache: updated KV cache for the decoder layer
        """
        # Get token embeddings (frozen)
        token_emb = self.embed_tokens(token_ids)  # (B, S, H)

        # Combine token embedding with fused target features
        combined = self.input_fc(torch.cat([token_emb, fused_hidden], dim=-1))  # (B, S, H)

        # Compute RoPE position embeddings (cos, sin) from position_ids
        position_embeddings = self.rotary_emb(combined, position_ids)

        # Run through single decoder layer
        layer_out = self.decoder_layer(
            combined,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_values,
            use_cache=use_cache,
        )

        hidden_state = layer_out[0]  # (B, S, H)
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
        new_kv = layer_out[2] if use_cache and len(layer_out) > 2 else None

        normed = self.norm(hidden_state)
        logits = self.lm_head(normed)  # (B, S, V)

        return logits, hidden_state, new_kv

    def fuse_target_features(
        self, hidden_states_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse hidden states from target model's feature_layers.

        Args:
            hidden_states_list: list of 3 tensors, each (B, S, H),
                from layers specified by config.feature_layers

        Returns:
            (B, S, H) fused feature tensor
        """
        concatenated = torch.cat(hidden_states_list, dim=-1)  # (B, S, 3*H)
        return self.fusion_fc(concatenated)  # (B, S, H)

    def trainable_parameters(self):
        """Yield only the trainable parameters (not frozen shared refs)."""
        yield from self.fusion_fc.parameters()
        yield from self.input_fc.parameters()
        yield from self.decoder_layer.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# ---------------------------------------------------------------------------
# Tree construction data structure
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """One node in the draft candidate tree."""

    token_id: int
    parent_idx: int  # -1 for root
    depth: int
    cum_logprob: float
    draft_probs: torch.Tensor  # (vocab_size,) distribution at this node
    draft_hidden: torch.Tensor  # (1, 1, H) hidden state for expansion
    draft_kv: object = None  # KV cache state for the decoder layer


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def build_draft_tree(
    draft_head: Eagle3DraftHead,
    last_token_id: int,
    fused_features: torch.Tensor,
    config: Eagle3Config,
    base_position: int,
    temperature: float = 0.0,
    generator: torch.Generator = None,
) -> List[TreeNode]:
    """
    Build a tree of candidate tokens via BFS expansion of the draft head.

    Args:
        draft_head: the EAGLE-3 draft head module
        last_token_id: the most recently accepted token
        fused_features: (1, 1, H) fused hidden state from target
        config: Eagle3Config with tree_budget, max_depth, top_k
        base_position: position ID for the root node
        temperature: sampling temperature for logit scaling
        generator: optional torch generator

    Returns:
        List of TreeNode forming the candidate tree (index 0 = root's first child)
    """
    device = fused_features.device
    tree_nodes: List[TreeNode] = []

    # Level 0: run draft head on last accepted token to get root candidates
    token_tensor = torch.tensor([[last_token_id]], device=device, dtype=torch.long)
    pos_ids = torch.tensor([[base_position]], device=device, dtype=torch.long)

    logits, hidden, kv = draft_head(
        token_ids=token_tensor,
        fused_hidden=fused_features,
        position_ids=pos_ids,
        past_key_values=None,
        use_cache=True,
    )

    logits_0 = logits[0, -1, :]  # (V,)
    if temperature > 0:
        probs_0 = F.softmax(logits_0 / temperature, dim=-1)
    else:
        probs_0 = torch.zeros_like(logits_0)
        probs_0[logits_0.argmax()] = 1.0

    log_probs_0 = torch.log(probs_0 + 1e-10)
    top_k_vals, top_k_ids = torch.topk(log_probs_0, min(config.top_k, config.tree_budget))

    for i in range(len(top_k_ids)):
        tree_nodes.append(TreeNode(
            token_id=top_k_ids[i].item(),
            parent_idx=-1,  # child of the virtual root
            depth=1,
            cum_logprob=top_k_vals[i].item(),
            draft_probs=probs_0.clone(),
            draft_hidden=hidden[:, -1:, :].clone(),
            draft_kv=_clone_draft_kv(kv),
        ))

    if len(tree_nodes) >= config.tree_budget:
        return tree_nodes[:config.tree_budget]

    # BFS expansion: levels 1 through max_depth-1
    for depth in range(2, config.max_depth + 1):
        if len(tree_nodes) >= config.tree_budget:
            break

        # Gather leaf nodes at depth-1 (candidates for expansion)
        leaves = [
            (idx, node) for idx, node in enumerate(tree_nodes)
            if node.depth == depth - 1
        ]

        if not leaves:
            break

        # Sort by cumulative log probability, pick top-n to expand
        remaining_budget = config.tree_budget - len(tree_nodes)
        n_expand = min(len(leaves), max(1, remaining_budget // config.top_k))
        leaves.sort(key=lambda x: x[1].cum_logprob, reverse=True)
        leaves = leaves[:n_expand]

        for leaf_idx, leaf_node in leaves:
            if len(tree_nodes) >= config.tree_budget:
                break

            # Run draft head for this leaf
            tok = torch.tensor([[leaf_node.token_id]], device=device, dtype=torch.long)
            pos = torch.tensor([[base_position + leaf_node.depth]], device=device, dtype=torch.long)

            # Feed the leaf's hidden state as fused features
            logits_d, hidden_d, kv_d = draft_head(
                token_ids=tok,
                fused_hidden=leaf_node.draft_hidden,
                position_ids=pos,
                past_key_values=leaf_node.draft_kv,
                use_cache=True,
            )

            logits_leaf = logits_d[0, -1, :]
            if temperature > 0:
                probs_leaf = F.softmax(logits_leaf / temperature, dim=-1)
            else:
                probs_leaf = torch.zeros_like(logits_leaf)
                probs_leaf[logits_leaf.argmax()] = 1.0

            log_probs_leaf = torch.log(probs_leaf + 1e-10)
            k = min(config.top_k, config.tree_budget - len(tree_nodes))
            top_vals, top_ids = torch.topk(log_probs_leaf, k)

            for i in range(len(top_ids)):
                if len(tree_nodes) >= config.tree_budget:
                    break
                tree_nodes.append(TreeNode(
                    token_id=top_ids[i].item(),
                    parent_idx=leaf_idx,
                    depth=depth,
                    cum_logprob=leaf_node.cum_logprob + top_vals[i].item(),
                    draft_probs=probs_leaf.clone(),
                    draft_hidden=hidden_d[:, -1:, :].clone(),
                    draft_kv=_clone_draft_kv(kv_d),
                ))

    return tree_nodes


# ---------------------------------------------------------------------------
# Tree attention mask
# ---------------------------------------------------------------------------


def build_tree_attention_mask(
    tree_nodes: List[TreeNode],
    prefix_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a 4D attention mask for tree-structured verification.

    Each tree node attends to:
      - All prefix positions (the prompt + previously accepted tokens)
      - Itself
      - Its ancestors in the tree

    Args:
        tree_nodes: list of TreeNode from build_draft_tree
        prefix_len: number of tokens in the prefix (prompt + accepted)
        device: target device

    Returns:
        (mask_4d, position_ids) where:
        - mask_4d: (1, 1, N_tree, prefix_len + N_tree) in BF16
          0.0 = attend, -inf = mask out
        - position_ids: (1, N_tree) position IDs for each tree node
    """
    N = len(tree_nodes)
    total_len = prefix_len + N

    # Start with everything masked
    mask = torch.full(
        (1, 1, N, total_len),
        float("-inf"),
        device=device,
        dtype=torch.bfloat16,
    )

    # All tree nodes attend to the full prefix
    mask[:, :, :, :prefix_len] = 0.0

    # Each node attends to itself
    for i in range(N):
        mask[0, 0, i, prefix_len + i] = 0.0

    # Each node attends to its ancestors
    for i in range(N):
        parent = tree_nodes[i].parent_idx
        while parent >= 0:
            mask[0, 0, i, prefix_len + parent] = 0.0
            parent = tree_nodes[parent].parent_idx

    # Position IDs: (prefix_len - 1) + depth (siblings share position for correct RoPE)
    # prefix_len includes the root token, so depth-1 nodes are at prefix_len - 1 + 1 = prefix_len
    position_ids = torch.tensor(
        [[prefix_len - 1 + node.depth for node in tree_nodes]],
        device=device,
        dtype=torch.long,
    )

    return mask, position_ids


# ---------------------------------------------------------------------------
# KV cache selection (arbitrary positions, not just prefix trimming)
# ---------------------------------------------------------------------------


def _select_kv_cache_positions(past_key_values, indices: torch.Tensor):
    """
    Select arbitrary positions from the KV cache.

    Unlike _trim_kv_cache (prefix slicing), this gathers specific positions
    by index. Used after tree verification to keep only prefix + accepted path.

    Args:
        past_key_values: DynamicCache or similar
        indices: (N,) long tensor of position indices to keep

    Returns:
        The modified cache (in-place for DynamicCache)
    """
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "key_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx] = (
                past_key_values.key_cache[layer_idx][:, :, indices, :]
            )
            past_key_values.value_cache[layer_idx] = (
                past_key_values.value_cache[layer_idx][:, :, indices, :]
            )
        return past_key_values

    if hasattr(past_key_values, "crop"):
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        for layer_data in past_key_values:
            k, v = layer_data[0], layer_data[1]
            new_cache.update(k[:, :, indices, :], v[:, :, indices, :], new_cache.get_seq_length())
        return new_cache

    return tuple(
        (entry[0][:, :, indices, :], entry[1][:, :, indices, :])
        for entry in past_key_values
    )


# ---------------------------------------------------------------------------
# Tree verification
# ---------------------------------------------------------------------------


def _get_all_paths(tree_nodes: List[TreeNode]) -> List[List[int]]:
    """Extract all root-to-leaf paths from the tree. Returns list of node index lists."""
    # Find leaf nodes (no children)
    has_children = set()
    for node in tree_nodes:
        if node.parent_idx >= 0:
            has_children.add(node.parent_idx)

    leaves = [i for i in range(len(tree_nodes)) if i not in has_children]

    paths = []
    for leaf_idx in leaves:
        path = []
        idx = leaf_idx
        while idx >= 0:
            path.append(idx)
            idx = tree_nodes[idx].parent_idx
        path.reverse()
        paths.append(path)

    return paths


def verify_tree(
    target_logits: torch.Tensor,
    tree_nodes: List[TreeNode],
    prefix_len: int,
    temperature: float,
    generator: torch.Generator = None,
    prefix_logits: torch.Tensor = None,
) -> Tuple[List[int], int, List[bool], List[int]]:
    """
    Verify tree candidates against target model logits.

    Walks each root-to-leaf path and finds the longest accepted prefix.

    Args:
        target_logits: (1, N_tree, V) logits from target model on tree nodes
        tree_nodes: candidate tree
        prefix_len: length of the prefix for position calculations
        temperature: sampling temperature
        generator: optional torch generator
        prefix_logits: (V,) logits from target model at root token position,
            used to verify root children (depth-1 nodes)

    Returns:
        (accepted_tokens, num_accepted, per_token_accepted, accepted_cache_indices)
        - accepted_tokens: list of token IDs (accepted + bonus/correction)
        - num_accepted: count of accepted draft tokens
        - per_token_accepted: bool list for each checked token
        - accepted_cache_indices: indices into the tree verification cache to keep
    """
    paths = _get_all_paths(tree_nodes)

    best_accepted_tokens: List[int] = []
    best_num_accepted = 0
    best_per_token: List[bool] = []
    best_path_node_indices: List[int] = []

    for path in paths:
        accepted_tokens: List[int] = []
        per_token: List[bool] = []
        num_acc = 0

        for step, node_idx in enumerate(path):
            node = tree_nodes[node_idx]
            # Target logits for this node predict the NEXT token
            # The target model was given all tree nodes, and logits[node_idx]
            # predicts what should come after this node
            t_logits = target_logits[0, node_idx, :]

            if temperature == 0.0:
                target_probs = torch.zeros_like(t_logits)
                target_probs[t_logits.argmax()] = 1.0
            else:
                target_probs = F.softmax(t_logits / temperature, dim=-1)

            # For the first node in path, verify against the parent's prediction
            # The draft proposed node.token_id; we check if the target agrees
            # Actually, we need the target logits at the PARENT position to verify this node
            # But with tree attention, the target processes all nodes simultaneously.
            # The logits at position node_idx predict what comes AFTER node_idx.

            # For verification: we check if the target model at the parent position
            # would have chosen this node's token.
            # In tree structure: logits at parent_pos predict child tokens.
            # For root children (parent_idx=-1), we need the logits from the prefix's last token.

            # Re-think: target_logits[0, i, :] are logits from position i predicting position i+1.
            # For a tree node at index i, its token was proposed by the draft.
            # To verify node i, we need: target's prediction at node i's parent position.
            # - If parent_idx == -1: the parent is the last prefix token, which isn't in tree_logits.
            #   We need logits from position (prefix_len - 1). But those aren't in target_logits
            #   (which only covers tree nodes). So for root children, they're verified by
            #   the target model's output at the virtual root position.
            #   Actually — we pass tree node tokens to the target with the tree mask.
            #   The target sees prefix (via KV cache) + tree nodes.
            #   target_logits[0, i, :] are the logits at tree position i.
            #   To verify tree node i's token_id:
            #     If i is a root child (parent_idx == -1): we need the logit from the last prefix position.
            #       But that's not in target_logits (it's from the prefix KV cache).
            #       HOWEVER, the tree mask allows root children to see the full prefix.
            #       The target_logits for root children are their own output — predicting their child.
            #
            # Let me reconsider the verification strategy:
            # The standard approach: target logits at position P predict token at P+1.
            # With tree attention and prefix KV cache:
            #   - We pass N tree node token IDs to the target.
            #   - The target outputs N sets of logits.
            #   - logits[i] predict what comes after tree node i.
            #
            # Verification of a path [n0, n1, n2, ...]:
            #   - n0 is a root child. To verify n0's token, we need the target's prediction
            #     at the last prefix position. This is handled OUTSIDE the tree — the caller
            #     should provide prefix_logits separately, OR we handle it in eagle3_decode.
            #   - n1's token is verified by logits at n0's position: target_logits[0, n0_idx, :]
            #   - n2's token is verified by logits at n1's position: target_logits[0, n1_idx, :]
            #   etc.
            #
            # So: to verify node path[step], we use target_logits at path[step-1].
            # For step=0 (root child), we need external prefix logits.

            if step == 0:
                # Root child: verify using prefix_logits (target's prediction at root position)
                if prefix_logits is not None:
                    if temperature == 0.0:
                        p_probs = torch.zeros_like(prefix_logits)
                        p_probs[prefix_logits.argmax()] = 1.0
                    else:
                        p_probs = F.softmax(prefix_logits / temperature, dim=-1)

                    accepted, token = rejection_sample_token(
                        p_probs,
                        node.draft_probs,
                        node.token_id,
                        temperature,
                        generator,
                    )

                    if accepted:
                        accepted_tokens.append(node.token_id)
                        per_token.append(True)
                        num_acc += 1
                    else:
                        accepted_tokens.append(token)
                        per_token.append(False)
                        break
                    continue
                else:
                    # No prefix logits provided — accept unconditionally (legacy fallback)
                    accepted_tokens.append(node.token_id)
                    per_token.append(True)
                    num_acc += 1
                    continue

            # Verify this node using parent's target logits
            parent_node_idx = path[step - 1]
            parent_target_logits = target_logits[0, parent_node_idx, :]

            if temperature == 0.0:
                parent_probs = torch.zeros_like(parent_target_logits)
                parent_probs[parent_target_logits.argmax()] = 1.0
            else:
                parent_probs = F.softmax(parent_target_logits / temperature, dim=-1)

            accepted, token = rejection_sample_token(
                parent_probs,
                node.draft_probs,
                node.token_id,
                temperature,
                generator,
            )

            if accepted:
                accepted_tokens.append(node.token_id)
                per_token.append(True)
                num_acc += 1
            else:
                # Rejected — add correction token
                accepted_tokens.append(token)
                per_token.append(False)
                break

        # Check if this path is the best so far
        if num_acc > best_num_accepted:
            best_num_accepted = num_acc
            best_accepted_tokens = accepted_tokens
            best_per_token = per_token
            best_path_node_indices = path[:num_acc]

    # Bonus token: if we accepted the entire best path, sample one more
    # from the last accepted node's target logits
    if best_num_accepted > 0 and len(best_per_token) == best_num_accepted:
        last_node_idx = best_path_node_indices[-1]
        if last_node_idx < target_logits.shape[1]:
            bonus_logits = target_logits[0, last_node_idx, :]
            if temperature == 0.0:
                bonus_probs = torch.zeros_like(bonus_logits)
                bonus_probs[bonus_logits.argmax()] = 1.0
            else:
                bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
            bonus_tok = sample_bonus_token(bonus_probs, temperature, generator)
            best_accepted_tokens.append(bonus_tok)

    # Build cache indices: prefix positions + accepted tree node positions
    accepted_cache_indices = list(range(prefix_len)) + [
        prefix_len + ni for ni in best_path_node_indices
    ]

    return (
        best_accepted_tokens,
        best_num_accepted,
        best_per_token,
        accepted_cache_indices,
    )


# ---------------------------------------------------------------------------
# Feature extraction from target model
# ---------------------------------------------------------------------------


def _extract_target_features(
    target_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    feature_layers: Tuple[int, ...],
    past_key_values=None,
    cache_position_start: int = 0,
) -> Tuple[torch.Tensor, List[torch.Tensor], object]:
    """
    Run target model forward pass and extract hidden states from feature_layers.

    Returns:
        (logits, feature_hidden_states, updated_kv_cache)
    """
    # Determine which tokens to feed (incremental if cache exists)
    if past_key_values is not None and cache_position_start > 0:
        feed_ids = input_ids[:, cache_position_start:]
        out = target_model(
            input_ids=feed_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
    else:
        out = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

    logits = out.logits
    all_hidden = out.hidden_states  # tuple of (B, S, H) for each layer
    features = [all_hidden[layer_idx] for layer_idx in feature_layers]

    return logits, features, out.past_key_values


# ---------------------------------------------------------------------------
# Main EAGLE-3 decode loop
# ---------------------------------------------------------------------------


@torch.inference_mode()
def eagle3_decode(
    target_model: AutoModelForCausalLM,
    draft_head: Eagle3DraftHead,
    eagle3_config: Eagle3Config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    max_new_tokens: int,
    tokenizer: AutoTokenizer,
    generator: torch.Generator = None,
) -> Dict:
    """
    Full EAGLE-3 speculative decoding for one prompt.

    Args:
        target_model: Large target model (with output_hidden_states support)
        draft_head: Trained Eagle3DraftHead
        eagle3_config: Eagle3Config hyperparameters
        input_ids: (1, prompt_len) input token IDs
        attention_mask: (1, prompt_len) attention mask
        temperature: Sampling temperature (0.0 = greedy)
        max_new_tokens: Maximum new tokens to generate
        tokenizer: For EOS detection and decoding
        generator: Optional torch generator

    Returns:
        dict with "output_ids", "output_text", and "metrics"
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    feature_layers = eagle3_config.feature_layers

    generated_ids: List[int] = []
    target_cache = None
    target_cache_len = 0

    all_rounds: List[RoundMetrics] = []
    total_draft_ms = 0.0

    reset_peak_vram()
    wall_start = time.perf_counter()
    ttft_ms = 0.0
    ttft_recorded = False

    # Save and restore attention implementation for tree verification
    original_attn_impl = getattr(target_model.config, "_attn_implementation", "sdpa")

    while len(generated_ids) < max_new_tokens:
        round_start = time.perf_counter()
        current_len = prompt_len + len(generated_ids)
        remaining = max_new_tokens - len(generated_ids)
        if remaining <= 0:
            break

        # Build current full sequence
        if generated_ids:
            gen_tensor = torch.tensor(
                [generated_ids], device=device, dtype=input_ids.dtype
            )
            full_ids = torch.cat([input_ids, gen_tensor], dim=1)
            full_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        1, len(generated_ids), device=device, dtype=attention_mask.dtype
                    ),
                ],
                dim=1,
            )
        else:
            full_ids = input_ids
            full_mask = attention_mask

        # ---- Step 1: Target forward for features + logits ----
        with CudaTimer() as feature_timer, torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, features, target_cache = _extract_target_features(
                target_model,
                full_ids,
                full_mask,
                feature_layers,
                past_key_values=target_cache,
                cache_position_start=target_cache_len,
            )

        # Get the last position's features and logits (for the token we just generated)
        last_logits = logits[0, -1, :]  # (V,)
        last_features = [f[:, -1:, :] for f in features]  # each (1, 1, H)
        fused_features = draft_head.fuse_target_features(last_features)  # (1, 1, H)
        target_cache_len = current_len

        # Sample the first token from target logits (this is the "root" token)
        root_token_id, root_probs = sample_from_logits(
            last_logits, temperature, generator
        )

        if not ttft_recorded:
            ttft_ms = (time.perf_counter() - wall_start) * 1000.0
            ttft_recorded = True

        # If tree_budget <= 1, just do linear (no tree)
        if eagle3_config.tree_budget <= 1 or remaining <= 1:
            generated_ids.append(root_token_id)
            round_ms = (time.perf_counter() - round_start) * 1000.0
            all_rounds.append(
                RoundMetrics(
                    round_index=len(all_rounds),
                    draft_tokens_proposed=0,
                    tokens_accepted=0,
                    bonus_token_generated=False,
                    total_tokens_produced=1,
                    per_token_accepted=[],
                    draft_time_ms=0.0,
                    verify_time_ms=feature_timer.elapsed_ms,
                    round_time_ms=round_ms,
                )
            )
            if root_token_id == tokenizer.eos_token_id:
                break
            continue

        # ---- Step 2: Build draft tree ----
        with CudaTimer() as draft_timer, torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tree_nodes = build_draft_tree(
                draft_head,
                root_token_id,
                fused_features,
                eagle3_config,
                base_position=current_len,
                temperature=temperature,
                generator=generator,
            )

        draft_ms = draft_timer.elapsed_ms
        total_draft_ms += draft_ms

        if not tree_nodes:
            # No tree candidates — just emit root token
            generated_ids.append(root_token_id)
            round_ms = (time.perf_counter() - round_start) * 1000.0
            all_rounds.append(
                RoundMetrics(
                    round_index=len(all_rounds),
                    draft_tokens_proposed=0,
                    tokens_accepted=0,
                    bonus_token_generated=False,
                    total_tokens_produced=1,
                    per_token_accepted=[],
                    draft_time_ms=draft_ms,
                    verify_time_ms=feature_timer.elapsed_ms,
                    round_time_ms=round_ms,
                )
            )
            if root_token_id == tokenizer.eos_token_id:
                break
            continue

        # ---- Step 3: Build tree attention mask and verify with target ----
        tree_mask, tree_position_ids = build_tree_attention_mask(
            tree_nodes, current_len + 1, device  # +1 for the root token
        )

        # Prepare tree node token IDs
        tree_token_ids = torch.tensor(
            [[node.token_id for node in tree_nodes]],
            device=device,
            dtype=input_ids.dtype,
        )

        # The root token needs to be added to the KV cache first
        # Add root token to the sequence
        root_tensor = torch.tensor([[root_token_id]], device=device, dtype=input_ids.dtype)
        root_mask = torch.cat(
            [full_mask, torch.ones(1, 1, device=device, dtype=full_mask.dtype)],
            dim=1,
        )

        # Run target on root token to update KV cache
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            root_out = target_model(
                input_ids=root_tensor,
                attention_mask=root_mask,
                past_key_values=target_cache,
                use_cache=True,
            )
        target_cache = root_out.past_key_values

        # Now run target on all tree nodes with 4D tree mask
        # Switch to eager attention for 4D mask support
        target_model.config._attn_implementation = "eager"
        # Also need to set on model's internal attention modules
        for layer in target_model.model.layers:
            layer.self_attn.config._attn_implementation = "eager"

        # Build full attention mask for tree nodes (they see prefix + root + tree)
        tree_full_mask = torch.cat(
            [
                root_mask,
                torch.ones(1, len(tree_nodes), device=device, dtype=root_mask.dtype),
            ],
            dim=1,
        )

        with CudaTimer() as verify_timer, torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tree_out = target_model(
                input_ids=tree_token_ids,
                attention_mask=tree_mask,
                past_key_values=target_cache,
                use_cache=True,
                position_ids=tree_position_ids,
            )

        # Restore flash attention
        target_model.config._attn_implementation = original_attn_impl
        for layer in target_model.model.layers:
            layer.self_attn.config._attn_implementation = original_attn_impl

        tree_logits = tree_out.logits  # (1, N_tree, V)
        tree_cache = tree_out.past_key_values

        # ---- Step 4: Verify tree candidates ----
        # root_out.logits[0, -1, :] are the target's predictions after the root token,
        # used to verify root children (depth-1 nodes)
        root_prefix_logits = root_out.logits[0, -1, :]
        accepted_tokens_tree, num_accepted_tree, per_token, accepted_indices = verify_tree(
            tree_logits,
            tree_nodes,
            current_len + 1,  # prefix includes root
            temperature,
            generator,
            prefix_logits=root_prefix_logits,
        )

        # The root token is always accepted (it came from the target model)
        accepted_tokens = [root_token_id] + accepted_tokens_tree
        total_accepted = 1 + num_accepted_tree

        # Trim the target KV cache to keep only prefix + root + accepted path
        accepted_indices_tensor = torch.tensor(
            accepted_indices, device=device, dtype=torch.long
        )
        target_cache = _select_kv_cache_positions(tree_cache, accepted_indices_tensor)
        target_cache_len = current_len + total_accepted

        # Trim to not exceed max_new_tokens
        space_left = max_new_tokens - len(generated_ids)
        if len(accepted_tokens) > space_left:
            accepted_tokens = accepted_tokens[:space_left]
            total_accepted = min(total_accepted, space_left)

        generated_ids.extend(accepted_tokens)

        verify_ms = verify_timer.elapsed_ms

        round_ms = (time.perf_counter() - round_start) * 1000.0
        all_rounds.append(
            RoundMetrics(
                round_index=len(all_rounds),
                draft_tokens_proposed=len(tree_nodes),
                tokens_accepted=num_accepted_tree,
                bonus_token_generated=len(accepted_tokens_tree) > num_accepted_tree,
                total_tokens_produced=len(accepted_tokens),
                per_token_accepted=per_token,
                draft_time_ms=draft_ms,
                verify_time_ms=verify_ms,
                round_time_ms=round_ms,
            )
        )

        # Check for EOS
        if tokenizer.eos_token_id in accepted_tokens:
            break

    # ---- Finalize metrics ----
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    total_tokens = len(generated_ids)
    tps = (total_tokens / wall_ms * 1000.0) if wall_ms > 0 else 0.0

    all_decisions = []
    for r in all_rounds:
        all_decisions.extend(r.per_token_accepted)
    acceptance_rate = (
        sum(all_decisions) / len(all_decisions) if all_decisions else 0.0
    )

    acceptance_length = (
        sum(r.tokens_accepted for r in all_rounds) / len(all_rounds)
        if all_rounds
        else 0.0
    )

    draft_overhead = total_draft_ms / wall_ms if wall_ms > 0 else 0.0

    metrics = GenerationMetrics(
        prompt_index=-1,
        total_tokens_generated=total_tokens,
        total_rounds=len(all_rounds),
        wall_clock_ms=wall_ms,
        ttft_ms=ttft_ms,
        tokens_per_second=tps,
        acceptance_rate=acceptance_rate,
        acceptance_length=acceptance_length,
        draft_overhead_ratio=draft_overhead,
        peak_vram_bytes=record_peak_vram(),
        rounds=all_rounds,
    )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "output_ids": generated_ids,
        "output_text": output_text,
        "metrics": metrics,
    }
