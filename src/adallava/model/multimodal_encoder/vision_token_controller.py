# Vision token controller (Exp1: adaptive vision token prune).
# Produces per-patch keep logits and selects exactly K patches via Gumbel-Softmax top-K.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scheduler.scheduler_utils import n_times_gumbel_softmax


class VisionTokenController(nn.Module):
    """
    Option B (Phase 1): vision_output [B, N, C] or [B, N+1, C] (if use_cls), budget_embedding [B, C],
    token_budget [B] (ratio).

    use_cls=False (LLaVA default): vision_output [B, N, C]; output [B, N, C] with masked patches.
    use_cls=True: vision_output [B, N+1, C] (CLS at 0); output [B, N+1, C] = [cls, masked_patches].
    Requires mm_vision_select_feature='cls_patch' so vision_tower returns [B, N+1, C].
    """

    def __init__(
        self,
        vision_dim: int,
        num_patches: int,
        tau: float = 5.0,
        use_cls: bool = False,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.num_patches = num_patches
        self.tau = tau
        self.use_cls = use_cls

        # Per-patch importance logits: [patch; budget_embed] -> logit. Single trainable linear.
        # Fusion is: concat(patch_tokens, budget_embedding) -> logit_head -> [B, N] logits.
        self.logit_head = nn.Linear(vision_dim * 2, 1)

    def forward(
        self,
        vision_output: torch.Tensor,
        token_budget: torch.Tensor,
        budget_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_output: [B, N, C] if use_cls=False; [B, N+1, C] (CLS at 0) if use_cls=True.
            token_budget: [B] float in [0,1] (ratio) or int K; used to get K = round(r*N) per batch.
            budget_embedding: [B, C] optional; if provided, concat with each patch for logits.

        Returns:
            selected_features: [B, N, C] or [B, N+1, C] with unselected patches set to 0.
            keep_mask: [B, N] binary mask over patches (1 = keep).
        """
        B, seq_len, C = vision_output.shape
        if self.use_cls:
            cls_tokens = vision_output[:, :1, :]   # [B, 1, C]
            patch_tokens = vision_output[:, 1:, :]  # [B, N, C]
        else:
            patch_tokens = vision_output  # [B, N, C]

        N = patch_tokens.size(1)

        # Map token_budget to integer K per batch
        if token_budget.dim() == 0:
            token_budget = token_budget.unsqueeze(0).expand(B)
        if token_budget.dtype in (torch.float32, torch.float16, torch.bfloat16):
            # ratio -> K
            K_per_batch = (token_budget * N).long().clamp(1, N)
        else:
            K_per_batch = token_budget.long().clamp(1, N)

        # Per-patch logits, optionally conditioned on budget_embedding
        if budget_embedding is not None:
            # [B, N, C] concat [B, C] broadcast -> [B, N, 2*C]
            patch_with_budget = torch.cat(
                [patch_tokens, budget_embedding.unsqueeze(1).expand(-1, N, -1)], dim=-1
            )
        else:
            patch_with_budget = torch.cat(
                [patch_tokens, patch_tokens], dim=-1
            )  # fallback: no extra info
        logits = self.logit_head(patch_with_budget).squeeze(-1)  # [B, N]

        # Select exactly K patches per batch via Gumbel; support different K per sample by loop
        keep_mask = torch.zeros(B, N, device=vision_output.device, dtype=vision_output.dtype)
        for b in range(B):
            k = min(int(K_per_batch[b].item()), N)
            mask_b = n_times_gumbel_softmax(
                logits[b : b + 1], n=k, tau=self.tau, hard=True, dim=-1, training=self.training
            )
            keep_mask[b] = mask_b.squeeze(0)

        # Fixed-length output for LLaVA: patch_tokens * mask -> [B, N, C].
        # Unselected tokens are masked to 0 (multiplied by keep_mask); gradient flows
        # through keep_mask via straight-through in Gumbel.
        masked_patches = patch_tokens * keep_mask.unsqueeze(-1)
        if self.use_cls:
            selected_features = torch.cat([cls_tokens, masked_patches], dim=1)
        else:
            selected_features = masked_patches

        return selected_features, keep_mask
