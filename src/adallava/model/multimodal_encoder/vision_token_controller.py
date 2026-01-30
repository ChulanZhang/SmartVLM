# Vision token controller (Exp1: adaptive vision token prune).
# Budget token is inside ViT; we use only the budget token output (fused with all patches)
# and one linear Linear(C, N) to get per-patch logits (same design as AdaLLaVA scheduler).

from typing import Tuple

import torch
import torch.nn as nn

from ..scheduler.scheduler_utils import n_times_gumbel_softmax


class VisionTokenController(nn.Module):
    """
    Input: vision_output [B, N+2, C] = [CLS, P_1..P_N, BUDGET] from ViT with budget token appended.
    We use only the last position (budget token output, fused with all patches via self-attention)
    -> one linear Linear(C, N) -> per-patch logits [B, N]. Same as AdaLLaVA: one fused vector -> mlp_head -> per-item logits.
    Output: [B, N, C] (only patch tokens, unselected masked to 0); projector input size unchanged.
    """

    def __init__(
        self,
        vision_dim: int,
        num_patches: int,
        tau: float = 5.0,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.num_patches = num_patches
        self.tau = tau
        self.logit_head = nn.Linear(vision_dim, num_patches)

    def forward(
        self,
        vision_output: torch.Tensor,
        token_budget: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_output: [B, N+2, C] = [CLS, P_1..P_N, BUDGET]; last position is budget token output.
            token_budget: [B] float in [0,1] (ratio) or int K; used to get K = round(r*N) per batch.

        Returns:
            selected_features: [B, N, C] with unselected patches set to 0 (only patch tokens).
            keep_mask: [B, N] binary mask over patches (1 = keep).
        """
        B, seq_len, C = vision_output.shape
        assert seq_len == self.num_patches + 2, "Expected [B, N+2, C] from vision encoder with budget token."
        patch_tokens = vision_output[:, 1:-1, :]   # [B, N, C]
        budget_repr = vision_output[:, -1, :]      # [B, C]

        N = patch_tokens.size(1)

        # Map token_budget to integer K per batch
        if token_budget.dim() == 0:
            token_budget = token_budget.unsqueeze(0).expand(B)
        if token_budget.dtype in (torch.float32, torch.float16, torch.bfloat16):
            K_per_batch = (token_budget * N).long().clamp(1, N)
        else:
            K_per_batch = token_budget.long().clamp(1, N)

        # Only fused budget token -> one linear -> per-patch logits (AdaLLaVA-style)
        logits = self.logit_head(budget_repr)  # [B, N]

        # Select exactly K patches per batch via Gumbel
        keep_mask = torch.zeros(B, N, device=vision_output.device, dtype=vision_output.dtype)
        for b in range(B):
            k = min(int(K_per_batch[b].item()), N)
            mask_b = n_times_gumbel_softmax(
                logits[b : b + 1], n=k, tau=self.tau, hard=True, dim=-1, training=self.training
            )
            keep_mask[b] = mask_b.squeeze(0)

        masked_patches = patch_tokens * keep_mask.unsqueeze(-1)
        return masked_patches, keep_mask
