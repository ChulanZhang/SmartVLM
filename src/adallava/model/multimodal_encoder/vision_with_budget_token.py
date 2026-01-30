# Exp1: Run vision encoder with budget token appended to patch sequence.
# [CLS, P_1, ..., P_N, BUDGET] -> ViT -> [B, N+2, C_v]. Budget fuses with patches via self-attention.
# C_v = vision encoder hidden dimension (e.g. 1024 for ViT-L).

from typing import Optional

import torch
import torch.nn as nn


def _get_inner_vision_model(vision_tower: nn.Module) -> nn.Module:
    """
    Return the inner ViT module that has .embeddings(pixel_values) and .encoder(...).
    - LLaVA CLIPVisionTower has .vision_tower = HuggingFace CLIPVisionModel.
    - HuggingFace CLIPVisionModel has .vision_model = CLIPVisionTransformer (has .embeddings and .encoder).
    """
    # LLaVA wrapper: vision_tower.vision_tower -> CLIPVisionModel
    inner = getattr(vision_tower, "vision_tower", vision_tower)
    # HuggingFace CLIPVisionModel: .vision_model -> CLIPVisionTransformer (has .embeddings and .encoder)
    inner = getattr(inner, "vision_model", inner)
    return inner


class VisionEncoderWithBudgetToken(nn.Module):
    """
    Wrapper that appends a budget token to the vision encoder input, runs the ViT,
    and returns [B, N+2, C_v] (CLS + N patches + budget token output at last position).
    C_v is the vision encoder hidden size (e.g. mm_hidden_size 1024).

    Requires the vision_tower's inner model to expose .embeddings(pixel_values) and
    .encoder(inputs_embeds=...). Compatible with LLaVA's CLIPVisionTower (path: .vision_tower.vision_model).
    """

    def __init__(
        self,
        budget_embedding_module: nn.Module,
        vision_hidden_size: int,
    ):
        super().__init__()
        self.budget_embedding_module = budget_embedding_module
        self.vision_hidden_size = vision_hidden_size
        # Learned position embedding for the budget token (last position).
        self.budget_pos_embed = nn.Parameter(torch.zeros(1, 1, vision_hidden_size))
        nn.init.normal_(self.budget_pos_embed, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        budget: torch.Tensor,
        vision_tower: nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W] from image processor.
            budget: [B] or [B, 1], values in [0, 1] (token budget ratio).
            vision_tower: LLaVA vision tower (inner path: .vision_tower.vision_model with .embeddings and .encoder).

        Returns:
            [B, N+2, C_v]: encoder output; last position is the budget token's output.
        """
        inner = _get_inner_vision_model(vision_tower)
        if not hasattr(inner, "embeddings") or not hasattr(inner, "encoder"):
            raise AttributeError(
                "Vision tower inner model must have .embeddings and .encoder. "
                "Got %s" % type(inner).__name__
            )

        # 1) Patch + CLS embedding -> [B, N+1, C_v]
        embedding_output = inner.embeddings(pixel_values)

        # 2) Budget embedding and append
        if budget.dim() == 2:
            budget = budget.squeeze(-1)
        budget_emb = self.budget_embedding_module(budget)  # [B, C_v]
        budget_emb = budget_emb.unsqueeze(1)  # [B, 1, C_v]
        budget_emb = budget_emb + self.budget_pos_embed  # add position for budget token
        # 3) [CLS, P_1, ..., P_N, BUDGET] -> [B, N+2, C_v]
        extended = torch.cat([embedding_output, budget_emb], dim=1)
        # Apply same pre_layernorm as in CLIPVisionTransformer before encoder
        if hasattr(inner, "pre_layrnorm"):
            extended = inner.pre_layrnorm(extended)

        # 4) Run encoder on extended sequence (same transformer stack)
        try:
            encoder_outputs = inner.encoder(inputs_embeds=extended, return_dict=True)
        except TypeError:
            encoder_outputs = inner.encoder(extended, return_dict=True)
        if hasattr(encoder_outputs, "last_hidden_state"):
            hidden_states = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, tuple):
            hidden_states = encoder_outputs[0]
        else:
            hidden_states = encoder_outputs
        return hidden_states  # [B, N+2, C_v]
