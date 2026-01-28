# Budget embedding for vision token controller (Exp1: adaptive vision token prune).
# Encodes scalar token budget (ratio in [0,1]) to a vector in vision space, following AdaLLaVA latency encoding.

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Small MLP for projecting sinusoidal budget encoding to vision dim."""

    def __init__(self, dim_in: int, hidden_dim: int, dim_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BudgetEmbedding(nn.Module):
    """
    Encode scalar token budget (ratio in [0, 1]) to a vector of size dim_out.
    Uses sinusoidal encoding + MLP, same style as AdaLLaVA's scheduler latency_encoding.

    dim_out must match the vision encoder output dim (e.g. mm_hidden_size 1024 for ViT-L),
    not the LLM hidden size, since the embedding is concatenated with vision patch features
    before the selection head.
    """

    def __init__(
        self,
        dim_out: int,
        num_freqs: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.dim_sincos = num_freqs * 2  # sin + cos
        self.scheduler_up_proj = FeedForward(self.dim_sincos, hidden_dim, dim_out)

    def forward(self, budget: torch.Tensor) -> torch.Tensor:
        """
        Args:
            budget: [B] or [B, 1], values in [0, 1] (fraction of tokens to keep).

        Returns:
            [B, dim_out]
        """
        if budget.dim() == 2:
            budget = budget.squeeze(-1)
        device = budget.device
        dtype = budget.dtype

        # Scale to [0, 2*pi]
        scaled = budget * 2 * torch.pi

        # Sinusoidal encoding
        frequencies = (
            1.0
            / (10000 ** (torch.arange(self.num_freqs, device=device, dtype=dtype) / self.num_freqs))
        )
        sin_vals = torch.sin(scaled.unsqueeze(1) * frequencies.unsqueeze(0))
        cos_vals = torch.cos(scaled.unsqueeze(1) * frequencies.unsqueeze(0))
        enc = torch.cat([sin_vals, cos_vals], dim=1)

        return self.scheduler_up_proj(enc)
