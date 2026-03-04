from __future__ import annotations

import torch
from torch import nn


class TransitionGenerator(nn.Module):
    """Inpainting/in-betweening transition generator in latent space."""

    def __init__(self, latent_dim: int, skill_count: int, horizon: int = 20, hidden_dim: int = 256):
        super().__init__()
        self.horizon = horizon
        self.skill_embedding = nn.Embedding(skill_count, hidden_dim)
        self.net = nn.GRU(input_size=latent_dim + hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_hist: torch.Tensor, target_skill: torch.Tensor) -> torch.Tensor:
        n, h, d = z_hist.shape
        skill = self.skill_embedding(target_skill).unsqueeze(1).expand(n, self.horizon, -1)
        z_anchor = z_hist[:, -1:].expand(n, self.horizon, d)
        gru_in = torch.cat([z_anchor, skill], dim=-1)
        out, _ = self.net(gru_in)
        return self.head(out)


def pseudo_transition_loss(
    pred_transition: torch.Tensor,
    target_start: torch.Tensor,
    prior_scores: torch.Tensor,
    smooth_weight: float = 1.0,
    align_weight: float = 2.0,
    prior_weight: float = 0.2,
) -> torch.Tensor:
    """Loss for no-transition supervision.

    - aligns final generated latent with target skill start segment,
    - regularizes smooth latent increments,
    - encourages high discriminator prior score.
    """
    align = torch.mean((pred_transition[:, -1] - target_start) ** 2)
    smooth = torch.mean((pred_transition[:, 1:] - pred_transition[:, :-1]) ** 2)
    prior_term = -prior_scores.mean()
    return align_weight * align + smooth_weight * smooth + prior_weight * prior_term
