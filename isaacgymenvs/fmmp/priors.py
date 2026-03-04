from __future__ import annotations

import torch
from torch import nn


class PartDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GlobalCouplingDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_cat: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_cat, global_features], dim=-1))


class FMMPPrior(nn.Module):
    """AMP-compatible reward heads in FMMP latent space."""

    def __init__(self, part_latent_dims: dict[str, int], global_feature_dim: int):
        super().__init__()
        self.part_names = list(part_latent_dims.keys())
        self.part_discriminators = nn.ModuleDict(
            {name: PartDiscriminator(dim) for name, dim in part_latent_dims.items()}
        )
        z_dim_total = sum(part_latent_dims.values())
        self.global_discriminator = GlobalCouplingDiscriminator(z_dim_total + global_feature_dim)

    @staticmethod
    def amp_style_reward(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # AMP-style discriminator reward: -log(1 - D)
        probs = torch.sigmoid(logits)
        return -torch.log(torch.clamp(1.0 - probs, min=eps))

    def forward(
        self, part_latents: dict[str, torch.Tensor], global_features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        part_rewards = []
        part_logits_out: dict[str, torch.Tensor] = {}
        ordered = [part_latents[name] for name in self.part_names]
        for name in self.part_names:
            logits = self.part_discriminators[name](part_latents[name])
            part_logits_out[name] = logits
            part_rewards.append(self.amp_style_reward(logits))

        global_logits = self.global_discriminator(torch.cat(ordered, dim=-1), global_features)
        reward_global = self.amp_style_reward(global_logits)
        reward_part = torch.stack(part_rewards, dim=0).sum(dim=0)
        total_reward = reward_part + reward_global
        diagnostics = {"global_logits": global_logits, "reward_part": reward_part, "reward_global": reward_global}
        diagnostics.update(part_logits_out)
        return total_reward.squeeze(-1), diagnostics
