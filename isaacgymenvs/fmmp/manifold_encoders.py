from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class PartManifoldEncoder(nn.Module):
    """Lightweight DeepPhase/FLD-style part encoder."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phase_head = nn.Linear(hidden_dim, 1)
        self.amp_head = nn.Linear(hidden_dim, latent_dim)
        self.bias_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [N, T, F] -> flatten temporal context
        h = self.backbone(x.flatten(start_dim=1))
        theta = self.phase_head(h)
        phase = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        return {
            "phase": phase,
            "amplitude": self.amp_head(h),
            "bias": self.bias_head(h),
            "z": torch.cat([phase, self.amp_head(h), self.bias_head(h)], dim=-1),
        }


@dataclass
class FMMPEncoderBank:
    encoders: dict[str, PartManifoldEncoder]

    def to(self, device: torch.device | str) -> "FMMPEncoderBank":
        for encoder in self.encoders.values():
            encoder.to(device)
        return self

    def encode(self, part_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_dict = {}
        for name, feature in part_features.items():
            z_dict[name] = self.encoders[name](feature)["z"]
        return z_dict

    def parameters(self):
        for encoder in self.encoders.values():
            yield from encoder.parameters()

    def state_dict(self) -> dict[str, dict]:
        return {name: encoder.state_dict() for name, encoder in self.encoders.items()}

    def load_state_dict(self, state: dict[str, dict]) -> None:
        for name, encoder_state in state.items():
            self.encoders[name].load_state_dict(encoder_state)


def build_encoder_bank(part_dims: dict[str, int], latent_dim: int = 16, hidden_dim: int = 128) -> FMMPEncoderBank:
    encoders = {
        name: PartManifoldEncoder(input_dim=dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        for name, dim in part_dims.items()
    }
    return FMMPEncoderBank(encoders)


def phase_smoothness_loss(phase_seq: torch.Tensor) -> torch.Tensor:
    return torch.mean((phase_seq[:, 1:] - phase_seq[:, :-1]) ** 2)
