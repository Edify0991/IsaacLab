from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PartSpec:
    name: str
    joint_names: tuple[str, ...]


DEFAULT_PART_SPECS: tuple[PartSpec, ...] = (
    PartSpec("left_leg", ("left_hip", "left_knee", "left_ankle")),
    PartSpec("right_leg", ("right_hip", "right_knee", "right_ankle")),
    PartSpec("torso", ("abdomen", "neck")),
    PartSpec("left_arm", ("left_shoulder", "left_elbow", "left_wrist")),
    PartSpec("right_arm", ("right_shoulder", "right_elbow", "right_wrist")),
)


def build_part_index_map(joint_names: list[str], specs: tuple[PartSpec, ...] = DEFAULT_PART_SPECS) -> dict[str, torch.Tensor]:
    """Maps each part to matching joint indices by fuzzy joint-name matching."""
    mapped: dict[str, torch.Tensor] = {}
    lowered = [name.lower() for name in joint_names]
    for spec in specs:
        ids = [i for i, name in enumerate(lowered) if any(key in name for key in spec.joint_names)]
        mapped[spec.name] = torch.tensor(ids, dtype=torch.long)
    return mapped


def extract_part_features(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    part_indices: dict[str, torch.Tensor],
    history: int = 12,
) -> dict[str, torch.Tensor]:
    """Extracts FMMP part features from temporal joint windows.

    Args:
        joint_pos: [N, T, D] or [N, D] tensor.
        joint_vel: [N, T, D] or [N, D] tensor.
        part_indices: Map returned by :func:`build_part_index_map`.
    """
    if joint_pos.ndim == 2:
        joint_pos = joint_pos.unsqueeze(1)
        joint_vel = joint_vel.unsqueeze(1)

    if joint_pos.shape[1] > history:
        joint_pos = joint_pos[:, -history:]
        joint_vel = joint_vel[:, -history:]

    features: dict[str, torch.Tensor] = {}
    for part_name, idx in part_indices.items():
        if idx.numel() == 0:
            features[part_name] = torch.zeros((joint_pos.shape[0], joint_pos.shape[1], 1), device=joint_pos.device)
            continue
        part_pos = joint_pos[:, :, idx]
        part_vel = joint_vel[:, :, idx]
        # lightweight feature stack: angle, velocity, finite difference velocity
        vel_delta = torch.diff(part_vel, dim=1, prepend=part_vel[:, :1])
        features[part_name] = torch.cat([part_pos, part_vel, vel_delta], dim=-1)
    return features


def extract_contact_features(feet_heights: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    """Binary contact heuristic from feet heights, shape [N, num_feet]."""
    return (feet_heights < threshold).float()
