#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from isaacgymenvs.fmmp.manifold_encoders import build_encoder_bank, phase_smoothness_loss


def main():
    parser = argparse.ArgumentParser(description="Train FMMP part manifold encoders.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to tensor dict checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=16)
    args = parser.parse_args()

    bundle = torch.load(args.dataset, map_location="cpu")
    part_tensors: dict[str, torch.Tensor] = bundle["part_windows"]  # [N, T, F]
    part_dims = {k: int(v.shape[1] * v.shape[2]) for k, v in part_tensors.items()}
    bank = build_encoder_bank(part_dims=part_dims, latent_dim=args.latent_dim)
    optim = torch.optim.Adam(bank.parameters(), lr=3e-4)
    recon_heads = nn.ModuleDict({k: nn.Linear(args.latent_dim * 2 + 2, v.shape[2]) for k, v in part_tensors.items()})

    for epoch in range(args.epochs):
        losses = []
        for part_name, windows in part_tensors.items():
            ds = TensorDataset(windows[:, :-1], windows[:, -1])
            loader = DataLoader(ds, batch_size=256, shuffle=True)
            for hist, target in loader:
                out = bank.encoders[part_name](hist)
                pred = recon_heads[part_name](out["z"])
                recon = torch.mean((pred - target) ** 2)
                smooth = phase_smoothness_loss(out["phase"].unsqueeze(1))
                loss = recon + 0.05 * smooth
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(float(loss.item()))
        print(f"[FMMP] epoch={epoch} loss={sum(losses) / max(len(losses), 1):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoders": bank.state_dict(), "part_dims": part_dims}, args.output)
    meta = args.output.with_suffix(".json")
    meta.write_text(json.dumps({"latent_dim": args.latent_dim, "part_dims": part_dims}, indent=2), encoding="utf-8")
    print(f"[FMMP] saved encoders to {args.output}")


if __name__ == "__main__":
    main()
