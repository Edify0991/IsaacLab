#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from isaacgymenvs.fmmp.metrics import compute_switching_metrics


def _mock_rollout(duration_s: float, dt: float = 1 / 60):
    t = np.arange(0, duration_s, dt)
    joint_pos = 0.5 * np.sin(t[:, None] * np.linspace(1.0, 3.0, 28)[None, :])
    com_pos = np.stack([0.1 * t, 0.0 * t, 1.0 + 0.02 * np.sin(3 * t)], axis=-1)
    foot_pos = np.zeros((t.shape[0], 2, 3))
    foot_pos[:, :, 0] = np.sin(t)[:, None] * 0.05
    foot_contact = (np.sin(t[:, None] * 4) > 0).astype(np.float32)
    foot_heights = 0.02 + 0.01 * np.sin(5 * t[:, None])
    torques = np.sin(t[:, None] * 2) * 50.0
    falls = np.array([False])
    survival = np.array([duration_s])
    return joint_pos, com_pos, foot_pos, foot_contact, foot_heights, torques, falls, survival


def main():
    parser = argparse.ArgumentParser(description="Evaluate switching stability for AMP/PMP/FMMP checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--record_video", action="store_true", default=False)
    args = parser.parse_args()

    _ = json.loads(args.commands.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = compute_switching_metrics(*_mock_rollout(args.duration), dt=1 / 60)

    csv_path = args.output / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.__dict__.items():
            writer.writerow([key, value])

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].bar(list(metrics.__dict__.keys()), list(metrics.__dict__.values()))
    ax[0].tick_params(axis="x", rotation=60)
    ax[0].set_title("Switching Evaluation Metrics")
    ax[1].plot(np.random.randn(300).cumsum())
    ax[1].set_title("Placeholder stability trace")
    fig.tight_layout()
    plot_path = args.output / "metrics.png"
    fig.savefig(plot_path, dpi=150)

    if args.record_video:
        print("[FMMP] TODO: hook Isaac Lab viewer recording into evaluation script.")

    print(f"[FMMP] checkpoint={args.checkpoint}")
    print(f"[FMMP] wrote metrics to {csv_path} and plot to {plot_path}")


if __name__ == "__main__":
    main()
