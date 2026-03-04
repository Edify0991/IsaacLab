#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def random_sequence(skills: list[str], duration_s: float, interval_range: tuple[float, float]) -> list[dict]:
    t = 0.0
    seq = []
    while t < duration_s:
        dt = random.uniform(*interval_range)
        seq.append({"start_s": round(t, 3), "end_s": round(min(t + dt, duration_s), 3), "skill": random.choice(skills)})
        t += dt
    return seq


def main():
    parser = argparse.ArgumentParser(description="Generate random FMMP skill switching commands.")
    parser.add_argument("--skills", nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--min_interval", type=float, default=1.0)
    parser.add_argument("--max_interval", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed)

    sequence = random_sequence(args.skills, args.duration, (args.min_interval, args.max_interval))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mode": "random", "sequence": sequence}, indent=2), encoding="utf-8")
    print(f"[FMMP] wrote {len(sequence)} commands to {args.output}")
    print("[FMMP] Real-time manual switching can be mapped to keyboard/gamepad in eval_switching.py TODO hook.")


if __name__ == "__main__":
    main()
