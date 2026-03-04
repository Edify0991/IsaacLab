#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def infer_skill_id(skill_name: str, skill_to_id: dict[str, int]) -> int:
    if skill_name not in skill_to_id:
        skill_to_id[skill_name] = len(skill_to_id)
    return skill_to_id[skill_name]


def main():
    parser = argparse.ArgumentParser(description="Build a no-transition motion list for FMMP.")
    parser.add_argument("--motions", nargs="+", required=True, help="Input npz files.")
    parser.add_argument("--skills", nargs="+", required=True, help="Skill label per motion file.")
    parser.add_argument("--trim_start", type=float, default=0.2)
    parser.add_argument("--trim_end", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.motions) != len(args.skills):
        raise ValueError("--motions and --skills must have the same length")

    skill_to_id: dict[str, int] = {}
    clips = []
    for motion_path, skill in zip(args.motions, args.skills, strict=True):
        data = np.load(motion_path)
        num_frames = int(data["dof_positions"].shape[0])
        start = int(num_frames * args.trim_start)
        end = int(num_frames * args.trim_end)
        clips.append(
            {
                "file": str(motion_path),
                "skill": skill,
                "skill_id": infer_skill_id(skill, skill_to_id),
                "start_frame": start,
                "end_frame": end,
                "num_frames": num_frames,
            }
        )

    payload = {"clips": clips, "skill_to_id": skill_to_id, "protocol": "no_inter_skill_transitions"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[FMMP] wrote {args.output} with {len(clips)} clips and {len(skill_to_id)} skills")


if __name__ == "__main__":
    main()
