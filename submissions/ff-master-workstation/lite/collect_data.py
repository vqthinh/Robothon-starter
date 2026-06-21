"""Collect a learned-policy dataset: many episodes with light domain
randomization, saved as per-episode .npz files plus a manifest.

Usage:
    python submissions/ff-master-workstation/lite/collect_data.py --episodes 20
Outputs (under ./dataset):
    dataset/ep###.npz        (per-episode controls/sensors/tips time series)
    dataset/manifest.json    (index + per-episode metrics and success)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fflite.scene import actuator_names, load_model
from fflite.policy import train_policy
from fflite.rollout import run_episode, TIP_GEOMS


def episode_to_arrays(traj: list[dict]) -> dict[str, np.ndarray]:
    t = np.asarray([f["t"] for f in traj])
    sig = {k: np.asarray([f[k] for f in traj]) for k in
           ("button_depth", "slider_pos", "keyg_depth", "keyb_depth", "wheel_angle")}
    tips = np.stack([np.asarray([f["tips"][g] for f in traj]) for g in TIP_GEOMS], axis=1)
    return {"t": t, "tips": tips, **sig}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect learned-policy dataset")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=0.004,
                        help="std of command-space domain randomization")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out_dir = os.path.join(HERE, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    model = load_model()
    policy = train_policy(actuator_names(model), model)

    manifest = {"controller": "learned imitation + sensor feedback",
                "episodes": [], "jitter": args.jitter}
    n_success = 0
    for ep in range(args.episodes):
        result = run_episode(policy, jitter=args.jitter, seed=ep, record=True, record_hz=args.fps)
        arrays = episode_to_arrays(result["trajectory"])
        fname = f"ep{ep:03d}.npz"
        np.savez_compressed(os.path.join(out_dir, fname), **arrays)
        passed = all(result["success"].values())
        n_success += int(passed)
        manifest["episodes"].append({
            "file": fname, "seed": ep, "frames": len(result["trajectory"]),
            "metrics": {k: round(v, 5) for k, v in result["metrics"].items()},
            "success": result["success"], "all_pass": passed,
        })
        print(f"  ep{ep:03d}: {'PASS' if passed else 'FAIL'}  -> {fname}")

    manifest["num_episodes"] = args.episodes
    manifest["num_success"] = n_success
    manifest["success_rate"] = round(n_success / max(args.episodes, 1), 4)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {args.episodes} episodes to {out_dir}")
    print(f"Dataset success rate: {n_success}/{args.episodes} = {manifest['success_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
