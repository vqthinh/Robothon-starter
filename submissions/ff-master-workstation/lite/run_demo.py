"""Train the learned policy, run one closed-loop episode, render the demo, and
write a trajectory record + summary.

Usage (from this directory or the repo root):
    python submissions/ff-master-workstation/lite/run_demo.py
Outputs (written next to this script, under ./outputs):
    outputs/demo.mp4
    outputs/trajectory.json
    outputs/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fflite.scene import actuator_names, load_model
from fflite.policy import train_policy
from fflite.rollout import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="FF Master Workstation lite demo")
    parser.add_argument("--no-video", action="store_true", help="skip MP4 rendering")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    model = load_model()
    actuators = actuator_names(model)
    print(f"[1/4] Training learned imitation policy ({model.nu} actuators, {model.nq} DOF)...")
    policy = train_policy(actuators, model)

    print("[2/4] Running closed-loop episode...")
    result = run_episode(policy, record=True, record_hz=args.fps)
    metrics, ok, traj = result["metrics"], result["success"], result["trajectory"]

    print("       measured metrics:")
    for key, val in metrics.items():
        print(f"         {key:14s} {val:.4f}")
    print("       task success:", ok, "->", "PASS" if all(ok.values()) else "FAIL")

    traj_path = os.path.join(out_dir, "trajectory.json")
    with open(traj_path, "w") as fh:
        json.dump({"metrics": metrics, "success": ok, "trajectory": traj}, fh, indent=2)
    print(f"[3/4] Wrote {traj_path}")

    summary = {"controller": "learned imitation + sensor feedback",
               "metrics": metrics, "success": ok, "all_pass": all(ok.values())}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if args.no_video:
        print("[4/4] Skipped video (--no-video).")
        return
    try:
        from fflite.viz import render_animation
        video_path = os.path.join(out_dir, "demo.mp4")
        render_animation(traj, video_path, fps=args.fps)
        print(f"[4/4] Wrote {video_path}")
    except Exception as exc:  # pragma: no cover - rendering is environment dependent
        print(f"[4/4] Video rendering skipped ({exc}). Trajectory + summary still written.")


if __name__ == "__main__":
    main()
