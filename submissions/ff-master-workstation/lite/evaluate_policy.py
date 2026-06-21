"""Evaluate the learned policy over many randomized rollouts and report the
per-task and overall success rate.

Usage:
    python submissions/ff-master-workstation/lite/evaluate_policy.py --episodes 20
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
    parser = argparse.ArgumentParser(description="Evaluate learned policy success rate")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=0.004)
    parser.add_argument("--no-feedback", action="store_true",
                        help="disable sensor feedback (ablation)")
    args = parser.parse_args()

    model = load_model()
    policy = train_policy(actuator_names(model), model, feedback=not args.no_feedback)

    tasks = ["button", "slider", "chord", "wheel"]
    per_task = {t: 0 for t in tasks}
    n_all = 0
    for ep in range(args.episodes):
        result = run_episode(policy, jitter=args.jitter, seed=ep)
        ok = result["success"]
        for t in tasks:
            per_task[t] += int(ok[t])
        n_all += int(all(ok.values()))

    n = args.episodes
    report = {
        "controller": "learned imitation" + ("" if args.no_feedback else " + sensor feedback"),
        "episodes": n, "jitter": args.jitter,
        "per_task_success_rate": {t: round(per_task[t] / n, 4) for t in tasks},
        "overall_success_rate": round(n_all / n, 4),
    }
    print(json.dumps(report, indent=2))
    print(f"\nOverall: {n_all}/{n} = {n_all / n * 100:.0f}% full-routine success")


if __name__ == "__main__":
    main()
