#!/usr/bin/env python
"""FF Master Workstation — single entry point.

  python run.py                 # quantitative benchmark -> results/benchmark.{json,csv}
  python run.py --quick         # fast benchmark (1 seed)
  python run.py --demo          # render the telemetry demo video -> results/demo.mp4
  python run.py --collect       # collect a state/action dataset -> results/dataset/
  python run.py --episodes N    #   (with --collect) number of episodes
  python run.py --teleop        # keyboard teleoperation (needs a display)

Pure CPU, no GPU. Deterministic.
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser(description="FF Master industrial-workstation operation (MuJoCo)")
    ap.add_argument("--demo", action="store_true", help="render the telemetry demo video")
    ap.add_argument("--benchmark", action="store_true", help="run the quantitative benchmark (default)")
    ap.add_argument("--quick", action="store_true", help="quick benchmark (1 seed)")
    ap.add_argument("--collect", action="store_true", help="collect a state/action dataset")
    ap.add_argument("--episodes", type=int, default=3, help="episodes for --collect")
    ap.add_argument("--teleop", action="store_true", help="keyboard teleoperation (needs a display)")
    args = ap.parse_args()

    if args.demo:
        from ffstation.record_demo import record
        print("[START] Rendering demo video...")
        print(f"[OK] {record()}")
        return
    if args.collect:
        from ffstation.dataio import collect
        print(f"[START] Collecting dataset ({args.episodes} episodes)...")
        collect(episodes=args.episodes)
        return
    if args.teleop:
        from ffstation.teleop import main as teleop_main
        teleop_main()
        return

    # default: benchmark
    from ffstation.benchmark import run_all
    import json
    seeds = (0,) if args.quick else (0, 1, 2)
    print(f"[START] FF Master workstation benchmark - seeds={seeds}")
    t0 = time.time()
    summary, rows = run_all(seeds=seeds)
    dt = time.time() - t0
    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\ntime: {dt:.1f}s, {len(rows)} trials")
    print("Full results: results/benchmark.json + results/benchmark.csv")


if __name__ == "__main__":
    main()
