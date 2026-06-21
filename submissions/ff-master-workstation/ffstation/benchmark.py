"""Quantitative benchmark — every number measured from the MuJoCo simulation.

For each control we sweep a range of command setpoints across several panel-jitter
seeds (the panel is randomly displaced a few mm, and the skills read the live
control positions, so this also measures robustness to placement error). We log
the achieved-vs-commanded result of every trial and aggregate per-task success
rate, setpoint-tracking error and the key physical metric.

Run:  python -m ffstation.benchmark      (or  python run.py --benchmark)
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import scene, tasks
from .controller import WholeBodyController

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# command setpoints per control (within the validated operating range)
VALVE_TARGETS = (15.0, 20.0, 25.0)
BUTTON_FORCES = (3.0, 3.5, 4.0)
LEVER_TARGETS = (40.0, 45.0, 55.0)
SLIDER_TARGETS = (12.0, 16.0, 20.0)


def _jitter(seed: int):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-0.003, 0.003, size=3))


def _trials_for_seed(seed: int) -> list[dict]:
    st = scene.build(jitter=_jitter(seed), seed=seed)
    c = WholeBodyController(st, record=False)
    rows = []

    def run(fn):
        c.reset(); c.step(60)
        r = fn(); r["seed"] = seed
        rows.append(r)

    for t in VALVE_TARGETS:
        run(lambda t=t: tasks.turn_valve(c, target_deg=t))
    for f in BUTTON_FORCES:
        run(lambda f=f: tasks.press_button(c, actuate_force=f))
    for t in LEVER_TARGETS:
        run(lambda t=t: tasks.flip_lever(c, target_deg=t))
    for t in SLIDER_TARGETS:
        run(lambda t=t: tasks.set_slider(c, target_mm=t))
    return rows


def summarize(rows: list[dict]) -> dict:
    def sub(task):
        return [r for r in rows if r["task"] == task]

    def rate(rs):
        return round(sum(r["success"] for r in rs) / max(1, len(rs)), 3)

    valve, button = sub("valve"), sub("button")
    lever, slider = sub("lever"), sub("slider")
    return {
        "n_trials": len(rows),
        "overall_success_rate": rate(rows),
        "valve": {
            "n": len(valve), "success_rate": rate(valve),
            "mean_opened_deg": round(float(np.mean([abs(r["turned_deg"]) for r in valve])), 1),
            "mean_held_deg": round(float(np.mean([abs(r["held_deg"]) for r in valve])), 1),
            "mean_grip_force_N": round(float(np.mean([r["grip_force_N"] for r in valve])), 1),
        },
        "button": {
            "n": len(button), "success_rate": rate(button),
            "mean_peak_force_N": round(float(np.mean([r["peak_force_N"] for r in button])), 2),
            "mean_hold_force_N": round(float(np.mean([r["hold_force_N"] for r in button])), 2),
            "max_peak_force_N": round(float(np.max([r["peak_force_N"] for r in button])), 2),
            "mean_return_mm": round(float(np.mean([r["released_mm"] for r in button])), 2),
        },
        "lever": {
            "n": len(lever), "success_rate": rate(lever),
            "mean_flipped_deg": round(float(np.mean([r["flipped_deg"] for r in lever])), 1),
        },
        "slider": {
            "n": len(slider), "success_rate": rate(slider),
            "mean_track_err_mm": round(float(np.mean([r["error_mm"] for r in slider])), 1),
        },
    }


def run_all(seeds=(0, 1, 2), verbose: bool = True):
    rows = []
    for s in seeds:
        if verbose:
            print(f"  [seed {s}] running {len(VALVE_TARGETS)+len(BUTTON_FORCES)+len(LEVER_TARGETS)+len(SLIDER_TARGETS)} trials...")
        rows.extend(_trials_for_seed(s))
    summary = summarize(rows)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "benchmark.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "trials": rows}, fp, indent=2, ensure_ascii=False)
    _write_csv(rows)
    return summary, rows


def _write_csv(rows):
    keys = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(RESULTS, "benchmark.csv"), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    summary, rows = run_all()
    print(json.dumps(summary, indent=2))
