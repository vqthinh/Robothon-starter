"""Trajectory data-collection pipeline (one of the contest's recommended directions).

Runs the autonomous workstation skills and records, at every control step, a
(state, action) tuple plus the live control/tactile readings — the raw material
for imitation / offline-RL training. Episodes are written as compact .npz arrays
with a JSON manifest describing the schema.

Run:  python -m ffstation.dataio       (or  python run.py --collect --episodes 3)
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import scene, tasks
from .controller import WholeBodyController

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DATA_DIR = os.path.join(RESULTS, "dataset")

TASK_FNS = {
    "valve": lambda c: tasks.turn_valve(c, 20.0),
    "button": lambda c: tasks.press_button(c),
    "lever": lambda c: tasks.flip_lever(c, 45.0),
    "slider": lambda c: tasks.set_slider(c, 18.0),
}


def _jitter(seed):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-0.003, 0.003, size=3))


def collect(episodes: int = 3, tasks_list=("valve", "button", "lever", "slider"),
            verbose: bool = True) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = {"episodes": [], "schema": {
        "qpos": "generalized positions (m, rad)", "qvel": "generalized velocities",
        "ctrl": "action = per-joint motor torque command (N·m)",
        "controls": "live valve_deg/button_mm/lever_deg/slider_mm + fingertip touch (N)"}}
    for ep in range(episodes):
        st = scene.build(jitter=_jitter(ep), seed=ep)
        c = WholeBodyController(st, record=False)
        c.log_data = True
        for tname in tasks_list:
            c.reset(); c.step(40)
            c.dataset = []
            TASK_FNS[tname](c)
            arr_qpos = np.array([d["qpos"] for d in c.dataset], dtype=np.float32)
            arr_qvel = np.array([d["qvel"] for d in c.dataset], dtype=np.float32)
            arr_ctrl = np.array([d["ctrl"] for d in c.dataset], dtype=np.float32)
            arr_t = np.array([d["t"] for d in c.dataset], dtype=np.float32)
            fn = f"ep{ep:02d}_{tname}.npz"
            np.savez_compressed(os.path.join(DATA_DIR, fn),
                                t=arr_t, qpos=arr_qpos, qvel=arr_qvel, ctrl=arr_ctrl)
            manifest["episodes"].append({"file": fn, "episode": ep, "task": tname,
                                         "steps": int(arr_qpos.shape[0]),
                                         "state_dim": int(arr_qpos.shape[1]),
                                         "action_dim": int(arr_ctrl.shape[1])})
            if verbose:
                print(f"  ep{ep} {tname:7s}: {arr_qpos.shape[0]} steps "
                      f"(state {arr_qpos.shape[1]}, action {arr_ctrl.shape[1]})")
    with open(os.path.join(DATA_DIR, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    total = sum(e["steps"] for e in manifest["episodes"])
    print(f"[OK] {len(manifest['episodes'])} trajectories, {total} transitions -> {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    collect()
