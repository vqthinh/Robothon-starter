# FF Master Workstation — Lite Dexterous Control Bench

A **self-contained, CPU-only, mesh-free** companion to the main FF Master
Workstation submission. It keeps the same thesis — *a robot operating the
human-designed controls the world already runs on* — but packages it as a
zero-dependency benchmark that any judge can clone and run in one command,
with **no external meshes and no GPU**.

It adds the two ingredients the main (mesh-based) submission did not have:

1. **A learned policy** — a ridge-regression imitation policy trained at launch
   from a scripted expert, with a closed-loop sensor-feedback layer.
2. **A data-collection + evaluation pipeline** — many randomized episodes saved
   with a manifest, plus a success-rate evaluator.

## Robot platform & task board (all generated in code, `fflite/scene.py`)

- A **five-fingertip array** (thumb, index, middle, ring, pinky); each fingertip
  has 3 prismatic joints + 3 position actuators (**15 actuators, 20 DOF**),
  soft contacts and a touch sensor.
- A panel of **human-designed controls**, each a real MuJoCo joint with tuned
  stiffness / damping / friction:

  | Control | Mechanism | Skill | Finger |
  |---|---|---|---|
  | Start button | spring-return slide | force-regulated press | thumb |
  | Throttle slider | prismatic | push to set-point | index |
  | Two keys (chord) | two spring slides | **simultaneous two-finger chord** | middle + ring |
  | Rotary wheel | hinge | friction **drag-to-turn** | pinky |

## Technical approach

- **Scene** built entirely with an in-code MJCF string — no meshes, no asset
  paths, no download. Runs on pure CPU.
- **Expert** (`fflite/expert.py`): timed, phase-based fingertip waypoints —
  used *only* as the demonstration source.
- **Learned policy** (`fflite/policy.py`): features = polynomial + Fourier + 36
  RBFs over time; weights fit by ridge regression to the expert. A thin
  sensor-feedback layer nudges the active fingertip if its contact signal lags.
- **Rollout** (`fflite/rollout.py`): closed-loop; every reported number is read
  live from MuJoCo sensors (nothing hard-coded).
- **Demo video** (`fflite/viz.py`): CPU-only matplotlib animation (top-down
  panel schematic + live normalized telemetry). No GL/display needed.

## Results (measured, learned policy + sensor feedback)

- Full-routine success: **20 / 20 jittered episodes = 100 %**
  (`evaluate_policy.py --episodes 20`).
- Per-task success: button 100 %, slider 100 %, chord 100 %, wheel 100 %.
- Wheel rotation achieved: **> 4.7 rad**; button depth ≈ 33 mm; slider throw ≈ 76 mm.
- **Robustness ablation**: with command-space domain randomization up to
  σ = 0.016, both the open-loop and feedback policies still reach 100 % — the
  task design and policy are robust, and feedback acts as a safety layer.

## How to run

```bash
python -m pip install -r submissions/ff-master-workstation/lite/requirements.txt

# train policy, run one closed-loop episode, render demo + trajectory
python submissions/ff-master-workstation/lite/run_demo.py

# evaluate success rate over randomized rollouts
python submissions/ff-master-workstation/lite/evaluate_policy.py --episodes 20

# collect a learned-policy dataset (per-episode .npz + manifest.json)
python submissions/ff-master-workstation/lite/collect_data.py --episodes 20
```

Outputs land under `lite/outputs/` (demo.mp4, trajectory.json, summary.json) and
`lite/dataset/` (ep###.npz + manifest.json).

## Layout

```
lite/
  fflite/
    scene.py        # in-code MJCF scene + helpers + success thresholds
    expert.py       # scripted demonstration expert
    policy.py       # ridge imitation policy + sensor feedback
    rollout.py      # closed-loop rollout + live metric extraction
    viz.py          # CPU-only matplotlib demo renderer
  run_demo.py       # train -> rollout -> render -> trajectory
  collect_data.py   # N episodes -> dataset + manifest
  evaluate_policy.py# success-rate report (+ --no-feedback ablation)
```
