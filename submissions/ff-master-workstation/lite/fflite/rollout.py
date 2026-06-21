"""Run a closed-loop rollout and measure task outcomes.

Nothing here is hard-coded: every reported number is read from MuJoCo sensors
during the rollout.
"""
from __future__ import annotations

import numpy as np

from .scene import EPISODE_DURATION_S, SENSORS, SUCCESS, actuator_names, load_model, sensor_value

import mujoco

# Fingertip body names whose world position we record for visualization.
TIP_GEOMS = ["tip_thumb", "tip_index", "tip_middle", "tip_ring", "tip_pinky"]


def evaluate_success(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "button": metrics["button_depth"] >= SUCCESS["button"],
        "slider": metrics["slider_pos"] >= SUCCESS["slider"],
        "chord": metrics["chord_sim"] >= SUCCESS["chord"],
        "wheel": metrics["wheel_angle"] >= SUCCESS["wheel"],
    }


def run_episode(policy, *, jitter: float = 0.0, seed: int = 0,
                duration: float = EPISODE_DURATION_S, record: bool = False,
                record_hz: int = 30):
    """Roll out ``policy`` (object with ``.target(t, sensors)``) and return results.

    Returns a dict with measured ``metrics``, per-task ``success`` flags, and
    (when ``record``) a ``trajectory`` of sampled sensor/tip values for plotting.
    """
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(seed)
    actuators = actuator_names(model)

    metrics = {"button_depth": 0.0, "slider_pos": 0.0, "keyg_depth": 0.0,
               "keyb_depth": 0.0, "wheel_angle": 0.0, "chord_sim": 0.0}
    traj: list[dict] = []
    record_every = max(1, int((1.0 / record_hz) / model.opt.timestep))
    n_steps = int(duration / model.opt.timestep)

    for k in range(n_steps):
        t = k * model.opt.timestep
        sensors = {n: sensor_value(model, data, n) for n in SENSORS}
        controls = policy.target(t, sensors)
        for act in actuators:
            cmd = controls.get(act, 0.0)
            if jitter:
                cmd += rng.normal(0.0, jitter)
            data.ctrl[model.actuator(act).id] = cmd
        mujoco.mj_step(model, data)

        keyg = -sensor_value(model, data, "keyg_depth")
        keyb = -sensor_value(model, data, "keyb_depth")
        metrics["button_depth"] = max(metrics["button_depth"], -sensor_value(model, data, "button_depth"))
        metrics["slider_pos"] = max(metrics["slider_pos"], sensor_value(model, data, "slider_pos"))
        metrics["keyg_depth"] = max(metrics["keyg_depth"], keyg)
        metrics["keyb_depth"] = max(metrics["keyb_depth"], keyb)
        metrics["wheel_angle"] = max(metrics["wheel_angle"], abs(sensor_value(model, data, "wheel_angle")))
        metrics["chord_sim"] = max(metrics["chord_sim"], min(keyg, keyb))

        if record and k % record_every == 0:
            traj.append({
                "t": round(t, 4),
                "button_depth": round(-sensor_value(model, data, "button_depth"), 5),
                "slider_pos": round(sensor_value(model, data, "slider_pos"), 5),
                "keyg_depth": round(keyg, 5),
                "keyb_depth": round(keyb, 5),
                "wheel_angle": round(sensor_value(model, data, "wheel_angle"), 5),
                "tips": {g: [round(float(v), 4) for v in data.geom(g).xpos] for g in TIP_GEOMS},
            })

    result = {"metrics": metrics, "success": evaluate_success(metrics)}
    if record:
        result["trajectory"] = traj
    return result
