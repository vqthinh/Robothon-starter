"""Learned imitation policy + sensor-feedback corrections.

A ridge-regression imitation policy is trained at launch from the scripted
expert. Its features are a time basis (polynomial + Fourier + radial basis
functions). At rollout time, a thin closed-loop layer reads live contact/joint
sensors and nudges the active fingertip when its target signal lags -- the kind
of correction learned open-loop policies need for contact-rich tasks.
"""
from __future__ import annotations

import math

import numpy as np

from .expert import expert_controls
from .scene import EPISODE_DURATION_S


N_RBF = 36
_CENTERS = np.linspace(0.0, EPISODE_DURATION_S, N_RBF)
_SIGMA = EPISODE_DURATION_S / (N_RBF - 2)


def features(t: float) -> np.ndarray:
    """Time-basis feature vector: polynomial + Fourier + RBF."""
    x = min(max(t / EPISODE_DURATION_S, 0.0), 1.0)
    feats = [1.0, x, x * x, x * x * x]
    for k in (1, 2, 3, 4, 5):
        feats.append(math.sin(2.0 * math.pi * k * x))
        feats.append(math.cos(2.0 * math.pi * k * x))
    feats.extend(np.exp(-0.5 * ((t - _CENTERS) / _SIGMA) ** 2).tolist())
    return np.asarray(feats, dtype=float)


# Sensor-feedback windows: (start, end, actuator, sensor, depth_threshold, gain).
_FEEDBACK = [
    (0.3, 2.0, "thumb_z", "button_depth", 0.018, 3.0),
    (4.6, 6.0, "middle_z", "keyg_depth", 0.015, 3.0),
    (4.6, 6.0, "ring_z", "keyb_depth", 0.015, 3.0),
]


class LearnedPolicy:
    """Ridge imitation policy with optional sensor-feedback corrections."""

    def __init__(self, weights: np.ndarray, actuators: list[str], model, feedback: bool = True):
        self.weights = weights
        self.actuators = actuators
        self.model = model
        self.feedback = feedback
        self._ranges = {a: tuple(model.actuator(a).ctrlrange) for a in actuators}

    def target(self, t: float, sensors: dict[str, float] | None = None) -> dict[str, float]:
        raw = features(t) @ self.weights
        controls = {a: float(raw[i]) for i, a in enumerate(self.actuators)}

        if self.feedback and sensors:
            for start, end, act, sen, thr, gain in _FEEDBACK:
                if start <= t <= end:
                    magnitude = -sensors.get(sen, 0.0)  # depths are negative joint values
                    if magnitude < thr:
                        controls[act] -= (thr - magnitude) * gain

        for act, value in controls.items():
            lo, hi = self._ranges[act]
            controls[act] = float(np.clip(value, lo, hi))
        return controls


def train_policy(actuators: list[str], model, hz: int = 100, ridge: float = 1e-4,
                 feedback: bool = True) -> LearnedPolicy:
    """Fit the imitation policy to the scripted expert via ridge regression."""
    times = np.arange(0.0, EPISODE_DURATION_S, 1.0 / hz)
    feat_matrix = np.vstack([features(t) for t in times])
    targets = np.vstack([[expert_controls(t).get(a, 0.0) for a in actuators] for t in times])
    reg = ridge * np.eye(feat_matrix.shape[1])
    weights = np.linalg.solve(feat_matrix.T @ feat_matrix + reg, feat_matrix.T @ targets)
    return LearnedPolicy(weights, actuators, model, feedback=feedback)
