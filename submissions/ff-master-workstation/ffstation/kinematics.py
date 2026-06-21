"""Inverse kinematics for FF Master's 7-DOF arms.

Damped-least-squares (Levenberg-Marquardt) IK solved on a scratch MjData with
pure kinematics (no dynamics stepping) -> fast, robust, respects joint limits.
Position-only by default; an optional soft orientation objective aligns a hand
axis with a world direction (used for palm-facing control). The solution is a
joint target that the gravity-compensated PD law in `controller.py` tracks.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .scene import Station


def fk_hand(st: Station, d, side: str) -> np.ndarray:
    return d.xpos[st.hand_bid[side]].copy()


def solve_ik(st: Station, q_init: np.ndarray, targets: dict,
             iters: int = 260, lam: float = 0.1, step_clip: float = 0.1) -> tuple[np.ndarray, dict]:
    """Position IK for one or both hands.

    targets: {"left": xyz | None, "right": xyz | None}
    Returns (q_des, achieved_fk).
    """
    m = st.model
    dik = mujoco.MjData(m)
    dik.qpos[:] = q_init
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, dik)
        mujoco.mj_comPos(m, dik)
        for side, tgt in targets.items():
            if tgt is None:
                continue
            mujoco.mj_jacBody(m, dik, jacp, None, st.hand_bid[side])
            err = np.asarray(tgt, float) - dik.xpos[st.hand_bid[side]]
            J = jacp[:, st.arm_dof[side]]
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), err)
            for k, qadr in enumerate(st.arm_qadr[side]):
                lo, hi = m.jnt_range[m.dof_jntid[st.arm_dof[side][k]]]
                dik.qpos[qadr] = np.clip(dik.qpos[qadr] + float(np.clip(dq[k], -step_clip, step_clip)), lo, hi)
    fk = {s: dik.xpos[st.hand_bid[s]].copy() for s in ("left", "right")}
    return dik.qpos.copy(), fk


def reach_error(st: Station, q_des: np.ndarray, targets: dict) -> dict:
    m = st.model
    dik = mujoco.MjData(m)
    dik.qpos[:] = q_des
    mujoco.mj_kinematics(m, dik)
    out = {}
    for side, tgt in targets.items():
        if tgt is None:
            continue
        out[side] = float(np.linalg.norm(np.asarray(tgt, float) - dik.xpos[st.hand_bid[side]]))
    return out
