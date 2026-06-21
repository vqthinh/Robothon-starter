"""Whole-body torque controller for the fixed-base FF Master humanoid.

Control law (computed-torque / gravity-compensated PD), evaluated every sim step:

    tau = qfrc_bias(q, qd)            # exact gravity + Coriolis compensation
        + Kp * (q_des - q) - Kd * qd  # joint-space PD toward the target config

FF Master is driven by direct-torque motors (gear 1), so `ctrl = tau`. Because
the base is fixed, this law is unconditionally stable and tracks a reference
posture to ~0 error (verified). Cartesian goals are reached by solving IK
(`kinematics.solve_ik`) for the arm joints and feeding the result in as q_des;
contact tasks add an outer closed loop that adjusts the Cartesian target from a
measured signal (contact force, control-joint angle).
"""
from __future__ import annotations

import numpy as np
import mujoco

from .scene import Station
from . import kinematics as K

KP_DEFAULT = 150.0
KD_DEFAULT = 11.0


class WholeBodyController:
    def __init__(self, st: Station, kp: float = KP_DEFAULT, kd: float = KD_DEFAULT,
                 substeps: int = 4, renderer=None, cam=None, record: bool = True):
        self.st = st
        self.m = st.model
        self.d = mujoco.MjData(self.m)
        self.kp = kp
        self.kd = kd
        self.substeps = substeps
        self.q_des = st.q_nom.copy()
        self.renderer = renderer
        self.cam = cam
        self.frames: list = []
        self.frame_every = 3
        self.overlay_fn = None          # set by record_demo to draw telemetry
        self.demo_title = ""            # set by record_demo per segment
        self.demo_sub = ""
        self.demo_index = "00"
        self.demo_active = ""
        self._k = 0
        self.record = record
        self.telemetry: list[dict] = []
        self.log_data = False           # set True to capture a state/action dataset
        self.dataset: list[dict] = []
        self.reset()

    # ------------------------------------------------------------------ infra
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.st.q_nom
        self.q_des = self.st.q_nom.copy()
        self.telemetry = []
        self._k = 0
        mujoco.mj_forward(self.m, self.d)

    @property
    def t(self) -> float:
        return self._k * self.substeps * self.m.opt.timestep

    def _torque(self):
        d = self.d
        bias = d.qfrc_bias
        for a, dofadr, qadr in self.st.act:
            d.ctrl[a] = bias[dofadr] + self.kp * (self.q_des[qadr] - d.qpos[qadr]) - self.kd * d.qvel[dofadr]

    def _capture(self):
        if self.renderer is not None and self._k % self.frame_every == 0:
            self.renderer.update_scene(self.d, self.cam)
            img = self.renderer.render()
            if self.overlay_fn is not None:
                img = self.overlay_fn(img, self)
            self.frames.append(np.asarray(img).copy())

    def step(self, n: int = 1):
        for _ in range(n):
            self._torque()
            for _ in range(self.substeps):
                mujoco.mj_step(self.m, self.d)
            self._k += 1
            if self.record:
                self.telemetry.append(self.snapshot())
            if self.log_data:
                self.dataset.append({
                    "t": round(self.t, 4),
                    "qpos": self.d.qpos.copy().round(5).tolist(),
                    "qvel": self.d.qvel.copy().round(5).tolist(),
                    "ctrl": self.d.ctrl.copy().round(4).tolist(),       # action = joint torques
                    "controls": self.snapshot(),
                })
            self._capture()

    # ------------------------------------------------------------------ goals
    def set_arm_target(self, targets: dict, ik_iters: int = 220):
        """Solve IK for the given hand targets and update q_des (arms only)."""
        q, _ = K.solve_ik(self.st, self.q_des, targets, iters=ik_iters)
        for side in targets:
            if targets[side] is None:
                continue
            for qadr in self.st.arm_qadr[side]:
                self.q_des[qadr] = q[qadr]
        return q

    def move_hands(self, targets: dict, steps: int, ik_iters: int = 220):
        self.set_arm_target(targets, ik_iters)
        self.step(steps)

    def set_posture(self, joint_values: dict):
        """Override q_des for named joints (e.g. waist lean, both legs squat)."""
        for jn, val in joint_values.items():
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self.q_des[self.m.jnt_qposadr[jid]] = val

    # ------------------------------------------------------------------ readouts
    def hand(self, side):
        return self.d.xpos[self.st.hand_bid[side]].copy()

    def pad_force(self, side, target_gid):
        return self.st.contact_force(self.d, side, target_gid)

    def ctrl_angle(self, joint_name):
        return self.st.ctrl_joint(self.d, joint_name)

    def control_pos(self, name):
        """Live world position of a control body (valve/button/lever/slider).
        Reading the actual position makes the skills robust to panel jitter."""
        return self.d.xpos[self.st.ids[name + "_bid"]].copy()

    def snapshot(self) -> dict:
        st = self.st
        return {
            "t": round(self.t, 4),
            "valve_deg": round(np.degrees(self.ctrl_angle("valve_hinge")), 2),
            "button_mm": round(self.ctrl_angle("button_slide") * 1000, 2),
            "lever_deg": round(np.degrees(self.ctrl_angle("lever_hinge")), 2),
            "slider_mm": round(self.ctrl_angle("slider_slide") * 1000, 2),
            "touch_left": round(st.touch(self.d, "left"), 3),
            "touch_right": round(st.touch(self.d, "right"), 3),
        }
