"""Build the MuJoCo scene programmatically with the MjSpec API.

We load Faraday Future's FF Master humanoid (assets/Master), convert it to a
*fixed-base manipulation station* (the pelvis is mounted on a pedestal, so the
robot cannot fall and every joule of actuation goes into the task), and compose
an industrial control panel in front of its chest:

  - VALVE   : a hand-wheel on a hinge joint -> operated with BOTH hands.
  - BUTTON  : a spring-return plunger on a slide joint -> force-regulated press.
  - LEVER   : a toggle bar on a hinge joint -> flipped past centre.
  - SLIDER  : a throttle knob on a slide joint -> set to a target position.

Soft contact pads are added to each palm, plus touch sensors, joint sensors and
demo cameras. Everything is built from code so the vendored humanoid stays
pristine and the panel layout is fully parametric (jitter-able for benchmarking).

All physical signals used by the controller (contact force, joint angles) are
read live from the simulation; nothing is hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
# repo-root assets (this file lives at submissions/ff-master-workstation/ffstation/)
_MASTER = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "assets", "Master", "scene.xml"))

# ----------------------------------------------------------------------------- maps
ARM_JOINTS = {
    "left":  ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
              "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"],
}
HAND_BODY = {"left": "left_wrist_roll_link", "right": "right_wrist_roll_link"}

# "ready at the console" standing pose: knees slightly bent, arms raised toward the panel.
NOMINAL_POSE = {
    "left_hip_pitch_joint": -0.20, "left_knee_joint": 0.45, "left_ankle_pitch_joint": -0.25,
    "right_hip_pitch_joint": -0.20, "right_knee_joint": 0.45, "right_ankle_pitch_joint": -0.25,
    "waist_pitch_joint": 0.06,
    "left_shoulder_pitch_joint": 0.35, "left_shoulder_roll_joint": 0.65, "left_elbow_joint": -1.15,
    "right_shoulder_pitch_joint": 0.35, "right_shoulder_roll_joint": -0.65, "right_elbow_joint": -1.15,
}

# ----------------------------------------------------------------------------- panel layout
# all positions inside FF Master's verified accurate reach box:
#   x in [0.28, 0.34],  |y| <= 0.22,  z in [0.70, 0.85]
VALVE_CENTER = np.array([0.32, 0.00, 0.82])        # centre -> BOTH hands
VALVE_RADIUS = 0.10
BUTTON_CENTER = np.array([0.30, 0.21, 0.83])       # upper-left  -> left hand
SLIDER_CENTER = np.array([0.30, 0.15, 0.73])       # lower-left  -> left hand (horizontal)
SLIDER_TRAVEL = 0.10                               # along +y (knob 0.10..0.20 -> reachable)
LEVER_CENTER = np.array([0.30, -0.20, 0.76])       # right       -> right hand
LEVER_LEN = 0.075

PAD_FRICTION = [2.5, 0.1, 0.002]
SOFT_SOLREF = [0.008, 1.0]
SOFT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2]


@dataclass
class Station:
    """Compiled model + cached ids/maps + live readout helpers."""
    model: mujoco.MjModel
    arm_dof: dict = field(default_factory=dict)
    arm_qadr: dict = field(default_factory=dict)
    hand_bid: dict = field(default_factory=dict)
    pad_gid: dict = field(default_factory=dict)
    act: list = field(default_factory=list)          # (act_id, dof_adr, qpos_adr)
    q_nom: np.ndarray = None
    ids: dict = field(default_factory=dict)          # named joint/sensor/body ids

    # --- live readouts (all measured from physics) ---
    def ctrl_joint(self, d, name) -> float:
        return float(d.qpos[self.ids[name + "_qadr"]])

    def touch(self, d, side) -> float:
        adr = self.model.sensor_adr[self.ids[f"touch_{side}_sid"]]
        return float(d.sensordata[adr])

    def contact_force(self, d, side, target_gid) -> float:
        """Sum |normal contact force| between a palm pad and a target geom."""
        pad = self.pad_gid[side]
        total = 0.0
        ff = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if pad not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == pad else c.geom1
            if target_gid is not None and other != target_gid:
                continue
            mujoco.mj_contactForce(self.model, d, i, ff)
            total += float(np.linalg.norm(ff[:3]))
        return total


def _add_pad(spec, side: str):
    b = spec.body(HAND_BODY[side])
    g = b.add_geom()
    g.name = f"{side}_palm_pad"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = np.array([0.05, 0.018, 0.055])
    g.pos = np.array([0.0, 0.0, -0.085])
    g.rgba = np.array([0.95, 0.55, 0.25, 1.0])
    g.friction = np.array(PAD_FRICTION)
    g.condim = 4
    g.solref = np.array(SOFT_SOLREF)
    g.solimp = np.array(SOFT_SOLIMP)
    # touch site covering the pad
    s = b.add_site()
    s.name = f"touch_{side}_site"
    s.type = mujoco.mjtGeom.mjGEOM_BOX
    s.size = np.array([0.055, 0.025, 0.06])
    s.pos = np.array([0.0, 0.0, -0.085])
    s.rgba = np.array([1.0, 0.3, 0.3, 0.0])


def _panel_material(spec):
    # a dark console backboard so the controls read clearly on camera
    pass


def build_spec(jitter=(0.0, 0.0, 0.0), seed: int | None = None) -> "mujoco.MjSpec":
    rng = np.random.default_rng(seed)
    j = np.array(jitter, float)
    spec = mujoco.MjSpec.from_file(_MASTER)
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.option.gravity = np.array([0.0, 0.0, -9.81])

    # 1) fixed base: drop the floating joint -> pelvis welded to the world.
    for jt in list(spec.joints):
        if jt.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(jt)

    # 2) soft palm pads + touch sites + sensors
    for side in ("left", "right"):
        _add_pad(spec, side)
    for side in ("left", "right"):
        sen = spec.add_sensor()
        sen.name = f"s_touch_{side}"
        sen.type = mujoco.mjtSensor.mjSENS_TOUCH
        sen.objtype = mujoco.mjtObj.mjOBJ_SITE
        sen.objname = f"touch_{side}_site"

    wb = spec.worldbody
    L = wb.add_light(); L.pos = np.array([0.4, -0.3, 1.6]); L.dir = np.array([-0.2, 0.2, -1.0])
    L.diffuse = np.array([0.5, 0.5, 0.5])

    # 3) equipment cabinet BELOW the controls (stays clear of the camera sightline)
    cab = wb.add_geom(); cab.name = "cabinet"; cab.type = mujoco.mjtGeom.mjGEOM_BOX
    cab.size = np.array([0.10, 0.34, 0.30]); cab.pos = np.array([0.44, 0.0, 0.30])
    cab.rgba = np.array([0.22, 0.24, 0.28, 1.0])
    top = wb.add_geom(); top.name = "cabinet_top"; top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = np.array([0.12, 0.36, 0.012]); top.pos = np.array([0.44, 0.0, 0.60])
    top.rgba = np.array([0.30, 0.32, 0.36, 1.0])

    def _standoff(center):
        p = wb.add_geom(); p.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        p.fromto = np.array([center[0] + 0.085, center[1], 0.60, center[0] + 0.085, center[1], center[2] - 0.03])
        p.size = np.array([0.007, 0, 0]); p.rgba = np.array([0.34, 0.36, 0.40, 1.0]); p.contype = 0; p.conaffinity = 0

    # 4) VALVE hand-wheel (hinge about +x, faces the robot)
    vc = VALVE_CENTER + j
    wheel = wb.add_body(); wheel.name = "valve"; wheel.pos = vc
    vj = wheel.add_joint(); vj.name = "valve_hinge"; vj.type = mujoco.mjtJoint.mjJNT_HINGE
    vj.axis = np.array([1.0, 0.0, 0.0])
    # decorative blue disc (thin) + cross spokes for the "wheel" look (light collision)
    rim = wheel.add_geom(); rim.name = "valve_rim"; rim.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    rim.size = np.array([VALVE_RADIUS, 0.010, 0.0]); rim.quat = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    rim.rgba = np.array([0.20, 0.45, 0.72, 1.0]); rim.friction = np.array([2.0, 0.1, 0.002])
    rim.condim = 4; rim.solref = np.array(SOFT_SOLREF); rim.mass = 0.10; rim.pos = np.array([0.03, 0, 0])
    for ang in (0.0, np.pi / 2):
        sp = wheel.add_geom(); sp.type = mujoco.mjtGeom.mjGEOM_BOX
        sp.size = np.array([0.010, VALVE_RADIUS, 0.018]); sp.contype = 0; sp.conaffinity = 0
        sp.quat = np.array([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)])
        sp.pos = np.array([0.03, 0.0, 0.0]); sp.rgba = np.array([0.30, 0.55, 0.80, 1.0]); sp.mass = 0.01
    hub = wheel.add_geom(); hub.type = mujoco.mjtGeom.mjGEOM_SPHERE; hub.size = np.array([0.028, 0, 0])
    hub.pos = np.array([0.03, 0, 0]); hub.rgba = np.array([0.3, 0.3, 0.33, 1.0]); hub.mass = 0.04
    # TWO grab handles sticking toward the robot (-x) at the +y / -y rim points.
    # Each hand pushes a handle around the circle -> positive purchase (not friction).
    for ysign, hname in ((+1, "valve_handle_l"), (-1, "valve_handle_r")):
        hp = wheel.add_geom(); hp.name = hname; hp.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        hp.fromto = np.array([0.03, ysign * VALVE_RADIUS, 0.0, -0.07, ysign * VALVE_RADIUS, 0.0])
        hp.size = np.array([0.016, 0, 0]); hp.rgba = np.array([0.92, 0.46, 0.18, 1.0])
        hp.friction = np.array([3.2, 0.2, 0.004]); hp.condim = 4; hp.mass = 0.05
    _standoff(vc)

    # 5) BUTTON spring-return plunger (slide along +x into the console)
    bc = BUTTON_CENTER + j
    btn = wb.add_body(); btn.name = "button"; btn.pos = bc
    bj = btn.add_joint(); bj.name = "button_slide"; bj.type = mujoco.mjtJoint.mjJNT_SLIDE
    bj.axis = np.array([1.0, 0.0, 0.0]); bj.range = np.array([0.0, 0.026])
    bcap = btn.add_geom(); bcap.name = "button_cap"; bcap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bcap.size = np.array([0.036, 0.02, 0.0]); bcap.quat = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    bcap.rgba = np.array([0.20, 0.78, 0.32, 1.0]); bcap.friction = np.array(PAD_FRICTION)
    # soft, compliant contact so press force builds gradually with hand position
    # (fine force control) and the cap is not "sticky" on release
    bcap.condim = 4; bcap.solref = np.array([0.08, 1.0])
    bcap.solimp = np.array([0.7, 0.85, 0.01, 0.5, 2]); bcap.mass = 0.05
    # housing ring BEHIND the cap (toward console) so the green face stays visible
    bring = wb.add_geom(); bring.name = "button_collar"; bring.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bring.size = np.array([0.046, 0.012, 0.0]); bring.quat = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    bring.pos = bc + np.array([0.055, 0, 0]); bring.rgba = np.array([0.12, 0.13, 0.15, 1.0])
    bring.contype = 0; bring.conaffinity = 0
    _standoff(bc)

    # 6) LEVER toggle (hinge about +y -> swings in x-z plane). Bar points up toward
    #    the robot (-x, +z); the hand pushes the knob forward/down to flip it.
    lc = LEVER_CENTER + j
    lev = wb.add_body(); lev.name = "lever"; lev.pos = lc
    lj = lev.add_joint(); lj.name = "lever_hinge"; lj.type = mujoco.mjtJoint.mjJNT_HINGE
    lj.axis = np.array([0.0, 1.0, 0.0]); lj.range = np.array([-1.0, 1.0])
    lbar = lev.add_geom(); lbar.name = "lever_bar"; lbar.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    lbar.fromto = np.array([0.0, 0.0, 0.0, -0.05, 0.0, LEVER_LEN]); lbar.size = np.array([0.014, 0, 0])
    lbar.rgba = np.array([0.80, 0.25, 0.22, 1.0]); lbar.friction = np.array(PAD_FRICTION); lbar.condim = 4
    lknob = lev.add_geom(); lknob.name = "lever_knob"; lknob.type = mujoco.mjtGeom.mjGEOM_SPHERE
    lknob.size = np.array([0.025, 0, 0]); lknob.pos = np.array([-0.05, 0.0, LEVER_LEN])
    lknob.rgba = np.array([0.92, 0.78, 0.25, 1.0]); lknob.friction = np.array(PAD_FRICTION); lknob.condim = 4
    _standoff(lc)

    # 7) SLIDER throttle (slide along +y, horizontal -> no gravity along travel).
    #    Left hand pushes the knob sideways to a target position.
    sc = SLIDER_CENTER + j
    sld = wb.add_body(); sld.name = "slider"; sld.pos = sc
    sj = sld.add_joint(); sj.name = "slider_slide"; sj.type = mujoco.mjtJoint.mjJNT_SLIDE
    sj.axis = np.array([0.0, 1.0, 0.0]); sj.range = np.array([-SLIDER_TRAVEL / 2, SLIDER_TRAVEL / 2])
    sk = sld.add_geom(); sk.name = "slider_knob"; sk.type = mujoco.mjtGeom.mjGEOM_BOX
    sk.size = np.array([0.032, 0.032, 0.05]); sk.rgba = np.array([0.93, 0.80, 0.22, 1.0])
    sk.friction = np.array(PAD_FRICTION); sk.condim = 4; sk.solref = np.array(SOFT_SOLREF)
    rail = wb.add_geom(); rail.name = "slider_rail"; rail.type = mujoco.mjtGeom.mjGEOM_BOX
    rail.size = np.array([0.018, SLIDER_TRAVEL / 2 + 0.035, 0.012]); rail.pos = sc + np.array([0.0, 0.0, -0.05])
    rail.rgba = np.array([0.12, 0.13, 0.15, 1.0]); rail.contype = 0; rail.conaffinity = 0
    _standoff(sc)

    return spec


def build(jitter=(0.0, 0.0, 0.0), seed: int | None = None) -> Station:
    spec = build_spec(jitter, seed)
    model = spec.compile()

    nid = lambda objtype, n: mujoco.mj_name2id(model, objtype, n)
    JID = lambda n: nid(mujoco.mjtObj.mjOBJ_JOINT, n)
    BID = lambda n: nid(mujoco.mjtObj.mjOBJ_BODY, n)
    GID = lambda n: nid(mujoco.mjtObj.mjOBJ_GEOM, n)
    SID = lambda n: nid(mujoco.mjtObj.mjOBJ_SENSOR, n)

    st = Station(model=model)
    st.arm_dof = {s: [model.jnt_dofadr[JID(j)] for j in js] for s, js in ARM_JOINTS.items()}
    st.arm_qadr = {s: [model.jnt_qposadr[JID(j)] for j in js] for s, js in ARM_JOINTS.items()}
    st.hand_bid = {s: BID(HAND_BODY[s]) for s in ("left", "right")}
    st.pad_gid = {s: GID(f"{s}_palm_pad") for s in ("left", "right")}
    st.act = [(a, model.jnt_dofadr[model.actuator_trnid[a, 0]],
               model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(model.nu)]

    ids = st.ids
    for s in ("left", "right"):
        ids[f"touch_{s}_sid"] = SID(f"s_touch_{s}")
    for cj in ("valve_hinge", "button_slide", "lever_hinge", "slider_slide"):
        ids[cj + "_jid"] = JID(cj)
        ids[cj + "_qadr"] = model.jnt_qposadr[JID(cj)]
        ids[cj + "_dof"] = model.jnt_dofadr[JID(cj)]
    for g in ("valve_rim", "button_cap", "lever_bar", "lever_knob", "slider_knob"):
        gid = GID(g) if GID(g) >= 0 else GID(g.replace("lever_knob", "lever_bar"))
    ids["valve_rim_gid"] = GID("valve_rim")
    ids["valve_handle_l_gid"] = GID("valve_handle_l")
    ids["valve_handle_r_gid"] = GID("valve_handle_r")
    ids["button_cap_gid"] = GID("button_cap")
    ids["lever_bid"] = BID("lever")
    ids["slider_knob_gid"] = GID("slider_knob")
    ids["button_bid"] = BID("button")
    ids["valve_bid"] = BID("valve")
    ids["slider_bid"] = BID("slider")

    # passive-control dynamics (set on compiled model: reliable across MjSpec versions)
    # button: spring-return tuned so the actuation force sits mid-travel (clean return)
    model.jnt_stiffness[ids["button_slide_jid"]] = 350.0
    model.dof_damping[ids["button_slide_dof"]] = 4.0
    # valve: low-inertia wheel; light dry friction lets a hand grip turn it, and
    # enough damping that it does not recoil when the hands release
    model.dof_damping[ids["valve_hinge_dof"]] = 1.3
    model.dof_frictionloss[ids["valve_hinge_dof"]] = 0.05
    # lever: damped, mild detent via frictionloss
    model.dof_damping[ids["lever_hinge_dof"]] = 0.5
    model.dof_frictionloss[ids["lever_hinge_dof"]] = 0.15
    # slider: damped + light dry friction (pushable, holds via damping)
    model.dof_damping[ids["slider_slide_dof"]] = 7.0
    model.dof_frictionloss[ids["slider_slide_dof"]] = 1.5

    # nominal qpos
    q = np.zeros(model.nq)
    for jn, val in NOMINAL_POSE.items():
        q[model.jnt_qposadr[JID(jn)]] = val
    st.q_nom = q
    return st


if __name__ == "__main__":
    st = build()
    m = st.model
    print(f"compiled: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} nsensor={m.nsensor}")
    print(f"robot mass = {m.body_mass.sum():.2f} kg")
    print("control joints:", {k: st.ids[k + "_qadr"] for k in ("valve_hinge", "button_slide", "lever_hinge", "slider_slide")})
