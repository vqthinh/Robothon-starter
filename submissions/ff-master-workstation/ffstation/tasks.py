"""Autonomous task primitives + the full machine start-up sequence.

Every primitive is *closed-loop on a physical signal* (control-joint angle or
contact force), not an open-loop trajectory: the hand keeps advancing until the
measured signal reaches its target, which makes the skills robust to contact
slip, placement jitter and modelling error. Each returns a dict of measured
outcomes for the benchmark — nothing is hard-coded.

Tasks (operating controls built for human hands):
    turn_valve  - bimanual: spin the hand-wheel to a target angle.
    press_button- force-regulated: press to a target contact force, no overshoot.
    flip_lever  - flip the toggle past its centre detent.
    set_slider  - push the throttle to a target position.
"""
from __future__ import annotations

import numpy as np

from . import scene
from .controller import WholeBodyController


# ------------------------------------------------------------------ valve
def turn_valve(c: WholeBodyController, target_deg: float = 90.0, direction: int = +1,
               settle: int = 16) -> dict:
    """Both hands push the two grab handles around the wheel axis. Closed loop on
    the measured wheel angle: the hands lead the wheel so slip is absorbed."""
    vc = c.control_pos("valve")
    r = scene.VALVE_RADIUS
    xg = -0.05                                       # x where the handle tips sit (robot side)
    lhg, rhg = c.st.ids["valve_handle_l_gid"], c.st.ids["valve_handle_r_gid"]

    def handle_targets(phi):
        return {"left": vc + [xg, r * np.cos(phi), r * np.sin(phi)],
                "right": vc + [xg, -r * np.cos(phi), -r * np.sin(phi)]}

    # approach the handles from the robot side, then settle onto them
    c.move_hands({"left": vc + [xg - 0.06, r, 0.0], "right": vc + [xg - 0.06, -r, 0.0]}, 130)
    c.move_hands(handle_targets(0.0), 110)
    start = c.ctrl_angle("valve_hinge")
    peak_force = 0.0
    phi = 0.0
    phi_max = np.radians(target_deg) * 1.6 + 0.5     # extra arc to absorb slip
    while abs(np.degrees(c.ctrl_angle("valve_hinge") - start)) < target_deg and abs(phi) < phi_max:
        phi += direction * 0.04
        c.move_hands(handle_targets(phi), settle)
        peak_force = max(peak_force, c.pad_force("left", lhg) + c.pad_force("right", rhg))
    turned = np.degrees(c.ctrl_angle("valve_hinge") - start)
    # release and hold -> confirm the wheel stays where it was set
    c.move_hands({"left": vc + [xg - 0.06, r * np.cos(phi), r * np.sin(phi)],
                  "right": vc + [xg - 0.06, -r * np.cos(phi), -r * np.sin(phi)]}, 120)
    held = np.degrees(c.ctrl_angle("valve_hinge") - start)
    return {"task": "valve", "target_deg": target_deg, "turned_deg": round(turned, 1),
            "held_deg": round(held, 1), "grip_force_N": round(peak_force, 1),
            "success": bool(abs(turned) >= 0.75 * target_deg)}


# ------------------------------------------------------------------ button
def press_button(c: WholeBodyController, actuate_force: float = 3.5,
                 crush_force: float = 15.0, hold_steps: int = 25) -> dict:
    """Press the spring-loaded start button with contact-force feedback: advance the
    hand in fine, settled increments until the measured contact force reaches the
    actuation threshold, hold the button engaged, then release and confirm the spring
    returns it. The contact force is read live (mj_contactForce); the press is firm
    enough to actuate yet stays well below the housing's crush rating."""
    bc = c.control_pos("button")
    cap = c.st.ids["button_cap_gid"]
    c.move_hands({"left": bc + [-0.16, 0.0, 0.06]}, 90)     # clean pre-approach waypoint
    c.move_hands({"left": bc + [-0.09, 0.0, 0.0]}, 120)
    x = -0.05
    peak_f = 0.0
    # advance in fine settled steps until the contact force crosses the actuation point
    for _ in range(160):
        f = c.pad_force("left", cap)
        peak_f = max(peak_f, f)
        if f >= actuate_force:
            break
        x = min(x + 0.0008, 0.05)
        c.move_hands({"left": bc + [x, 0.0, 0.0]}, 8)
    depth_mm = c.ctrl_angle("button_slide") * 1000.0
    # hold the button engaged, record the steady contact force
    forces = []
    for _ in range(hold_steps):
        c.move_hands({"left": bc + [x, 0.0, 0.0]}, 2)
        f = c.pad_force("left", cap)
        forces.append(f)
        peak_f = max(peak_f, f)
    # release -> spring returns the plunger
    c.move_hands({"left": bc + [-0.12, 0.0, 0.0]}, 180)
    released_mm = c.ctrl_angle("button_slide") * 1000.0
    forces = np.array(forces) if forces else np.array([0.0])
    return {"task": "button", "actuate_force_N": actuate_force,
            "press_depth_mm": round(depth_mm, 1), "peak_force_N": round(peak_f, 2),
            "hold_force_N": round(float(forces.mean()), 2), "released_mm": round(released_mm, 1),
            "success": bool(peak_f >= actuate_force and peak_f <= crush_force and released_mm < 4.0)}


# ------------------------------------------------------------------ lever
def flip_lever(c: WholeBodyController, target_deg: float = 45.0) -> dict:
    lc = c.control_pos("lever")
    knob = lc + [-0.05, 0.0, scene.LEVER_LEN]
    start = np.degrees(c.ctrl_angle("lever_hinge"))
    c.move_hands({"right": knob + [-0.09, 0.0, 0.06]}, 160)
    # push the knob forward/down through the detent
    for dx, dz in [(0.0, 0.0), (0.05, -0.03), (0.09, -0.05), (0.12, -0.06)]:
        c.move_hands({"right": knob + [dx, 0.0, dz]}, 110)
        if np.degrees(c.ctrl_angle("lever_hinge")) - start >= target_deg:
            break
    moved = np.degrees(c.ctrl_angle("lever_hinge")) - start
    c.move_hands({"right": knob + [-0.10, 0.0, 0.08]}, 110)
    held = np.degrees(c.ctrl_angle("lever_hinge")) - start
    return {"task": "lever", "target_deg": target_deg, "flipped_deg": round(moved, 1),
            "held_deg": round(held, 1), "success": bool(abs(moved) >= target_deg)}


# ------------------------------------------------------------------ slider
def set_slider(c: WholeBodyController, target_mm: float = 35.0) -> dict:
    """Push the throttle knob to a target position. The knob is engaged from its
    -y side (approached IN FRONT so the descent never knocks it), then carried to
    the target — closed loop on the measured slider position."""
    sc = c.control_pos("slider")
    target_m = target_mm / 1000.0
    y_lo = -(scene.SLIDER_TRAVEL / 2) - 0.03            # just left of the knob's travel
    # 1) clean pre-approach waypoint high in front, then descend in FRONT of the panel
    #    (x small) so the hand never sweeps through the knob
    c.move_hands({"left": sc + [-0.16, y_lo, 0.12]}, 90)
    c.move_hands({"left": sc + [-0.11, y_lo, 0.0]}, 110)
    # 2) move in to the engage plane, still left of the knob, pressed lightly +x so the
    #    pad keeps contact with the knob's -y face while carrying
    c.move_hands({"left": sc + [0.02, y_lo, 0.0]}, 90)
    # 3) carry the knob along +y, closed loop on slider position
    y = y_lo
    for _ in range(70):
        if c.ctrl_angle("slider_slide") >= target_m:
            break
        y = min(y + 0.004, scene.SLIDER_TRAVEL / 2 + 0.04)
        c.move_hands({"left": sc + [0.02, y, 0.0]}, 12)
    c.move_hands({"left": sc + [0.02, y, 0.0]}, 40)        # settle -> kill knob velocity
    pos = c.ctrl_angle("slider_slide") * 1000.0
    # retract by lifting STRAIGHT UP from the final carry pose (no y motion -> no drag)
    c.move_hands({"left": sc + [0.02, y, 0.14]}, 80)
    c.move_hands({"left": sc + [-0.12, y, 0.14]}, 70)
    held = c.ctrl_angle("slider_slide") * 1000.0
    return {"task": "slider", "target_mm": target_mm, "pos_mm": round(pos, 1),
            "held_mm": round(held, 1), "error_mm": round(abs(pos - target_mm), 1),
            "success": bool(abs(pos - target_mm) <= 8.0)}


# ------------------------------------------------------------------ full sequence
SEQUENCE = [
    ("valve", lambda c: turn_valve(c, target_deg=20.0)),
    ("button", lambda c: press_button(c)),
    ("lever", lambda c: flip_lever(c, target_deg=45.0)),
    ("slider", lambda c: set_slider(c, target_mm=30.0)),
]


def return_to_ready(c: WholeBodyController, steps: int = 60):
    """Bring both arms back to the nominal ready posture."""
    for side in ("left", "right"):
        for qadr in c.st.arm_qadr[side]:
            c.q_des[qadr] = c.st.q_nom[qadr]
    c.step(steps)


def run_sequence(c: WholeBodyController, settle_first: int = 80) -> list[dict]:
    """Machine start-up: pressurise (valve) -> start (button) -> mode (lever) ->
    throttle (slider). Returns the measured result of each step."""
    c.reset()
    c.step(settle_first)
    results = []
    for _name, fn in SEQUENCE:
        results.append(fn(c))
        return_to_ready(c)
    return results
