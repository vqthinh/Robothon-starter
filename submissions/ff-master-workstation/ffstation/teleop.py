"""Keyboard teleoperation (one of the contest's recommended directions).

Drive either hand's Cartesian target with the keyboard; the same gravity-compensated
whole-body controller tracks it in real time, so you can operate the panel by hand.
Requires a display (opens the MuJoCo viewer).

Controls:
    Tab        switch active hand (left / right)
    W / S      move target  +x / -x   (toward / away from panel)
    A / D      move target  +y / -y   (left / right)
    R / F      move target  +z / -z   (up / down)
    [ / ]      both hands inward / outward (bimanual squeeze, e.g. valve)
    Space      reset to the ready pose
    Esc        quit

Run:  python -m ffstation.teleop      (or  python run.py --teleop)
"""
from __future__ import annotations

import numpy as np

from . import scene
from .controller import WholeBodyController

STEP = 0.01


def main():
    try:
        import mujoco.viewer
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Teleop needs the MuJoCo viewer (a display): {exc}")

    st = scene.build()
    c = WholeBodyController(st, record=False)
    c.reset()
    targets = {"left": c.hand("left"), "right": c.hand("right")}
    active = ["left"]

    def on_key(keycode):
        key = chr(keycode) if 0 <= keycode < 0x110000 else ""
        a = active[0]
        if keycode == 258:  # Tab
            active[0] = "right" if a == "left" else "left"
            return
        if key in ("W", "w"): targets[a][0] += STEP
        elif key in ("S", "s"): targets[a][0] -= STEP
        elif key in ("A", "a"): targets[a][1] += STEP
        elif key in ("D", "d"): targets[a][1] -= STEP
        elif key in ("R", "r"): targets[a][2] += STEP
        elif key in ("F", "f"): targets[a][2] -= STEP
        elif key == "[":
            targets["left"][1] -= STEP; targets["right"][1] += STEP
        elif key == "]":
            targets["left"][1] += STEP; targets["right"][1] -= STEP
        elif key == " ":
            c.reset(); targets["left"][:] = c.hand("left"); targets["right"][:] = c.hand("right")

    print(__doc__)
    with mujoco.viewer.launch_passive(c.m, c.d, key_callback=on_key) as v:
        while v.is_running():
            c.set_arm_target({s: targets[s] for s in ("left", "right")}, ik_iters=60)
            c.step(1)
            v.sync()


if __name__ == "__main__":
    main()
