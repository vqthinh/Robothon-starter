"""Scripted expert that operates the panel start-up routine.

The expert is a set of timed, phase-based fingertip waypoints. It is used only
as the demonstration source the learned imitation policy is trained from -- the
learned policy, not this script, drives the benchmark and demo.
"""
from __future__ import annotations

from .scene import EPISODE_DURATION_S, smoothstep


def expert_controls(t: float) -> dict[str, float]:
    """Return target actuator commands for the scripted expert at time ``t``."""
    s = smoothstep
    c: dict[str, float] = {}

    # Phase 1 - press the spring-return button with the thumb.
    if t < 2.1:
        c["thumb_z"] = -0.17 * s(0.3, 1.3, t) * (1.0 - s(1.8, 2.1, t))

    # Phase 2 - push the throttle slider to its set-point with the index finger.
    if 2.1 <= t < 4.4:
        c["index_z"] = -0.155 * s(2.1, 2.7, t)
        c["index_x"] = -0.03 + 0.12 * s(2.9, 4.2, t)

    # Phase 3 - two-finger chord: middle + ring press both keys simultaneously.
    if 4.4 <= t < 6.4:
        press = -0.16 * s(4.6, 5.6, t) * (1.0 - s(6.0, 6.3, t))
        c["middle_z"] = press
        c["ring_z"] = press

    # Phase 4 - turn the rotary thumbwheel: pinky presses then drags in +x.
    if 6.4 <= t < EPISODE_DURATION_S - 0.1:
        c["pinky_z"] = -0.135 * s(6.4, 7.2, t)
        c["pinky_x"] = 0.16 * s(7.3, 9.7, t)

    return c


def expert_label(t: float) -> str:
    if t < 2.1:
        return "press start button (thumb)"
    if t < 4.4:
        return "set throttle slider (index)"
    if t < 6.4:
        return "two-finger chord (middle+ring)"
    return "turn rotary wheel (pinky)"
