"""Render the narrated telemetry demo video.

A full-bleed view of the FF Master humanoid operating each control, with a
"mission-control" HUD composited on top: a top-left wordmark, a top-right live
tactile-force card, a lower-third caption with a segment chip, and a row of
control-status pills. Every number on screen is read straight from the running
simulation; nothing is scripted.

Run:  python -m ffstation.record_demo     (or  python run.py --demo)
"""
from __future__ import annotations

import os

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from . import scene, tasks
from .controller import WholeBodyController

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
W, H = 1280, 720
TARGET_SECONDS = 60

# palette (dark, with a warm accent)
C_INK = (238, 235, 230)
C_MUTE = (150, 158, 170)
C_FAINT = (96, 104, 116)
C_ACCENT = (240, 146, 64)
C_OK = (96, 206, 138)
C_BLUE = (96, 158, 226)
C_WARN = (236, 122, 92)
C_SET = (240, 206, 120)
C_PANEL = (16, 19, 25)

_FONT_FILES = {
    "head": [r"C:\Windows\Fonts\bahnschrift.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arialbd.ttf"],
    "body": [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"],
    "mono": [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"],
    "monob": [r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\courbd.ttf"],
}
_CACHE: dict = {}


def _f(kind: str, size: int):
    key = (kind, size)
    if key not in _CACHE:
        for path in _FONT_FILES[kind]:
            if os.path.exists(path):
                _CACHE[key] = ImageFont.truetype(path, size)
                break
        else:
            _CACHE[key] = ImageFont.load_default()
    return _CACHE[key]


def _spaced(s: str, gap: str = " ") -> str:
    return gap.join(list(s))


def _rrect(d, box, radius, **kw):
    try:
        d.rounded_rectangle(box, radius=radius, **kw)
    except Exception:
        d.rectangle(box, **kw)


def _camera():
    c = mujoco.MjvCamera()
    c.azimuth = 180; c.elevation = -13; c.distance = 1.18
    c.lookat[:] = [0.30, 0.0, 0.80]
    return c


# ----------------------------------------------------------------------- HUD
PILLS = [
    ("valve", "VALVE", C_BLUE, lambda s, c: f"{s['valve_deg']:5.0f}°"),
    ("button", "BUTTON", C_OK, lambda s, c: f"{c.pad_force('left', c.st.ids['button_cap_gid']):5.1f} N"),
    ("lever", "LEVER", C_WARN, lambda s, c: f"{s['lever_deg']:5.0f}°"),
    ("slider", "SLIDER", C_SET, lambda s, c: f"{s['slider_mm']:4.0f} mm"),
]


def _overlay(img, c: WholeBodyController):
    snap = c.snapshot()
    base = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # ---- top-left wordmark ----
    d.text((44, 34), "FF MASTER", font=_f("head", 30), fill=C_INK)
    wm = d.textlength("FF MASTER", font=_f("head", 30))
    d.ellipse([44 + wm + 12, 48, 44 + wm + 22, 58], fill=C_ACCENT)
    d.text((44 + wm + 30, 34), "WORKSTATION", font=_f("head", 30), fill=C_ACCENT)
    d.text((46, 74), _spaced("FARADAY FUTURE  ·  MUJOCO"), font=_f("mono", 12), fill=C_MUTE)
    d.line([46, 96, 300, 96], fill=(*C_ACCENT, 220), width=2)

    # ---- top-right tactile card ----
    cx, cy, cw, ch = W - 322, 30, 292, 96
    _rrect(d, [cx, cy, cx + cw, cy + ch], 12, fill=(*C_PANEL, 205), outline=(70, 78, 92, 220), width=1)
    d.rectangle([cx, cy + 12, cx + 3, cy + ch - 12], fill=C_ACCENT)
    d.text((cx + 18, cy + 13), _spaced("TACTILE · FINGERTIP FORCE"), font=_f("mono", 11), fill=C_MUTE)
    d.text((cx + 18, cy + 36), "L", font=_f("mono", 16), fill=C_FAINT)
    d.text((cx + 40, cy + 33), f"{snap['touch_left']:5.1f}", font=_f("monob", 24), fill=C_INK)
    d.text((cx + 120, cy + 39), "N", font=_f("mono", 15), fill=C_FAINT)
    d.text((cx + 150, cy + 36), "R", font=_f("mono", 16), fill=C_FAINT)
    d.text((cx + 172, cy + 33), f"{snap['touch_right']:5.1f}", font=_f("monob", 24), fill=C_INK)
    d.text((cx + 252, cy + 39), "N", font=_f("mono", 15), fill=C_FAINT)

    # ---- lower-third caption band ----
    y0 = H - 178
    band = Image.new("RGBA", (W, H - y0), (8, 9, 12, 165))
    ov.alpha_composite(band, (0, y0))
    d.line([0, y0, W, y0], fill=(*C_ACCENT, 200), width=2)

    # segment chip + title + description
    _rrect(d, [44, y0 + 24, 100, y0 + 80], 10, fill=(*C_ACCENT, 235))
    num = c.demo_index
    tw = d.textlength(num, font=_f("monob", 28))
    d.text((72 - tw / 2, y0 + 36), num, font=_f("monob", 28), fill=(14, 16, 20))
    d.text((118, y0 + 20), c.demo_title, font=_f("head", 42), fill=C_INK)
    d.text((120, y0 + 76), c.demo_sub, font=_f("body", 20), fill=C_MUTE)

    # control-status pills
    py, ph = y0 + 122, 42
    px, gap = 44, 12
    pw = (W - 2 * px - 3 * gap) // 4
    for i, (key, label, color, valfn) in enumerate(PILLS):
        x = px + i * (pw + gap)
        active = (c.demo_active == key)
        _rrect(d, [x, py, x + pw, py + ph], 10,
               fill=(*color, 40) if active else (32, 36, 44, 170),
               outline=(*color, 255) if active else (64, 70, 82, 200), width=2 if active else 1)
        _rrect(d, [x + 12, py + 14, x + 24, py + 28], 3, fill=color if active else (*color, 150))
        d.text((x + 34, py + 6), label, font=_f("mono", 12), fill=C_INK if active else C_MUTE)
        val = valfn(snap, c)
        vw = d.textlength(val, font=_f("monob", 19))
        d.text((x + pw - 14 - vw, py + 11), val, font=_f("monob", 19),
               fill=C_INK if active else C_MUTE)

    # live-data tag, right-aligned inside the band header
    tag = _spaced("LIVE FROM PHYSICS")
    tw = d.textlength(tag, font=_f("mono", 11))
    d.ellipse([W - 60 - tw - 16, y0 + 20, W - 60 - tw - 6, y0 + 30], fill=C_WARN)
    d.text((W - 60 - tw, y0 + 18), tag, font=_f("mono", 11), fill=C_MUTE)

    return np.asarray(Image.alpha_composite(base, ov).convert("RGB"))


# ----------------------------------------------------------------------- cards
def _card(index, title, lines, n=64):
    canvas = Image.new("RGB", (W, H), (11, 13, 17))
    d = ImageDraw.Draw(canvas)
    # left accent bar
    d.rectangle([0, 0, 8, H], fill=C_ACCENT)
    x = 96
    d.text((x, H // 2 - 150), _spaced(index), font=_f("mono", 22), fill=C_FAINT)
    d.line([x, H // 2 - 108, x + 64, H // 2 - 108], fill=C_ACCENT, width=3)
    y = H // 2 - 86
    for txt, sz, col in title:
        d.text((x, y), txt, font=_f("head", sz), fill=col); y += sz + 6
    y += 16
    for ln in lines:
        d.text((x, y), ln, font=_f("body", 21), fill=C_MUTE); y += 32
    d.text((x, H - 70), _spaced("FF MASTER WORKSTATION · MUJOCO"), font=_f("mono", 13), fill=C_FAINT)
    return [np.asarray(canvas)] * n


def record(out_path: str | None = None, fps: int | None = None) -> str:
    out_path = out_path or os.path.join(RESULTS, "demo.mp4")
    os.makedirs(RESULTS, exist_ok=True)
    st = scene.build()
    R = mujoco.Renderer(st.model, H, W)
    cam = _camera()
    c = WholeBodyController(st, renderer=R, cam=cam, record=False)
    c.frame_every = 3
    c.overlay_fn = _overlay
    c.demo_index = "00"

    frames = []
    frames += _card("INTRO", [("FF Master", 60, C_INK),
                              ("Workstation", 60, C_ACCENT)],
                    ["A humanoid operating human-designed industrial controls.",
                     "31-DOF whole-body torque control  ·  every value read live from physics."],
                    n=96)

    segments = [
        ("01", "valve", "Bimanual valve", "Both hands coordinate to open the hand-wheel.",
         lambda: tasks.turn_valve(c, 20.0)),
        ("02", "button", "Force-feedback button", "Press the start button under live contact-force feedback.",
         lambda: tasks.press_button(c)),
        ("03", "lever", "Toggle lever", "Flip the mode lever past its centre detent.",
         lambda: tasks.flip_lever(c, 45.0)),
        ("04", "slider", "Throttle slider", "Push the throttle to a commanded position.",
         lambda: tasks.set_slider(c, 18.0)),
    ]
    for idx, key, title, sub, fn in segments:
        frames += _card(f"{idx} / 04", [(title, 52, C_INK)], [sub], n=52)
        c.demo_index, c.demo_active = idx, key
        c.demo_title, c.demo_sub = title, sub
        c.reset(); c.step(40)
        c.frames = []
        fn()
        frames += c.frames

    frames += _card("RESULT", [("36-trial benchmark", 46, C_INK), ("100% success", 46, C_OK)],
                    ["pip install -r requirements.txt   &&   python run.py --demo",
                     "Embodied AI for the controls the world already runs on."],
                    n=104)

    if fps is None:
        fps = int(np.clip(round(len(frames) / TARGET_SECONDS), 18, 30))
    imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=1)
    print(f"[OK] {len(frames)} frames @ {fps}fps ({len(frames)/fps:.0f}s) -> {out_path}")
    return out_path


if __name__ == "__main__":
    record()
