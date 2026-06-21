"""Render a demo video from a recorded rollout trajectory.

Primary path: a CPU-only matplotlib animation (top-down panel schematic + live
telemetry) that runs anywhere -- no GPU or display required. If a working
MuJoCo OpenGL backend is available, ``render_mujoco`` can be used instead.
"""
from __future__ import annotations

import numpy as np

from .scene import CTRL_POS, SUCCESS

_CTRL_XY = {
    "button": (CTRL_POS["button"], 0.0),
    "slider": (CTRL_POS["slider"], 0.0),
    "key_g": (CTRL_POS["key_g"], 0.0),
    "key_b": (CTRL_POS["key_b"], 0.0),
    "wheel": (CTRL_POS["wheel"], 0.0),
}
_TIP_COLORS = {
    "tip_thumb": "#e74c3c", "tip_index": "#2ecc71", "tip_middle": "#f1c40f",
    "tip_ring": "#3498db", "tip_pinky": "#9b59b6",
}


def render_animation(trajectory: list[dict], out_path: str, fps: int = 30) -> str:
    """Write an MP4 telemetry/schematic animation from ``trajectory``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    ts = [f["t"] for f in trajectory]
    series = {
        "button_depth": [f["button_depth"] for f in trajectory],
        "slider_pos": [f["slider_pos"] for f in trajectory],
        "chord": [min(f["keyg_depth"], f["keyb_depth"]) for f in trajectory],
        "wheel_angle": [abs(f["wheel_angle"]) for f in trajectory],
    }
    labels = {
        "button_depth": ("Button depth (m)", SUCCESS["button"]),
        "slider_pos": ("Slider pos (m)", SUCCESS["slider"]),
        "chord": ("Chord min depth (m)", SUCCESS["chord"]),
        "wheel_angle": ("Wheel angle (rad)", SUCCESS["wheel"]),
    }

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                macro_block_size=None, quality=7)
    try:
        for i in range(len(trajectory)):
            fig, (ax_panel, ax_tel) = plt.subplots(1, 2, figsize=(8.0, 3.6), dpi=120)
            fig.patch.set_facecolor("#0e1116")

            # --- top-down panel schematic ---
            ax_panel.set_facecolor("#0e1116")
            ax_panel.add_patch(plt.Rectangle((-0.30, -0.15), 0.60, 0.30,
                                             facecolor="#272b33", edgecolor="#4a4f59"))
            for name, (cx, cy) in _CTRL_XY.items():
                ax_panel.scatter([cx], [cy], s=260, marker="s",
                                 facecolor="#3a3f4a", edgecolor="#7a8090", zorder=2)
                ax_panel.annotate(name, (cx, cy + 0.035), color="#aab0bb",
                                  ha="center", fontsize=7)
            for g, color in _TIP_COLORS.items():
                x, y, _ = trajectory[i]["tips"][g]
                ax_panel.scatter([x], [y], s=90, color=color, edgecolor="white",
                                 linewidth=0.6, zorder=4)
            ax_panel.set_xlim(-0.34, 0.34)
            ax_panel.set_ylim(-0.22, 0.22)
            ax_panel.set_aspect("equal")
            ax_panel.set_xticks([]); ax_panel.set_yticks([])
            ax_panel.set_title("FF Master Workstation (lite) - top view",
                               color="#e6e6e6", fontsize=9)

            # --- telemetry ---
            ax_tel.set_facecolor("#0e1116")
            for key, color in (("button_depth", "#e74c3c"), ("slider_pos", "#2ecc71"),
                               ("chord", "#f1c40f"), ("wheel_angle", "#9b59b6")):
                vals = series[key]
                label, thr = labels[key]
                norm = np.asarray(vals) / max(thr * 1.6, 1e-6)
                ax_tel.plot(ts[:i + 1], norm[:i + 1], color=color, lw=1.6, label=label)
            ax_tel.axhline(1.0 / 1.6, color="#888", ls="--", lw=0.8)
            ax_tel.set_xlim(0, ts[-1]); ax_tel.set_ylim(0, 1.25)
            ax_tel.set_xlabel("time (s)", color="#aab0bb", fontsize=8)
            ax_tel.set_yticks([])
            ax_tel.tick_params(colors="#aab0bb", labelsize=7)
            ax_tel.set_title("normalized task signals (dashed = success)",
                             color="#e6e6e6", fontsize=9)
            ax_tel.legend(loc="upper left", fontsize=6, facecolor="#1a1d24",
                          edgecolor="#3a3f4a", labelcolor="#d0d0d0")

            fig.tight_layout()
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            writer.append_data(frame)
            plt.close(fig)
    finally:
        writer.close()
    return out_path
