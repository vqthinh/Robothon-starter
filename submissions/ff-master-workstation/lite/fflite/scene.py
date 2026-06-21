"""Self-contained MuJoCo control-panel scene for FF Master Workstation (lite).

A fixed five-fingertip array operates a panel of human-designed controls:
a spring-return push button, a throttle slider, a two-key chord, and a
rotary thumbwheel. The whole model is generated in code as a single MJCF
string -- no external meshes, no GPU, CPU-only. This makes the benchmark
trivially reproducible on any machine.
"""
from __future__ import annotations

import mujoco

EPISODE_DURATION_S = 10.0

# Fingertip home x-positions, each above the control it operates.
FINGERS = {"thumb": -0.15, "index": -0.08, "middle": 0.03, "ring": 0.07, "pinky": 0.15}

# Control x-positions on the panel (must match FINGERS for clean reaches).
CTRL_POS = {"button": -0.15, "slider": -0.08, "key_g": 0.03, "key_b": 0.07, "wheel": 0.15}

# Success thresholds (metric -> minimum value), every number measured live.
SUCCESS = {"button": 0.02, "slider": 0.04, "chord": 0.012, "wheel": 0.4}

SENSORS = ["button_depth", "slider_pos", "keyg_depth", "keyb_depth", "wheel_angle"]


def build_xml() -> str:
    """Return the full MJCF for the lite control-panel scene."""
    finger_bodies = ""
    for name, x in FINGERS.items():
        finger_bodies += f"""
        <body name="ft_{name}" pos="{x} 0 0.20">
          <joint name="{name}_x" type="slide" axis="1 0 0" range="-0.24 0.24" damping="8"/>
          <joint name="{name}_y" type="slide" axis="0 1 0" range="-0.24 0.24" damping="8"/>
          <joint name="{name}_z" type="slide" axis="0 0 1" range="-0.22 0.04" damping="8"/>
          <geom name="tip_{name}" type="sphere" size="0.012" rgba="0.95 0.75 0.2 1"
                condim="4" friction="2.2 0.04 0.002" solref="0.008 1"/>
          <site name="s_{name}" type="sphere" size="0.013" rgba="0 0 0 0"/>
        </body>"""

    actuators = ""
    for name in FINGERS:
        for axis, rng in (("x", "-0.24 0.24"), ("y", "-0.24 0.24"), ("z", "-0.22 0.04")):
            actuators += (f'    <position name="{name}_{axis}" joint="{name}_{axis}" '
                          f'kp="550" ctrlrange="{rng}"/>\n')

    touch = "".join(f'    <touch name="touch_{n}" site="s_{n}"/>\n' for n in FINGERS)

    return f"""
<mujoco model="ff_master_lite">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="720" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 1.3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="cam" pos="0 -0.45 0.55" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom name="panel" type="box" pos="0 0 0" size="0.30 0.15 0.02" rgba="0.24 0.26 0.3 1"/>

    <body name="button" pos="{CTRL_POS['button']} 0 0.075">
      <joint name="button_j" type="slide" axis="0 0 1" range="-0.035 0" stiffness="90" damping="3"/>
      <geom name="button_g" type="box" size="0.024 0.024 0.018" rgba="0.85 0.2 0.2 1"
            condim="4" friction="1.5 0.02 0.001"/>
    </body>

    <body name="slider" pos="{CTRL_POS['slider']} 0 0.04">
      <joint name="slider_j" type="slide" axis="1 0 0" range="0 0.075" damping="2"/>
      <geom name="slider_g" type="box" size="0.014 0.024 0.016" rgba="0.2 0.6 0.85 1"
            condim="4" friction="1.5 0.02 0.001"/>
    </body>

    <body name="key_g" pos="{CTRL_POS['key_g']} 0 0.065">
      <joint name="keyg_j" type="slide" axis="0 0 1" range="-0.03 0" stiffness="80" damping="3"/>
      <geom name="keyg_g" type="box" size="0.016 0.02 0.016" rgba="0.3 0.8 0.4 1"
            condim="4" friction="1.5 0.02 0.001"/>
    </body>

    <body name="key_b" pos="{CTRL_POS['key_b']} 0 0.065">
      <joint name="keyb_j" type="slide" axis="0 0 1" range="-0.03 0" stiffness="80" damping="3"/>
      <geom name="keyb_g" type="box" size="0.016 0.02 0.016" rgba="0.3 0.5 0.9 1"
            condim="4" friction="1.5 0.02 0.001"/>
    </body>

    <body name="wheel" pos="{CTRL_POS['wheel']} 0 0.045">
      <joint name="wheel_j" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="wheel_g" type="cylinder" fromto="0 -0.022 0 0 0.022 0" size="0.03"
            rgba="0.7 0.6 0.2 1" condim="4" friction="2.5 0.05 0.002"/>
    </body>
{finger_bodies}
  </worldbody>
  <actuator>
{actuators}  </actuator>
  <sensor>
{touch}    <jointpos name="button_depth" joint="button_j"/>
    <jointpos name="slider_pos" joint="slider_j"/>
    <jointpos name="keyg_depth" joint="keyg_j"/>
    <jointpos name="keyb_depth" joint="keyb_j"/>
    <jointpos name="wheel_angle" joint="wheel_j"/>
  </sensor>
</mujoco>"""


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml())


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [model.actuator(i).name for i in range(model.nu)]


def sensor_value(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    return float(data.sensor(name).data[0])


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)
