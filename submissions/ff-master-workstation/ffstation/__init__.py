"""FF Master Workstation — bimanual whole-body operation of human-designed
industrial controls (hand-wheel valve, force-regulated button, lever, slider)
on the Faraday Future Master humanoid in MuJoCo.

Public modules:
  scene       - programmatic MjSpec scene (robot + workstation + sensors + cameras)
  kinematics  - joint maps, FK helpers, damped-least-squares IK (pos + orientation)
  controller  - gravity-compensated whole-body torque control (PD + Cartesian + force)
  tasks       - autonomous task primitives + the full start-up sequence FSM
  benchmark   - quantitative, multi-seed, measured-from-physics evaluation
  record_demo - narrated telemetry demo video
  dataio      - trajectory data-collection export (state / action / contact)
"""
__version__ = "1.0.0"
