"""Synthetic kinematic fixtures for HSiKAN validation.

Builds small URDF strings (returned as XML text or written to disk)
for canonical mechanisms with known cycle structure:

  * **four_bar_linkage**       — a single k=4 closed loop
  * **stewart_platform_6dof**  — 6 closed loops, k=4 each
  * **delta_robot_3rrr**       — 3 closed loops sharing the end-effector
  * **serial_arm_n**           — serial chain (no cycles)

Useful as smoke fixtures for the kinematic graph adapter and as
synthetic training data for mechanism-property prediction tasks.
"""
from __future__ import annotations

import tempfile
from pathlib import Path


_FOUR_BAR_URDF = """<?xml version="1.0"?>
<robot name="four_bar">
  <link name="ground"/>
  <link name="crank"/>
  <link name="coupler"/>
  <link name="rocker"/>

  <joint name="j_ground_crank" type="revolute">
    <parent link="ground"/><child link="crank"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2"/>
  </joint>
  <joint name="j_crank_coupler" type="revolute">
    <parent link="crank"/><child link="coupler"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2"/>
  </joint>
  <joint name="j_coupler_rocker" type="revolute">
    <parent link="coupler"/><child link="rocker"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2"/>
  </joint>
  <joint name="j_rocker_ground" type="revolute">
    <parent link="rocker"/><child link="ground"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2"/>
  </joint>
</robot>
"""


_STEWART_PLATFORM_URDF = """<?xml version="1.0"?>
<robot name="stewart_platform">
  <link name="base"/>
  <link name="ee"/>
  <link name="leg1_lower"/><link name="leg1_upper"/>
  <link name="leg2_lower"/><link name="leg2_upper"/>
  <link name="leg3_lower"/><link name="leg3_upper"/>
  <link name="leg4_lower"/><link name="leg4_upper"/>
  <link name="leg5_lower"/><link name="leg5_upper"/>
  <link name="leg6_lower"/><link name="leg6_upper"/>
  <!-- Each of 6 legs: base→lower (revolute), lower→upper (prismatic),
        upper→ee (revolute). Each leg forms a k=4 loop with the base
        and ee. -->
""" + "".join(
    f"""
  <joint name="j{i}_base"  type="revolute">
    <parent link="base"/><child link="leg{i}_lower"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="j{i}_strut" type="prismatic">
    <parent link="leg{i}_lower"/><child link="leg{i}_upper"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1" effort="100" velocity="1"/>
  </joint>
  <joint name="j{i}_ee"    type="revolute">
    <parent link="leg{i}_upper"/><child link="ee"/>
    <axis xyz="0 0 1"/>
  </joint>"""
    for i in range(1, 7)
) + "\n</robot>\n"


_DELTA_ROBOT_URDF = """<?xml version="1.0"?>
<robot name="delta_robot">
  <link name="base"/>
  <link name="ee"/>
  <link name="arm1_upper"/><link name="arm1_lower"/>
  <link name="arm2_upper"/><link name="arm2_lower"/>
  <link name="arm3_upper"/><link name="arm3_lower"/>
""" + "".join(
    f"""
  <joint name="j{i}_shoulder" type="revolute">
    <parent link="base"/><child link="arm{i}_upper"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="j{i}_elbow" type="revolute">
    <parent link="arm{i}_upper"/><child link="arm{i}_lower"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="j{i}_wrist" type="revolute">
    <parent link="arm{i}_lower"/><child link="ee"/>
    <axis xyz="0 1 0"/>
  </joint>"""
    for i in range(1, 4)
) + "\n</robot>\n"


def _serial_arm_urdf(n_links: int) -> str:
    """Serial open-chain arm with n_links + 1 bodies and n_links revolute
    joints. No closed loops — should give 0 cycles at all arities."""
    parts = ['<?xml version="1.0"?>', f'<robot name="serial_arm_{n_links}">',
              '  <link name="link0"/>']
    for i in range(1, n_links + 1):
        parts.append(f'  <link name="link{i}"/>')
    for i in range(1, n_links + 1):
        parts.append(
            f'  <joint name="j{i}" type="revolute">\n'
            f'    <parent link="link{i-1}"/><child link="link{i}"/>\n'
            f'    <axis xyz="0 0 1"/>\n'
            f'  </joint>'
        )
    parts.append('</robot>')
    return "\n".join(parts)


_FIXTURES = {
    "four_bar":      _FOUR_BAR_URDF,
    "stewart":       _STEWART_PLATFORM_URDF,
    "delta_3rrr":    _DELTA_ROBOT_URDF,
    "serial_4":      _serial_arm_urdf(4),
    "serial_7":      _serial_arm_urdf(7),
}


def write_fixture(name: str, out_dir: str | Path | None = None) -> Path:
    """Write fixture URDF to disk and return its path. Without ``out_dir``,
    uses a temp file (caller should keep the path)."""
    if name not in _FIXTURES:
        raise KeyError(f"unknown fixture {name!r}; "
                        f"available: {list(_FIXTURES.keys())}")
    if out_dir is None:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{name}.urdf", delete=False,
        )
        f.write(_FIXTURES[name])
        f.close()
        return Path(f.name)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.urdf"
    path.write_text(_FIXTURES[name])
    return path


if __name__ == "__main__":
    from .graph import urdf_to_signed_graph, kinematic_loop_summary
    print("=== Synthetic mechanism cycle structure ===")
    print(f"{'fixture':<14s}  {'links':>5s}  {'joints':>6s}  "
          f"{'k=3':>4s}  {'k=4':>4s}  {'k=5':>4s}  {'k=6':>4s}")
    for name in _FIXTURES:
        path = write_fixture(name)
        try:
            g, links, joints = urdf_to_signed_graph(path)
            summary = kinematic_loop_summary(g, joints, max_k=6)
            cycles = summary["cycles_per_arity"]
            print(f"{name:<14s}  {summary['n_links']:>5d}  "
                  f"{summary['n_joints']:>6d}  "
                  f"{cycles.get(3, 0):>4d}  {cycles.get(4, 0):>4d}  "
                  f"{cycles.get(5, 0):>4d}  {cycles.get(6, 0):>4d}")
        finally:
            path.unlink(missing_ok=True)
