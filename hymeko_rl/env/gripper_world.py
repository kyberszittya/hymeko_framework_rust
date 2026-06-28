"""Pick-and-place scene helpers: compose a table + liftable box + place-target around a gripper MJCF.

Phase 0 of the grasping plan (``docs/plans/2026-06-22-pick-and-place-grasping/``). The gripper itself is
**described in HyMeKo** (``data/robotics/gripper.hymeko`` — a 3-DOF gantry base + two prismatic fingers,
emitted via ``emit_arm_mjcf``); this module only injects the *environment* around it (floor, the freejoint
object, the target site), exactly as ``planar_grasp_env.compose_planar_scene`` does for the coin. The
kinematic hypergraph is derived from the gripper MJCF by ``HypergraphState.from_mjcf``.

``make_gripper_mjcf`` is a hand-authored gripper-only fallback (no CLI needed), mirroring
``planar_grasp_env.make_planar_arms_mjcf``; the env defaults to the emitted HyMeKo gripper.
"""
from __future__ import annotations


def make_gripper_mjcf(*, open_gap: float = 0.035, finger_len: float = 0.06, base_z: float = 0.22) -> str:
    """A hand-authored gripper-only fallback — the CLI-free mirror of ``gripper.hymeko``.

    Matches the *emitted* topology exactly (gantry ``carriage_x -> carriage_y -> palm -> finger_l/r``, so
    5 link-vertices and the same body/joint names), so the env treats emitted and fallback identically.
    ``slide_z`` is relative to the gantry anchor at ``base_z`` (palm world-z ``= base_z + slide_z``)."""
    fr = f"{finger_len / 2:g}"
    return f'''<mujoco model="gripper">
  <compiler angle="radian"/>
  <default><joint damping="2.0"/><geom friction="1.2 0.05 0.001"/></default>
  <worldbody>
    <body name="carriage_x" pos="0 0 {base_z:g}">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <geom name="carriage_x" type="box" size="0.02 0.02 0.01" rgba="0.25 0.35 0.8 1" mass="0.1"/>
      <body name="carriage_y">
        <joint name="slide_y" type="slide" axis="0 1 0"/>
        <geom name="carriage_y" type="box" size="0.02 0.02 0.01" rgba="0.25 0.35 0.8 1" mass="0.1"/>
        <body name="palm">
          <joint name="slide_z" type="slide" axis="0 0 1"/>
          <geom name="palm" type="box" size="0.05 0.025 0.012" rgba="0.25 0.35 0.8 1" mass="0.3"/>
          <body name="finger_l" pos="-{open_gap:g} 0 -{fr}">
            <joint name="grip_l" type="slide" axis="1 0 0"/>
            <geom name="finger_l" type="box" size="0.006 0.018 {fr}" rgba="0.85 0.45 0.2 1" mass="0.02"/>
          </body>
          <body name="finger_r" pos="{open_gap:g} 0 -{fr}">
            <joint name="grip_r" type="slide" axis="1 0 0"/>
            <geom name="finger_r" type="box" size="0.006 0.018 {fr}" rgba="0.85 0.45 0.2 1" mass="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_slide_x" joint="slide_x" kp="60" kv="4"/>
    <position name="a_slide_y" joint="slide_y" kp="60" kv="4"/>
    <position name="a_slide_z" joint="slide_z" kp="120" kv="8"/>
    <position name="a_grip_l" joint="grip_l" kp="60" kv="3"/>
    <position name="a_grip_r" joint="grip_r" kp="60" kv="3"/>
  </actuator>
</mujoco>'''


def attach_gripper(arm_mjcf: str, *, ee_body: str = "tool", mount_z: float = 0.1,
                   open_gap: float = 0.035, finger_len: float = 0.06) -> str:
    """Attach a two-finger parallel-jaw gripper to ``ee_body`` of an *emitted arm* — the faithful
    decomposition: the SAME ``anthropomorphic_arm.hymeko`` as the reach scenario, plus an end-effector.

    Splices two prismatic fingers (``grip_l``/``grip_r``) + their position actuators into the arm's tool
    body, so the kinematic hypergraph spans the arm links AND the fingers. # Preconditions ``arm_mjcf`` has a
    ``<body name="{ee_body}" ...>`` and an ``</actuator>``. # Postconditions returns the arm+gripper MJCF."""
    fr = f"{finger_len / 2:g}"
    fingers = (
        f'<body name="finger_l" pos="-{open_gap:g} 0 {mount_z:g}">'
        f'<joint name="grip_l" type="slide" axis="1 0 0" damping="2"/>'
        f'<geom name="g_finger_l" type="box" size="0.006 0.018 {fr}" rgba="0.85 0.45 0.2 1" mass="0.02"/>'
        f'</body>'
        f'<body name="finger_r" pos="{open_gap:g} 0 {mount_z:g}">'
        f'<joint name="grip_r" type="slide" axis="1 0 0" damping="2"/>'
        f'<geom name="g_finger_r" type="box" size="0.006 0.018 {fr}" rgba="0.85 0.45 0.2 1" mass="0.02"/>'
        f'</body>')
    # insert the fingers before the tool body's closing </body> (the tool is a leaf body).
    anchor = arm_mjcf.find(f'name="{ee_body}"')
    if anchor < 0:
        raise ValueError(f"ee_body {ee_body!r} not found in arm MJCF")
    close = arm_mjcf.find("</body>", anchor)
    arm_mjcf = arm_mjcf[:close] + fingers + arm_mjcf[close:]
    acts = ('<position name="a_grip_l" joint="grip_l" kp="60" kv="3"/>'
            '<position name="a_grip_r" joint="grip_r" kp="60" kv="3"/>')
    return arm_mjcf.replace("</actuator>", acts + "</actuator>", 1)


def compose_pick_place_scene(gripper_mjcf: str, *, box_half: float = 0.02, box_mass: float = 0.05,
                             table_z: float = 0.0, target_xy: tuple[float, float] = (0.15, 0.0),
                             table_top: float | None = None, pedestal: float = 0.0,
                             box_z_half: float | None = None, target_bin: bool = False) -> str:
    """Inject a floor, a ``<freejoint>`` box object, and a place-target site into a gripper MJCF; with
    ``table_top`` set, also inject an elevated **table** so the box is grasped ABOVE the ground (the robot
    never reaches the floor — see the ground-collision concern).

    Works on either the emitted (HyMeKo) or the hand-authored gripper. Appended before ``</worldbody>``.
    # Preconditions ``gripper_mjcf`` has a ``</worldbody>``; ``table_top`` (if given) ``> table_z``.
    # Postconditions adds geoms ``floor``/``object`` (+ ``table`` if elevated), body ``object`` (freejoint
    #   ``obj``), site ``target``; the work surface is ``table_top`` if given, else ``table_z``."""
    tx, ty = target_xy
    surf = table_z if table_top is None else float(table_top)
    floor = (f'<geom name="floor" type="plane" material="grid" pos="0 0 {table_z:g}" size="2 2 0.05"/>')
    table = ""
    if table_top is not None:
        h = (surf - table_z) / 2.0                    # static box from the ground up to the work surface
        table = (f'<geom name="table" type="box" pos="{tx:g} {ty:g} {table_z + h:g}" '
                 f'size="0.20 0.26 {h:g}" rgba="0.55 0.42 0.30 1" friction="1.0 0.05 0.001"/>')
    bz = box_half if box_z_half is None else float(box_z_half)   # tall object → grasp upper body, clear table
    obj = (f'<body name="object" pos="0 0 {surf + bz:g}">'
           f'<freejoint name="obj"/>'
           f'<geom name="object" type="box" size="{box_half:g} {box_half:g} {bz:g}" '
           f'rgba="0.85 0.3 0.2 1" friction="1.2 0.05 0.001" mass="{box_mass:g}"/></body>')
    target = (f'<site name="target" type="cylinder" size="0.04 0.002" '
              f'pos="{tx:g} {ty:g} {surf + 0.002:g}" rgba="0.2 0.8 0.3 0.4"/>')
    if target_bin:                                     # an OPEN BOX at the target: the object must be LIFTED
        bw, bh, bt = 0.06, 0.05, 0.006                # over the walls and dropped in — sliding/pushing fails.
        wz = surf + bh / 2.0
        target += (
            f'<geom name="bin_xp" type="box" pos="{tx + bw:g} {ty:g} {wz:g}" size="{bt:g} {bw + bt:g} {bh/2:g}" rgba="0.2 0.6 0.35 0.8"/>'
            f'<geom name="bin_xn" type="box" pos="{tx - bw:g} {ty:g} {wz:g}" size="{bt:g} {bw + bt:g} {bh/2:g}" rgba="0.2 0.6 0.35 0.8"/>'
            f'<geom name="bin_yp" type="box" pos="{tx:g} {ty + bw:g} {wz:g}" size="{bw + bt:g} {bt:g} {bh/2:g}" rgba="0.2 0.6 0.35 0.8"/>'
            f'<geom name="bin_yn" type="box" pos="{tx:g} {ty - bw:g} {wz:g}" size="{bw + bt:g} {bt:g} {bh/2:g}" rgba="0.2 0.6 0.35 0.8"/>')
    stand = ""
    if pedestal > 0.0:                                 # cosmetic stand under the (raised) arm base
        ph = pedestal / 2.0
        stand = (f'<geom name="pedestal" type="box" pos="0 0 {ph:g}" size="0.10 0.10 {ph:g}" '
                 f'rgba="0.30 0.30 0.34 1" contype="0" conaffinity="0"/>')
    # Render quality: lit scene (the emitted MJCF has no lights → dark), so two fill lights go in the
    # worldbody unconditionally; a sky-gradient skybox + checker-floor material; soft headlight + haze.
    lights = ('<light pos="0.4 0.2 1.6" dir="-0.25 -0.1 -1" directional="false" diffuse="0.7 0.7 0.7" '
              'specular="0.2 0.2 0.2" castshadow="true"/>'
              '<light pos="-0.5 -0.4 1.2" dir="0.4 0.3 -1" diffuse="0.35 0.35 0.4" castshadow="false"/>')
    scene = gripper_mjcf.replace("</worldbody>",
                                 f"    {floor}{table}{stand}{obj}{target}{lights}\n  </worldbody>", 1)
    # skybox + checker-floor material: MERGE into the emitted <asset> (it exists, holding the link materials);
    # create one only if absent. Merging is essential — skipping left material 'grid' undefined.
    asset_defs = ('<texture name="skybox" type="skybox" builtin="gradient" rgb1="0.55 0.72 0.92" '
                  'rgb2="0.92 0.95 0.99" width="512" height="512"/>'
                  '<texture name="grid" type="2d" builtin="checker" rgb1="0.80 0.81 0.85" '
                  'rgb2="0.66 0.69 0.76" width="512" height="512"/>'
                  '<material name="grid" texture="grid" texrepeat="10 10" reflectance="0.08"/>')
    if "<asset>" in scene:
        scene = scene.replace("<asset>", "<asset>" + asset_defs, 1)
    else:
        scene = scene.replace("<worldbody>", "<asset>" + asset_defs + "</asset>\n  <worldbody>", 1)
    if "<visual" not in scene:                          # the emitted MJCF has none → add sky/haze/shadows
        scene = scene.replace("<worldbody>",
            '<visual><headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7" specular="0.15 0.15 0.15"/>'
            '<rgba haze="0.86 0.91 0.96 1"/><quality shadowsize="4096"/>'
            '<map shadowclip="2"/></visual>\n  <worldbody>', 1)
    if "<option" not in scene:                          # ensure gravity (the hand-authored gripper omits it)
        scene = scene.replace("<worldbody>",
                              '<option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>\n'
                              '  <worldbody>', 1)
    return scene
