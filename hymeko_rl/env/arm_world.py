"""The canonical 4-DOF arm scene — the *geometric description* role.

Kinematic chain (Listing A.6.1 topology, the same hypergraph as the kinematic
``.hymeko`` robot): base --j1(Z)-- shoulder --j2(Y)-- elbow --j3(Y)-- wrist
--j4(Z)-- flange. The body is fixed; only the **actuator interface** is parameterised
by ``control_mode`` so the env can ablate torque vs position vs velocity control
(torque is the headline — the standard MuJoCo-RL and real-robot interface, and what the
emitted ``<motor>`` arms use once B-005's axes are fixed).

(Follow-up dedup: ``hymeko_bench/paper_figs/sim_mujoco_scenario.py`` should import from
here. Phase 2 adds a parallel-jaw gripper + a graspable box + contacts for grasping.)
"""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import mujoco
import numpy as np

from hymeko_rl.env.constants import Physics

_REPO = Path(__file__).resolve().parents[2]

N_JOINTS = 4
CONTROL_MODES = ("torque", "position", "velocity")

# The actuator class set in <default>; the per-joint entries use the matching tag.
_ACTUATOR_DEFAULT = {
    "torque": '<motor ctrlrange="-25 25"/>',
    "position": '<position kp="40" kv="4" ctrlrange="-2.5 2.5"/>',
    "velocity": '<velocity kv="6" ctrlrange="-4 4"/>',
}
_ACTUATOR_TAG = {"torque": "motor", "position": "position", "velocity": "velocity"}

_BODY = """  <worldbody>
    <geom name="floor" type="plane" size="1.2 1.2 0.05"/>
    <body name="base_link" pos="0 0 0">
      <inertial pos="0 0 0.04" mass="1.5" diaginertia="0.005 0.005 0.003"/>
      <geom type="cylinder" size="0.06 0.04" pos="0 0 0.04"/>
      <body name="shoulder_link" pos="0 0 0.08">
        <inertial pos="0 0 0.06" mass="0.8" diaginertia="0.002 0.002 0.001"/>
        <joint name="j1" type="hinge" axis="0 0 1" pos="0 0 0"/>
        <geom type="cylinder" size="0.035 0.06" pos="0 0 0.06"/>
        <body name="elbow_link" pos="0 0 0.12">
          <inertial pos="0 0 0.12" mass="0.6" diaginertia="0.004 0.004 0.0005"/>
          <joint name="j2" type="hinge" axis="0 1 0" pos="0 0 0"/>
          <geom type="box" size="0.03 0.025 0.12" pos="0 0 0.12"/>
          <body name="wrist_link" pos="0 0 0.24">
            <inertial pos="0 0 0.09" mass="0.4" diaginertia="0.002 0.002 0.0003"/>
            <joint name="j3" type="hinge" axis="0 1 0" pos="0 0 0"/>
            <geom type="box" size="0.025 0.02 0.09" pos="0 0 0.09"/>
            <body name="flange_link" pos="0 0 0.18">
              <inertial pos="0 0 0.025" mass="0.15" diaginertia="0.0002 0.0002 0.0001"/>
              <joint name="j4" type="hinge" axis="0 0 1" pos="0 0 0"/>
              <geom type="cylinder" size="0.03 0.025" pos="0 0 0.025"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>"""


def make_arm_mjcf(control_mode: str = "torque") -> str:
    """The arm MJCF with the actuator interface for ``control_mode``.

    # Preconditions ``control_mode in CONTROL_MODES``.
    # Postconditions ``model.nu == model.nq == N_JOINTS``; the kinematic hypergraph
    (links + joints) is identical across modes — only the actuators differ.
    """
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"control_mode must be in {CONTROL_MODES}; got {control_mode!r}")
    tag = _ACTUATOR_TAG[control_mode]
    acts = "\n    ".join(
        f'<{tag} name="m{i}" joint="j{i}"/>' for i in range(1, N_JOINTS + 1))
    return f"""<mujoco model="hymeko_grasping_arm">
  <compiler angle="radian"/>
  <option {Physics.option_attrs()}/>
  <default>
    <joint damping="2.0" limited="true" range="-2.5 2.5"/>
    <geom friction="0.8 0.05 0.0005"/>
    {_ACTUATOR_DEFAULT[control_mode]}
  </default>
{_BODY}
  <actuator>
    {acts}
  </actuator>
</mujoco>"""


# Default scene = torque control (the headline interface).
ARM_MJCF = make_arm_mjcf("torque")


def with_collision_floor(mjcf: str, size: float = 2.0, z: float = 0.0) -> str:
    """Ensure the scene has a ground plane (at height ``z``) so ground contact is physically possible.

    Emitted arms (``emit_arm_mjcf``) carry no floor; the hand-authored ``make_arm_mjcf`` arm and a
    beautified render scene already do. **Idempotent:** if any ``plane`` geom is present the MJCF is
    returned unchanged, so this never adds a second floor. ``z`` lets the caller seat the plane just
    below the arm's home extent (the emitted arm's lower link clips ``z = 0`` at its mount).

    # Preconditions ``mjcf`` has a ``<worldbody>``.
    # Postconditions the result has exactly one ground plane (the original, or one injected here).
    """
    if 'type="plane"' in mjcf:
        return mjcf
    floor = f'<geom name="floor" type="plane" pos="0 0 {z:g}" size="{size:g} {size:g} 0.1"/>'
    return mjcf.replace("<worldbody>", "<worldbody>\n    " + floor, 1)


def slim_arm_collision(mjcf: str, scale: float = 0.4) -> str:
    """Shrink the radius of the arm's cylinder geoms so non-adjacent links stop *spuriously*
    self-colliding at rest.

    The emitted 6-DOF arm's fat cylinders (r=0.075) overlap given the lateral joint offsets, so a
    self-collision death would fire in ordinary poses. Slimming the radius removes those false
    positives while genuine folds (links brought within the slim radius) still collide. Only
    2-component ``size`` attributes (cylinders/capsules) are touched — the plane floor and box tool
    (3-component) are left alone. Visual and collision shrink together (the safety env does not render).

    # Preconditions ``0 < scale <= 1``.
    # Postconditions every cylinder geom's radius is multiplied by ``scale``; half-length unchanged.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1]; got {scale}")

    def shrink(m: "re.Match[str]") -> str:
        radius, half = float(m.group(1)), m.group(2)
        return f'size="{radius * scale:g} {half}"'

    return re.sub(r'size="([\d.]+) ([\d.]+)"', shrink, mjcf)


def decorate_scene(mjcf: str) -> str:
    """Add render decoration to an MJCF — a gradient skybox, a checkered ground material, soft lighting and
    haze — so the offscreen render is not a black void. Visual-only (no DOF added), so a model built from the
    decorated MJCF shares the original's ``qpos`` and can be driven by the undecorated env's state.

    # Preconditions ``mjcf`` has an ``<asset>`` block (emitted robots do). # Postconditions returns a valid
    MJCF that renders with a sky + (where a floor plane exists) a textured ground.
    """
    visual = ('  <visual>\n'
              '    <headlight diffuse="0.55 0.55 0.55" ambient="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>\n'
              '    <rgba haze="0.72 0.81 0.92 1"/>\n'
              '    <global azimuth="140" elevation="-22"/>\n'
              '  </visual>')
    assets = ('    <texture type="skybox" builtin="gradient" rgb1="0.45 0.62 0.82" rgb2="0.85 0.9 0.96"'
              ' width="512" height="512"/>\n'
              '    <texture name="grid" type="2d" builtin="checker" rgb1="0.52 0.55 0.6"'
              ' rgb2="0.7 0.72 0.77" width="300" height="300"/>\n'
              '    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.05"/>')
    light = ('    <light pos="1.0 -1.0 3.0" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7"'
             ' specular="0.2 0.2 0.2"/>')
    out = re.sub(r"(<option[^>]*/>)", r"\1\n" + visual, mjcf, count=1)
    out = out.replace("<asset>", "<asset>\n" + assets, 1)
    out = out.replace("<worldbody>", "<worldbody>\n" + light, 1)
    out = out.replace('type="plane"', 'type="plane" material="grid"', 1)   # texture an existing floor
    return out


def strip_actuators(mjcf: str, joint_names: "Iterable[str]") -> str:
    """Remove the ``<motor>`` actuators that drive the named joints — leaving those joints *passive*.

    The emitter actuates **every** non-fixed joint, but the canonical inverted pendulum is
    under-actuated: only the cart's rail is driven, the pole hinge swings freely. Dropping the pole's
    motor here (rather than in the ``.hymeko``) keeps the source description canonical and confines the
    emitter quirk to the env layer — the same pattern as :func:`with_collision_floor` /
    :func:`slim_arm_collision`.

    # Preconditions ``mjcf`` is a valid scene; each name in ``joint_names`` is a joint with a
    ``<motor joint="name" …/>`` actuator.
    # Postconditions exactly the motors whose ``joint=`` is in ``joint_names`` are removed; every other
    actuator is intact (so the model's ``nu`` drops by the number of distinct names actually matched).
    """
    targets = set(joint_names)
    if not targets:
        return mjcf
    # A self-closing <motor .../> whose joint= attribute is one of the targets (attribute order-agnostic).
    pattern = re.compile(r'\s*<motor\b[^>]*\bjoint="([^"]+)"[^>]*/>')
    return pattern.sub(lambda m: "" if m.group(1) in targets else m.group(0), mjcf)


def actuated_dof_addrs(model: "mujoco.MjModel") -> np.ndarray:
    """The ``qvel``/``qacc`` addresses of the joints driven by actuators (transmission type ``joint``).

    Lets a reward term read the *actuated* joints' velocities/accelerations (for smoothness/effort penalties)
    without each env re-deriving the mapping. # Postconditions returns an ``int64`` array of length ``nu``
    (one dof address per actuator) for single-dof (hinge/slide) transmissions."""
    out = []
    for i in range(int(model.nu)):
        if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            out.append(int(model.jnt_dofadr[int(model.actuator_trnid[i, 0])]))
    return np.asarray(out, dtype=np.int64)


def actuator_vertices(model: "mujoco.MjModel") -> np.ndarray:
    """Each actuator's **hypergraph vertex** — the driven joint's child body remapped ``b -> b-1`` (the same
    body->vertex convention as :meth:`hymeko_rl.agents.hypergraph_state.HypergraphState.from_mjcf`).

    Lets a per-node actor head read each joint's action from *its own* graph vertex. # Preconditions every
    actuator is a single-dof ``joint`` transmission. # Postconditions returns an ``int64`` array of length
    ``nu``, each entry in ``[0, nbody-1)``."""
    out = []
    for i in range(int(model.nu)):
        if int(model.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"actuator {i} is not a joint transmission; per-node mapping needs joint actuators")
        out.append(int(model.jnt_bodyid[int(model.actuator_trnid[i, 0])]) - 1)
    return np.asarray(out, dtype=np.int64)


def load_arm(control_mode: str = "torque") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the arm MJCF for ``control_mode`` and return ``(model, data)``."""
    model = mujoco.MjModel.from_xml_string(make_arm_mjcf(control_mode))
    return model, mujoco.MjData(model)


# ── the canonical .hymeko → MJCF bridge ──────────────────────────────────────
# The hand-authored arm above is the fallback / control-mode ablation scene. The
# *canonical* scene comes from a ``.hymeko`` robot, emitted via the same CLI path
# that produces the kinematic hypergraph and the observation space — one source,
# three products (the Kato framing). Routes through the CLI because the PyO3 import
# resolver is still B-003-blocked; ``hymeko emit`` resolves imports correctly.


def _hymeko_cli() -> Path:
    """Locate the built ``hymeko`` CLI binary under ``target/{debug,release}``.

    # Postconditions Returns an existing executable path.
    # Errors ``FileNotFoundError`` if the CLI has not been built.
    """
    exe_name = "hymeko.exe" if os.name == "nt" else "hymeko"
    for profile in ("debug", "release"):
        candidate = _REPO / "target" / profile / exe_name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"hymeko CLI not found under {_REPO / 'target'}/(debug|release)/{exe_name}; "
        "build it with `cargo build -p hymeko_cli`.")


def emit_arm_mjcf(hymeko_path: str | os.PathLike[str], *, name: str = "arm",
                  control_mode: str = "torque") -> str:
    """Emit the MuJoCo scene for a ``.hymeko`` robot via the canonical CLI.

    The emitter produces ``<motor>`` (torque) actuators, nested bodies with the source's
    per-joint axes / origins / limits, and the EE flange body. For ``position`` / ``velocity``
    control the torque actuators are retargeted to PD / velocity servos (the emitter only
    emits torque) and joint damping is added for stable closed-loop tracking — mirroring
    :func:`make_arm_mjcf`'s control-mode ablation. The *geometry* is always canonical (from
    the ``.hymeko``); only the actuator interface is parameterised.

    # Preconditions ``hymeko_path`` exists, the CLI is built, ``control_mode in CONTROL_MODES``.
    # Postconditions Returns a well-formed MJCF string loadable by MuJoCo.
    # Errors ``FileNotFoundError`` (missing source or CLI); ``RuntimeError`` (emit failed).
    """
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"control_mode must be in {CONTROL_MODES}; got {control_mode!r}")
    path = Path(hymeko_path)
    if not path.is_file():
        raise FileNotFoundError(f"hymeko robot source not found: {path}")
    proc = subprocess.run(
        [str(_hymeko_cli()), "emit", "-f", "mjcf", str(path), "-n", name],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`hymeko emit -f mjcf {path}` failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}")
    mjcf = proc.stdout
    return mjcf if control_mode == "torque" else _retarget_actuators(mjcf, control_mode)


def _retarget_actuators(mjcf: str, control_mode: str) -> str:
    """Rewrite the emitted torque ``<motor>`` actuators to position / velocity servos and add
    joint damping. Position ctrlrange is the joint angle range (±π); gains match
    :data:`_ACTUATOR_DEFAULT`."""
    servo = {
        "position": r'<position \1 kp="40" kv="4" ctrlrange="-3.1416 3.1416"/>',
        "velocity": r'<velocity \1 kv="6" ctrlrange="-4 4"/>',
    }[control_mode]
    mjcf = re.sub(r'<motor (name="act_\w+" joint="\w+")[^/]*/>', servo, mjcf)
    # damping stabilises the PD/velocity loop (the emitter does not declare it).
    return mjcf.replace('<joint name=', '<joint damping="2.0" name=')
