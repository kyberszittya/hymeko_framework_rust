"""The canonical manipulation-rig generator (R11.7A unification, stage U3).

One entry point — :func:`build_manipulation_rig` — turns a *typed* object description
(:class:`hymeko_rl.env.object_spec.ObjectSpec`, itself read from the HyMeKo scene) plus a
:class:`RobotContract` into a reconstructed environment and a stable-name registry. Both the R11.6C
exact-zero acquisition path (via ``bv_identification_benchmark._make_env``) and the video/object-variant
tools consume this instead of hand-passing loose ``coin_shape``/``hx``/``hy`` literals, so the object's
identity lives in one place and downstream code resolves the manipuland by a stable handle rather than a
shape-specific MuJoCo name.

This is a *thin* generator: the physically-authoritative MuJoCo model is still built by
:func:`hymeko_rl.env.planar_grasp_env.compose_planar_scene` (reached through the existing
``reconstruct_handoff`` chain). The generator adds (1) the ObjectSpec→builder mapping in one spot and
(2) the :class:`RigRegistry` of stable semantic handles. It does **not** change the model, the collision
contract, or the delivery rollout — those remain the frozen R11.6C stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco  # type: ignore[import-untyped]  # mujoco ships no stubs / py.typed marker

from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.env.object_spec import ObjectSpec


@dataclass(frozen=True)
class RobotContract:
    """The robot/embodiment half of the scene contract — fixed for a given experiment family, orthogonal
    to the manipulated object. R11.6C exact-zero delivery uses the ball-tip embodiment
    (:data:`geom` ``"POINT"`` + a ball-tip ``arm_mjcf_transform``). ``geom=None`` selects the canonical
    concave-clamp eval robot.

    The ``arm_mjcf_transform`` is supplied by the caller (it lives in a sys.path-injected experiment
    module), keeping this generator free of that import and making the robot contract explicit data.
    """

    geom: str | None = None
    arm_mjcf_transform: Any = None
    label: str = "clamp"


@dataclass(frozen=True)
class RigRegistry:
    """Stable semantic handles for the compiled model, independent of object shape. The manipuland geom
    and body are named ``"disk"`` for every :class:`hymeko_rl.env.object_spec.Shape`, so consumers resolve
    the object through :meth:`object_geom_id` / :meth:`object_body_id` rather than hard-coding the literal.

    # Invariants the compiled model contains the declared handles; :meth:`resolve` fails loudly otherwise.
    """

    object_body: str = "disk"
    object_geom: str = "disk"
    object_site: str | None = None

    @classmethod
    def resolve(cls, model: Any) -> "RigRegistry":
        """Verify the stable object handle is present in ``model`` (guards against a silent rename in a
        future ``compose_planar_scene`` shape branch).

        # Errors ``ValueError`` if the compiled model lacks the ``"disk"`` geom/body handle.
        """
        reg = cls()
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, reg.object_geom) < 0:
            raise ValueError(f"compiled model lacks the stable object geom handle {reg.object_geom!r}")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, reg.object_body) < 0:
            raise ValueError(f"compiled model lacks the stable object body handle {reg.object_body!r}")
        return reg

    def object_geom_id(self, model: Any) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.object_geom))

    def object_body_id(self, model: Any) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.object_body))


@dataclass
class ManipulationRig:
    """A reconstructed environment plus the data that generated it. ``rl`` is the ``CoinRL4Dof`` env,
    ``gate`` its handoff gate — exactly what ``reconstruct_handoff`` returns — so this is a drop-in for
    the ``(rl, gate)`` pairs the pipeline already threads."""

    rl: Any
    gate: Any
    registry: RigRegistry
    object_spec: ObjectSpec
    robot: RobotContract


def build_manipulation_rig(pi0: Any, demo_state: Any, *, object_spec: ObjectSpec,
                           robot: RobotContract, horizon: int = 360) -> ManipulationRig:
    """Reconstruct one manipulation rig for ``demo_state`` under ``object_spec`` + ``robot``.

    The single ObjectSpec→builder mapping: ``coin_shape``/``disk_radius_override``/
    ``disk_radius_y_override`` come from the object; ``geom``/``arm_mjcf_transform`` from the robot.

    # Preconditions ``object_spec`` is a valid :class:`ObjectSpec`; ``demo_state`` is a reconstructable
      handoff (a ``LateStart``). The full object (shape/size AND mass/friction) is threaded as a single
      carrier through the reconstruct chain — no axis is silently dropped.
    # Postconditions returns a :class:`ManipulationRig` whose ``rl``/``gate`` equal
      ``reconstruct_handoff(..., object_spec=object_spec)[:2]``, and whose registry's handles exist in the
      compiled model. For the reference coin the built model is bit-identical to the pre-unification
      ``reconstruct_handoff(coin_shape="cylinder", disk_radius_override=0.02, geom=robot.geom, …)``.
    # Errors ``ValueError`` from :meth:`RigRegistry.resolve` if the object handle is missing.
    """
    rl, gate, _h, _r = reconstruct_handoff(
        pi0, demo_state, horizon=horizon, object_spec=object_spec,
        arm_mjcf_transform=robot.arm_mjcf_transform, geom=robot.geom)
    registry = RigRegistry.resolve(rl.inner.model)
    return ManipulationRig(rl=rl, gate=gate, registry=registry, object_spec=object_spec, robot=robot)
