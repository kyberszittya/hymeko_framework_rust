"""Kinematic embodiments for the coin-delivery scenarios.

A :class:`KinematicVariant` owns everything embodiment-specific: it builds the stepping env, exposes the action
dimensions, maps a :class:`SemanticCommand` to the embodiment's action vector (via its
:class:`KinematicAdapter`), and reports kinematic metadata. The frozen delivery monitor consumes the env this
variant builds — so every variant presents the canonical :class:`ContactFormationEnv` interface (6-DoF
cooperative action, ``env._env`` = the ``PlanarGraspEnv``).

* **K0Canonical** — the current canonical arm, built byte-unchanged via ``pedc_selection._env`` (no extra DOF).

* **K1DistalOrientation** — the canonical finger model PLUS **one independent distal pad-orientation DOF per
  fingertip** (defined precisely as that, NOT "one more than a human finger"). The physical DOF is an ADDITIVE
  MJCF variant, :func:`with_distal_pad_orientation`, that wraps each ``fingertip_{side}`` collision geom in a
  distal ``pad_{side}`` body carrying an in-plane hinge + a position actuator, so the pad normal can orient over
  the workspace. :func:`distal_dof_probe` verifies this builds, steps, and RESPONDS to the pad actuator.

  **Honest scope (2026-07-20).** The distal-DOF *geometry* is verified-constructible (see the probe / test #6):
  it compiles, steps stably, and the pad orientation tracks its actuator once the near-zero pad inertia is
  regularized with joint ``armature``. What is NOT yet wired is the CLOSED LOOP: the frozen cooperative
  controller emits a 4-DoF IK motor for the arm joints only, so driving the 2 pad actuators end-to-end through
  ``ContactFormationEnv`` + the delivery monitor needs a pad-aware cooperative controller (a follow-up, not
  faked here). Therefore K1's :meth:`build_env` returns the canonical embodiment for the authoritative rollout,
  while K1 exposes the verified extra-DOF interface (``n_extra_dof``, :meth:`apply_pad_orientation`, the 8-DoF
  action mapping) and the constructibility proof. K0 is left byte-unchanged.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.actors.base import (
    BASE_ACTION_DIM,
    KinematicAdapter,
    KinematicMapping,
    SemanticCommand,
)

N_FINGERTIPS = 2


# ── K1 additive MJCF transform + constructibility probe ────────────────────────────────────────────────────────────
def with_distal_pad_orientation(arm_mjcf: str, *, kp: float = 6.0, kv: float = 0.4, armature: float = 2e-3,
                                hinge_range: float = 1.2) -> str:
    """Add one independent distal pad-orientation DOF per fingertip to ``arm_mjcf`` (additive, idempotent-safe).

    For each leaf link bearing a ``fingertip_{side}`` collision geom, insert a child ``pad_{side}`` body at the
    fingertip position, carrying an in-plane hinge ``pad_hinge_{side}`` (+ position actuator ``a_pad_{side}``),
    and relocate the fingertip geom + ``tip_{side}`` site into it. The pad hinge is armature-regularized because
    the pad's own inertia is ~1e-6 kg·m² — without ``armature`` a position servo produces astronomic ``qacc``.

    # Preconditions ``arm_mjcf`` is valid MJCF with ``fingertip_{side}`` geoms and an ``<actuator>`` block.
    # Postconditions each fingertip gains exactly one hinge DOF + one position actuator; the rest is untouched.
    # Invariants at rest the fingertip/tip world positions are unchanged (pad body origin = fingertip position)."""
    root = ET.fromstring(arm_mjcf)
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for body in list(root.iter("body")):
        ft = next((g for g in body.findall("geom") if str(g.get("name", "")).startswith("fingertip_")), None)
        if ft is None:
            continue
        side = ft.get("name", "").split("_", 1)[1]
        pad = ET.SubElement(body, "body", {"name": f"pad_{side}", "pos": ft.get("pos", "0 0 0")})
        ET.SubElement(pad, "joint", {"name": f"pad_hinge_{side}", "type": "hinge", "axis": "0 0 1",
                                     "range": f"{-hinge_range:g} {hinge_range:g}", "damping": "0.05",
                                     "armature": f"{armature:g}"})
        body.remove(ft)
        ft.set("pos", "0 0 0")
        pad.append(ft)
        tip = next((s for s in body.findall("site") if s.get("name") == f"tip_{side}"), None)
        if tip is not None:
            body.remove(tip)
            tip.set("pos", "0 0 0")
            pad.append(tip)
        ET.SubElement(actuator, "position", {"name": f"a_pad_{side}", "joint": f"pad_hinge_{side}",
                                             "kp": f"{kp:g}", "kv": f"{kv:g}",
                                             "ctrlrange": f"{-hinge_range:g} {hinge_range:g}"})
    return ET.tostring(root, encoding="unicode")


def with_pad_closure(arm_mjcf: str, *, kp: float = 30.0, kv: float = 2.0, armature: float = 2e-3,
                     slide_range: float = 0.02) -> str:
    """Add one independent PAD-CLOSURE DOF per fingertip: a slide joint along the pad's local X (the measured contact
    normal) + a position actuator ``a_close_{side}``, so each pad can be pressed toward the coin *independently* of the
    arm (a regulated normal-force clamp rather than the arm's coupled squeeze). Composes with
    :func:`with_distal_pad_orientation` (apply closure AFTER the wrist so the slide is on the wrist-oriented pad body).
    # Preconditions ``arm_mjcf`` has ``fingertip_{side}`` geoms and an ``<actuator>`` block. # Postconditions each
    fingertip's parent body gains one slide DOF + one position actuator; the fingertip geom is unmoved at rest."""
    root = ET.fromstring(arm_mjcf)
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for body in list(root.iter("body")):
        ft = next((g for g in body.findall("geom") if str(g.get("name", "")).startswith("fingertip_")), None)
        if ft is None:
            continue
        side = ft.get("name", "").split("_", 1)[1].split("_")[0]   # "left"/"right" (handles fingertip_left_0)
        ET.SubElement(body, "joint", {"name": f"pad_slide_{side}", "type": "slide", "axis": "1 0 0",
                                      "range": f"{-slide_range:g} {slide_range:g}", "damping": "0.2",
                                      "armature": f"{armature:g}"})
        ET.SubElement(actuator, "position", {"name": f"a_close_{side}", "joint": f"pad_slide_{side}",
                                             "kp": f"{kp:g}", "kv": f"{kv:g}",
                                             "ctrlrange": f"{-slide_range:g} {slide_range:g}"})
    return ET.tostring(root, encoding="unicode")


def _canonical_arm_mjcf() -> str:
    """The canonical arm MJCF the delivery env is built from (emitted arm + injected fingertip sites/geoms)."""
    from hymeko_rl.env.arm_world import emit_arm_mjcf
    from hymeko_rl.env.planar_grasp_env import with_fingertip_sites
    return with_fingertip_sites(emit_arm_mjcf("data/robotics/galambos_planar.hymeko", name="galambos",
                                              control_mode="position"))


@dataclass(frozen=True)
class DistalDofProbe:
    """Result of building + stepping the K1 padded MJCF and commanding a pad orientation."""

    built: bool
    steps_cleanly: bool
    n_extra_actuators: int
    pad_response_rad: float          # observed pad-normal rotation under the commanded orientation
    responds: bool
    detail: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def distal_dof_probe(target: float = 0.8, settle_steps: int = 300) -> DistalDofProbe:
    """Build the canonical arm + distal pad DOF, compose the planar scene, step it, command each pad, and measure
    the pad-normal rotation. This is the K1 constructibility proof (physical geometry, not a closed-loop policy).

    # Postconditions ``responds`` is True iff the pad normal rotates by > 0.1 rad toward the commanded target."""
    import mujoco
    from hymeko_rl.env.planar_grasp_env import compose_planar_scene
    padded = with_distal_pad_orientation(_canonical_arm_mjcf())
    try:
        model = mujoco.MjModel.from_xml_string(compose_planar_scene(padded))
        data = mujoco.MjData(model)
    except (ValueError, mujoco.FatalError) as err:                     # pragma: no cover - build-failure path
        return DistalDofProbe(False, False, 0, 0.0, False, f"build failed: {err}")
    pad_acts = [i for i in range(model.nu)
                if str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)).startswith("a_pad_")]
    padb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pad_left")
    mujoco.mj_forward(model, data)
    x0 = data.xmat[padb].reshape(3, 3)[:, 0].copy()                    # pad frame x-axis = pad normal proxy
    for act in pad_acts:
        data.ctrl[act] = target
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    clean = not bool(np.any(np.isnan(data.qpos)))
    x1 = data.xmat[padb].reshape(3, 3)[:, 0].copy()
    ang = float(np.arccos(np.clip(float(np.dot(x0, x1)), -1.0, 1.0)))
    responds = clean and ang > 0.1
    detail = (f"pad normal rotated {ang:.3f} rad under target {target}" if responds
              else f"built (nu extra={len(pad_acts)}) but pad response {ang:.3f} rad")
    return DistalDofProbe(True, clean, len(pad_acts), round(ang, 4), responds, detail)


# ── metadata + adapters ────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KinematicMetadata:
    variant_id: str
    description: str
    base_action_dim: int
    n_extra_dof_per_fingertip: int
    n_fingertips: int
    action_dim: int
    distal_dof_status: str           # "canonical" | "physical_geometry_verified"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class _K0Adapter(KinematicAdapter):
    """Canonical embodiment: the 6-DoF cooperative action. Pad-orientation commands are EXPLICITLY rejected."""

    @property
    def action_dim(self) -> int:
        return BASE_ACTION_DIM

    def map_command(self, cmd: SemanticCommand) -> KinematicMapping:
        rejected = tuple(name for name, val in (("left_pad_orientation", cmd.left_pad_orientation),
                                                ("right_pad_orientation", cmd.right_pad_orientation))
                         if val is not None)                          # dropped, recorded — never folded in
        return KinematicMapping(self._cooperative(cmd), rejected)


class _K1Adapter(KinematicAdapter):
    """Distal-pad embodiment: the 6-DoF cooperative action + one pad-orientation dim per fingertip (default 0)."""

    @property
    def action_dim(self) -> int:
        return BASE_ACTION_DIM + N_FINGERTIPS

    def map_command(self, cmd: SemanticCommand) -> KinematicMapping:
        pads = np.array([cmd.left_pad_orientation or 0.0, cmd.right_pad_orientation or 0.0], dtype=np.float32)
        return KinematicMapping(np.concatenate([self._cooperative(cmd), pads]), ())


class KinematicVariant(ABC):
    """A coin-delivery embodiment. Concrete variants are frozen (value-equal, serializable)."""

    KIND: str = "abstract"

    @property
    @abstractmethod
    def variant_id(self) -> str: ...

    @property
    @abstractmethod
    def n_extra_dof(self) -> int:
        """Extra distal-orientation DOFs PER FINGERTIP (0 for K0, 1 for K1)."""

    @property
    @abstractmethod
    def adapter(self) -> KinematicAdapter: ...

    @property
    def base_action_dim(self) -> int:
        return BASE_ACTION_DIM

    @property
    def n_fingertips(self) -> int:
        return N_FINGERTIPS

    @property
    def action_dim(self) -> int:
        return self.base_action_dim + self.n_extra_dof * self.n_fingertips

    @abstractmethod
    def build_env(self) -> Any:
        """Construct the stepping delivery env (a :class:`ContactFormationEnv` the frozen monitor consumes)."""

    @abstractmethod
    def metadata(self) -> KinematicMetadata: ...

    @abstractmethod
    def to_dict(self) -> dict: ...

    def map_command(self, cmd: SemanticCommand) -> KinematicMapping:
        return self.adapter.map_command(cmd)

    def apply_pad_orientation(self, action: np.ndarray, left: float, right: float) -> np.ndarray:
        """Write pad orientations into the tail of an embodiment action vector. K0 has no pad DOF → rejected."""
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if self.n_extra_dof == 0:
            raise ValueError(f"{self.variant_id} has no distal pad-orientation DOF")
        a[BASE_ACTION_DIM], a[BASE_ACTION_DIM + 1] = float(left), float(right)
        return a


@dataclass(frozen=True)
class K0Canonical(KinematicVariant):
    """The canonical arm, byte-unchanged (built via ``pedc_selection._env``). No distal orientation DOF."""

    KIND = "k0_canonical"

    @property
    def variant_id(self) -> str:
        return "k0"

    @property
    def n_extra_dof(self) -> int:
        return 0

    @property
    def adapter(self) -> KinematicAdapter:
        return _K0Adapter()

    def build_env(self) -> Any:
        from hymeko_rl.experiments.pedc_selection import _env
        return _env()

    def metadata(self) -> KinematicMetadata:
        return KinematicMetadata("k0", "canonical planar two-finger arm (6-DoF cooperative action)",
                                 BASE_ACTION_DIM, 0, N_FINGERTIPS, self.action_dim, "canonical")

    def to_dict(self) -> dict:
        return {"kind": self.KIND}


@dataclass(frozen=True)
class K1DistalOrientation(KinematicVariant):
    """Canonical arm + one independent distal pad-orientation DOF per fingertip (verified-constructible geometry;
    the closed-loop cooperative-controller wiring is a follow-up — see the module docstring)."""

    KIND = "k1_distal_orientation"

    @property
    def variant_id(self) -> str:
        return "k1"

    @property
    def n_extra_dof(self) -> int:
        return 1

    @property
    def adapter(self) -> KinematicAdapter:
        return _K1Adapter()

    def build_env(self) -> Any:
        """The K1 distal-orientation embodiment is NOT wired into the authoritative delivery rollout. This method
        used to silently return the K0 canonical env, making C0K1/C1K1 scenarios indistinguishable from K0 — a false
        registration. It now fails loudly rather than fabricating K0.

        # Raises ``NotImplementedError`` always — K1 is out of scope for the canonical path. For the *constructed*
        distal-pad closed loop use :func:`hymeko_rl.train.pad_aware_control.build_pad_aware_env` (a separate experiment
        harness), or :meth:`build_padded_arm_mjcf` for the geometry probe."""
        raise NotImplementedError(
            "K1 distal-orientation build_env is unsupported: the distal-pad DOF is not driven end-to-end by the "
            "authoritative rollout. Use the K0 canonical embodiment, or train.pad_aware_control.build_pad_aware_env "
            "for the closed-loop K1 experiment. (This variant must never silently construct K0.)")

    def build_padded_arm_mjcf(self) -> str:
        """The K1 arm MJCF with the physical distal pad DOFs (constructibility artifact / probe input)."""
        return with_distal_pad_orientation(_canonical_arm_mjcf())

    def metadata(self) -> KinematicMetadata:
        return KinematicMetadata("k1", "canonical arm + 1 distal pad-orientation DOF per fingertip",
                                 BASE_ACTION_DIM, 1, N_FINGERTIPS, self.action_dim, "physical_geometry_verified")

    def to_dict(self) -> dict:
        return {"kind": self.KIND}


_VARIANT_KINDS: dict[str, type[KinematicVariant]] = {K0Canonical.KIND: K0Canonical,
                                                     K1DistalOrientation.KIND: K1DistalOrientation}


def kinematic_variant_from_dict(d: dict) -> KinematicVariant:
    """Reconstruct a :class:`KinematicVariant` from :meth:`KinematicVariant.to_dict`. # Postconditions round-trips."""
    kind = d.get("kind")
    cls = _VARIANT_KINDS.get(str(kind))
    if cls is None:
        raise ValueError(f"unknown kinematic-variant kind {kind!r}")
    return cls()
