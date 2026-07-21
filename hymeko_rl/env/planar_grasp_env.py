"""Galambos planar two-finger grasping env — TOP-DOWN table; two connected planar arms pull a coin.

Galambos-sensei's "hello world" (docs/demo/galambos_scenario, local-only): two 2-link planar arms
(thumb + index) lie flat on a table and sweep in the XY plane (Z-axis hinges). A coin is **placed at
a random reachable spot on the table** (it does not fall); the arms pull it into a fixed zone between
them. Pure PPO; the policy reads the **kinematic hypergraph** of the two arms.

The arms are hand-authored MJCF (connected capsule links via ``fromto``, bases sized so the workspace
is well inside reach) — the emitter's geometry conventions cannot express connected planar rods, so
this is a hand-authored scene like ``arm_world``; the hypergraph is still derived from the MJCF. The
coin is a planar body (slide-x/slide-y/hinge-z) confined to the arms' plane; the zone is a marker site.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.env.arm_world import actuated_dof_addrs, emit_arm_mjcf, with_collision_floor
from hymeko_rl.env.constants import Collision, Physics
from hymeko_rl.env.contact_legality import (
    ContactLegalitySpec, ContactLegalityState, ContactMode, classify_contacts,
)
from hymeko_rl.env.env_spec import DEFAULT_ENV, EnvSpec
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.agents.hypergraph_state import HypergraphState

_PLANAR_ARM = "data/robotics/galambos_planar.hymeko"
_PLANAR_ENV = "data/robotics/galambos_env.hymeko"
_PLANAR_TASK = "data/robotics/galambos_task.hymeko"
_PLANE_Z = 0.04
_ARM_REACH = 0.28   # each 2-link finger reaches ~l1+l2=0.30 m; 0.28 leaves a margin


def make_planar_arms_mjcf(*, base_x: float = 0.14, l1: float = 0.16, l2: float = 0.14,
                          z: float = _PLANE_Z) -> str:
    """A hand-authored two-arm planar scene (floor + two connected 2-link Z-hinge arms + position
    servos). Each arm: base hub → link1 (length ``l1``) → link2 (length ``l2``) → fingertip site,
    capsules connected ``fromto`` (no gaps). Bases at ``x = ±base_x``; at home both arms point +Y."""
    def arm(side: int, nm: str) -> str:
        # PHYSICAL-CONTACT contract (2026-07-22): every structural arm geom carries ARM_LEGALITY (contype 1 /
        # conaffinity 3) so it PHYSICALLY collides with the coin (bit 2) — the arm links can no longer pass through
        # the coin. (Previously the structural geoms had no collision attr → MuJoCo default 1/1 → filtered from the
        # coin; only the fingertip collided. Whole-arm contact is now legal, not a body-shove failure.)
        al = Collision.attr(Collision.ARM_LEGALITY)
        return f'''<body name="base_{nm}" pos="{side * base_x:g} -0.02 {z:g}">
        <geom type="cylinder" size="0.022 0.012" {al}/>
        <body name="link1_{nm}">
          <joint name="j1_{nm}" type="hinge" axis="0 0 1" range="-4.0 4.0"/>
          <geom type="capsule" fromto="0 0 0 0 {l1:g} 0" size="0.012" {al}/>
          <body name="link2_{nm}" pos="0 {l1:g} 0">
            <joint name="j2_{nm}" type="hinge" axis="0 0 1" range="-4.0 4.0"/>
            <geom type="capsule" fromto="0 0 0 0 {l2:g} 0" size="0.01" rgba="0.4 0.6 0.95 1" {al}/>
            <geom name="fingertip_{nm}" type="sphere" size="0.014" pos="0 {l2:g} 0" rgba="0.9 0.7 0.1 1" {Collision.attr(Collision.FINGERTIP)}/>
            <site name="tip_{nm}" pos="0 {l2:g} 0" size="0.013" rgba="0.9 0.7 0.1 1"/>
          </body>
        </body>
      </body>'''
    return f'''<mujoco model="galambos_planar">
  <compiler angle="radian"/>
  <option {Physics.option_attrs()}/>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/></visual>
  <default><joint damping="1.5"/><geom rgba="0.2 0.5 0.9 1" friction="1 0.05 0.001"/></default>
  <worldbody>
    <light pos="0 0.1 1.2" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.82 0.82 0.85 1" conaffinity="{int(Collision.Affinity.ANY)}"/>
    {arm(-1, "left")}
    {arm(1, "right")}
  </worldbody>
  <actuator>
    <position name="a_j1_left" joint="j1_left" kp="40" kv="4.0" ctrlrange="-4.0 4.0"/>
    <position name="a_j2_left" joint="j2_left" kp="40" kv="4.0" ctrlrange="-4.0 4.0"/>
    <position name="a_j1_right" joint="j1_right" kp="40" kv="4.0" ctrlrange="-4.0 4.0"/>
    <position name="a_j2_right" joint="j2_right" kp="40" kv="4.0" ctrlrange="-4.0 4.0"/>
  </actuator>
</mujoco>'''


# Tip-dominant blend for the dense approach distance: 0.75·fingertip + 0.25·elbow. The fingertip is the
# grasping point (the term that bridges 'near' → contact must shape *it*), while the elbow keeps a
# far-field gradient alive when the arm is fully extended away from the coin. Pure tip would be `1.0`.
_TIP_BLEND = 0.75


def _vec3(text: str) -> np.ndarray:
    """Parse a whitespace-separated MJCF 3-vector (``"a b c"``) into a float array, padding to 3."""
    parts = [float(v) for v in text.split()]
    return np.asarray((parts + [0.0, 0.0, 0.0])[:3], dtype=np.float64)


def _leaf_body(body: ET.Element) -> ET.Element:
    """Descend the single-chain ``<body>`` tree to its distal leaf (the link bearing the fingertip).

    # Postconditions the returned element has no ``<body>`` child (it may be ``body`` itself)."""
    cur = body
    while (child := cur.find("body")) is not None:
        cur = child
    return cur


def with_fingertip_sites(arm_mjcf: str) -> str:
    """Inject a massless ``tip_{side}`` site at each arm's distal-link far end, if absent (idempotent).

    The emitted planar arm (``base_/upper_/lower_``) carries **no fingertip site** — only the scene's
    ``target_zone``. The dense approach reward and the BC demonstrator both need the true tool point, not
    a body origin (the elbow/base): ``compute_planar_metrics`` mis-shaped the approach as a min over body
    origins, and ``galambos_demo._extract_arms`` fell back to ``tip_site=-1`` (reading ``target_zone``).
    Adding the site fixes both at the source. A site is massless and collisionless, so the compiled
    **dynamics are unchanged** — only the metric/demonstrator read it.

    The site goes at the leaf geom's far end (``pos ± size`` along its longest axis), matching how the
    hand-authored scene declares ``tip_{side}`` directly. Arms that already declare one are left intact.

    # Preconditions valid MJCF with ≥1 top-level arm body whose name ends ``_left``/``_right``.
    # Postconditions every such arm's leaf link carries a ``site name="tip_{side}"``; non-arm bodies,
      joints, geoms, inertials, and actuators are untouched.
    """
    root = ET.fromstring(arm_mjcf)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return arm_mjcf
    changed = False
    for arm in list(worldbody):
        if arm.tag != "body":
            continue
        name = arm.get("name", "")
        side = "left" if name.endswith("_left") else "right" if name.endswith("_right") else None
        if side is None:
            continue
        leaf = _leaf_body(arm)
        geom = leaf.find("geom")
        if geom is None:
            continue
        if any(s.get("name") == f"tip_{side}" for s in leaf.findall("site")):
            continue  # idempotent: the hand-authored scene already declares its tips
        pos, size = _vec3(geom.get("pos", "0 0 0")), _vec3(geom.get("size", "0 0 0"))
        axis = int(np.argmax(size))                       # the link's long axis
        far = pos.copy()
        far[axis] = pos[axis] + math.copysign(size[axis], pos[axis] if pos[axis] != 0.0 else 1.0)
        farstr = f"{far[0]:g} {far[1]:g} {far[2]:g}"
        ET.SubElement(leaf, "site", {"name": f"tip_{side}", "pos": farstr,
                                     "size": "0.012", "rgba": "0.9 0.7 0.1 1"})
        # The COLLISION fingertip (Galambos 2026-07-03): a small geom on conaffinity 3 so ONLY the fingertip
        # (not the arm links, MuJoCo-default 1/1) can touch the coin (bit 2). Mirrors the hand-authored scene.
        ET.SubElement(leaf, "geom", {"name": f"fingertip_{side}", "type": "sphere", "size": "0.014",
                                     "pos": farstr, "rgba": "0.9 0.7 0.1 1",
                                     "contype": str(int(Collision.FINGERTIP[0])),
                                     "conaffinity": str(int(Collision.FINGERTIP[1]))})
        changed = True
    return ET.tostring(root, encoding="unicode") if changed else arm_mjcf


def with_fingertip_shape(arm_mjcf: str, shape: str, size: str, friction: "float | None" = None,
                         compliance: "tuple[tuple[float, float], tuple[float, float, float, float, float] | None] | None" = None) -> str:
    """DIAGNOSTIC (COIN-GRIPPER-GEOMETRY-1 / GRIP-CONTROL-1 / COMPLIANT-PAD-1): configure the collision
    ``fingertip_{side}`` geoms — retype to a flat/pad ``shape`` (a sphere fingertip is a rolling point contact),
    set the tangential ``friction`` (pad-material variant), and/or set the ``compliance`` = (solref, solimp) for a
    *simulated compliant contact* (softer normal response; NOT literal material elasticity). When compliance is given
    the fingertip geoms get ``priority="1"`` so the fingertip↔coin contact params are governed by the fingertip (the
    coin/floor stay canonical). Keeps the fingertip site/frame/collision mask + mirror symmetry. No-op only if
    ``shape == "sphere"`` AND no friction AND no compliance override. # Preconditions ``shape`` in MuJoCo geom types;
    ``solref`` = (timeconst>0, dampratio>0); ``solimp`` = 5-tuple in the standard MuJoCo form or ``None`` to keep default."""
    if shape == "sphere" and friction is None and compliance is None:
        return arm_mjcf
    root = ET.fromstring(arm_mjcf)
    for geom in root.iter("geom"):
        if str(geom.get("name", "")).startswith("fingertip_"):
            if shape != "sphere":
                geom.set("type", shape)
                geom.set("size", size)
            if friction is not None:
                geom.set("friction", f"{friction:g} 0.05 0.001")   # tangential / torsional / rolling
            if compliance is not None:
                solref, solimp = compliance
                geom.set("solref", f"{solref[0]:g} {solref[1]:g}")
                geom.set("priority", "1")                          # fingertip governs the fingertip↔coin contact
                if solimp is not None:
                    geom.set("solimp", " ".join(f"{v:g}" for v in solimp))
    return ET.tostring(root, encoding="unicode")


def with_fingertip_clamp(arm_mjcf: str, coin_radius: float = 0.02, half_span: float = 0.020,
                         prong_radius: float = 0.012) -> str:
    """CONCAVE_CLAMP (COIN-CLAMP embodiment): replace each single point/flat ``fingertip_{side}`` geom with a shallow
    concave CRADLE — two prong geoms straddling the coin so their contact normals span the coin centre → force closure
    (the flat pad had one central normal → the cylinder rolled).

    FRAME (verified 2026-07-21, correcting a tandem-prong bug): the fingertip's local **X** is the CONTACT NORMAL (the
    FLAT_PAD box is thin in X; the approach face points along X), local **Y** is the in-plane TANGENT, local **Z** is the
    cylinder-axis vertical. So the two prongs are offset **±half_span along local Y** (side-by-side around the coin's
    circular cross-section), NOT along the contact normal (which would place one prong behind the other — the front
    shadowing the rear, the original defect). Both prongs keep the ``fingertip_`` prefix + side substring + collision
    mask (contact attribution/legality treat them as the fingertip). No new actuator (the existing aperture/squeeze
    closes the opposing cradles). # Preconditions ``arm_mjcf`` has ``fingertip_{side}`` geoms; ``half_span < coin_radius``
    so the prongs grip inside the coin silhouette. # Postconditions each is replaced by two ``fingertip_{side}_{0,1}``
    prong spheres offset along the tangent."""
    import math
    root = ET.fromstring(arm_mjcf)
    for body in root.iter("body"):
        for geom in [g for g in list(body) if g.tag == "geom" and str(g.get("name", "")).startswith("fingertip_")]:
            name = str(geom.get("name"))
            pos = _vec3(geom.get("pos", "0 0 0"))
            keep = {k: geom.get(k) for k in ("contype", "conaffinity", "rgba", "friction") if geom.get(k) is not None}
            body.remove(geom)
            # A horizontal RING of prongs (in the local X-Y plane; local Z = world vertical = the cylinder axis). The
            # 2-link arm has no wrist DoF to ORIENT a fixed 2-prong cradle toward the coin (the fingertip→coin direction
            # is config-dependent — measured), so a directional cradle is tandem for some grasps. A ring is
            # orientation-INVARIANT: whatever horizontal direction the coin contacts from, the two nearest prongs on the
            # ring straddle it → force closure. Radius ``half_span`` < coin_radius so the near arc grips inside the coin.
            n_prong = 6
            for i in range(n_prong):
                th = 2.0 * math.pi * i / n_prong
                px, py = pos[0] + half_span * math.cos(th), pos[1] + half_span * math.sin(th)
                body.append(ET.fromstring(
                    f'<geom name="{name}_{i}" type="sphere" size="{prong_radius:g}" '
                    f'pos="{px:g} {py:g} {pos[2]:g}" '
                    + " ".join(f'{k}="{v}"' for k, v in keep.items()) + "/>"))
    return ET.tostring(root, encoding="unicode")


def with_arm_coin_collision(arm_mjcf: str) -> str:
    """v2 physics: make the arm-body geoms physically collide with the coin (they pass through it in v1).

    Sets :data:`Collision.ARM_LEGALITY` (``1/3``) on every geom under an arm body so MuJoCo generates
    arm--coin contacts; the fingertip geoms are already ``1/3`` so this is a no-op for them. This function
    only *enables* the contact at the physics layer — **which** contacts are legal (fingertip) versus
    forbidden (any other arm link) is the declarative :class:`ContactLegalitySpec`'s job, decided by geom
    role downstream, never by the mask or by a name check here.

    # Preconditions ``arm_mjcf`` has top-level arm bodies named ``*_left``/``*_right`` bearing geoms.
    # Postconditions every geom under those arms carries ``contype=1 conaffinity=3``; the coin, floor,
      zone, joints, and inertials are untouched. Idempotent.
    """
    root = ET.fromstring(arm_mjcf)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return arm_mjcf
    ct, ca = str(int(Collision.ARM_LEGALITY[0])), str(int(Collision.ARM_LEGALITY[1]))
    changed = False
    for arm in list(worldbody):
        if arm.tag != "body":
            continue
        name = arm.get("name", "")
        if not (name.endswith("_left") or name.endswith("_right")):
            continue
        for geom in arm.iter("geom"):
            geom.set("contype", ct)
            geom.set("conaffinity", ca)
            changed = True
    return ET.tostring(root, encoding="unicode") if changed else arm_mjcf


def compose_planar_scene(arm_mjcf: str, *, disk_radius: float = 0.035, disk_half: float = 0.02,
                         zone_x: float = 0.0, zone_y: float = 0.16, zone_half: float = 0.055,
                         plane_z: float = _PLANE_Z, coin_damping: float = 2.5,
                         coin_density: float | None = None, coin_shape: str = "cylinder",
                         spin_damping: float = 0.8, coin_frictionloss: float = 0.0) -> str:
    """Inject a planar table object (slide-x/slide-y/hinge-z, confined to the arms' plane; placed, not
    dropped) + a target-zone marker site into a two-arm MJCF. Appended before ``</worldbody>``.

    ``coin_damping`` is the slide-joint resistance — for a planar slide-joint object this **is** the table
    friction. HIGHER (e.g. 8–12) makes it **hard to shove**: a knock/impulse dissipates almost at once, so
    only a *sustained two-finger grip-and-drag* moves it — this kills the ``knock'' shortcut (2026-06-27: 94% of
    deliveries were knocks, fingers never both touching) and makes delivery require real contact. ``coin_density``
    (kg/m³, HIGHER = heavier object, needs a firmer grip). Defaults reproduce the original (easy-to-shove) coin.

    ``coin_shape`` selects the manipuland: ``"cylinder"`` = the original round coin — which **rolls out of a
    two-finger clamp when dragged perpendicular** (a point contact on a circle is an unstable antipodal grasp;
    documented in ``galambos_bc``). ``"box"`` = a square prism whose **flat faces give the fingertips a stable
    contact** so the clamp holds under a drag — the graspable object (a cube-like puck). ``spin_damping`` is the
    ``disk_rz`` hinge resistance (raise it to further suppress the object spinning out of the clamp)."""
    arm_mjcf = with_collision_floor(arm_mjcf, z=0.0)   # emitted arms carry no floor
    dens = f' density="{coin_density:g}"' if coin_density is not None else ""
    # Collision bitmask (Galambos 2026-07-03): the coin is on bit 2 ONLY, so arm links (MuJoCo default 1/1)
    # cannot touch it — only the fingertip geoms (conaffinity 3) and the floor (conaffinity 3) can. The coin is
    # thus moved ONLY by the yellow fingertips, never knocked by an arm body. `cc` is applied to the manipuland.
    cc = " " + Collision.attr(Collision.COIN)
    if coin_shape == "box":
        geom = (f'<geom name="disk" type="box" size="{disk_radius:g} {disk_radius:g} {disk_half:g}" '
                f'rgba="0.85 0.3 0.2 1" friction="1.0 0.05 0.001"{dens}{cc}/>')
    elif coin_shape == "cylinder":
        geom = (f'<geom name="disk" type="cylinder" size="{disk_radius:g} {disk_half:g}" '
                f'rgba="0.85 0.3 0.2 1" friction="1.0 0.05 0.001"{dens}{cc}/>')
    else:
        raise ValueError(f"coin_shape must be 'cylinder' or 'box'; got {coin_shape!r}")
    # `frictionloss` is DRY (Coulomb) friction on the slide joints — a FORCE THRESHOLD, not a rate: the coin does
    # not move until the applied push exceeds it. Set between one arm's and two arms' push force → a SINGLE arm
    # cannot move the coin, only two together (Galambos 2026-07-03: "két robot ereje kelljen a henger
    # megmozdításához"). 0 = the original free-sliding coin (unchanged default).
    fl = f' frictionloss="{coin_frictionloss:g}"' if coin_frictionloss > 0.0 else ""
    coin = (
        f'<body name="disk" pos="0 0 {plane_z:g}">'
        f'<joint name="disk_x" type="slide" axis="1 0 0" damping="{coin_damping:g}"{fl}/>'
        f'<joint name="disk_y" type="slide" axis="0 1 0" damping="{coin_damping:g}"{fl}/>'
        f'<joint name="disk_rz" type="hinge" axis="0 0 1" damping="{spin_damping:g}"/>'
        f'{geom}'
        f'</body>'
    )
    zone = (
        f'<site name="target_zone" type="cylinder" size="{zone_half:g} 0.002" '
        f'pos="{zone_x:g} {zone_y:g} 0.004" rgba="0.2 0.8 0.3 0.4"/>'
    )
    return arm_mjcf.replace("</worldbody>", "    " + coin + zone + "\n  </worldbody>", 1)


@dataclass(frozen=True)
class PlanarGraspMetrics:
    """Live task state on the table: coin centre ``(x, y)``, in-plane distance to the zone, per-finger
    contact with the coin, whether the coin is inside the zone, and each arm's in-plane **approach
    distance** to the coin — a tip-dominant blend ``0.75·fingertip + 0.25·elbow`` (the dense signal
    that bridges 'near' → 'contact', shaping the actual grasping point rather than a body origin)."""

    disk_pos: np.ndarray
    disk_to_zone: float
    left_contact: bool
    right_contact: bool
    in_zone: bool
    left_tip_dist: float
    right_tip_dist: float
    disk_speed: float
    arm_speed: float
    arm_self_contact: bool
    disk_vel: np.ndarray                  # in-plane coin velocity (vx, vy) — projects onto coin→zone for the
                                          # target-directed delivery-progress metric (eval, not reward)
    fingers_self_contact: bool = False   # the two FINGERTIP links touching EACH OTHER (a crash) — NOT a coin-pinch
    # v2 contact-legality result (None in v1): the classified object contacts against the declarative
    # ContactLegalitySpec — carries the forbidden-contact flag/count/impulse and both-fingertip state.
    legality: "ContactLegalityState | None" = None


CLEAN_PLANAR = PlanarGraspMetrics(
    np.zeros(2, dtype=np.float32), 0.0, False, False, False, 0.0, 0.0, 0.0, 0.0, False,
    np.zeros(2, dtype=np.float32))


def compute_planar_metrics(model: Any, data: Any, *, disk_body: int, disk_geom: int,
                           left_bodies: frozenset[int], right_bodies: frozenset[int],
                           zone_x: float, zone_y: float, zone_half: float,
                           disk_dofx: int, disk_dofy: int, arm_dofs: tuple[int, ...],
                           tip_sites: tuple[int, int], elbow_bodies: tuple[int, int],
                           tip_blend: float = _TIP_BLEND,
                           contact_spec: "ContactLegalitySpec | None" = None) -> PlanarGraspMetrics:
    """Derive :class:`PlanarGraspMetrics` from a stepped ``(model, data)`` — all in the table plane.

    ``tip_sites`` / ``elbow_bodies`` are the ``(left, right)`` fingertip-site and distal-link-body ids;
    the approach distance is ``tip_blend·d(tip) + (1-tip_blend)·d(elbow)`` — the fingertip is the
    grasping point, the elbow keeps a far-field gradient when the arm is fully extended.

    ``contact_spec`` selects the contact-legality mode. **None (v1 prototype):** only fingertips can
    physically touch the coin (the ``2/2`` bitmask), so a coin contact on any arm body is a fingertip
    contact — classified body-level, no forbidden contact possible, ``legality is None``. **A spec (v2):**
    arm links physically collide with the coin, so :func:`classify_contacts` decides legality per geom-role
    (fingertip = valid left/right, any other arm geom = forbidden) and returns a :class:`ContactLegalityState`.
    Arm self-contact (a crash) is classified here in either mode — it is not a coin-legality concern.

    # Preconditions ``tip_sites`` valid site ids (``>= 0``); ``elbow_bodies`` valid body ids;
      ``0 <= tip_blend <= 1``; ``contact_spec`` (if given) was built for this ``model``.
    # Postconditions ``left/right_tip_dist`` finite ``>= 0``; ``legality`` is ``None`` iff ``contact_spec``
      is ``None`` (v1 parity)."""
    pos = data.xpos[disk_body]
    disk_xy = np.array([float(pos[0]), float(pos[1])], dtype=np.float32)
    to_zone = float(np.hypot(pos[0] - zone_x, pos[1] - zone_y))
    disk_vel = np.array([float(data.qvel[disk_dofx]), float(data.qvel[disk_dofy])], dtype=np.float32)
    disk_speed = float(np.hypot(disk_vel[0], disk_vel[1]))
    arm_speed = float(sum(abs(float(data.qvel[d])) for d in arm_dofs))
    left = right = arm_self_contact = fingers_self_contact = False
    finger_pair = {int(elbow_bodies[0]), int(elbow_bodies[1])}   # the two fingertip-bearing distal links
    # v2: the declarative spec owns coin-contact legality (valid fingertip vs forbidden arm link).
    legality: "ContactLegalityState | None" = None
    if contact_spec is not None:
        legality = classify_contacts(model, data, contact_spec)
        left, right = legality.left_fingertip_contact, legality.right_fingertip_contact
    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if contact_spec is None and (g1 == disk_geom or g2 == disk_geom):
            ob = b2 if g1 == disk_geom else b1                   # v1: only fingertips can touch → body-level
            if ob in left_bodies:
                left = True
            elif ob in right_bodies:
                right = True
        elif (b1 in left_bodies and b2 in right_bodies) or \
                (b1 in right_bodies and b2 in left_bodies):
            arm_self_contact = True   # the two arms collided with each other (any body pair)
            if {b1, b2} == finger_pair:   # NARROW: the two FINGERS hitting each other (a crash, not a coin-pinch)
                fingers_self_contact = True

    def _planar_dist(p: Any) -> float:
        return float(np.hypot(p[0] - disk_xy[0], p[1] - disk_xy[1]))

    def _approach(tip_site: int, elbow_body: int) -> float:
        """Tip-dominant approach distance: blend the fingertip (the grasping point) with the elbow
        (the distal-link origin, a far-field anchor). Falls back to elbow-only if no tip site."""
        d_elbow = _planar_dist(data.xpos[elbow_body])
        if tip_site < 0:
            return d_elbow
        return tip_blend * _planar_dist(data.site_xpos[tip_site]) + (1.0 - tip_blend) * d_elbow

    return PlanarGraspMetrics(disk_xy, to_zone, left, right, to_zone < zone_half,
                              _approach(tip_sites[0], elbow_bodies[0]),
                              _approach(tip_sites[1], elbow_bodies[1]), disk_speed, arm_speed,
                              arm_self_contact, disk_vel, fingers_self_contact, legality=legality)


def coin_zone_direction(disk_pos, zone_x: float, zone_y: float) -> tuple[np.ndarray, float]:
    """Pure coin→zone geometry: the unit direction from the coin centre ``disk_pos`` (x,y[,z]) to the delivery zone
    ``(zone_x, zone_y)`` and the distance ``n``. Shared by :meth:`PlanarGraspEnv.direction_to_zone` and unit tests
    (so the math has a single home, testable without constructing a MuJoCo env).

    # Postconditions ``||u|| ≈ 1`` unless the coin is exactly on the zone (then ``u == 0``); ``n ≥ 0``."""
    coin = np.asarray(disk_pos[:2], np.float64)
    d = np.array([zone_x, zone_y], np.float64) - coin
    n = float(np.linalg.norm(d))
    return d / (n + 1e-9), n


class PlanarGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Top-down planar grasping: pull a coin placed on the table into the zone between two arms.

    Observation = per-vertex features on the two-arm hypergraph ``(n_vertices, 8)``: each link vertex
    carries its driving joint ``(qpos, qvel)``, its world position ``(x, y)``, its **vector to the
    coin** ``(coin - vertex)``, and the broadcast **coin→zone** vector. Action = 4 arm joint targets
    (the coin is unactuated). Reward = the declarative ``galambos_task.hymeko`` spec. The coin is
    placed in reach at reset (no fall). Ends on success (coin in zone for ``success_steps``), death
    (coin knocked out of the workspace), or ``max_steps``.
    """

    metadata = {"render_modes": []}
    _FEAT = 8
    privileged_dim = 5   # asymmetric-CTDE critic state z(s): [left_contact, right_contact, phase-onehot(3)]

    def __init__(self, *, robot: str | None = _PLANAR_ARM, reward_spec: RewardSpec | None = None,
                 env: EnvSpec = DEFAULT_ENV, frame_skip: int = 5, max_steps: int = 160,
                 difficulty: float = 1.0, task_graph: bool = False,
                 coin_damping: float = 2.5, coin_density: float | None = None,
                 coin_shape: str = "cylinder", spin_damping: float = 0.8,
                 coin_frictionloss: float = 0.0,
                 fingertip_shape: str = "sphere", fingertip_size: str = "0.006 0.016 0.02",
                 fingertip_friction: "float | None" = None,
                 fingertip_compliance: "tuple[tuple[float, float], tuple[float, float, float, float, float] | None] | None" = None,
                 arm_mjcf_transform: "Callable[[str], str] | None" = None,
                 terminate_on_success: bool = True,
                 contact_legality: bool | None = None) -> None:
        super().__init__()
        if frame_skip < 1 or max_steps < 1:
            raise ValueError("frame_skip/max_steps must be >= 1")
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        # The scene geometry is one config struct (EnvSpec), read from galambos_env.hymeko by
        # `from_hymeko`. `difficulty` (curriculum) and `max_steps` are runtime knobs, not scene data.
        self._env = env
        self._zone_x, self._zone_y, self._zone_half = env.zone_x, env.zone_y, env.zone_half
        self._zone_region, self._coin_region = env.zone_region, env.coin_region
        self._coin_clearance, self._randomize_zone = env.coin_clearance, env.randomize_zone
        self._out_bound, self._y_min, self._y_max = env.out_bound, env.y_min, env.y_max
        self.success_steps = env.success_steps
        # De-farm (2026-07-02, oracle-verified): ending the episode on sustained in-zone forfeits the per-step
        # in-zone annuity, so the reward-optimum FARMS (oscillates in/out) instead of committing. With
        # `terminate_on_success=False` the episode runs the full horizon, holding-in-zone becomes optimal
        # (annuity for all remaining steps > oscillating), and the farm incentive vanishes. Default True
        # preserves the historical behaviour; the de-farmed task sets it False. Death/out-of-bounds still ends.
        self.terminate_on_success = bool(terminate_on_success)
        # v2 contact legality (2026-07-06): in v1 the coin passes through the arm bodies (an abstract
        # fingertip-only prototype). v2 makes the arm bodies physically collide with the coin and treats any
        # non-fingertip arm↔coin contact as a FORBIDDEN constraint violation that HARD-invalidates delivery
        # (a reward penalty alone is farmable). The contract is declared in EnvSpec (`contact` term);
        # `contact_legality` here overrides it (None → use the declared value). Off by default keeps v1
        # bit-reproducible (§6.5 #19: no metric-changing default flip before the scripted baseline re-validates).
        self._contact_legality = env.contact_legality if contact_legality is None else bool(contact_legality)
        self._contact_spec: "ContactLegalitySpec | None" = None   # built once below (v2 only)
        self.difficulty = difficulty
        # The robot is DESCRIBED IN HYMEKO (galambos_planar.hymeko) and emitted; `robot=None` falls
        # back to the hand-authored baseline (make_planar_arms_mjcf).
        arm_mjcf = (emit_arm_mjcf(robot, name="galambos", control_mode="position")
                    if robot is not None else make_planar_arms_mjcf())
        # The emitted arm carries no fingertip site; inject one at each leaf link's far end so the dense
        # approach reward (and the BC demonstrator) shape the true tool point, not a body origin. Massless
        # → dynamics unchanged. Idempotent on the hand-authored scene (it declares its tips already).
        arm_mjcf = with_fingertip_sites(arm_mjcf)
        # DIAGNOSTIC (COIN-GRIPPER-GEOMETRY-1): flat/pad fingertip geometry so a parallel clamp catches the cylinder
        # (no-op for the canonical sphere). Same site/frame/mask + mirror symmetry; only the collision geom shape.
        arm_mjcf = with_fingertip_shape(arm_mjcf, fingertip_shape, fingertip_size, fingertip_friction, fingertip_compliance)
        # PAD-AWARE-COOPERATIVE-CONTROL-0: an OPTIONAL additive arm-MJCF transform (e.g. add a distal pad-orientation
        # DOF per fingertip). Default None → K0 byte-unchanged. The caller supplies the transform (no layering cycle).
        if arm_mjcf_transform is not None:
            arm_mjcf = arm_mjcf_transform(arm_mjcf)
        # v2 only: give the arm-link geoms a coin-collidable mask so the coin cannot pass through them.
        # Collision-mask change only (the hypergraph reads body/joint structure, not masks) → the graph is
        # unaffected; legality is still decided by geom id at metric time.
        if self._contact_legality:
            arm_mjcf = with_arm_coin_collision(arm_mjcf)
        self.hg = HypergraphState.from_mjcf(arm_mjcf, is_path=False)
        # `task_graph` (the discriminating ablation): put the COIN and ZONE in the graph as vertices joined to
        # the robot by a GRASP hyperedge {fingertips, coin} + a GOAL hyperedge {coin, zone}, so the structural
        # prior can finally reason about the task objective instead of reading it as flat broadcast features
        # (the HSiKAN==MLP tie root cause; reports/2026-06-24-galambos-hyperedge-ab.md). Baseline = robot-only.
        self._n_robot = self.hg.n_vertices
        self._task_graph = task_graph
        self._coin_vtx = self._zone_vtx = -1
        if task_graph:
            down_parents = {int(p) for (p, c), s in zip(self.hg.edges, self.hg.signs) if s > 0}
            grasp_links = tuple(v for v in range(self.hg.n_vertices) if v not in down_parents)  # arm leaves
            self._coin_vtx, self._zone_vtx = self._n_robot, self._n_robot + 1
            self.hg = self.hg.with_task_hyperedges(
                new_entities=["coin", "zone"],
                hyperedges=[("grasp_hub", [*grasp_links, self._coin_vtx]),
                            ("goal_hub", [self._coin_vtx, self._zone_vtx])])
        # The emitter now emits parent→child <contact><exclude> directly (hymeko_formats), so the
        # adjacent-link self-contact no longer pins the shoulder joint — no env-level workaround.
        mjcf = compose_planar_scene(arm_mjcf, zone_x=env.zone_x, zone_y=env.zone_y,
                                    zone_half=env.zone_half, disk_radius=env.disk_radius,
                                    coin_damping=coin_damping, coin_density=coin_density,
                                    coin_shape=coin_shape, spin_damping=spin_damping,
                                    coin_frictionloss=coin_frictionloss)
        self._mjcf = mjcf   # kept so the renderer can re-skin the scene (decorate_scene)
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self._act_dofs = actuated_dof_addrs(self.model)   # for joint-velocity/acceleration reward terms
        self.reward_spec = reward_spec if reward_spec is not None else \
            RewardSpec.from_hymeko(_PLANAR_TASK)
        # Stable integration (blow-up fix, measured 2026-06-30): the 2e-3 s sub-step lets the planar arms
        # DETONATE on contact under aggressive actions (|qacc| ~8e3 -> bodies ejected, which can knock the coin
        # into the zone = a false delivery). Shrink the sub-step and raise the substep count so the control
        # interval (frame_skip * timestep) is preserved EXACTLY; trained weights transfer unchanged.
        _stable_dt = Physics.STABLE_DT
        _control_dt = self.model.opt.timestep * int(frame_skip)
        if self.model.opt.timestep > _stable_dt:
            substeps = max(1, round(_control_dt / _stable_dt))
            self.model.opt.timestep = _control_dt / substeps
            frame_skip = substeps
        self.frame_skip, self.max_steps = frame_skip, max_steps

        self._disk_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "disk")
        self._disk_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
        self._zone_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_zone")
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                 for b in range(self.model.nbody)]
        self._left_bodies = frozenset(b for b, n in enumerate(names) if n and n.endswith("_left"))
        self._right_bodies = frozenset(b for b, n in enumerate(names) if n and n.endswith("_right"))
        # Fingertip sites (the grasping point) + distal-link 'elbow' bodies (far-field anchor) per arm,
        # for the tip-dominant approach reward. The elbow is each arm's leaf body — the link the tip
        # site sits on; resolved structurally (the leaf is nobody's parent within the arm's body set).
        self._tip_sites = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_left"),
                           mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_right"))
        self._elbow_bodies = (self._distal_body(self._left_bodies),
                              self._distal_body(self._right_bodies))
        # v2 only: build the declarative contact contract ONCE from the compiled model (geom roles assigned
        # inside ContactLegalitySpec.from_model, not scattered here). None in v1 → compute_planar_metrics
        # falls back to the abstract body-level fingertip-only classification. Fail-loud if unsatisfiable.
        # GRADED (default): arm-body↔coin contact is allowed, tracked, and reward-penalised, never voids the
        # episode. STRICT: additionally invalidates the delivery / terminates (clean-paper validation).
        if self._contact_legality:
            self._contact_spec = ContactLegalitySpec.from_model(
                self.model, object_geoms={self._disk_geom},
                arm_bodies_left=self._left_bodies, arm_bodies_right=self._right_bodies,
                fingertip_prefix=env.valid_contact_prefix,
                mode=ContactMode(env.contact_mode))
        # Each arm's base (a direct child of the worldbody) anchors its reach circle; read from the
        # model so zone/coin sampling tracks the .hymeko stance (e.g. the wider ±0.18 bases).
        self._reach_centers = [
            (float(self.model.body_pos[b][0]), float(self.model.body_pos[b][1]))
            for b in sorted(self._left_bodies | self._right_bodies)
            if int(self.model.body_parentid[b]) == 0]
        # Coin-clearance: the arms reset to the home pose (qpos=0), so freeze every arm-link geom's XY there.
        # The coin must NOT spawn inside a link (measured 2026-06-30: it could, and the penetration impulse +
        # the policy's first motion flung it toward the zone, making a non-toss read as a delivery).
        _arm_bodies = self._left_bodies | self._right_bodies
        _arm_geoms = [g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in _arm_bodies]
        mujoco.mj_forward(self.model, self.data)                  # qpos=0 home pose → arm-link geom XY
        # Freeze each arm-link geom as a CAPSULE SEGMENT (endpoints + radius), not just its centroid: a capsule
        # extends ±half-length from its centre, so a coin near a capsule END clears the centroid yet penetrates the
        # link (the seed-1011 defect). Point-to-segment distance is exact. Physical-contact contract 2026-07-22.
        segs = []
        for g in _arm_geoms:
            c = self.data.geom_xpos[g][:2].copy()
            r = float(self.model.geom_size[g][0])
            if int(self.model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                axis = self.data.geom_xmat[g].reshape(3, 3)[:2, 2]   # capsule long axis (local +Z) projected to XY
                h = float(self.model.geom_size[g][1])
                segs.append((c - h * axis, c + h * axis, r))
            else:
                segs.append((c, c.copy(), r))                        # hub/sphere: degenerate segment
        self._rest_arm_segs = segs
        self._coin_arm_clear = float(env.disk_radius) + 0.003     # coin radius + 3 mm margin (> 0.5 mm penetration tol)
        self._arm_joints = [
            j for j in range(self.model.njnt)
            if not (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    or "").startswith("disk")]
        self._arm_dofs = tuple(int(self.model.jnt_dofadr[j]) for j in self._arm_joints)
        jid_dx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "disk_x")
        jid_dy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "disk_y")
        self._disk_x_adr = self.model.jnt_qposadr[jid_dx]
        self._disk_y_adr = self.model.jnt_qposadr[jid_dy]
        self._disk_dofx = int(self.model.jnt_dofadr[jid_dx])
        self._disk_dofy = int(self.model.jnt_dofadr[jid_dy])

        self.n_actions = int(self.model.nu)
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(self._ctrl_lo, self._ctrl_hi, dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.hg.n_vertices, self._FEAT), dtype=np.float32)
        self._planar_metrics = CLEAN_PLANAR
        self._disk_out = False
        self._step = 0
        self._success = 0
        self._reset_contact_accum()   # v2 arm-body↔coin contact accumulators (ever / duration / impulse)

    @classmethod
    def from_hymeko(cls, *, robot: str = _PLANAR_ARM, env: str = _PLANAR_ENV,
                    task: str = _PLANAR_TASK, max_steps: int = 160,
                    difficulty: float = 1.0, task_graph: bool = False,
                    contact_legality: bool | None = None) -> "PlanarGraspEnv":
        """Build the whole MDP from three ``.hymeko`` sources — the robot, the environment scene,
        and the reward task. The env geometry (zone, coin spawn, workspace, success) comes from
        ``env`` via :class:`EnvSpec`; the reward from ``task`` via :class:`RewardSpec`. Mirrors
        :meth:`hymeko_rl.env.arm_reach_env.ArmReachEnv.from_hymeko` — one source per concern.
        ``task_graph`` puts the coin+zone in the graph (grasp/goal hyperedges) — the structural ablation.
        ``contact_legality`` selects v2 physics (arm bodies collide with the coin; a non-fingertip
        arm↔coin contact is forbidden and hard-invalidates delivery)."""
        return cls(robot=robot, reward_spec=RewardSpec.from_hymeko(task),
                   env=EnvSpec.from_hymeko(env), max_steps=max_steps, difficulty=difficulty,
                   task_graph=task_graph, contact_legality=contact_legality)

    def _distal_body(self, bodies: frozenset[int]) -> int:
        """The arm's distal (leaf) body — the fingertip-bearing link, nobody's parent within ``bodies``.

        # Preconditions ``bodies`` is non-empty and forms a single kinematic chain.
        # Postconditions returns a body id in ``bodies``."""
        parents = {int(self.model.body_parentid[b]) for b in bodies}
        leaves = sorted(b for b in bodies if b not in parents)
        if not leaves:
            raise ValueError("arm body set has no leaf (cyclic?) — cannot resolve the distal link")
        return leaves[-1]

    def _reset_contact_accum(self) -> None:
        """Reset the per-episode arm-body↔coin contact accumulators (v2). ``_arm_body_ever`` is the latch,
        ``_arm_body_steps`` the contact duration (in env steps), ``_arm_body_impulse_sum`` the summed force.
        ``_fingertip_progress``/``_body_progress`` accumulate the coin's toward-zone displacement attributed to
        fingertip vs body-only contact — the grade signal a contact-quality-gated reward reads."""
        self._arm_body_ever = False
        self._arm_body_steps = 0
        self._arm_body_impulse_sum = 0.0
        self._prev_disk_to_zone: float | None = None
        self._fingertip_progress = 0.0
        self._body_progress = 0.0

    @property
    def planar_metrics(self) -> PlanarGraspMetrics:
        """Public read-only view of the current planar contact/geometry metrics — the single source the canonical
        delivery ``rollout`` and the strict monitor read (callers must not touch ``_planar_metrics`` directly).

        # Postconditions returns the metrics computed after the most recent ``reset``/``step``; never ``None``."""
        return self._planar_metrics

    @property
    def arm_body_steps(self) -> int:
        """Public read-only count of arm-body↔coin contact steps (the body-shove attribution source)."""
        return int(self._arm_body_steps)

    def direction_to_zone(self) -> tuple[np.ndarray, float]:
        """Unit direction from the coin centre to the delivery zone, and the coin→zone distance ``n``.

        The canonical target-relative geometry every scripted delivery primitive/actor uses. Lives on the env (which
        owns both the coin metrics and the zone) so library/train code no longer reaches into an experiment module.

        # Postconditions returns ``(u, n)`` with ``||u|| ≈ 1`` (``u == 0`` only if the coin sits exactly on the zone)
        and ``n == disk_to_zone ≥ 0``."""
        return coin_zone_direction(self._planar_metrics.disk_pos, self._zone_x, self._zone_y)

    def _metrics(self) -> PlanarGraspMetrics:
        return compute_planar_metrics(
            self.model, self.data, disk_body=self._disk_body, disk_geom=self._disk_geom,
            left_bodies=self._left_bodies, right_bodies=self._right_bodies,
            zone_x=self._zone_x, zone_y=self._zone_y, zone_half=self._zone_half,
            disk_dofx=self._disk_dofx, disk_dofy=self._disk_dofy, arm_dofs=self._arm_dofs,
            tip_sites=self._tip_sites, elbow_bodies=self._elbow_bodies,
            contact_spec=self._contact_spec)

    def node_features(self) -> np.ndarray:
        feat = np.zeros((self.hg.n_vertices, self._FEAT), dtype=np.float32)
        cx, cy = float(self._planar_metrics.disk_pos[0]), float(self._planar_metrics.disk_pos[1])
        nr = self._n_robot                                     # robot-link rows (== n_vertices unless augmented)
        for j in self._arm_joints:
            v = int(self.model.jnt_bodyid[j]) - 1
            if 0 <= v < nr:
                feat[v, 0] = self.data.qpos[self.model.jnt_qposadr[j]]
                feat[v, 1] = self.data.qvel[self.model.jnt_dofadr[j]]
        for v in range(nr):
            vx, vy = float(self.data.xpos[v + 1][0]), float(self.data.xpos[v + 1][1])
            feat[v, 2], feat[v, 3] = vx, vy
            feat[v, 4], feat[v, 5] = cx - vx, cy - vy           # this link's vector to the coin
        feat[:nr, 6] = cx - self._zone_x                        # broadcast coin -> zone
        feat[:nr, 7] = cy - self._zone_y
        if self._task_graph:                                   # coin/zone are real vertices; hubs stay zero
            feat[self._coin_vtx, 2], feat[self._coin_vtx, 3] = cx, cy
            feat[self._coin_vtx, 6], feat[self._coin_vtx, 7] = cx - self._zone_x, cy - self._zone_y
            feat[self._zone_vtx, 2], feat[self._zone_vtx, 3] = self._zone_x, self._zone_y
            feat[self._zone_vtx, 4], feat[self._zone_vtx, 5] = cx - self._zone_x, cy - self._zone_y
        return feat

    def privileged_state(self) -> np.ndarray:
        """Privileged global state ``z(s)`` for the centralized (asymmetric CTDE / MADDPG) critic — read by the
        critic ONLY, never by the decentralized actors, and NOT written into :meth:`node_features` (so
        decentralized execution needs only the geometry obs). It carries what the geometry obs cannot supply:
        both arms' coin **contact** (needs contact forces) and the coarse task **phase** (the
        ``_ever_grasped`` history latch).

        # Preconditions the env has been ``reset`` (``_planar_metrics`` populated).
        # Postconditions float32 ``(5,)`` = ``[left_contact, right_contact, onehot(reach, carry, in_zone)]``;
          exactly one phase bit set; contact bits in ``{0, 1}``.
        """
        m = self._planar_metrics
        phase = 2 if m.in_zone else (1 if getattr(self, "_ever_grasped", False) else 0)
        z = np.zeros(self.privileged_dim, dtype=np.float32)
        z[0] = 1.0 if m.left_contact else 0.0
        z[1] = 1.0 if m.right_contact else 0.0
        z[2 + phase] = 1.0
        return z

    def _reachable_by_any(self, x: float, y: float) -> bool:
        """True if at least one arm base is within reach of ``(x, y)`` (so an arm can fetch it)."""
        return any(np.hypot(x - cx, y - cy) <= _ARM_REACH for cx, cy in self._reach_centers)

    def _clear_of_arms(self, x: float, y: float) -> bool:
        """True if a coin at ``(x, y)`` clears every arm-link CAPSULE at the home pose (point-to-segment distance minus
        the capsule radius exceeds the coin+margin clearance) — the coin must not spawn inside an arm."""
        p = np.array([x, y], dtype=np.float64)
        for a, b, r in self._rest_arm_segs:
            ab = b - a
            denom = float(ab @ ab)
            t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
            seg_dist = float(np.linalg.norm(p - (a + t * ab)))
            if seg_dist - r <= self._coin_arm_clear:
                return False
        return True

    def _sample_zone_xy(self) -> tuple[float, float]:
        """A small target zone re-placed each episode, kept within reach of **both** arms (so the
        coin can be delivered there). Falls back to the central point if rejection sampling fails."""
        rx_lo, rx_hi, ry_lo, ry_hi = self._zone_region
        for _ in range(64):
            x = float(self.np_random.uniform(rx_lo, rx_hi))
            y = float(self.np_random.uniform(ry_lo, ry_hi))
            if all(np.hypot(x - cx, y - cy) <= _ARM_REACH for cx, cy in self._reach_centers):
                return x, y
        return 0.0, 0.15

    def _sample_coin_xy(self) -> tuple[float, float]:
        """A coin placement over the **reachable table** (it MAY be outside the between-arms band, so
        an arm must reach out and corral it), always OUTSIDE the zone and reachable by at least one
        arm. The curriculum ``difficulty`` caps how far from the (per-episode) zone the coin may
        spawn: at difficulty 1 the cap covers the full reachable table; lower difficulty keeps the
        coin in a shell near the zone (an easier start)."""
        inner = self._zone_half + self._coin_clearance
        cap = inner + 0.02 + 0.30 * self.difficulty   # d=1 → covers the whole reachable table
        rx_lo, rx_hi, ry_lo, ry_hi = self._coin_region
        for _ in range(128):
            x = float(self.np_random.uniform(rx_lo, rx_hi))
            y = float(self.np_random.uniform(ry_lo, ry_hi))
            d = float(np.hypot(x - self._zone_x, y - self._zone_y))
            if inner <= d <= cap and self._reachable_by_any(x, y) and self._clear_of_arms(x, y):
                return x, y
        return self._zone_x + inner + 0.01, self._zone_y   # reachable, just outside the zone

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = 0.0
        if self._randomize_zone:
            self._zone_x, self._zone_y = self._sample_zone_xy()
            if self._zone_site >= 0:
                self.model.site_pos[self._zone_site] = [self._zone_x, self._zone_y, 0.004]
        x, y = self._sample_coin_xy()
        self.data.qpos[self._disk_x_adr] = x
        self.data.qpos[self._disk_y_adr] = y
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._planar_metrics = self._metrics()
        self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()   # jerk term baseline
        self._step = 0
        self._success = 0
        self._reset_contact_accum()                                   # v2: reset arm-body contact accumulators
        self._ever_grasped = False                                    # gates the pre-grasp stillness term
        self._pbrs_prev_zone = None                                   # PBRS potentials re-initialise each episode
        self._pbrs_prev_grasp = None                                  # (Ng-Harada-Russell: γΦ(s')-Φ(s), see reward.py)
        self._pbrs_prev_conj = None                                   # conjunctive potential Φ=-max(zone,tips)
        return self.node_features(), {"disk_to_zone": self._planar_metrics.disk_to_zone}

    def step(self, action: np.ndarray,
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ctrl = np.clip(np.asarray(action, dtype=np.float32), self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[:] = ctrl
        try:
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
        except mujoco.FatalError:
            # A degenerate heavy-contact pose overflowed MuJoCo's per-pair contact buffer. End the episode as a
            # death rather than crash the run — a strong contact reward can drive the arms into such a pose.
            zeros = np.zeros(self.observation_space.shape, dtype=np.float32)
            return zeros, -1.0, True, False, {"disk_to_zone": 0.0, "in_zone": False, "death": True,
                                              "both_contact": False, "physics_failure": True}
        self._step += 1
        self._planar_metrics = self._metrics()
        m = self._planar_metrics
        both_fingertip = m.left_contact and m.right_contact
        if both_fingertip:
            self._ever_grasped = True                                 # latch: pre-grasp stillness turns off
        lg = m.legality                                               # v2 contact-legality state (None in v1)
        spec = self._contact_spec
        arm_body_now = lg.arm_body_contact if lg is not None else False
        if arm_body_now:                                             # v2: track the non-preferred contact
            self._arm_body_ever = True                              # (GRADED: tracked+penalised, not fatal)
            self._arm_body_steps += 1                               # duration (# of steps in contact)
            self._arm_body_impulse_sum += lg.arm_body_contact_impulse
        # Attribute the coin's toward-zone displacement THIS step to fingertip vs body-only contact (BEFORE the
        # reward, so a contact-quality-gated terminal term sees the up-to-now grade). Fingertip contact wins the
        # attribution when both a fingertip and an arm body touch (the grasp is doing the work).
        if self._prev_disk_to_zone is not None:
            toward = max(0.0, self._prev_disk_to_zone - m.disk_to_zone)
            if m.left_contact or m.right_contact:
                self._fingertip_progress += toward
            elif arm_body_now:
                self._body_progress += toward                       # body-only push (no fingertip this step)
        self._prev_disk_to_zone = m.disk_to_zone
        # Compute out-of-bounds BEFORE the reward so the `out_of_bounds` term can penalise it (the
        # disk knocked off the table). Death only terminating was not enough — over-pushing was free.
        cx, cy = float(m.disk_pos[0]), float(m.disk_pos[1])
        self._disk_out = abs(cx) > self._out_bound or cy < self._y_min or cy > self._y_max
        reward = self.reward_spec.evaluate(self, m.disk_to_zone, ctrl)   # arm-body penalty is a declared reward term
        self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()   # for next step's jerk term
        death = self._disk_out
        self._success = self._success + 1 if m.in_zone else 0
        held = self._success >= self.success_steps
        # STRICT mode only: an arm-body contact voids the delivery and (if asked) terminates. GRADED (default)
        # never voids — the delivery still counts as raw, and the arm-body contact grades it (clean/assisted/
        # exploit) at eval time and is penalised by the reward. In v1 (no spec) this reduces to the old rule.
        invalidated = self._arm_body_ever and spec is not None and spec.invalidates_on_arm_body
        delivered = held and not invalidated
        strict_term = (self._arm_body_ever and spec is not None
                       and spec.invalidates_on_arm_body and spec.strict_terminate)
        terminated = bool((delivered and self.terminate_on_success) or death or strict_term)
        truncated = self._step >= self.max_steps
        return (self.node_features(), reward, terminated, truncated,
                {"disk_to_zone": m.disk_to_zone, "in_zone": m.in_zone, "death": death,
                 "both_contact": both_fingertip, "both_fingertip_contact": both_fingertip,
                 "fingertip_contact": m.left_contact or m.right_contact,
                 "arm_body_contact": self._arm_body_ever, "arm_body_contact_this_step": arm_body_now,
                 "arm_body_contact_count": lg.arm_body_contact_count if lg is not None else 0,
                 "arm_body_contact_steps": self._arm_body_steps,
                 "arm_body_contact_impulse": lg.arm_body_contact_impulse if lg is not None else 0.0,
                 "arm_body_impulse_sum": self._arm_body_impulse_sum,
                 "delivered": delivered, "delivered_valid": delivered})

    def close(self) -> None:
        self.data = None
        self.model = None
        super().close()
