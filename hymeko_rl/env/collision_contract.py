"""COIN_ARM_COLLISION_CONTRACT — an explicit per-side collision-category mask contract for the planar coin scene.

Rationale (2026-07-27): the coin<->arm collision is verified functional (shallow-penetration + swept tests), but same-arm
isolation was enforced only by the emitter's ADJACENT-link `<exclude>` pairs — non-adjacent same-arm geoms remained
mask-collidable. That is a hidden contact-semantics difference that would surface as reachable-state discrepancies on
transfer (AIBO / humanoid / pick-and-place). This module sets an explicit, category-based mask so the contract holds for
EVERY pair, independent of the exclude list or the arm morphology:

    LEFT_ARM=1  RIGHT_ARM=2  COIN=4  WORLD=8

    left-arm  geom : contype=LEFT_ARM,  conaffinity=RIGHT_ARM|COIN|WORLD   (1 / 14)
    right-arm geom : contype=RIGHT_ARM, conaffinity=LEFT_ARM|COIN|WORLD    (2 / 13)
    coin      geom : contype=COIN,      conaffinity=LEFT_ARM|RIGHT_ARM|WORLD (4 / 11)
    world/floor    : contype=WORLD,     conaffinity=LEFT_ARM|RIGHT_ARM|COIN  (8 / 7)

MuJoCo's predicate ``(contype_A & conaffinity_B) | (contype_B & conaffinity_A) != 0`` then gives, for free:
same-arm ⇒ NO collide; left↔right ⇒ collide; arm↔coin ⇒ collide; arm/coin↔world ⇒ collide. Masks-only: the hypergraph
reads body/joint structure, not masks, so the graph and all morphology/dynamics are unchanged.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

LEFT_ARM, RIGHT_ARM, COIN, WORLD = 1, 2, 4, 8

_ROLE_MASK = {
    "left": (LEFT_ARM, RIGHT_ARM | COIN | WORLD),        # 1 / 14
    "right": (RIGHT_ARM, LEFT_ARM | COIN | WORLD),       # 2 / 13
    "coin": (COIN, LEFT_ARM | RIGHT_ARM | WORLD),        # 4 / 11
    "world": (WORLD, LEFT_ARM | RIGHT_ARM | COIN),       # 8 / 7
}


def role_masks(role: str) -> tuple[int, int]:
    """(contype, conaffinity) for a collision role ∈ {left, right, coin, world}. # Postconditions: same-arm masks share
    no bit (contype ∉ conaffinity of the same side)."""
    return _ROLE_MASK[role]


def _geom_role(geom: ET.Element, side: "str | None") -> "str | None":
    name = geom.get("name", "") or ""
    gtype = geom.get("type", "") or ""
    if name == "disk":
        return "coin"
    if gtype == "plane" or name == "floor":
        return "world"
    if side in ("left", "right"):
        return side
    return None                                          # unclassifiable (e.g. a stray visual geom) — left untouched


def apply_collision_contract(mjcf: str) -> str:
    """Rewrite every collision geom's contype/conaffinity in a composed planar-coin scene MJCF to the per-side contract.
    Arm side is taken from the nearest ancestor body whose name ends ``_left`` / ``_right``. Coin (``name="disk"``) and the
    floor/plane are classified by name/type. Returns the transformed MJCF string (recompilable). # Preconditions: a valid
    ``<mujoco>`` MJCF with a ``<worldbody>``. # Postconditions: only contype/conaffinity attributes change; geom types,
    sizes, poses, bodies, joints, and the `<contact>` section are untouched."""
    root = ET.fromstring(mjcf)

    def walk(elem: ET.Element, side: "str | None") -> None:
        for child in list(elem):
            if child.tag == "body":
                nm = child.get("name", "") or ""
                s = "left" if nm.endswith("_left") else ("right" if nm.endswith("_right") else side)
                walk(child, s)
            elif child.tag == "geom":
                role = _geom_role(child, side)
                if role is not None:
                    ct, ca = role_masks(role)
                    child.set("contype", str(ct))
                    child.set("conaffinity", str(ca))
            else:
                walk(child, side)
    wb = root.find("worldbody")
    if wb is not None:
        walk(wb, None)
    return ET.tostring(root, encoding="unicode")
