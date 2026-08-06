"""R11.7A U6 — the object-variant curriculum, as HyMeKo scene declarations.

Each variant is a full ``.hymeko`` scene identical to the reference Galambos scene except for the ``@dsk``
object declaration, so an object family is *another HyMeKo file*, not a Python branch. The :class:`ObjectSpec`
is read from that scene via :meth:`EnvSpec.from_hymeko` — the same path the deployed pipeline uses — so the
curriculum is generated from HyMeKo, not hard-coded.

First round (user-scoped): O0 reference coin + three single-axis ablations, one variant per family:
  * **O1-L** SIZE — radius 1.20x, mass+friction held = O0 (density lowered), inertia recomputed from geometry.
  * **O2-M** DYNAMICS — geometry = O0, mass 2x (density doubled), friction = O0.
  * **O4-S** SHAPE — square prism, equal projected area ⇒ mass = O0, friction = O0.
O3 (ellipse/capsule) is intentionally parked until O1/O2/O4 run clean (it needs a new ``Shape`` member + a
``compose_planar_scene`` geom branch — an architectural change we do not want to make mid-measurement).
"""
from __future__ import annotations

from dataclasses import dataclass

from hymeko_rl.env.env_spec import EnvSpec
from hymeko_rl.env.object_spec import ObjectSpec

_ROBOTICS = "data/robotics"


@dataclass(frozen=True)
class ObjectVariant:
    """One curriculum entry: a stable id, the HyMeKo scene that declares it, and the single axis it ablates.

    # Invariants ``scene`` is a ``.hymeko`` profile whose ``@dsk`` declares the manipuland; ``object_spec``
      is read from it (never constructed inline), so the variant is HyMeKo-sourced.
    """

    variant_id: str
    scene: str
    ablation: str

    @property
    def object_spec(self) -> ObjectSpec:
        """The manipuland :class:`ObjectSpec`, read from this variant's HyMeKo scene."""
        return EnvSpec.from_hymeko(self.scene).object


# The reference coin (O0) is the frozen control; the three variants are single-axis ablations.
U6A_CURRICULUM: tuple[ObjectVariant, ...] = (
    ObjectVariant("O0", f"{_ROBOTICS}/galambos_env.hymeko", "reference"),
    ObjectVariant("O1-L", f"{_ROBOTICS}/galambos_env_o1_large.hymeko", "size"),
    ObjectVariant("O2-M", f"{_ROBOTICS}/galambos_env_o2_heavy.hymeko", "dynamics"),
    ObjectVariant("O4-S", f"{_ROBOTICS}/galambos_env_o4_square.hymeko", "shape"),
)


def variant(variant_id: str) -> ObjectVariant:
    """Look up a curriculum entry by id. # Errors ``KeyError`` if unknown (fail loud)."""
    for v in U6A_CURRICULUM:
        if v.variant_id == variant_id:
            return v
    raise KeyError(f"unknown object variant {variant_id!r}; known: {[v.variant_id for v in U6A_CURRICULUM]}")
