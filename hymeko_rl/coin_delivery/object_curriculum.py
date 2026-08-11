"""R11.7A U6 — the object-variant curriculum, as HyMeKo scene declarations.

Each variant is a full ``.hymeko`` scene identical to the reference Galambos scene except for the ``@dsk``
object declaration, so an object family is *another HyMeKo file*, not a Python branch. The :class:`ObjectSpec`
is read from that scene via :meth:`EnvSpec.from_hymeko` — the same path the deployed pipeline uses — so the
curriculum is generated from HyMeKo, not hard-coded.

First round (user-scoped): O0 reference coin + three single-axis ablations, one variant per family:
  * **O1-L** SIZE — radius 1.20x, mass+friction held = O0 (density lowered), inertia recomputed from geometry.
  * **O2-M** DYNAMICS — geometry = O0, mass 2x (density doubled), friction = O0.
  * **O4-S** SHAPE — square prism, equal projected area ⇒ mass = O0, friction = O0.
Later single-axis SHAPE ablations reuse the existing ``compose_planar_scene`` branches (no new geom code):
  * **O5-R** SHAPE-elongated — 1.5:1 rectangle (``radius_y``), 180°-symmetric; the orientation family (R12.2).
  * **O6-T** SHAPE-corner — equilateral triangular prism, equal projected area ⇒ mass = O0; the first
    manipuland with SHARP CORNERS (footprint circumradius ≈ 31 mm > the coin's 20 mm). Track A2, 2026-08-12.
  * **O7-P / O8-H** SHAPE-corner-count — regular pentagonal (n=5) / hexagonal (n=6) prisms, equal area ⇒
    mass = O0. Together with O6-T (n=3) they span the corner-count axis 3→5→6→…→circle, all from the one
    ``compose_planar_scene`` "ngon" generator (triangle = its n=3 special case). Track A2, 2026-08-12.
O3 (ellipse/capsule) is intentionally parked until the others run clean (it needs a new ``Shape`` member + a
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
    # R12.2-A2: an elongated (1.5:1) rectangle — 180° symmetry, so distinct orientations span [0,180°) (vs the
    # square's [0,45°]); the object family for the orientation-aware geometric model. Equal area ⇒ mass = O0.
    ObjectVariant("O5-R", f"{_ROBOTICS}/galambos_env_o5_rect.hymeko", "shape-elongated"),
    # Track A2 (2026-08-12): the first corner-bearing object — an equilateral triangular prism, equal projected
    # area ⇒ mass = O0. Three sharp vertices; footprint circumradius ≈ 31 mm (> coin 20 mm). Reuses the existing
    # compose_planar_scene "triangle" branch (Shape.TRIANGLE), no new geometry code.
    ObjectVariant("O6-T", f"{_ROBOTICS}/galambos_env_o6_triangle.hymeko", "shape-corner"),
    # Track A2 N-gon family (2026-08-12): pentagon (n=5) + hexagon (n=6), equal area ⇒ mass = O0. With O6-T (n=3)
    # they span the corner-count axis via the unified Shape.NGON "ngon" generator. Footprints ~23.0 mm / ~22.0 mm.
    ObjectVariant("O7-P", f"{_ROBOTICS}/galambos_env_o7_pentagon.hymeko", "shape-corner-count"),
    ObjectVariant("O8-H", f"{_ROBOTICS}/galambos_env_o8_hexagon.hymeko", "shape-corner-count"),
)


def variant(variant_id: str) -> ObjectVariant:
    """Look up a curriculum entry by id. # Errors ``KeyError`` if unknown (fail loud)."""
    for v in U6A_CURRICULUM:
        if v.variant_id == variant_id:
            return v
    raise KeyError(f"unknown object variant {variant_id!r}; known: {[v.variant_id for v in U6A_CURRICULUM]}")
