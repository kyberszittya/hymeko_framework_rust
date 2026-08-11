"""The manipulated object as a declarative, typed spec — the object half of the HyMeKo scene.

R11.7A unification: the manipuland's *full* physical description (shape, principal dimensions,
out-of-plane thickness, mass/density, friction, slide/spin damping, semantic family) is read from the
``.hymeko`` scene's ``disk`` config term, rather than split between HyMeKo (radius only) and Python
``PlanarGraspEnv``/``compose_planar_scene`` keyword arguments. :class:`ObjectSpec` is the in-memory
form; :class:`EnvSpec` *has-a* :class:`ObjectSpec`.

The string field ``shape`` is fine at the HyMeKo boundary — it is parsed to the :class:`Shape` enum on
the way in (a mismatched string fails at parse, not at a deep ``_ => panic`` arm), per the repo's
string-at-boundary/enum-internally rule. The object handle in the compiled MuJoCo model is *always*
``"disk"`` regardless of shape, so downstream code resolves the object by a stable name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


def equal_area_regular_ngon_circumradius(equal_area_radius: float, n_sides: int) -> float:
    """Circumradius of a regular ``n_sides``-gon whose area equals a disk of ``equal_area_radius``.

    A regular ``n``-gon with circumradius ``R`` has area ``½·n·R²·sin(2π/n)``; setting it equal to
    ``π·r²`` gives ``R = sqrt(2π r² / (n·sin(2π/n)))``. This is the single source of truth shared by
    :meth:`ObjectSpec.footprint_radius` and ``compose_planar_scene``'s prism mesh generator, so the
    footprint standoff and the built geometry can never drift.

    # Preconditions ``equal_area_radius > 0``; ``n_sides >= 3``.
    # Postconditions ``> 0``; for ``n_sides == 3`` this equals ``sqrt(π r²/(3√3/4))`` bit-identically.
    """
    assert equal_area_radius > 0.0, f"equal_area_radius must be > 0, got {equal_area_radius}"
    assert n_sides >= 3, f"n_sides must be >= 3, got {n_sides}"
    return math.sqrt(2.0 * math.pi * equal_area_radius * equal_area_radius
                     / (n_sides * math.sin(2.0 * math.pi / n_sides)))


class Shape(Enum):
    """Manipuland geometry family. The value is exactly the ``coin_shape`` string
    :func:`hymeko_rl.env.planar_grasp_env.compose_planar_scene` dispatches on."""

    CYLINDER = "cylinder"   # the original round coin — rolls out of a straddle unless squeezed
    BOX = "box"             # a rectangular prism (``radius_y`` ⇒ rectangle) — flat faces, clamp-holdable
    TRIANGLE = "triangle"   # equal-area equilateral prism — the n=3 special case of NGON (kept for back-compat)
    NGON = "ngon"           # equal-area regular n-gon prism; ``n_sides`` picks the corner count (3→∞≈circle)
    ELLIPSE = "ellipse"     # flat elliptical prism (``radius``=r_x, ``radius_y``=r_y) — smooth, no corners
    CAPSULE = "capsule"     # flat stadium/discorectangle prism (``radius``=half-length a, ``radius_y``=cap radius b)

    @classmethod
    def from_str(cls, text: str) -> "Shape":
        """Parse a HyMeKo shape string → :class:`Shape`.

        # Preconditions ``text`` is one of the enum values (case-insensitive, trimmed).
        # Errors ``ValueError`` naming the offending string and the valid set (fails at the boundary).
        """
        key = text.strip().lower()
        for member in cls:
            if member.value == key:
                return member
        valid = ", ".join(repr(m.value) for m in cls)
        raise ValueError(f"unknown object shape {text!r}; valid shapes: {valid}")


# HyMeKo field defaults for the reference coin (OBJ_O0). Kept here so both the dataclass default and
# `from_fields` agree, and so the default ObjectSpec reproduces today's `compose_planar_scene` coin call.
_COIN_RADIUS = 0.02
_COIN_HALF = 0.02
_COIN_SLIDE_DAMPING = 2.5
_COIN_SPIN_DAMPING = 0.8


@dataclass(frozen=True)
class ObjectSpec:
    """The manipulated object's physical description. Defaults reproduce the reference coin (OBJ_O0).

    # Preconditions ``radius > 0``; ``half_thickness > 0``; ``radius_y`` is ``None`` or ``> 0``;
      ``density`` is ``None`` (MuJoCo default) or ``> 0``; ``frictionloss >= 0``; dampings ``>= 0``.
    # Postconditions :meth:`compose_kwargs` returns exactly the keyword set
      :func:`compose_planar_scene` accepts, so the object is one source of truth (no split).
    # Invariants the compiled model names the object geom/body ``"disk"`` for every :class:`Shape`.
    """

    family: str = "coin"
    shape: Shape = Shape.CYLINDER
    radius: float = _COIN_RADIUS
    radius_y: float | None = None
    half_thickness: float = _COIN_HALF
    density: float | None = None
    frictionloss: float = 0.0
    slide_damping: float = _COIN_SLIDE_DAMPING
    spin_damping: float = _COIN_SPIN_DAMPING
    n_sides: int | None = None      # NGON: corner count (>= 3); ignored for other shapes (TRIANGLE implies 3)

    def __post_init__(self) -> None:
        # Preconditions (DbC): a violated precondition is a bug in the caller / scene author.
        assert self.radius > 0.0, f"ObjectSpec.radius must be > 0, got {self.radius}"
        assert self.half_thickness > 0.0, f"ObjectSpec.half_thickness must be > 0, got {self.half_thickness}"
        assert self.radius_y is None or self.radius_y > 0.0, f"ObjectSpec.radius_y must be > 0 or None, got {self.radius_y}"
        assert self.density is None or self.density > 0.0, f"ObjectSpec.density must be > 0 or None, got {self.density}"
        assert self.frictionloss >= 0.0, f"ObjectSpec.frictionloss must be >= 0, got {self.frictionloss}"
        assert self.slide_damping >= 0.0 and self.spin_damping >= 0.0, "dampings must be >= 0"
        assert self.n_sides is None or self.n_sides >= 3, f"ObjectSpec.n_sides must be >= 3 or None, got {self.n_sides}"
        # An NGON needs its corner count declared; other shapes must not smuggle one in.
        if self.shape is Shape.NGON:
            assert self.n_sides is not None, "ObjectSpec.shape=NGON requires n_sides (the corner count)"
        # The round family reuses radius/radius_y as the two semi-axes; both need radius_y (the y extent).
        if self.shape in (Shape.ELLIPSE, Shape.CAPSULE):
            assert self.radius_y is not None, f"ObjectSpec.shape={self.shape.name} requires radius_y"
            if self.shape is Shape.CAPSULE:
                # Stadium: half-length a=radius must exceed cap radius b=radius_y (non-negative straight section).
                assert self.radius > self.radius_y, "ObjectSpec.shape=CAPSULE requires radius (half-length) > radius_y (cap radius)"

    def polygon_sides(self) -> int | None:
        """Corner count if this is a regular-polygon prism, else ``None``.

        # Postconditions 3 for TRIANGLE (its fixed corner count), ``n_sides`` for NGON, ``None`` for
          CYLINDER/BOX. Invariant: the return is ``None`` or ``>= 3``.
        """
        if self.shape is Shape.TRIANGLE:
            return 3
        if self.shape is Shape.NGON:
            return self.n_sides
        return None

    @classmethod
    def from_fields(cls, fields: dict[str, float | tuple[float, ...] | str]) -> "ObjectSpec":
        """Build an :class:`ObjectSpec` from a parsed ``disk`` config-term body
        (:func:`hymeko_rl.env._profile.parse_fields` output). Absent fields fall back to the coin
        default, so a bare ``{ radius 0.02; }`` reproduces OBJ_O0 exactly.

        # Preconditions ``shape``/``family`` (if present) are strings; numeric fields (if present) parse
          as floats. # Errors ``ValueError`` on an unknown ``shape`` string (via :meth:`Shape.from_str`).
        """
        def _f(name: str, default: float) -> float:
            v = fields.get(name)
            return float(v) if isinstance(v, (int, float)) else default

        def _opt(name: str) -> float | None:
            v = fields.get(name)
            return float(v) if isinstance(v, (int, float)) else None

        def _opt_int(name: str) -> int | None:
            v = fields.get(name)
            return int(v) if isinstance(v, (int, float)) else None

        shape_raw = fields.get("shape")
        shape = Shape.from_str(shape_raw) if isinstance(shape_raw, str) else Shape.CYLINDER
        family_raw = fields.get("family")
        family = family_raw if isinstance(family_raw, str) else "coin"
        return cls(
            family=family,
            shape=shape,
            radius=_f("radius", _COIN_RADIUS),
            radius_y=_opt("radius_y"),
            half_thickness=_f("half", _COIN_HALF),
            density=_opt("density"),
            frictionloss=_f("frictionloss", 0.0),
            slide_damping=_f("damping", _COIN_SLIDE_DAMPING),
            spin_damping=_f("spin_damping", _COIN_SPIN_DAMPING),
            n_sides=_opt_int("n_sides"),
        )

    def compose_kwargs(self) -> dict[str, object]:
        """The exact keyword set :func:`hymeko_rl.env.planar_grasp_env.compose_planar_scene` accepts.

        Centralizing this mapping is the point of the unification: callers pass one ``ObjectSpec``
        instead of six loose ``coin_*`` kwargs, and the object's identity lives in one place.

        # Postconditions keys are a subset of ``compose_planar_scene``'s manipuland parameters; the
          returned ``coin_shape`` equals ``self.shape.value``.
        """
        return {
            "disk_radius": self.radius,
            "disk_half": self.half_thickness,
            "coin_damping": self.slide_damping,
            "coin_density": self.density,
            "coin_shape": self.shape.value,
            "spin_damping": self.spin_damping,
            "coin_frictionloss": self.frictionloss,
            "disk_radius_y": self.radius_y,
            "disk_n_sides": self.polygon_sides(),
        }

    def planar_env_kwargs(self) -> dict[str, object]:
        """The object keyword set :class:`hymeko_rl.env.planar_grasp_env.PlanarGraspEnv` accepts (the
        ``__init__`` override surface). This is the mapping the reconstruct chain threads end-to-end —
        distinct from :meth:`compose_kwargs` (which targets ``compose_planar_scene`` directly).

        # Postconditions the returned ``coin_shape`` equals ``self.shape.value``; ``disk_radius_override``
          equals ``self.radius``. For the reference coin these values match ``PlanarGraspEnv``'s defaults,
          so an explicit spec and the pre-unification loose kwargs build the same model.
        """
        return {
            "coin_shape": self.shape.value,
            "disk_radius_override": self.radius,
            "disk_radius_y_override": self.radius_y,
            "disk_n_sides_override": self.polygon_sides(),
            "coin_density": self.density,
            "coin_frictionloss": self.frictionloss,
            "coin_damping": self.slide_damping,
            "spin_damping": self.spin_damping,
        }

    def footprint_radius(self) -> float:
        """The in-plane circumscribing radius of the object's footprint — the standoff a straddle
        approach must clear. For a cylinder this is ``radius``; for a rectangle it is the diagonal
        half-extent; for a regular-polygon prism (TRIANGLE / NGON) it is the equal-area circumradius
        of the cross-section. Used by the capture straddle-target placement (U3) to generalize off the
        coin-radius hard-code.

        # Postconditions ``> 0``.
        """
        if self.shape is Shape.BOX:
            hy = self.radius if self.radius_y is None else self.radius_y
            return math.hypot(self.radius, hy)
        if self.shape is Shape.ELLIPSE:
            # Circumscribing radius = the larger semi-axis (radius_y guaranteed by the precondition).
            return max(self.radius, self.radius_y if self.radius_y is not None else self.radius)
        if self.shape is Shape.CAPSULE:
            return self.radius     # the stadium's half-length a is its farthest point (a > b)
        sides = self.polygon_sides()
        if sides is not None:
            # Shared with compose_planar_scene's mesh generator ⇒ standoff and geometry cannot drift.
            return equal_area_regular_ngon_circumradius(self.radius, sides)
        return self.radius     # CYLINDER


# The reference coin (OBJ_O0) as a module constant — matches galambos_env.hymeko's @dsk declaration.
COIN_OBJECT = ObjectSpec()
