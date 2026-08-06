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

from dataclasses import dataclass
from enum import Enum


class Shape(Enum):
    """Manipuland geometry family. The value is exactly the ``coin_shape`` string
    :func:`hymeko_rl.env.planar_grasp_env.compose_planar_scene` dispatches on."""

    CYLINDER = "cylinder"   # the original round coin — rolls out of a straddle unless squeezed
    BOX = "box"             # a rectangular prism (``radius_y`` ⇒ rectangle) — flat faces, clamp-holdable
    TRIANGLE = "triangle"   # an equal-area equilateral triangular prism

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

    def __post_init__(self) -> None:
        # Preconditions (DbC): a violated precondition is a bug in the caller / scene author.
        assert self.radius > 0.0, f"ObjectSpec.radius must be > 0, got {self.radius}"
        assert self.half_thickness > 0.0, f"ObjectSpec.half_thickness must be > 0, got {self.half_thickness}"
        assert self.radius_y is None or self.radius_y > 0.0, f"ObjectSpec.radius_y must be > 0 or None, got {self.radius_y}"
        assert self.density is None or self.density > 0.0, f"ObjectSpec.density must be > 0 or None, got {self.density}"
        assert self.frictionloss >= 0.0, f"ObjectSpec.frictionloss must be >= 0, got {self.frictionloss}"
        assert self.slide_damping >= 0.0 and self.spin_damping >= 0.0, "dampings must be >= 0"

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
            "coin_density": self.density,
            "coin_frictionloss": self.frictionloss,
            "coin_damping": self.slide_damping,
            "spin_damping": self.spin_damping,
        }

    def footprint_radius(self) -> float:
        """The in-plane circumscribing radius of the object's footprint — the standoff a straddle
        approach must clear. For a cylinder this is ``radius``; for a rectangle it is the diagonal
        half-extent; for the equal-area triangular prism it is the circumradius of the equilateral
        cross-section. Used by the capture straddle-target placement (U3) to generalize off the
        coin-radius hard-code.

        # Postconditions ``> 0``.
        """
        import math

        if self.shape is Shape.CYLINDER:
            return self.radius
        if self.shape is Shape.BOX:
            hy = self.radius if self.radius_y is None else self.radius_y
            return math.hypot(self.radius, hy)
        # TRIANGLE: compose_planar_scene builds an equal-area equilateral prism whose side s satisfies
        # (3√3/4)·(s/√3)²… ; its circumradius equals the equal-area radius parameter R = √(π r² /(3√3/4)).
        return math.sqrt(math.pi * self.radius * self.radius / (3.0 * math.sqrt(3.0) / 4.0))


# The reference coin (OBJ_O0) as a module constant — matches galambos_env.hymeko's @dsk declaration.
COIN_OBJECT = ObjectSpec()
