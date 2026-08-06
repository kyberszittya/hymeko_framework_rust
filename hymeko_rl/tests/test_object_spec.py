"""U2 (R11.7A unification) — the typed object schema and its HyMeKo sourcing.

Covers: :class:`Shape` parsing (string-at-boundary → enum, incl. the failure case), :class:`ObjectSpec`
contracts + defaults reproducing the coin, ``from_fields`` sourcing, ``compose_kwargs`` staying in lockstep
with :func:`compose_planar_scene`'s signature, ``footprint_radius`` geometry, and the O0-unchanged
invariant: ``EnvSpec.from_hymeko(galambos_env.hymeko).object`` is exactly the reference coin and
``disk_radius`` still equals 0.02 (regression against the pre-unification scalar the builder consumes).
"""
from __future__ import annotations

import inspect
import math

import mujoco
import pytest

from hymeko_rl.env.env_spec import EnvSpec
from hymeko_rl.env.object_spec import COIN_OBJECT, ObjectSpec, Shape
from hymeko_rl.env.planar_grasp_env import compose_planar_scene

_SCENE = "data/robotics/galambos_env.hymeko"


# --- Shape (string-at-boundary → enum) -------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("cylinder", Shape.CYLINDER), ("box", Shape.BOX), ("triangle", Shape.TRIANGLE),
    ("  CyLiNdEr ", Shape.CYLINDER),   # trimmed + case-insensitive
])
def test_shape_from_str_valid(text: str, expected: Shape) -> None:
    assert Shape.from_str(text) is expected


def test_shape_from_str_unknown_fails_at_boundary() -> None:
    with pytest.raises(ValueError, match="unknown object shape 'sphere'"):
        Shape.from_str("sphere")


# --- ObjectSpec defaults + contracts ---------------------------------------------------------------
def test_default_objectspec_is_the_coin() -> None:
    o = ObjectSpec()
    assert o is not None and o == COIN_OBJECT
    assert (o.shape, o.radius, o.half_thickness, o.density, o.frictionloss) == (
        Shape.CYLINDER, 0.02, 0.02, None, 0.0)
    assert (o.slide_damping, o.spin_damping, o.family) == (2.5, 0.8, "coin")


@pytest.mark.parametrize("bad", [
    {"radius": -0.01}, {"radius": 0.0}, {"half_thickness": 0.0},
    {"radius_y": -0.1}, {"density": 0.0}, {"frictionloss": -1.0},
])
def test_objectspec_precondition_violations_assert(bad: dict) -> None:
    with pytest.raises(AssertionError):
        ObjectSpec(**bad)


# --- from_fields (HyMeKo sourcing) -----------------------------------------------------------------
def test_from_fields_bare_radius_reproduces_coin() -> None:
    # A pre-unification `{ radius 0.02; }` body must still yield the reference coin.
    assert ObjectSpec.from_fields({"radius": 0.02}) == COIN_OBJECT


def test_from_fields_box_variant() -> None:
    o = ObjectSpec.from_fields({"shape": "box", "family": "puck", "radius": 0.03,
                                "radius_y": 0.015, "density": 800.0, "frictionloss": 0.1})
    assert (o.shape, o.family, o.radius, o.radius_y) == (Shape.BOX, "puck", 0.03, 0.015)
    assert (o.density, o.frictionloss) == (800.0, 0.1)


def test_from_fields_unknown_shape_raises() -> None:
    with pytest.raises(ValueError, match="unknown object shape"):
        ObjectSpec.from_fields({"shape": "sphere", "radius": 0.02})


# --- compose_kwargs stays in lockstep with the builder signature -----------------------------------
def test_compose_kwargs_are_a_subset_of_the_builder_signature() -> None:
    params = set(inspect.signature(compose_planar_scene).parameters)
    kwargs = ObjectSpec().compose_kwargs()
    assert set(kwargs) <= params, f"compose_kwargs drifted from compose_planar_scene: {set(kwargs) - params}"
    assert kwargs["coin_shape"] == "cylinder"


def test_compose_kwargs_maps_variant() -> None:
    kw = ObjectSpec(shape=Shape.BOX, radius=0.03, radius_y=0.02, density=700.0,
                    frictionloss=0.2).compose_kwargs()
    assert kw["coin_shape"] == "box" and kw["disk_radius"] == 0.03
    assert kw["disk_radius_y"] == 0.02 and kw["coin_density"] == 700.0 and kw["coin_frictionloss"] == 0.2


# --- planar_env_kwargs (the chain-threaded PlanarGraspEnv mapping) --------------------------------
def test_planar_env_kwargs_for_coin() -> None:
    kw = COIN_OBJECT.planar_env_kwargs()
    assert kw["coin_shape"] == "cylinder" and kw["disk_radius_override"] == 0.02
    assert kw["disk_radius_y_override"] is None and kw["coin_density"] is None
    assert kw["coin_frictionloss"] == 0.0 and kw["coin_damping"] == 2.5 and kw["spin_damping"] == 0.8


def test_planar_env_kwargs_builds_box_density_model() -> None:
    # Endpoint proof that density/friction reach the compiled model (U3b gap closed): a box+density spec
    # builds a box geom whose mass exceeds the same box at the MuJoCo default density.
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

    dense = ObjectSpec(shape=Shape.BOX, radius=0.03, radius_y=0.02, density=4000.0)
    plain = ObjectSpec(shape=Shape.BOX, radius=0.03, radius_y=0.02)
    env_d = PlanarGraspEnv(**dense.planar_env_kwargs())
    env_p = PlanarGraspEnv(**plain.planar_env_kwargs())
    gid_d = mujoco.mj_name2id(env_d.model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    bid_d = mujoco.mj_name2id(env_d.model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    bid_p = mujoco.mj_name2id(env_p.model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    assert int(env_d.model.geom_type[gid_d]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    assert env_d.model.body_mass[bid_d] > env_p.model.body_mass[bid_p], "coin_density did not reach the model"


# --- footprint_radius (straddle standoff for U3) ---------------------------------------------------
def test_footprint_radius_by_shape() -> None:
    assert ObjectSpec(shape=Shape.CYLINDER, radius=0.02).footprint_radius() == pytest.approx(0.02)
    box = ObjectSpec(shape=Shape.BOX, radius=0.03, radius_y=0.04).footprint_radius()
    assert box == pytest.approx(math.hypot(0.03, 0.04))
    tri = ObjectSpec(shape=Shape.TRIANGLE, radius=0.02).footprint_radius()
    assert tri > 0.02  # circumradius of an equal-area equilateral cross-section exceeds the disk radius


# --- O0-unchanged invariant: the scene sources the coin, disk_radius regression -----------------
def test_scene_sources_the_reference_coin() -> None:
    spec = EnvSpec.from_hymeko(_SCENE)
    assert spec.object == COIN_OBJECT, "galambos_env.hymeko no longer sources the reference coin (O0 drift)"


def test_disk_radius_scalar_still_002_and_equals_object_radius() -> None:
    spec = EnvSpec.from_hymeko(_SCENE)
    assert spec.disk_radius == pytest.approx(0.02)          # pre-unification builder scalar unchanged
    assert spec.disk_radius == spec.object.radius            # single source of truth (no split)
