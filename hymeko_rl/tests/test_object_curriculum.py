"""R11.7A U6A — the object-variant curriculum loads from HyMeKo and realizes the intended single-axis ablations.

Each variant is read from its ``.hymeko`` scene (not constructed inline) and compiled to a MuJoCo model; the
tests assert the physical intent actually lands: O1-L holds mass = O0 while the footprint grows, O2-M doubles
mass at fixed geometry, O4-S is a square prism at O0's mass. This is the "mass/inertia/friction actually
differ" + "stable disk handle" + "HyMeKo-generated" portion of the U6A generation gate.
"""
from __future__ import annotations

import mujoco
import pytest

from hymeko_rl.coin_delivery.object_curriculum import U6A_CURRICULUM, variant
from hymeko_rl.env.object_spec import Shape
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

_M0 = 0.050265          # O0 reference disk mass (kg), from the compiled model
_MTOL = 1e-4


def _model_of(variant_id: str):
    env = PlanarGraspEnv(**variant(variant_id).object_spec.planar_env_kwargs())
    m = env.model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disk")
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    assert bid >= 0 and gid >= 0, f"{variant_id}: lost the stable 'disk' handle"
    return m, bid, gid


def test_all_variants_load_from_hymeko_with_stable_handle() -> None:
    # Intentional-membership pin: the curriculum is deliberately THIS set of single-axis ablations.
    # (Was stale at ["O0","O1-L","O2-M","O4-S"] after O5-R was added for R12.2; corrected + extended with O6-T.)
    assert [v.variant_id for v in U6A_CURRICULUM] == ["O0", "O1-L", "O2-M", "O4-S", "O5-R", "O6-T"]
    for v in U6A_CURRICULUM:
        spec = v.object_spec                      # read from the .hymeko scene
        assert spec.radius > 0.0
        _model_of(v.variant_id)                   # compiles + resolves "disk"


def test_o0_is_the_reference_coin() -> None:
    m, bid, gid = _model_of("O0")
    assert int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert m.body_mass[bid] == pytest.approx(_M0, abs=_MTOL)


def test_o1_large_holds_mass_grows_footprint() -> None:
    # SIZE ablation: bigger radius, mass held = O0, footprint (circumscribing radius) larger.
    spec = variant("O1-L").object_spec
    assert spec.shape is Shape.CYLINDER and spec.radius == pytest.approx(0.024)
    m, bid, _ = _model_of("O1-L")
    assert m.body_mass[bid] == pytest.approx(_M0, abs=_MTOL), "O1-L mass must match O0 (size-only ablation)"
    assert spec.footprint_radius() > variant("O0").object_spec.footprint_radius()


def test_o2_heavy_doubles_mass_fixed_geometry() -> None:
    # DYNAMICS ablation: same geometry as O0, 2x mass.
    spec = variant("O2-M").object_spec
    assert spec.shape is Shape.CYLINDER and spec.radius == pytest.approx(0.02)
    m, bid, _ = _model_of("O2-M")
    assert m.body_mass[bid] == pytest.approx(2.0 * _M0, abs=2 * _MTOL), "O2-M must be 2x O0 mass"


def test_o4_square_is_box_at_o0_mass() -> None:
    # SHAPE ablation: square prism, equal projected area ⇒ mass = O0.
    spec = variant("O4-S").object_spec
    assert spec.shape is Shape.BOX
    m, bid, gid = _model_of("O4-S")
    assert int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    assert m.body_mass[bid] == pytest.approx(_M0, abs=_MTOL), "O4-S must match O0 mass (equal-area, equal-thickness)"


def test_o6_triangle_is_corner_prism_at_o0_mass() -> None:
    # SHAPE-corner ablation: equilateral triangular prism (mesh), equal projected area ⇒ mass = O0, with a
    # LARGER footprint (circumradius) than the coin — the first sharp-cornered manipuland.
    spec = variant("O6-T").object_spec
    assert spec.shape is Shape.TRIANGLE and spec.radius == pytest.approx(0.02)
    m, bid, gid = _model_of("O6-T")
    assert int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH), "O6-T must be a convex MESH prism"
    assert m.body_mass[bid] == pytest.approx(_M0, abs=_MTOL), "O6-T must match O0 mass (equal-area, equal-thickness)"
    # equal-area equilateral circumradius = sqrt(pi r^2 / (3√3/4)) ≈ 1.5554 r > r; strictly exceeds the coin.
    assert spec.footprint_radius() == pytest.approx(0.031102, abs=1e-5)
    assert spec.footprint_radius() > variant("O0").object_spec.footprint_radius()


def test_variant_lookup_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown object variant"):
        variant("O3")
