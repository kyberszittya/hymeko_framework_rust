"""O3 — triangular-prism footprint geometry + full-footprint certificate (pure geometry, no MuJoCo)."""
import math

import numpy as np

from hymeko_rl.coin_delivery.triangle_footprint import (
    footprint_margin, full_footprint_certified, leading_feature, orientation_strata,
    triangle_base_vertices, triangle_circumradius, triangle_footprint_world)


def test_circumradius_is_equal_area_to_cylinder():
    r = 0.020
    R = triangle_circumradius(r)
    tri_area = 3.0 * math.sqrt(3.0) / 4.0 * R * R
    assert abs(tri_area - math.pi * r * r) < 1e-9        # equal-area to the r-cylinder


def test_base_vertices_centroid_at_origin():
    v = triangle_base_vertices(triangle_circumradius(0.020))
    assert v.shape == (3, 2)
    assert np.allclose(v.mean(0), 0.0, atol=1e-9)        # centroid at the body origin
    assert np.allclose(np.linalg.norm(v, axis=1), triangle_circumradius(0.020))  # all at the circumradius


def test_footprint_world_translation_and_rotation():
    R = triangle_circumradius(0.020)
    at = np.array([0.1, 0.2])
    w0 = triangle_footprint_world(at, 0.0, R)
    assert np.allclose(w0.mean(0), at, atol=1e-9)        # centroid translates
    w120 = triangle_footprint_world(at, 2 * math.pi / 3, R)   # 120° = the 3-fold symmetry ⇒ same vertex SET
    assert np.allclose(np.sort(w0, axis=0), np.sort(w120, axis=0), atol=1e-6)


def test_full_footprint_certificate_stricter_than_centroid():
    R = triangle_circumradius(0.020)
    zone, zh = np.array([0.0, 0.16]), 0.055
    assert full_footprint_certified(zone, 0.0, R, zone, zh)          # centred triangle fits
    # a shift keeping the centroid inside but pushing a vertex out
    shift = zone + np.array([0.0, zh - R + 0.005])
    assert np.linalg.norm(shift - zone) <= zh                        # centroid still in the zone
    assert not full_footprint_certified(shift, 0.0, R, zone, zh)     # but a corner is out ⇒ NOT certified
    assert not full_footprint_certified(zone + np.array([0.3, 0.0]), 0.0, R, zone, zh)


def test_footprint_margin_sign_matches_certificate():
    R = triangle_circumradius(0.020)
    zone, zh = np.array([0.0, 0.16]), 0.055
    assert footprint_margin(zone, 0.0, R, zone, zh) >= 0             # certified ⇒ margin ≥ 0
    out = zone + np.array([0.0, 0.05])
    assert (footprint_margin(out, 0.0, R, zone, zh) >= 0) == full_footprint_certified(out, 0.0, R, zone, zh)


def test_leading_feature_and_strata():
    assert leading_feature(0.0) == "vertex"                          # apex at +y is vertex-leading toward the zone
    strata = orientation_strata(12)
    assert len(strata) == 12 and {lf for lf, _ in strata} == {"vertex", "edge"}   # both leading features present
