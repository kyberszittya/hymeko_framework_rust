"""O3 — triangular-prism footprint geometry + the FULL-FOOTPRINT delivery certificate.

The cylinder/box delivery certificate is centre-based (distance-to-zone ≤ tol). A triangle has corners that can stick out
of the zone even when its centroid is inside, and its delivery depends on ORIENTATION (a vertex vs an edge leading into
the zone). So O3 needs a stricter geometric predicate: **every base vertex of the (rotated, translated) triangle lies
within the target zone**. This module is pure geometry (no MuJoCo) — the physical env supplies the disk pose; the
validation script (o3_triangle_physical_prep) checks it against the compiled mesh + runtime inertia.
"""
from __future__ import annotations

import math

import numpy as np


def triangle_circumradius(disk_radius: float) -> float:
    """The circumradius of an equilateral triangle EQUAL-AREA to a ``disk_radius`` cylinder: (3√3/4)R² = π·disk_radius².
    Matches ``compose_planar_scene(coin_shape='triangle')``. # Postconditions R > 0."""
    return math.sqrt(math.pi * disk_radius * disk_radius / (3.0 * math.sqrt(3.0) / 4.0))


def triangle_base_vertices(circumradius: float) -> np.ndarray:
    """The 3 body-frame (x, y) base vertices (apex at +y), centroid at the origin — the same layout the MJCF mesh uses."""
    angs = [math.pi / 2 + k * 2 * math.pi / 3 for k in range(3)]
    return np.array([[circumradius * math.cos(a), circumradius * math.sin(a)] for a in angs], np.float64)


def triangle_footprint_world(disk_xy: np.ndarray, disk_rz: float, circumradius: float) -> np.ndarray:
    """World (x, y) of the 3 base vertices after the object's in-plane rotation ``disk_rz`` + translation ``disk_xy``.
    # Postconditions shape (3, 2)."""
    v = triangle_base_vertices(circumradius)
    c, s = math.cos(disk_rz), math.sin(disk_rz)
    rot = np.array([[c, -s], [s, c]], np.float64)
    return v @ rot.T + np.asarray(disk_xy, np.float64)


def full_footprint_certified(disk_xy: np.ndarray, disk_rz: float, circumradius: float,
                             zone_xy: np.ndarray, zone_half: float) -> bool:
    """FULL-FOOTPRINT certificate: EVERY base vertex is within ``zone_half`` of the zone centre (the whole triangle,
    corners included, is inside the target). Stricter than the centroid-in-zone test."""
    verts = triangle_footprint_world(disk_xy, disk_rz, circumradius)
    return bool(np.all(np.linalg.norm(verts - np.asarray(zone_xy, np.float64), axis=1) <= zone_half))

def footprint_margin(disk_xy: np.ndarray, disk_rz: float, circumradius: float,
                     zone_xy: np.ndarray, zone_half: float) -> float:
    """Signed clearance of the WORST vertex: ``zone_half − max_vertex_distance``. ≥0 ⇔ full-footprint certified."""
    verts = triangle_footprint_world(disk_xy, disk_rz, circumradius)
    return float(zone_half - np.max(np.linalg.norm(verts - np.asarray(zone_xy, np.float64), axis=1)))


def leading_feature(disk_rz: float, *, toward: float = math.pi / 2) -> str:
    """Whether a VERTEX or an EDGE leads toward the zone direction ``toward`` (default +y). The apex vertex is at
    ``π/2 + disk_rz``; the triangle has 3-fold symmetry (period 2π/3). 'vertex' if a vertex is within 30° of ``toward``,
    else 'edge'. Used for orientation stratification of the panel."""
    apex = math.pi / 2 + disk_rz
    delta = (apex - toward) % (2 * math.pi / 3)
    delta = min(delta, 2 * math.pi / 3 - delta)                 # angular distance to the nearest vertex direction
    return "vertex" if delta <= math.radians(30) else "edge"


def orientation_strata(n: int = 12) -> list[tuple[str, float]]:
    """A deterministic orientation panel over one 3-fold period, labelled vertex-leading / edge-leading — the O3
    ``vertex/edge/orientation`` stratification. # Postconditions n entries spanning [0, 2π/3)."""
    return [(leading_feature(rz), round(float(rz), 4)) for rz in np.linspace(0.0, 2 * math.pi / 3, n, endpoint=False)]
