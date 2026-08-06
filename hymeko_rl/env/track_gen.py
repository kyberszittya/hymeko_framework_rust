"""Smooth race-track generation from **cubic Bézier splines**. A track is a closed loop of anchor points
turned into a C1-continuous curve (Catmull-Rom → Bézier handles), sampled densely into the waypoint array the
`WheeledVehicleEnv` follows. Smooth curvature matters for a fast car: a polygon of sharp corners makes
pure-pursuit chord-cut / spin out, whereas a Bézier loop has bounded, continuous curvature.

`race_circuit` presets a large flat GP-style circuit whose corner radii are big enough for the diff-drive
race car to hold at speed (a 55 m/s corner needs radius ≳ v²/(μg) ≈ 300 m); scale it down for slower cars."""
from __future__ import annotations

import numpy as np


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic Bézier point(s) at parameter ``t`` (scalar or array) for control points ``p0..p3``."""
    t = np.asarray(t, dtype=np.float64)[..., None]
    mt = 1.0 - t
    return mt ** 3 * p0 + 3 * mt ** 2 * t * p1 + 3 * mt * t ** 2 * p2 + t ** 3 * p3


def bezier_track(anchors: np.ndarray, *, samples_per_segment: int = 24, closed: bool = True) -> np.ndarray:
    """A smooth closed (or open) cubic-Bézier spline **through** ``anchors``, sampled into waypoints. The Bézier
    handles are the Catmull-Rom tangents (``bᵢ = Pᵢ ± (P₊−P₋)/6``), so the curve passes through every anchor
    with continuous tangents. # Preconditions ``len(anchors) >= 3`` (closed) or ``>= 2`` (open).
    # Postconditions returns ``(n_seg·samples_per_segment, 2)`` float64 waypoints along the track."""
    a = np.asarray(anchors, dtype=np.float64)
    n = len(a)
    if (closed and n < 3) or (not closed and n < 2):
        raise ValueError("need >= 3 anchors (closed) or >= 2 (open)")
    ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    segs = range(n) if closed else range(n - 1)
    out = []
    for i in segs:
        p0, p1 = a[(i - 1) % n], a[i]
        p2, p3 = a[(i + 1) % n], a[(i + 2) % n]
        b1 = p1 + (p2 - p0) / 6.0
        b2 = p2 - (p3 - p1) / 6.0
        out.append(cubic_bezier(p1, b1, b2, p2, ts))
    return np.concatenate(out, axis=0)


def race_circuit(scale: float = 100.0, *, samples_per_segment: int = 24) -> np.ndarray:
    """A large flat GP-style circuit (a long start straight + sweeping corners + a chicane), starting at the
    origin heading +x so the car spawns on the track. ``scale`` is the base unit in metres — at the default
    100 m the corner radii (a few hundred m) suit the 200 km/h diff-drive race car; scale down for slower cars.
    Anchors define the racing line; :func:`bezier_track` smooths them into waypoints."""
    s = float(scale)
    anchors = np.array([
        (0.0, 0.0),          # start / finish, heading +x
        (6.0, 0.0),          # long start straight
        (9.0, 1.5),          # turn 1 (right-hand sweep up)
        (10.0, 4.5),
        (8.0, 7.0),          # turn 2
        (4.5, 7.5),          # back straight
        (2.5, 6.0),          # chicane in
        (3.2, 4.2),          # chicane out
        (1.0, 2.5),          # turn 3 (hairpin-ish, largest curvature)
        (-1.0, 1.0),
    ], dtype=np.float64) * s
    return bezier_track(anchors, samples_per_segment=samples_per_segment, closed=True)
