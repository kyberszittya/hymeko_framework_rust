"""Procedural terrain for the locomotion substrates via MuJoCo **heightfields** (``<hfield>``). MuJoCo
supports terrain natively: a heightfield is an ``nrow×ncol`` elevation grid (values in ``[0,1]``, scaled by
``z_top``) rendered + collided as a mesh. :func:`procedural_hfield` fills that grid (rolling hills / random
bumps / ramps); :func:`with_hfield` swaps the flat collision floor of an emitted MJCF for a heightfield geom;
:func:`fill_hfield` writes the elevation into a built model's ``hfield_data``.

The border is flattened to a ramp-in skirt so a vehicle can drive onto the terrain from the flat spawn."""
from __future__ import annotations

import re

import mujoco
import numpy as np

TerrainKind = str  # "flat" | "hills" | "bumps" | "ramps"


def procedural_hfield(kind: TerrainKind, n: int = 96, *, seed: int = 0, skirt: float = 0.12) -> np.ndarray:
    """An ``(n, n)`` elevation grid in ``[0, 1]``. ``skirt`` is the fraction of the border flattened to 0 (a
    flat ramp-in ring around the terrain). # Preconditions ``n >= 8``, ``0 <= skirt < 0.5``.
    # Postconditions ``min == 0`` on the border, values in ``[0, 1]``."""
    if n < 8 or not 0.0 <= skirt < 0.5:
        raise ValueError("n >= 8 and 0 <= skirt < 0.5 required")
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="ij")
    if kind == "flat":
        h = np.zeros((n, n), dtype=np.float64)
    elif kind == "hills":
        h = (np.sin(2 * np.pi * 2 * xx) * np.cos(2 * np.pi * 2 * yy)
             + 0.5 * np.sin(2 * np.pi * 3.5 * xx + 0.7) + 0.5 * np.cos(2 * np.pi * 3 * yy + 1.3))
    elif kind == "bumps":
        h = np.zeros((n, n), dtype=np.float64)
        for _ in range(24):
            cx, cy = rng.uniform(0.1, 0.9, 2)
            s = rng.uniform(0.03, 0.07)
            h += rng.uniform(0.4, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    elif kind == "ramps":
        h = np.clip(np.sin(2 * np.pi * 1.5 * xx), 0.0, None) * (0.5 + 0.5 * yy)
    else:
        raise ValueError(f"unknown terrain kind {kind!r}")
    h = h - h.min()
    if h.max() > 1e-9:
        h = h / h.max()
    # flatten a border skirt (Hann taper) so the spawn/edge is flat ground the vehicle rolls in from
    if skirt > 0:
        ramp = np.ones(n)
        w = max(1, int(skirt * n))
        ramp[:w] = np.linspace(0.0, 1.0, w)
        ramp[-w:] = np.linspace(1.0, 0.0, w)
        h = h * np.outer(ramp, ramp)
    return h.astype(np.float32)


def with_hfield(mjcf: str, *, radius: float, z_top: float, z_pos: float, nrow: int = 96, ncol: int = 96,
                material: str | None = "hk_grid_mat") -> str:
    """Replace the emitted flat collision floor (``name="floor"``) with a heightfield geom. The ``<hfield>``
    asset is added (data filled later by :func:`fill_hfield`); the geom spans ``2·radius`` m and rises to
    ``z_top`` m, based at ``z_pos``. Reuses the beautify checker material if present (``material``)."""
    mjcf = re.sub(r'<geom name="floor" type="plane"[^/]*/>', "", mjcf)
    hfield = (f'<hfield name="hk_terrain" nrow="{nrow}" ncol="{ncol}" '
              f'size="{radius:g} {radius:g} {z_top:g} 0.1"/>')
    mat = f' material="{material}"' if material else ' rgba="0.42 0.45 0.35 1"'
    geom = f'<geom name="hk_terrain_geom" type="hfield" hfield="hk_terrain" pos="0 0 {z_pos:g}"{mat}/>'
    if "<asset>" in mjcf:
        mjcf = mjcf.replace("<asset>", "<asset>" + hfield, 1)
    else:
        mjcf = re.sub(r"(<mujoco[^>]*>)", r"\1<asset>" + hfield + "</asset>", mjcf, count=1)
    return mjcf.replace("<worldbody>", "<worldbody>" + geom, 1)


def fill_hfield(model: mujoco.MjModel, elevation: np.ndarray, name: str = "hk_terrain") -> None:
    """Write an ``(n, n)`` elevation grid (``[0, 1]``) into the model's ``hfield_data``. # Preconditions the
    grid size matches the hfield's ``nrow·ncol``."""
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    if hid < 0:
        raise ValueError(f"heightfield {name!r} not in model")
    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    if elevation.shape != (nrow, ncol):
        raise ValueError(f"elevation {elevation.shape} != hfield ({nrow}, {ncol})")
    adr = int(model.hfield_adr[hid])
    model.hfield_data[adr:adr + nrow * ncol] = np.clip(elevation, 0.0, 1.0).astype(np.float32).ravel()
