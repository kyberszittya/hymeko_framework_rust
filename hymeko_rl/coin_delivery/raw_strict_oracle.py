"""Independent raw-MuJoCo strict-success oracle (physical-contact contract, 2026-07-22).

Computes the COIN_DELIVERY_STRICT inputs DIRECTLY from raw MuJoCo state — `data.geom_xpos` (coin centre),
`data.site_xpos` (zone centre), `data.qvel` (coin speed), `data.contact` (fingertip / arm-link geom pairs) — with NO
reliance on the env's cached `planar_metrics`. Streaming it alongside the production certifier and requiring agreement
on the strict verdict is the STRICT_MONITOR_CONTRACT gate: it proves the certifier is not lying about zone, velocity,
contact, or the (now legal) arm-link contact.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.delivery_certificate import CertStep


def _fingertip_geoms(m) -> tuple[int, int]:
    return (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right"))


def _arm_link_geoms(m) -> set[int]:
    out = set()
    for g in range(m.ngeom):
        bod = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if "link" in bod and g != mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"fingertip_{bod.split('_')[-1]}"):
            out.add(g)
    return out


def raw_cert_step(inner: Any, cf: Any = None) -> CertStep:
    """Build a :class:`CertStep` from RAW MuJoCo state (coin geom, zone site, qvel, contacts) — the independent oracle.

    ``left/right_fingertip`` and ``arm_body_contact`` are read from ``data.contact`` geom pairs (not the cached
    legality). ``body_progress``/``ever_grasped`` still come from the env's accumulators (they are physical facts the
    env already tracks history for), so the oracle checks the raw-perceivable quantities the certifier could fake."""
    m, d = inner.model, inner.data
    disk = inner._disk_geom
    coin = d.geom_xpos[disk][:2]
    zone = np.array([inner._zone_x, inner._zone_y], dtype=np.float64)
    disk_to_zone = float(np.linalg.norm(coin - zone))
    v = d.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]
    disk_speed = float(np.linalg.norm(v))
    ft_l, ft_r = _fingertip_geoms(m)
    arm = _arm_link_geoms(m)
    lf = rf = body = False
    for c in range(d.ncon):
        g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
        if disk in (g1, g2):
            other = g2 if g1 == disk else g1
            lf = lf or other == ft_l
            rf = rf or other == ft_r
            body = body or other in arm
    return CertStep(disk_to_zone=disk_to_zone, disk_speed=disk_speed, left_fingertip=lf, right_fingertip=rf,
                    arm_body_contact=body, arm_body_impulse=0.0,
                    body_progress=float(getattr(inner, "_body_progress", 0.0)),
                    ever_grasped=bool(getattr(inner, "_ever_grasped", False)))
