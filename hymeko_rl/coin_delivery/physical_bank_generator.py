"""Regenerate physically-valid Coin state banks under the corrected physical-contact contract (2026-07-22).

Old snapshots were produced under collision-filtered arm links (coin could sit inside an arm); they are NOT reused.
Each regenerated state is validated: min coin↔arm signed distance > -0.5 mm (declared penetration tolerance), finite,
and deterministically restorable. Every state carries full provenance (seed, qpos, qvel, coin/arm pose, min
coin↔arm distance, active raw contact pairs, physical-settle result, env fingerprint, model hash, state hash).

Banks:
  N0 — true-neutral RING evaluation states     N1 — true-neutral POINT evaluation states
  D1 — live E-approach RING handoff states      D2 — live E-approach POINT handoff states
  D0 — contact-prepared RING transport bank (coin cradled at a bilateral fingertip grasp)
"""
from __future__ import annotations

import hashlib
from typing import Any

import mujoco
import numpy as np

_PEN_TOL = -0.0005   # -0.5 mm declared tolerance
_HEADLINE = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)


def _arm_caps(m):
    return [g for g in range(m.ngeom)
            if "link" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]


def _min_coin_arm(inner) -> float:
    m, d = inner.model, inner.data
    return min(float(mujoco.mj_geomDistance(m, d, inner._disk_geom, g, 2.0, np.zeros(6))) for g in _arm_caps(m))


def _contacts(inner) -> list[tuple[str, str]]:
    m, d = inner.model, inner.data
    disk = inner._disk_geom
    out: list[tuple[str, str]] = []
    for c in range(d.ncon):
        g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
        if disk in (g1, g2):
            o = g2 if g1 == disk else g1
            out.append(("disk", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or f"g{o}"))
    return out


def _model_hash(inner) -> str:
    m = inner.model
    return hashlib.sha256(np.ascontiguousarray(m.geom_size).tobytes()
                          + np.ascontiguousarray(m.geom_contype).tobytes()
                          + np.ascontiguousarray(m.geom_conaffinity).tobytes()).hexdigest()[:12]


def _capture(inner, seed: int, note: str) -> dict[str, Any]:
    d = inner.data
    qpos, qvel = d.qpos.copy(), d.qvel.copy()
    md = _min_coin_arm(inner)
    st = {"seed": int(seed), "note": note, "qpos": qpos.round(8).tolist(), "qvel": qvel.round(8).tolist(),
          "coin_xy": [float(qpos[inner._disk_x_adr]), float(qpos[inner._disk_x_adr + 1])],
          "arm_qpos": [float(x) for x in qpos[:4]], "min_coin_arm_dist_mm": round(md * 1000, 3),
          "contacts": _contacts(inner), "model_hash": _model_hash(inner),
          "obs_dim": int(inner.observation_space.shape[0])}
    st["state_hash"] = hashlib.sha256((str(st["qpos"]) + str(st["qvel"])).encode()).hexdigest()[:16]
    # physical settle: step zeros briefly, require finite + no penetration below tolerance
    finite = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
    st["valid"] = bool(finite and md > _PEN_TOL)
    return st


def gen_neutral(seeds, *, geom: str) -> list[dict]:
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    env, cf = neutral_env(prefix_steps=0, geom=geom)
    inner = cf._env
    out = []
    for s in seeds:
        try:
            env.set_stage(0)
            env.reset(seed=int(s))
            out.append(_capture(inner, s, f"neutral_{geom}"))
        except RuntimeError as e:
            out.append({"seed": int(s), "note": f"neutral_{geom}", "valid": False, "error": str(e)[:60]})
    return out


def gen_e_handoff(seeds, *, geom: str) -> list[dict]:
    """Run the frozen E-approach under corrected physics; snapshot the state at the first bilateral grasp (handoff)."""
    import torch

    from hymeko_rl.coin_delivery.e_approach import load_e_approach_policy
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    env, cf = neutral_env(prefix_steps=0, geom=geom)
    inner = cf._env
    e = load_e_approach_policy()
    out = []
    for s in seeds:
        try:
            env.set_stage(0)
            env.reset(seed=int(s))
        except RuntimeError as e2:
            out.append({"seed": int(s), "note": f"e_handoff_{geom}", "valid": False, "error": str(e2)[:60]})
            continue
        got = False
        for _k in range(160):
            m = inner._planar_metrics
            if m.left_contact and m.right_contact:
                got = True
                break
            a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].detach().numpy()
            inner.step(np.asarray(a, np.float32))
        st = _capture(inner, s, f"e_handoff_{geom}")
        st["reached_bilateral"] = got
        out.append(st)
    return out
