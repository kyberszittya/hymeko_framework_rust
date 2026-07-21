"""Standalone FULL-ACTION coin delivery (2026-07-22).

The RL contract of the residual campaign was ``u_exec = clip(grasp_carry + delta·tanh(policy))`` — an always-active
scripted base. This module instead defines the STANDALONE contract the full-action experiment requires:

    u_exec = clip(policy(observation))          # the full 6-DoF action, NO scripted base, NO residual composition

:class:`FullActionDeliveryEnv` reuses the corrected-physics dynamics, the 6→4 cooperative action mapping, and the
delivery reward/transition bookkeeping of :class:`CoinDeliveryTrainEnv`, but:
  * ``reset`` establishes the corrected-physics transport-prepared INITIAL STATE and runs NO scripted acquisition
    prefix (a start condition, not an online command);
  * ``step`` applies the policy's full action directly — ``_base()`` is never called, so the scripted controller
    contributes no online actuator command.

The scripted expert used to generate demonstrations is ``grasp_carry`` applied as the FULL action (verified to deliver
~8/9 on the corrected-physics panel). The BC target is therefore ``u_target = u_expert_executed`` (the full action),
never a zero residual.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np

from hymeko_rl.train.coin_delivery_rl import CoinDeliveryTrainEnv, DeliveryRLConfig, make_delivery_rl_env, p_grasp_carry

ActionFn = Callable[[np.ndarray], np.ndarray]


class FullActionDeliveryEnv(CoinDeliveryTrainEnv):
    """Standalone full-action delivery env. ``u_exec = clip(action, lo, hi)`` — no scripted base, no prefix during
    the policy rollout. # Invariants ``_base()`` is never invoked; every actuator command in a rollout is the
    policy's own full action."""

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, _info = self.env.reset(seed=seed)              # corrected-physics transport-prepared START (no prefix)
        self.env._horizon = self.cfg.prefix_cap + self.cfg.horizon + 8
        self._reset_state()
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._start_obs = self._last_obs.copy()
        self._prev_dtz = self._start_dtz = self._dtz()
        self._prev_both = self._both()
        return self._last_obs, {"handoff_event": False}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a_exec = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), self.cfg.lo, self.cfg.hi)   # FULL action
        self._record_telemetry(np.zeros(6, np.float32), np.asarray(action, np.float32), a_exec)        # base ≡ 0
        obs, _r, _t, _tr, sinfo = self.env.step(a_exec)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._suffix_t += 1
        reward, _terminated_center, truncated, safety, dtz = self._transition(sinfo)
        # full-action DELIVERY: do NOT terminate at center-reach — the strict certificate needs a 6-step HELD dwell,
        # which cannot accumulate if the episode ends the instant the coin first touches the centre. Run to the
        # horizon (or a safety violation); reward still credits the center event via _transition.
        terminated = bool(safety)
        return self._last_obs, reward, terminated, truncated, self._info(safety=safety, dtz=dtz)


def make_full_action_env(cfg: DeliveryRLConfig | None = None, *, fingertip_geometry: str = "POINT",
                         horizon: int = 160) -> FullActionDeliveryEnv:
    """A standalone full-action delivery env. ``horizon`` (default 160 = prefix_cap + transport) gives the policy time
    to acquire + transport + settle from the start state, since there is no scripted acquisition prefix."""
    base = make_delivery_rl_env(cfg, fingertip_geometry=fingertip_geometry)
    fa = FullActionDeliveryEnv(base.env, DeliveryRLConfig(**{**base.cfg.__dict__, "horizon": horizon}))
    return fa


def scripted_expert_fn(env: FullActionDeliveryEnv) -> ActionFn:
    """The scripted expert as a FULL action (``grasp_carry``: grasp midpoint → zone + squeeze) — what BC clones
    (``u_target = this``), NOT a zero residual. Reads only ``env.inner``/``suffix_t`` (deployment-available)."""
    return lambda _obs: p_grasp_carry(env.inner, env._suffix_t)


def _state_hash(env: FullActionDeliveryEnv) -> str:
    d = env.inner.data
    return hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes()).hexdigest()[:16]


def _phase(env: FullActionDeliveryEnv) -> str:
    """Deployment-available phase/context from the metrics (both-contact ⇒ carrying, else approaching)."""
    m = env.inner._planar_metrics
    if m.disk_to_zone <= env.cfg.zone_half:
        return "IN_ZONE"
    return "CARRY" if (m.left_contact and m.right_contact) else "APPROACH"


def rollout_expert(env: FullActionDeliveryEnv, seed: int, *, record: bool = True) -> dict[str, Any]:
    """Roll the scripted full-action expert on ``seed`` and (optionally) record per-step (obs, full action, phase,
    contacts, cert inputs, hashes). Returns the trajectory + success flags. The expert reads only ``env.inner``/
    ``suffix_t`` (deployment-available), and every executed action is stored as the BC target."""
    env.reset(seed=int(seed))
    steps: list[dict[str, Any]] = []
    entered = center = False
    for _t in range(env.cfg.horizon):
        obs = env._last_obs.copy()
        a = np.asarray(p_grasp_carry(env.inner, env._suffix_t), np.float32)   # the FULL expert action
        m = env.inner._planar_metrics
        if record:
            steps.append({"obs": obs.astype(np.float32), "action": a.astype(np.float32), "phase": _phase(env),
                          "left_contact": bool(m.left_contact), "right_contact": bool(m.right_contact),
                          "disk_to_zone": float(m.disk_to_zone), "state_hash": _state_hash(env)})
        env.step(a)
        dz = float(env.inner._planar_metrics.disk_to_zone)
        entered = entered or dz <= env.cfg.zone_half
        center = center or dz <= env.cfg.center_tol
    return {"seed": int(seed), "steps": steps, "entered": entered, "center": center, "n": len(steps)}


def eval_full_action(action_fn: ActionFn, seeds, env: FullActionDeliveryEnv, *, gamma: float = 0.99) -> dict[str, Any]:
    """Evaluate one action source (scripted expert or a learned policy) on ``seeds`` through the standalone
    full-action env — ONE consistent env-native + strict metric, at the declared horizon. Reports final success
    (native center + strict certificate) AND the temporal diagnostics (first zone-entry / first strict step,
    success-by-time, time-to-success stats, discounted return, success-curve AUC).

    ``action_fn(obs) -> 6-DoF`` for a policy; the scripted expert reads ``env`` state so pass
    ``lambda _o: p_grasp_carry(env.inner, env._suffix_t)``. Every executed command comes from ``action_fn`` — no
    scripted base (proven by the zero-action control)."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.coin_delivery.raw_strict_oracle import raw_cert_step
    horizon = env.cfg.horizon
    probes = (30, 60, 90, 120, horizon)
    rows = []
    for s in seeds:
        env.reset(seed=int(s))
        cert = DeliveryCertifier(initial_clearance=max(_min_coin_arm(env.inner), 1e-4))
        first_zone = first_strict = None
        center = False
        ret = 0.0
        for t in range(horizon):
            cert.update(raw_cert_step(env.inner))
            dz = float(env.inner._planar_metrics.disk_to_zone)
            if first_zone is None and dz <= env.cfg.zone_half:
                first_zone = t
            if first_strict is None and cert.delivery_certified:
                first_strict = t
            center = center or dz <= env.cfg.center_tol
            _o, r, term, trunc, _i = env.step(np.asarray(action_fn(env._last_obs), np.float32))
            ret += (gamma ** t) * float(r)
            if term or trunc:
                break
        cert.update(raw_cert_step(env.inner))
        strict = bool(cert.delivery_certified)
        if strict and first_strict is None:
            first_strict = horizon
        rows.append({"seed": int(s), "center": center, "strict": strict, "first_zone": first_zone,
                     "first_strict": first_strict, "return": ret})
    n = max(1, len(rows))
    tts = [r["first_strict"] for r in rows if r["first_strict"] is not None]
    succ_by_time = {k: int(sum(1 for r in rows if r["first_strict"] is not None and r["first_strict"] <= k))
                    for k in probes}
    # success-curve AUC: mean over time of the success-by-time fraction (normalised to [0,1])
    grid = list(range(0, horizon + 1, 5))
    auc = float(np.mean([sum(1 for r in rows if r["first_strict"] is not None and r["first_strict"] <= g) / n
                         for g in grid]))
    return {"n": len(rows), "center_rate": round(sum(r["center"] for r in rows) / n, 4),
            "strict_count": int(sum(r["strict"] for r in rows)), "strict_rate": round(sum(r["strict"] for r in rows) / n, 4),
            "success_by_time": succ_by_time, "tts_median": float(np.median(tts)) if tts else None,
            "tts_iqr": [float(np.percentile(tts, 25)), float(np.percentile(tts, 75))] if tts else None,
            "first_zone_median": float(np.median([r["first_zone"] for r in rows if r["first_zone"] is not None]))
            if any(r["first_zone"] is not None for r in rows) else None,
            "return_median": float(np.median([r["return"] for r in rows])), "success_curve_auc": round(auc, 4),
            "per_seed": rows}


def _min_coin_arm(inner) -> float:
    import mujoco
    m, d = inner.model, inner.data
    caps = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            and "link" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")]
    return min(float(mujoco.mj_geomDistance(m, d, inner._disk_geom, g, 2.0, np.zeros(6))) for g in caps)


def collect_expert_dataset(seeds, *, fingertip_geometry: str = "POINT", horizon: int = 160,
                           successful_only: bool = True) -> dict[str, Any]:
    """Generate the full-action expert dataset over ``seeds``: (obs → full executed action) pairs, keeping only
    trajectories where the scripted expert reaches the center (a *successful* full-action trajectory). Returns
    ``{"obs", "act", "meta"}`` with the flat BC arrays and provenance."""
    env = make_full_action_env(fingertip_geometry=fingertip_geometry, horizon=horizon)
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    kept, dropped = [], []
    for s in seeds:
        tr = rollout_expert(env, s)
        ok = tr["center"] if successful_only else True
        (kept if ok else dropped).append(int(s))
        if ok:
            for st in tr["steps"]:
                obs_all.append(st["obs"])
                act_all.append(st["action"])
    obs = np.asarray(obs_all, np.float32)
    act = np.asarray(act_all, np.float32)
    meta = {"n_seeds": len(list(seeds)), "n_kept": len(kept), "n_dropped": len(dropped), "kept": kept,
            "dropped": dropped, "n_transitions": int(obs.shape[0]), "horizon": horizon,
            "fingertip_geometry": fingertip_geometry, "bc_target": "u_expert_executed (full action, not residual)",
            "obs_hash": hashlib.sha256(obs.tobytes()).hexdigest()[:16] if obs.size else "",
            "act_hash": hashlib.sha256(act.tobytes()).hexdigest()[:16] if act.size else ""}
    return {"obs": obs, "act": act, "meta": meta}
