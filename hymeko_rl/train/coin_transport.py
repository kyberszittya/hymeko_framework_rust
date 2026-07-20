"""COIN-TRANSPORT-1 — object-centric clamp-and-translate transport action space.

COIN-DELIVERY-OVERNIGHT-2 (F-COIN-DELIVERY-ACQ) left the decisive gap: the acquisition primitive recovers 12/19 hard
states but grasp_carry cannot then transport them (chained 0/0). Hypothesis: grasp_carry under-uses the action space —
transport needs COORDINATED control of the grasp MIDPOINT, the CLAMP aperture, and a DIFFERENTIAL finger correction,
not a naive translate+squeeze.

The env's 6-DoF cooperative action is ALREADY object-centric (verified cooperative_contact_controller.targets):
  a[0:2] = midpoint translation Δc   ⊥   a[2] = aperture Δa, a[3] = squeeze (clamp retention)   ⊥   a[4] = differential Δd
  ⊥ a[5] = rotate.
So the object-centric transport action u = [Δc_x, Δc_y, Δa, Δd] maps to the env action with a primitive-set squeeze
(clamp) and rotate. The INVARIANT is midpoint ⊥ clamp ⊥ differential — NOT independent unconstrained fingertip deltas.

FROZEN upstream: env/dynamics/reward/monitor/acquisition FSM (no-regrasp), the acquired cohort, grasp_carry baseline,
easy-state benchmark. Transport begins from exact handoff snapshots (planar_snapshot). NO RL here; NO env/CORE change.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar
from hymeko_rl.train.coin_delivery_acquisition import (
    MID_TO_COIN, L_TO_COIN, R_TO_COIN, BOTH_C, APERTURE,
    AcqParams, AcquisitionPrimitive, Phase,
)

_CENTER_TOL = 0.02
_ZONE_HALF = 0.04


# ── object-centric action mapping (midpoint ⊥ clamp ⊥ differential) ──────────────────────────────────────────────────
def obj_to_env(dc: np.ndarray, da: float, dd: float, squeeze: float, rotate: float = 0.0) -> np.ndarray:
    """Map an object-centric transport command to the env's 6-DoF cooperative action. dc = midpoint Δ (2), da =
    aperture Δ, dd = differential Δ, squeeze = clamp-retention control, rotate = orientation. Invariant: these are the
    factored control axes (a[0:2] midpoint, a[2] aperture, a[3] squeeze, a[4] differential, a[5] rotate)."""
    return np.clip(np.array([dc[0], dc[1], da, squeeze, dd, rotate], np.float32), -1.0, 1.0)


def fingertip_targets(mid: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p_L = c + d/2, p_R = c - d/2 (the object-centric decomposition, for the roundtrip test)."""
    return mid + 0.5 * d, mid - 0.5 * d


# ── handoff snapshots (exact restorable acquisition terminals) ────────────────────────────────────────────────────────
@dataclass
class Handoff:
    seed: int
    snap: object                       # PlanarSnapshot (restorable)
    obs: np.ndarray
    coin_xy: tuple
    aperture: float
    both_contact: bool
    terminal_phase: str
    state_hash: str


def _hash_state(inner) -> str:
    d = inner.data
    return hashlib.sha1(np.concatenate([d.qpos, d.qvel]).tobytes()).hexdigest()[:16]


def extract_handoffs(env, seeds, acq_params: AcqParams, *, horizon: int = 90) -> list[Handoff]:
    """Freeze the acquisition primitive; roll each seed to its stable-acquisition terminal and snapshot the exact state.
    Only states that reach DONE (stable two-finger acquisition) produce a handoff."""
    inner = env._env
    out = []
    for sd in seeds:
        obs, _i = env.reset(seed=int(sd))
        env._horizon = horizon + 200
        prim = AcquisitionPrimitive(acq_params)
        prim.reset()
        for _ in range(horizon):
            obs, _r, _t, _tr, info = env.step(np.clip(prim.action(obs), -1, 1).astype(np.float32))
            if prim.phase == Phase.DONE:
                m = inner._planar_metrics
                out.append(Handoff(seed=int(sd), snap=snapshot_planar(inner), obs=np.asarray(obs, np.float32),
                                   coin_xy=(float(m.disk_pos[0]), float(m.disk_pos[1])), aperture=float(obs[APERTURE]),
                                   both_contact=bool(m.left_contact and m.right_contact),
                                   terminal_phase=prim.phase.value, state_hash=_hash_state(inner)))
                break
            if info.get("safety_violation"):
                break
    return out


# ── transport parameters + primitive families ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TransportParams:
    v_translate: float = 0.8       # midpoint translation velocity toward the zone
    a_target: float = 0.85         # clamp squeeze (retention hold)
    k_a: float = 0.0               # compliant aperture gain (T2): Δa = -k_a·(contact_deficit)
    k_d: float = 0.0               # differential recentering gain (T3)
    h_clearance: float = 0.0       # clearance/stabilise steps before translate (T4)
    t_dwell: int = 4               # settle dwell after zone entry
    r_zone_approach: float = 0.5   # slow-down radius near the zone (proportional approach)

    @staticmethod
    def from_unit(u: np.ndarray, base: "TransportParams | None" = None) -> "TransportParams":
        v = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
        b = base or TransportParams()
        return replace(b, v_translate=0.3 + v[0] * 0.7, a_target=0.3 + v[1] * 0.7, k_a=v[2] * 1.5,
                       k_d=v[3] * 1.5, r_zone_approach=0.02 + v[4] * 0.1)


class TState(str, Enum):
    HANDOFF = "HANDOFF"
    CLAMP_VERIFY = "CLAMP_VERIFY"
    CLEAR = "CLEAR"
    TRANSLATE = "TRANSLATE"
    ZONE_ENTRY = "ZONE_ENTRY"
    SETTLE = "SETTLE"
    RELEASE = "RELEASE"


class TransportFamily(str, Enum):
    T0_GRASP_CARRY = "T0_grasp_carry"
    T1_RIGID = "T1_rigid_clamp"
    T2_COMPLIANT = "T2_compliant_clamp"
    T3_DIFFERENTIAL = "T3_differential_recenter"
    T4_STAGED = "T4_staged"
    T5_RECOVERY = "T5_recovery"


def _approach_scale(n: float, r: float, v: float) -> float:
    """Proportional slow-down near the zone: full v when far, scales to a floor near the zone centre."""
    return v * float(np.clip(n / max(1e-6, r), 0.15, 1.0))


def _differential(obs: np.ndarray) -> float:
    """Coin off-centre between the fingertips → a differential correction to recentre it (uses l/r-to-coin asymmetry)."""
    ld = float(np.hypot(*obs[list(L_TO_COIN)])); rd = float(np.hypot(*obs[list(R_TO_COIN)]))
    return float(np.clip(rd - ld, -1.0, 1.0))          # + shifts toward the farther (right) finger


def _contact_deficit(obs: np.ndarray) -> float:
    """A bounded geometric contact-retention proxy (NOT force): how far the fingertips are from the coin."""
    return float(np.clip(0.5 * (np.hypot(*obs[list(L_TO_COIN)]) + np.hypot(*obs[list(R_TO_COIN)])), 0.0, 1.0))


class TransportPrimitive:
    """Deterministic object-centric transport primitive. `family` selects T0-T5; params tune it. Stateful (FSM for T4,
    recovery latch for T5)."""

    def __init__(self, family: TransportFamily, params: TransportParams) -> None:
        self.family = family
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.state = TState.HANDOFF
        self.clear_t = 0
        self.recovered = False
        self.last_stable = None

    def action(self, inner, obs: np.ndarray) -> np.ndarray:
        d, n = inner.direction_to_zone()
        both = bool(obs[BOTH_C] > 0.5)
        if self.family == TransportFamily.T0_GRASP_CARRY:
            return obj_to_env(np.array([d[0], d[1]]), 0.0, 0.0, 0.8)      # historical baseline
        if self.family == TransportFamily.T5_RECOVERY:
            return self._t5(inner, obs, d, n, both)
        if self.family == TransportFamily.T4_STAGED:
            return self._t4(inner, obs, d, n, both)
        return self._translate(obs, d, n)                                # T1/T2/T3 differ only by clamp/differential

    def _translate(self, obs: np.ndarray, d: np.ndarray, n: float) -> np.ndarray:
        v = _approach_scale(n, self.p.r_zone_approach, self.p.v_translate)
        dc = v * d
        da = -self.p.k_a * _contact_deficit(obs) if self.family in (TransportFamily.T2_COMPLIANT,) else 0.0
        dd = self.p.k_d * _differential(obs) if self.family in (TransportFamily.T3_DIFFERENTIAL,) else 0.0
        return obj_to_env(dc, da, dd, self.p.a_target)

    def _t4(self, inner, obs, d, n, both) -> np.ndarray:
        if self.state == TState.HANDOFF:
            self.state = TState.CLAMP_VERIFY
        if self.state == TState.CLAMP_VERIFY:
            if both:
                self.state = TState.CLEAR
            return obj_to_env(np.zeros(2), -0.3, 0.0, self.p.a_target)    # verify clamp: close aperture, hold
        if self.state == TState.CLEAR and self.clear_t < int(self.p.h_clearance):
            self.clear_t += 1
            return obj_to_env(np.zeros(2), 0.0, 0.0, self.p.a_target)     # stabilise in place
        self.state = TState.TRANSLATE
        v = _approach_scale(n, self.p.r_zone_approach, self.p.v_translate)
        dd = self.p.k_d * _differential(obs)
        return obj_to_env(v * d, -self.p.k_a * _contact_deficit(obs), dd, self.p.a_target)

    def _t5(self, inner, obs, d, n, both) -> np.ndarray:
        if both:
            self.last_stable = (obs[list(MID_TO_COIN)].copy(), self.p.a_target)
            return self._translate(obs, d, n)
        if not self.recovered:                                            # ONE bounded re-clamp on loss
            self.recovered = True
            return obj_to_env(np.zeros(2), -0.5, 0.0, min(1.0, self.p.a_target + 0.1))
        return obj_to_env(np.zeros(2), 0.0, 0.0, self.p.a_target)         # give up translation after one recovery


# ── scramble controls (transport-level falsification) ────────────────────────────────────────────────────────────────
class Scramble(str, Enum):
    S0_CORRECT = "S0_correct"
    S1_MIDPOINT = "S1_midpoint_scramble"
    S2_APERTURE_SIGN = "S2_aperture_sign_scramble"
    S3_DIFFERENTIAL = "S3_differential_scramble"
    S4_INDEP_FINGER = "S4_independent_finger"
    S5_RANDOM = "S5_random_matched"


def scramble_action(a: np.ndarray, mode: Scramble, rng: np.random.Generator) -> np.ndarray:
    """Apply a matched-magnitude scramble to the object-centric action (the load-bearing falsification control)."""
    a = a.copy()
    if mode == Scramble.S0_CORRECT:
        return a
    if mode == Scramble.S1_MIDPOINT:
        a[0], a[1] = a[1], -a[0]                                          # rotate the midpoint direction 90° (wrong midpoint)
    elif mode == Scramble.S2_APERTURE_SIGN:
        a[2] = -a[2]                                                      # invert the clamp correction, same magnitude
    elif mode == Scramble.S3_DIFFERENTIAL:
        a[4] = -a[4]                                                      # invert the differential relation
    elif mode == Scramble.S4_INDEP_FINGER:
        perm = rng.permutation(np.abs(a[[0, 1, 2, 4]]))                  # same params, no coordinated factorisation
        a[0], a[1], a[2], a[4] = perm * np.sign(rng.uniform(-1, 1, 4))
    elif mode == Scramble.S5_RANDOM:
        mag = float(np.linalg.norm(a[:2]))
        r = rng.uniform(-1, 1, 2); a[0], a[1] = mag * r / (np.linalg.norm(r) + 1e-9)  # matched-amplitude random midpoint
    return np.clip(a, -1.0, 1.0)


# ── gate evaluation G0-G5 ────────────────────────────────────────────────────────────────────────────────────────────
def _run_episode(env, inner, prim, obs, horizon, scramble, rng) -> dict:
    """Roll the transport primitive from a restored state; return the raw transport signals (contact/dtz/times)."""
    start_dtz = float(inner._planar_metrics.disk_to_zone)
    retain_run = 0
    zone = center = False
    min_dtz = start_dtz
    t_center = None
    for t in range(horizon):
        a = np.clip(prim.action(inner, obs), -1, 1).astype(np.float32)
        if scramble != Scramble.S0_CORRECT:
            a = scramble_action(a, scramble, rng)
        obs, _r, _term, _trunc, info = env.step(a)
        m = inner._planar_metrics
        both = bool(m.left_contact and m.right_contact)
        dtz = float(m.disk_to_zone); min_dtz = min(min_dtz, dtz)
        retain_run = retain_run + 1 if both else 0
        zone = zone or (dtz <= _ZONE_HALF and both)
        if dtz <= _CENTER_TOL and t_center is None:
            center, t_center = True, t
        if info.get("safety_violation"):
            break
    return {"start_dtz": start_dtz, "min_dtz": min_dtz, "retain_run": retain_run, "zone": zone, "center": center,
            "t_center": t_center}


def roll_transport(env, handoff: Handoff, family: TransportFamily, params: TransportParams, *, horizon: int = 140,
                   scramble: Scramble = Scramble.S0_CORRECT, rng: np.random.Generator | None = None) -> dict:
    """Restore the handoff, run the transport primitive, report the highest gate reached (G0..G5)."""
    inner = env._env
    restore_planar(inner, handoff.snap)
    env._horizon = horizon + 200
    both0 = bool(inner._planar_metrics.left_contact and inner._planar_metrics.right_contact)
    prim = TransportPrimitive(family, params)
    prim.reset()
    s = _run_episode(env, inner, prim, handoff.obs.copy(), horizon, scramble, rng or np.random.default_rng(0))
    g = [both0, (s["retain_run"] >= 8 or s["zone"] or s["center"]), (s["start_dtz"] - s["min_dtz"]) >= 0.02,
         s["zone"], s["center"], (s["center"] or (s["zone"] and s["min_dtz"] <= _ZONE_HALF)) and s["center"]]
    highest = max([i for i, gi in enumerate(g) if gi], default=-1)
    return {"seed": handoff.seed, "g0": g[0], "g1": g[1], "g2": g[2], "g3_zone": g[3], "g4_center": g[4],
            "g5_release": g[5], "highest_gate": highest, "start_dtz": round(s["start_dtz"], 4),
            "min_dtz": round(s["min_dtz"], 4), "progress": round(s["start_dtz"] - s["min_dtz"], 4),
            "time_to_center": s["t_center"]}


def eval_transport(env, handoffs: list[Handoff], family: TransportFamily, params: TransportParams, *,
                   scramble: Scramble = Scramble.S0_CORRECT, horizon: int = 140) -> dict:
    """Aggregate the transport gates over the handoff cohort."""
    rng = np.random.default_rng(0)
    rows = [roll_transport(env, h, family, params, horizon=horizon, scramble=scramble, rng=rng) for h in handoffs]
    n = max(1, len(rows))
    return {"family": family.value, "scramble": scramble.value, "n": len(rows),
            "clamp_retention": round(sum(r["g1"] for r in rows) / n, 4),
            "transport_progress": round(sum(r["g2"] for r in rows) / n, 4),
            "zone_entry": sum(r["g3_zone"] for r in rows), "center_reach": sum(r["g4_center"] for r in rows),
            "zone_entry_rate": round(sum(r["g3_zone"] for r in rows) / n, 4),
            "progress_med": round(float(np.median([r["progress"] for r in rows])), 4),
            "rows": rows}
