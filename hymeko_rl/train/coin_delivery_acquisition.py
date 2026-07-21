"""COIN-DELIVERY-OVERNIGHT-2 Track A — a genuinely NEW acquisition/regrasp control family (NOT a grasp_carry residual).

RL-2 (F-COIN-DELIVERY-RL2) + the Track-B centering result (F-COIN-DELIVERY-PRIMB) left the ACQUISITION wall untested:
15 GEOMETRIC_HARD (never grasp — grasp_carry drives the grasp MIDPOINT toward the ZONE, so it never straddles a coin
that is not already between the fingers) + 4 CONTACT_LOSS (grasp then drop). This module builds a distinct control
family that drives the FINGERTIPS TO THE COIN, aligns, then closes on a schedule — with retreat/retry and regrasp:

  phase FSM:  APPROACH → ALIGN → CLOSE → STABILIZE → (DONE)   with  RETREAT/RETRY  and  REGRASP-on-loss edges.

Modes (for the selector gate + ablation): symmetric / asymmetric-left / asymmetric-right approach; staggered closure;
retreat-and-retry; contact-loss regrasp; geometry-conditioning (approach angle from coin/fingertip/zone geometry).

Success is STABLE two-finger acquisition (both fingertips in contact for a dwell window) — an acquisition SUBTASK
success, NOT delivery success. Handoff remains a phase event and never terminates the rollout. NO env/reward/dynamics/
CORE change; the primitive reads the obs and emits the same 6-DoF cooperative action the env already accepts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from hymeko_rl.coin_delivery.env_factory import make_coin_contact_env as _env

# ── obs field indices (ACTOR_FIELDS) the primitive reads ─────────────────────────────────────────────────────────────
COIN = (0, 1)
L_TO_COIN = (14, 15)
R_TO_COIN = (16, 17)
COIN_TO_TARGET = (18, 19)
MID_TO_COIN = (20, 21)
APERTURE = 22
LEFT_C, RIGHT_C, BOTH_C, ARM_BODY = 26, 27, 28, 29

# CooperativeContactController action layout: a[0:2] translate midpoint, a[2] aperture, a[3] squeeze,
# a[4] balance shift along the finger axis, a[5] rotate.
ACT_DIM = 6


class Phase(str, Enum):
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    CLOSE = "CLOSE"
    STABILIZE = "STABILIZE"
    RETREAT = "RETREAT"
    DONE = "DONE"


class ApproachMode(str, Enum):
    SYMMETRIC = "symmetric"
    ASYM_LEFT = "asym_left"
    ASYM_RIGHT = "asym_right"


@dataclass(frozen=True)
class AcqParams:
    """Acquisition-primitive parameters + mode flags. Continuous params are CEM/grid-searched; the flags select
    modes (for the selector gate) and are toggled for the component ablation."""

    pregrasp_radius: float = 0.05      # mid_to_coin distance below which ALIGN engages
    approach_gain: float = 20.0        # proportional gain driving the midpoint to the coin
    open_amount: float = 0.7           # aperture command during APPROACH/ALIGN (open the fingers)
    close_amount: float = 0.85         # squeeze command during CLOSE/STABILIZE
    approach_angle: float = 0.0        # geometry-conditioned rotate command during ALIGN (a[5])
    asym_offset: float = 0.0           # asymmetric balance shift (a[4]) magnitude
    stagger: float = 0.0               # staggered-closure balance bias (close one finger first)
    align_steps: int = 3               # steps spent in ALIGN before CLOSE
    stabilize_dwell: int = 6           # both-contact steps required for STABLE acquisition (success)
    retry_count: int = 2               # max retreat/retry cycles
    retreat_steps: int = 4             # steps spent retreating before re-approach
    retry_angle_offset: float = 0.4    # rotate offset added each retry
    stuck_steps: int = 8               # consecutive no-progress steps that trigger RETREAT
    # mode flags
    approach_mode: ApproachMode = ApproachMode.SYMMETRIC
    staggered: bool = False
    retry: bool = True
    regrasp: bool = True
    geometry_conditioned: bool = True  # if False, no approach-angle / asym conditioning (ablation)

    @staticmethod
    def from_unit(u: np.ndarray, base: "AcqParams | None" = None) -> "AcqParams":
        """Map a normalised ``[0,1]^8`` vector to the CEM-searched continuous parameters (mode flags from ``base``)."""
        v = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        b = base or AcqParams()
        return replace(b,
                       pregrasp_radius=0.02 + v[0] * 0.08, approach_gain=8.0 + v[1] * 32.0,
                       open_amount=0.3 + v[2] * 0.7, close_amount=0.4 + v[3] * 0.6,
                       approach_angle=(v[4] - 0.5) * 1.2, asym_offset=v[5] * 0.6,
                       stagger=v[6] * 0.6, retry_angle_offset=(v[7] - 0.5) * 1.2)


def scramble_geometry(obs: np.ndarray, rng_perm: np.ndarray) -> np.ndarray:
    """Scramble the geometry fields the primitive conditions on (the equal-budget structural control): permute + sign-
    flip the mid_to_coin / l_to_coin / r_to_coin / coin_to_target vectors so geometry-conditioning is destroyed while
    the search budget/capacity is identical. ``rng_perm`` is a fixed 4-permutation with signs (len 4, values in
    {+idx,-idx})."""
    o = obs.copy()
    vecs = [MID_TO_COIN, L_TO_COIN, R_TO_COIN, COIN_TO_TARGET]
    orig = [obs[list(v)].copy() for v in vecs]
    for dst, src_signed in zip(vecs, rng_perm):
        src = abs(int(src_signed))
        sign = 1.0 if src_signed >= 0 else -1.0
        o[list(dst)] = sign * orig[src][::-1]      # permute source + swap x/y + optional sign flip
    return o


class AcquisitionPrimitive:
    """Stateful acquisition FSM. ``action(obs)`` returns a 6-DoF cooperative action; ``reset()`` clears episode state."""

    def __init__(self, params: AcqParams) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.APPROACH
        self.dwell = 0
        self.align_t = 0
        self.retreat_t = 0
        self.retries = 0
        self.angle = self.p.approach_angle
        self.had_both = False
        self.stuck = 0
        self.prev_mid = None

    # ── geometry helpers ──────────────────────────────────────────────────────────────────────────────────────────
    def _mid_to_coin(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        v = obs[list(MID_TO_COIN)].astype(np.float64)
        return v, float(np.hypot(v[0], v[1]))

    def _approach_translate(self, obs: np.ndarray) -> np.ndarray:
        v, n = self._mid_to_coin(obs)
        d = v / (n + 1e-9)
        return np.clip(self.p.approach_gain * n, 0.0, 1.0) * d      # proportional: drive the midpoint ONTO the coin

    def _asym_balance(self, obs: np.ndarray) -> float:
        if not self.p.geometry_conditioned:
            return 0.0
        if self.p.approach_mode == ApproachMode.ASYM_LEFT:
            return self.p.asym_offset
        if self.p.approach_mode == ApproachMode.ASYM_RIGHT:
            return -self.p.asym_offset
        # symmetric: correct toward whichever fingertip is farther from the coin
        ld = float(np.hypot(*obs[list(L_TO_COIN)])); rd = float(np.hypot(*obs[list(R_TO_COIN)]))
        return self.p.asym_offset * np.sign(ld - rd)

    # ── per-phase actions ─────────────────────────────────────────────────────────────────────────────────────────
    def _a_approach(self, obs: np.ndarray, mid_n: float) -> np.ndarray:
        t = self._approach_translate(obs)
        return np.array([t[0], t[1], self.p.open_amount, 0.0, self._asym_balance(obs), 0.0], np.float32)

    def _a_align(self, obs: np.ndarray) -> np.ndarray:
        t = 0.4 * self._approach_translate(obs)
        ang = self.angle if self.p.geometry_conditioned else 0.0
        return np.array([t[0], t[1], self.p.open_amount * 0.5, 0.1, self._asym_balance(obs), ang], np.float32)

    def _a_close(self, obs: np.ndarray) -> np.ndarray:
        bal = self.p.stagger if (self.p.staggered and not self.had_both) else 0.0
        return np.array([0.0, 0.0, -0.2, self.p.close_amount, bal, 0.0], np.float32)

    def _a_stabilize(self) -> np.ndarray:
        return np.array([0.0, 0.0, -0.1, self.p.close_amount, 0.0, 0.0], np.float32)

    def _a_retreat(self, obs: np.ndarray) -> np.ndarray:
        v, n = self._mid_to_coin(obs)
        d = v / (n + 1e-9)
        return np.array([-0.4 * d[0], -0.4 * d[1], self.p.open_amount, -0.5, 0.0, self.angle], np.float32)

    # ── FSM step ──────────────────────────────────────────────────────────────────────────────────────────────────
    def action(self, obs: np.ndarray) -> np.ndarray:
        both = bool(obs[BOTH_C] > 0.5)
        self.had_both = self.had_both or both
        _v, mid_n = self._mid_to_coin(obs)
        self._update_stuck(mid_n)
        self._maybe_regrasp(both)
        if self.phase == Phase.APPROACH:
            return self._step_approach(obs, mid_n)
        if self.phase == Phase.ALIGN:
            return self._step_align(obs)
        if self.phase == Phase.CLOSE:
            return self._step_close(obs, both)
        if self.phase == Phase.STABILIZE:
            return self._step_stabilize(both)
        if self.phase == Phase.RETREAT:
            return self._step_retreat(obs)
        return self._a_stabilize()                                 # DONE — hold the grasp

    def _update_stuck(self, mid_n: float) -> None:
        if self.prev_mid is not None and self.phase == Phase.APPROACH:
            self.stuck = self.stuck + 1 if (self.prev_mid - mid_n) < 1e-3 else 0
        self.prev_mid = mid_n

    def _maybe_regrasp(self, both: bool) -> None:
        if self.p.regrasp and self.had_both and (not both) and self.phase in (Phase.CLOSE, Phase.STABILIZE):
            self.phase = Phase.RETREAT
            self.retreat_t = 0
            self.dwell = 0

    def _step_approach(self, obs: np.ndarray, mid_n: float) -> np.ndarray:
        if mid_n <= self.p.pregrasp_radius:
            self.phase, self.align_t = Phase.ALIGN, 0
        elif self.p.retry and self.stuck >= self.p.stuck_steps:
            self.phase, self.retreat_t = Phase.RETREAT, 0
        return self._a_approach(obs, mid_n)

    def _step_align(self, obs: np.ndarray) -> np.ndarray:
        self.align_t += 1
        if self.align_t >= self.p.align_steps:
            self.phase = Phase.CLOSE
        return self._a_align(obs)

    def _step_close(self, obs: np.ndarray, both: bool) -> np.ndarray:
        if both:
            self.phase, self.dwell = Phase.STABILIZE, 0
        return self._a_close(obs)

    def _step_stabilize(self, both: bool) -> np.ndarray:
        self.dwell = self.dwell + 1 if both else 0
        if self.dwell >= self.p.stabilize_dwell:
            self.phase = Phase.DONE
        return self._a_stabilize()

    def _step_retreat(self, obs: np.ndarray) -> np.ndarray:
        self.retreat_t += 1
        if self.retreat_t >= self.p.retreat_steps:
            self.retries += 1
            self.angle = self.p.approach_angle + self.retries * self.p.retry_angle_offset
            self.phase, self.stuck = (Phase.APPROACH if self.retries <= self.p.retry_count else Phase.RETREAT), 0
        return self._a_retreat(obs)


# ── rollout + funnel metrics (handoff NON-terminal; success = stable two-finger dwell) ───────────────────────────────
def roll_acquisition(env, seed: int, params: AcqParams, *, horizon: int = 120,
                     scramble_perm: "np.ndarray | None" = None) -> dict:
    """Roll the acquisition primitive from a pre-contact snapshot; report the acquisition FUNNEL + metrics.

    Funnel: valid_pregrasp → first_contact → two_finger_contact → stable_dwell (= acquisition SUCCESS). Handoff is a
    phase event (never terminates); the rollout stops on stable acquisition, safety, or horizon."""
    inner = env._env
    obs, _info = env.reset(seed=int(seed))
    env._horizon = horizon + 4
    prim = AcquisitionPrimitive(params)
    prim.reset()
    pregrasp = first_contact = two_finger = stable = safety = False
    had_both = False
    t_stable = None
    t = 0
    for t in range(horizon):
        o_read = obs if scramble_perm is None else scramble_geometry(obs, scramble_perm)
        a = np.clip(prim.action(o_read), -1.0, 1.0).astype(np.float32)
        obs, _r, _term, _trunc, info = env.step(a)
        m = inner._planar_metrics
        left, right = bool(m.left_contact), bool(m.right_contact)
        both = left and right
        pregrasp = pregrasp or (float(np.hypot(*obs[list(MID_TO_COIN)])) <= params.pregrasp_radius)
        first_contact = first_contact or left or right
        two_finger = two_finger or both
        had_both = had_both or both
        if prim.phase == Phase.DONE and not stable:
            stable, t_stable = True, t
        if info.get("safety_violation"):
            safety = True
            break
        if stable:
            break
    return {"seed": int(seed), "pregrasp_aligned": pregrasp, "first_contact": first_contact,
            "two_finger_contact": two_finger, "stable_acquisition": stable, "time_to_stable": t_stable,
            "retries": prim.retries, "contact_lost": bool(had_both and not both), "safety": safety,
            "steps": t + 1, "final_phase": prim.phase.value}


def make_acq_env():
    return _env()


def eval_acquisition(params: AcqParams, seeds, *, env=None, horizon: int = 120,
                     scramble_perm: "np.ndarray | None" = None) -> dict:
    """Aggregate the acquisition funnel over ``seeds`` (stable-acquisition rate + per-stage funnel + times)."""
    env = env or make_acq_env()
    rows = [roll_acquisition(env, s, params, horizon=horizon, scramble_perm=scramble_perm) for s in seeds]
    n = max(1, len(rows))
    st = [r for r in rows if r["stable_acquisition"]]
    return {"n": len(rows), "rows": rows,
            "pregrasp_rate": round(sum(r["pregrasp_aligned"] for r in rows) / n, 4),
            "first_contact_rate": round(sum(r["first_contact"] for r in rows) / n, 4),
            "two_finger_rate": round(sum(r["two_finger_contact"] for r in rows) / n, 4),
            "stable_acquisition_rate": round(sum(r["stable_acquisition"] for r in rows) / n, 4),
            "n_stable": sum(r["stable_acquisition"] for r in rows),
            "contact_lost_rate": round(sum(r["contact_lost"] for r in rows) / n, 4),
            "safety_rate": round(sum(r["safety"] for r in rows) / n, 4),
            "time_to_stable_med": (round(float(np.median([r["time_to_stable"] for r in st])), 1) if st else None),
            "recovered_seeds": [r["seed"] for r in st]}


@dataclass(frozen=True)
class AcqOracleConfig:
    pop: int = 16
    elite: int = 5
    iters: int = 6
    horizon: int = 120


def cem_acquisition(seeds, ocfg: AcqOracleConfig, rng: np.random.Generator, *, env, base: AcqParams | None = None,
                    scramble_perm: "np.ndarray | None" = None, log=None) -> dict:
    """CEM over the 8 continuous acquisition params (mode flags fixed by ``base``) to maximise stable-acquisition count
    on ``seeds``. Objective: n_stable, tie-broken by the funnel depth (two_finger + first_contact)."""
    dim = 8
    mean, sigma = np.full(dim, 0.5), np.full(dim, 0.3)
    best = {"score": -1e9, "params": base or AcqParams(), "n_stable": 0, "eval": {}}
    for it in range(ocfg.iters):
        pops = np.clip(rng.normal(mean, sigma, size=(ocfg.pop, dim)), 0.0, 1.0)
        if it == 0:
            pops[0] = 0.5
        scored = []
        for u in pops:
            p = AcqParams.from_unit(u, base)
            ev = eval_acquisition(p, seeds, env=env, horizon=ocfg.horizon, scramble_perm=scramble_perm)
            score = ev["n_stable"] + 0.1 * ev["two_finger_rate"] + 0.05 * ev["first_contact_rate"]
            scored.append((score, u, p, ev))
        scored.sort(key=lambda x: x[0])
        elites = np.stack([u for _s, u, _p, _e in scored[-ocfg.elite:]])
        mean, sigma = elites.mean(0), elites.std(0) + 1e-2
        top = scored[-1]
        if top[0] > best["score"]:
            best = {"score": top[0], "params": top[2], "n_stable": top[3]["n_stable"], "eval": top[3]}
        if log is not None:
            log(f"  [cem-acq it {it + 1}/{ocfg.iters}] best n_stable={best['n_stable']}/{len(list(seeds))} "
                f"pregrasp={best['eval'].get('pregrasp_rate')} two_finger={best['eval'].get('two_finger_rate')}")
    return best
