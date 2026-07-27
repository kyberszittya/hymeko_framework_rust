"""R4 Stage 1 — the causal closed-loop `ResponseState` + energy/stopping diagnostics.

The R1–R3 arc proved an open-loop fixed intent does not cover held-out (all 2/4, held 0/2); the concrete witness is s4,
whose cradle tolerates only a gentle transport (peak coin speed 0.263 m/s) where the R3 predictor asks for a dev-like
strong one (0.47). R4's hypothesis is that the over-speed, undecidable *before* the trajectory, is *measurable within the
first few physical steps* and correctable online. This module measures that response from the LIVE branched `rl`
mid-rollout using only **past + present** physical signals — the same class the frozen K6 monitor and the option's
velocity-feedback brake already read. No future value, no K6-as-input, no teacher intent/θ, no held-out label.

The load-bearing diagnostic is the **stopping margin**: whether the coin's current kinetic energy can be dissipated before
the delivery zone under the realistically-reachable braking authority. Expressed two equivalent ways (the guard uses the
mass-free form, so it does not depend on a coin-mass lookup):

    E_kin = ½·m_coin·speed²                          W_brake = m_coin·a_brake·d_safe
    d_stop = v_parallel² / (2·a_brake)               guard:  d_stop > d_safe   ⇔   E_kin > W_brake

with a_brake the governor-realised, slew-admissible one-step reachable deceleration (`brake_opposed_reach / control_dt`),
floored so a vanishing authority reads as *un-brakable* (conservatively over-speed), never as a division blow-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL
from hymeko_rl.coin_delivery.contact_velocity import (
    frozen_contact_frames, measure_contact_velocities, primary_fingertip_contacts)
from hymeko_rl.coin_delivery.forward_displacement import _coin_speed, _coin_xy

_EPS = 1e-9


# ── pure energy / stopping algebra (physics-free, unit-tested directly) ──────────────────────────────────────────────
def kinetic_energy(m_coin: float, speed: float) -> float:
    """# Postconditions: ≥ 0; 0 iff speed 0."""
    return 0.5 * float(m_coin) * float(speed) ** 2


def braking_decel(brake_reach: float, control_dt: float, *, gain: float = 1.0, floor: float = 1e-4) -> float:
    """Governor-realised achievable deceleration a_brake ≈ (one-step reachable coin Δv along −v) / control_dt, scaled by a
    dev-tuned ``gain`` and floored. FLOORED (not ε-added): a vanishing braking authority yields the floor, so d_stop is
    LARGE (conservatively over-speed) rather than infinite. # Postconditions: ≥ floor > 0."""
    a = float(gain) * max(float(brake_reach), 0.0) / max(float(control_dt), _EPS)
    return max(a, float(floor))


def stopping_distance(v_parallel: float, a_brake: float) -> float:
    """d_stop = v_par² / (2·a_brake) — the along-track distance to bleed v_parallel to rest at a_brake. # Postconditions:
    ≥ 0; monotone increasing in |v_parallel|; monotone decreasing in a_brake."""
    v = abs(float(v_parallel))
    return v * v / (2.0 * max(float(a_brake), _EPS))


def brake_work(m_coin: float, a_brake: float, d_safe: float) -> float:
    """Available braking work over the safe remaining distance W_brake = m_coin·a_brake·d_safe. # Postconditions: ≥ 0."""
    return float(m_coin) * float(a_brake) * max(float(d_safe), 0.0)


def over_speed(d_stop: float, d_safe: float) -> bool:
    """The mass-free stopping guard d_stop > d_safe, identical in sign to E_kin > W_brake (both divide out m_coin·a_brake).
    # Postconditions: True ⇔ the coin cannot stop before the zone at the reachable braking authority."""
    return bool(float(d_stop) > float(d_safe))


# ── R5: online effective-coast-deceleration estimator (physics-free; unit-tested directly) ────────────────────────────
@dataclass
class CoastEstimator:
    """Online estimate of the effective coast deceleration `â_eff` from the coin's own along-track velocity decay — the R5
    fix for R4's fixed `a_friction` (which mistimed release per cradle). A sample `−Δv∥/Δt` is admitted only from a window
    where the coin is genuinely coasting/braking: no active forward push, speed above `v_min` (stable ratio), and physically
    decelerating (Δv∥ < 0). The estimate is the robust MEDIAN over the last `k_est` samples, clamped to a physical band; a
    prior is used below `n_min` samples. Deterministic; no future / K6 / teacher.

    # Preconditions: `update` is called once per control step with the (prev, now) along-track velocities and whether the
    step drove an active forward push. # Postconditions: `estimate()` ∈ [a_lo, a_hi]; equals `a_prior` below n_min samples;
    invariant to sample order except through the sliding window; a single outlier cannot move the median by more than one
    rank."""

    a_prior: float = 0.55
    a_lo: float = 0.20
    a_hi: float = 1.20
    v_min: float = 0.08
    k_est: int = 8
    n_min: int = 2
    _samples: "list[float]" = field(default_factory=list)

    def update(self, v_prev: float, v_now: float, dt: float, *, active_push: bool) -> None:
        if active_push or float(v_prev) <= self.v_min or float(dt) <= _EPS:
            return
        dvdt = (float(v_now) - float(v_prev)) / float(dt)
        if dvdt >= 0.0:                                        # accelerating / flat ⇒ not a coast sample
            return
        self._samples.append(-dvdt)
        if len(self._samples) > self.k_est:
            self._samples.pop(0)

    def estimate(self) -> float:
        if len(self._samples) < self.n_min:
            return float(self.a_prior)
        return float(np.clip(np.median(self._samples), self.a_lo, self.a_hi))

    @property
    def n_samples(self) -> int:
        return len(self._samples)


# ── the probe reference (captured at the probe start; deltas are measured against it) ────────────────────────────────
@dataclass(frozen=True)
class ProbeReference:
    """The coin state at the probe start, so a window measures a genuine Δ (displacement, velocity gain) rather than an
    absolute. Captured once from the live rl; carries no future information."""

    coin_xy0: np.ndarray
    speed0: float
    t0: int

    @staticmethod
    def capture(rl: Any, t: int) -> "ProbeReference":
        return ProbeReference(coin_xy0=np.asarray(_coin_xy(rl), np.float64), speed0=float(_coin_speed(rl)), t0=int(t))


@dataclass(frozen=True)
class MeasureContext:
    """The causal, cradle-fixed scalars the measurement needs but cannot read from the live rl alone: the handoff-measured
    authority (same source the R3 decoder uses — object B_coin reaches + contact/internal reaches), the control timestep,
    coin mass, the slew step, the actuator box, the live rate-limited torque, and the a_brake gain (dev-tuned)."""

    authority: "dict[str, float]"
    control_dt: float
    m_coin: float
    slew: float
    lo: np.ndarray
    hi: np.ndarray
    prev_tau: np.ndarray
    brake_gain: float = 1.0


@dataclass
class ResponseState:
    """One causal closed-loop response measurement. `as_dict()` is the audit record (rounded)."""

    dtz: float
    d_safe: float
    v_parallel: float
    v_perpendicular: float
    coin_speed: float
    coin_spin: float
    probe_forward_disp: float
    probe_perp_disp: float
    probe_accel_mean: float
    contact_v_n: "tuple[float, float]"
    contact_v_t: "tuple[float, float]"
    friction_util: "tuple[float, float]"
    fn: "tuple[float, float]"
    both_contact: bool
    straddle: "float | None"
    forward_push_reach: float
    brake_opposed_reach: float
    lateral_reach: float
    normal_force_reach: float
    balance_reach_signed: float
    slew_headroom_up: float
    slew_headroom_dn: float
    e_kin: float
    a_brake: float
    d_stop: float
    w_brake_available: float
    over_speed: bool
    phase: str
    elapsed_horizon: int

    def as_dict(self) -> "dict[str, Any]":
        def r(x: Any) -> Any:
            return None if x is None else round(float(x), 5)
        return {"dtz": r(self.dtz), "d_safe": r(self.d_safe), "v_parallel": r(self.v_parallel),
                "v_perpendicular": r(self.v_perpendicular), "coin_speed": r(self.coin_speed), "coin_spin": r(self.coin_spin),
                "probe_forward_disp": r(self.probe_forward_disp), "probe_perp_disp": r(self.probe_perp_disp),
                "probe_accel_mean": r(self.probe_accel_mean),
                "contact_v_n": [r(x) for x in self.contact_v_n], "contact_v_t": [r(x) for x in self.contact_v_t],
                "friction_util": [r(x) for x in self.friction_util], "fn": [r(x) for x in self.fn],
                "both_contact": bool(self.both_contact), "straddle": r(self.straddle),
                "forward_push_reach": r(self.forward_push_reach), "brake_opposed_reach": r(self.brake_opposed_reach),
                "lateral_reach": r(self.lateral_reach), "normal_force_reach": r(self.normal_force_reach),
                "balance_reach_signed": r(self.balance_reach_signed),
                "slew_headroom_up": r(self.slew_headroom_up), "slew_headroom_dn": r(self.slew_headroom_dn),
                "e_kin": r(self.e_kin), "a_brake": r(self.a_brake), "d_stop": r(self.d_stop),
                "w_brake_available": r(self.w_brake_available), "over_speed": bool(self.over_speed),
                "phase": self.phase, "elapsed_horizon": int(self.elapsed_horizon)}


def _target_frame_live(rl: Any) -> "tuple[np.ndarray, np.ndarray, float]":
    """(e_par = coin→zone unit, e_perp = ⊥, dtz) from the LIVE rl (the target direction moves as the coin moves)."""
    u, dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float64)[:2]
    n = float(np.linalg.norm(e_par))
    e_par = e_par / n if n > 1e-9 else np.array([1.0, 0.0])
    return e_par, np.array([-e_par[1], e_par[0]]), float(dtz)


def _coin_spin(rl: Any) -> float:
    """The disk spin (planar rotational dof, index disk_dofy+1 = 6 on the 7-dof model); 0 if not surfaced."""
    d = rl.inner.data
    idx = int(rl.inner._disk_dofy) + 1
    return float(d.qvel[idx]) if idx < d.qvel.shape[0] else 0.0


def _live_contacts(rl: Any) -> "dict[str, Any]":
    """Per-side (v_n, v_t, friction-util slip, fn), both-contact, straddle from the live rl. Missing contact ⇒ zeros;
    friction slip is the BOUNDED |v_t|/(|v_t|+|v_n|+ε) proxy (the same as canonical_frame, not the unbounded ratio)."""
    con = primary_fingertip_contacts(rl)
    fn = (float(con["left"]["fn"]) if con["left"] is not None else 0.0,
          float(con["right"]["fn"]) if con["right"] is not None else 0.0)
    both = con["left"] is not None and con["right"] is not None
    straddle = float(con["left"]["n"] @ con["right"]["n"]) if both else None
    vn = vt = (0.0, 0.0)
    try:
        per = measure_contact_velocities(rl, frozen_contact_frames(rl))["per"]
        vn = (float(per["left"]["v_n"]) if per["left"] else 0.0, float(per["right"]["v_n"]) if per["right"] else 0.0)
        vt = (float(per["left"]["v_t"]) if per["left"] else 0.0, float(per["right"]["v_t"]) if per["right"] else 0.0)
    except Exception:                                          # contact identity churn ⇒ best-effort zeros (documented)
        pass

    def _slip(t: float, n: float) -> float:
        return abs(t) / (abs(t) + abs(n) + 1e-3)
    return {"v_n": vn, "v_t": vt, "friction_util": (_slip(vt[0], vn[0]), _slip(vt[1], vn[1])),
            "fn": fn, "both_contact": bool(both), "straddle": straddle}


def measure_response(rl: Any, ref: ProbeReference, ctx: MeasureContext, *, phase: str, t: int, horizon: int) -> ResponseState:
    """Read the causal closed-loop response at control step ``t`` from the live rl + the probe reference. # Preconditions:
    rl mid-rollout (post-step); ref captured at the probe start; ctx carries the handoff authority + control scalars.
    # Postconditions: every field finite; `over_speed` = the mass-free stopping guard; uses ONLY past+present signals."""
    e_par, e_perp, dtz = _target_frame_live(rl)
    v = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    v_par, v_perp = float(v @ e_par), float(v @ e_perp)
    speed = float(_coin_speed(rl))
    coin_xy = np.asarray(_coin_xy(rl), np.float64)
    d_probe = coin_xy - ref.coin_xy0
    dt_probe = max(int(t) - int(ref.t0), 1) * float(ctx.control_dt)
    accel = (speed - float(ref.speed0)) / max(dt_probe, _EPS)
    c = _live_contacts(rl)
    a = ctx.authority
    d_safe = max(dtz - CENTER_TOL, 0.0)
    a_brake = braking_decel(a.get("brake_opposed_reach", 0.0), ctx.control_dt, gain=ctx.brake_gain)
    d_stop = stopping_distance(v_par, a_brake)
    e_kin = kinetic_energy(ctx.m_coin, speed)
    w_brake = brake_work(ctx.m_coin, a_brake, d_safe)
    up = float(np.mean(np.minimum(np.asarray(ctx.hi) - np.asarray(ctx.prev_tau), ctx.slew) / max(ctx.slew, _EPS)))
    dn = float(np.mean(np.minimum(np.asarray(ctx.prev_tau) - np.asarray(ctx.lo), ctx.slew) / max(ctx.slew, _EPS)))
    return ResponseState(
        dtz=dtz, d_safe=d_safe, v_parallel=v_par, v_perpendicular=v_perp, coin_speed=speed, coin_spin=_coin_spin(rl),
        probe_forward_disp=float(d_probe @ e_par), probe_perp_disp=float(d_probe @ e_perp), probe_accel_mean=float(accel),
        contact_v_n=c["v_n"], contact_v_t=c["v_t"], friction_util=c["friction_util"], fn=c["fn"],
        both_contact=c["both_contact"], straddle=c["straddle"],
        forward_push_reach=float(a.get("forward_push_reach", 0.0)), brake_opposed_reach=float(a.get("brake_opposed_reach", 0.0)),
        lateral_reach=float(a.get("lateral_reach", 0.0)), normal_force_reach=float(a.get("normal_force_reach", 0.0)),
        balance_reach_signed=float(a.get("balance_reach_signed", 0.0)),
        slew_headroom_up=up, slew_headroom_dn=dn, e_kin=e_kin, a_brake=a_brake, d_stop=d_stop, w_brake_available=w_brake,
        over_speed=over_speed(d_stop, d_safe), phase=str(phase), elapsed_horizon=int(horizon))
