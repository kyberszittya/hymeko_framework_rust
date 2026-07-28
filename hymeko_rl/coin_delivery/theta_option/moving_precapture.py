"""R10 Stage 1B — phase-shaped moving-precursor dynamic capture: READY -> moving handoff -> frozen downstream -> K6.

The cradle handoff ``(q*, qvel*, tau*)`` is a *phase-point of a moving orbit*, not a global capture target: ``q`` is the
position and ``qvel`` the *tangent* of the same trajectory, ``tau*`` the control-history memory. Hitting all three
jointly under a terminal-state objective is ill-conditioned (position xor velocity, measured). Instead we arrive *moving
along the orbit's tangent*:

  1. a quintic terminal segment drives the tips from READY to a backward-tangent precursor ``q_pre(dt) = q* - dt*qvel*``
     with terminal velocity ``~ s*qvel*`` (direction is what matters — the deliverable tube is wide on direction cosine,
     tight on preload);
  2. the ``prev_tau`` preload ramps *during* that moving segment (applying ``tau*`` afterward accelerates the arm out of
     the deliverable velocity band — ``tau*`` is the accelerating torque, ``qdot`` jumps 0.66->1.07 in one step);
  3. a short residual under a low-dimensional structured CEM closes the last margin.

The frozen downstream (APPROACH -> HANDOFF_RESET -> R2 -> coast -> K6) then generates the coupling itself. No state edit,
no snapshot injection: every command passes the governed ``step_ablation`` stack. Deterministic given the planner seed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.env.motion_contract import govern_torque

R2_ALPHA = 0.15
CAPTURE_HORIZON = 80


# --------------------------------------------------------------------------------------------------------------------
# Handoff reference (the moving orbit's phase-point) and the quintic terminal segment
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HandoffReference:
    """The cradle handoff as a phase-point of the winning orbit: position ``q_star``, tangent ``qvel_star``, preload
    ``tau_star``. The capture targets the backward-tangent precursor ``q_star - n*dt*qvel_star`` (a deliverable tube)."""

    q_star: np.ndarray
    qvel_star: np.ndarray
    tau_star: np.ndarray
    control_dt: float

    @classmethod
    def from_cradle(cls, cradle: Any) -> "HandoffReference":
        d = cradle.branch().inner.data
        return cls(d.qpos[:4].copy(), d.qvel[:4].copy(), np.asarray(cradle.prev_tau, float).copy(),
                   float(cradle.stack.control_dt))

    def precursor(self, n: float) -> np.ndarray:
        """Backward-tangent precursor ``q_star - n*control_dt*qvel_star`` (n=0 is the handoff itself)."""
        return self.q_star - n * self.control_dt * self.qvel_star

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.qvel_star))


def quintic_coeffs(q0: np.ndarray, v0: np.ndarray, qf: np.ndarray, vf: np.ndarray,
                   T: float) -> "tuple[np.ndarray, ...]":
    """Per-joint quintic with boundary ``(q0,v0,0) -> (qf,vf,0)`` over ``[0,T]`` (zero terminal acceleration)."""
    a3 = (20 * (qf - q0) - (8 * vf + 12 * v0) * T) / (2 * T ** 3)
    a4 = (30 * (q0 - qf) + (14 * vf + 16 * v0) * T) / (2 * T ** 4)
    a5 = (12 * (qf - q0) - (6 * vf + 6 * v0) * T) / (2 * T ** 5)
    return q0, v0, np.zeros(4), a3, a4, a5


def quintic_eval(c: "tuple[np.ndarray, ...]", t: float) -> "tuple[np.ndarray, np.ndarray]":
    """Position and velocity of the quintic at time ``t``."""
    a0, a1, a2, a3, a4, a5 = c
    q = a0 + a1 * t + a2 * t * t + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    v = a1 + 2 * a2 * t + 3 * a3 * t * t + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
    return q, v


# --------------------------------------------------------------------------------------------------------------------
# Capture parameters + outcome
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureParams:
    """The structured, bounded capture parameters (what the CEM searches; everything else is fixed contract)."""

    n: float = 3.0                 # backward-tangent precursor index (Delta t in control steps)
    s: float = 0.3                 # terminal velocity scale (of qvel_star)
    preload_start: float = 0.5     # fraction of the segment at which the prev_tau ramp begins
    bmax: float = 0.6              # max preload-blend weight
    residual: np.ndarray = field(default_factory=lambda: np.zeros((2, 4)))
    steps: int = 20                # segment length
    kp: float = 16.0
    kd: float = 8.0


@dataclass(frozen=True)
class CaptureOutcome:
    """Result of a phase-shaped capture: the terminal snapshot + the tube-diagnostic metrics + downstream verdict."""

    snapshot: Any
    cos_dir: float                 # cosine(terminal qvel, qvel_star)
    vel_scale: float               # |terminal qvel| / |qvel_star|
    dtau: float                    # ||terminal prev_tau - tau_star||
    contacts: int
    k6: bool = False
    min_dtz_mm: float = float("nan")
    safe: bool = True


@dataclass(frozen=True)
class CaptureSearchSpec:
    """Bounds for the structured CEM (bounded structural parameters only — never raw per-step torques)."""

    s_max: float = 0.6
    steps: int = 20
    kp: float = 16.0
    kd: float = 8.0
    knots: int = 2
    population: int = 44
    iters: int = 14
    elite: int = 9
    init_std: float = 0.35


# --------------------------------------------------------------------------------------------------------------------
# The phase-shaped capture roller
# --------------------------------------------------------------------------------------------------------------------
class PhaseShapeCapture:
    """Roll a phase-shaped moving precursor from READY through the governed servo (no state edit, no hidden force).

    Tracks a quintic ``READY -> q_pre`` (terminal velocity ``s*qvel_star``) with a bounded residual, and ramps the
    commanded torque toward ``tau_star`` *during* the moving segment. Every command passes ``step_ablation`` + governor.
    """

    def __init__(self, ready: Any, ref: HandoffReference, stack: Any) -> None:
        self.ready = ready
        self.ref = ref
        self.stack = stack
        self.slew = float(stack.tau_rate * stack.control_dt)
        self.lo = np.asarray(ready.lo, dtype=float)
        self.hi = np.asarray(ready.hi, dtype=float)
        d = ready.branch().inner.data
        self.q0, self.v0 = d.qpos[:4].copy(), d.qvel[:4].copy()
        self.prev0 = np.asarray(ready.prev_tau, float).copy()

    def roll(self, params: CaptureParams) -> CaptureOutcome:
        """Execute the capture; return the terminal snapshot + tube metrics (downstream verdict filled in by the caller).

        # Postconditions: deterministic given ``(ready, params)``; the coin is untouched by direct edit; the returned
        #   snapshot's ``prev_tau`` is the last governed command.
        """
        dt = self.ref.control_dt
        q_pre = self.ref.precursor(params.n)
        coeffs = quintic_coeffs(self.q0, self.v0, q_pre, params.s * self.ref.qvel_star, params.steps * dt)
        mujoco.set_mjcb_control(self._governor())
        try:
            rl, prev = self._track(coeffs, params)
        finally:
            mujoco.set_mjcb_control(None)
        return self._outcome(rl, prev)

    def _governor(self) -> Callable:
        gov = self.stack.gov

        def cb(_model: Any, data: Any) -> None:
            data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)

        return cb

    def _track(self, coeffs: "tuple[np.ndarray, ...]", params: CaptureParams) -> "tuple[Any, np.ndarray]":
        rl = self.ready.branch()
        prev = self.prev0.copy()
        dt = self.ref.control_dt
        knot_t = np.linspace(0, len(params.residual) - 1, params.steps)
        for i in range(params.steps):
            q_ref, v_ref = quintic_eval(coeffs, i * dt)
            data = rl.inner.data
            servo = np.clip(params.kp * (q_ref - data.qpos[:4]) + params.kd * (v_ref - data.qvel[:4]),
                            -self.slew, self.slew)
            res = self._residual(params.residual, knot_t, i) * self.slew
            target = self._preload_blend(prev + servo + res, i, params)
            prev = np.clip(prev + np.clip(target - prev, -self.slew, self.slew), self.lo, self.hi)
            step_ablation(rl, np.asarray(prev, np.float32), "A")
        return rl, prev

    @staticmethod
    def _residual(knots: np.ndarray, knot_t: np.ndarray, i: int) -> np.ndarray:
        idx = range(len(knots))
        return np.array([np.interp(knot_t[i], idx, knots[:, j]) for j in range(4)])

    def _preload_blend(self, target: np.ndarray, i: int, params: CaptureParams) -> np.ndarray:
        """Blend the tracking target toward ``tau_star`` once the segment fraction passes ``preload_start``."""
        frac = i / (params.steps - 1)
        if frac < params.preload_start:
            return target
        weight = params.bmax * (frac - params.preload_start) / (1.0 - params.preload_start + 1e-9)
        return (1.0 - weight) * target + weight * self.ref.tau_star

    def _outcome(self, rl: Any, prev: np.ndarray) -> CaptureOutcome:
        d = rl.inner.data
        qv = d.qvel[:4]
        speed = float(np.linalg.norm(qv))
        cos = float(qv @ self.ref.qvel_star / (speed * self.ref.speed + 1e-9))
        con = primary_fingertip_contacts(rl)
        snap = kc.TransportSnapshot.from_live(copy.deepcopy(rl), self.stack, prev.copy())
        return CaptureOutcome(snapshot=snap, cos_dir=round(cos, 3), vel_scale=round(speed / (self.ref.speed + 1e-9), 3),
                              dtau=round(float(np.linalg.norm(prev - self.ref.tau_star)), 3),
                              contacts=int(con["left"] is not None) + int(con["right"] is not None))


# --------------------------------------------------------------------------------------------------------------------
# Frozen downstream adapter + structured CEM planner
# --------------------------------------------------------------------------------------------------------------------
class FrozenDownstream:
    """Thin adapter over the frozen HANDOFF_RESET -> R2 -> coast -> K6 chain (imported as-is; never modified here)."""

    def __init__(self, model: Any, norm: Any, r2_fn: Any, stack: Any, horizon: int = CAPTURE_HORIZON) -> None:
        self.model, self.norm, self.r2_fn, self.stack = model, norm, r2_fn, stack
        self.cfg = replace(DELIVERY_CFG, horizon=horizon)

    def deliver(self, snapshot: Any) -> "tuple[bool, float, bool]":
        """Roll the frozen downstream from ``snapshot``; return (strict K6, min_dtz_mm, safe)."""
        k6, md, safe, _ = self.deliver_with_trace(snapshot)
        return k6, md, safe

    def deliver_with_trace(self, snapshot: Any) -> "tuple[bool, float, bool, list]":
        """As ``deliver`` but also return the controller's per-step kind trace (for the boundary-transition gates)."""
        from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
        ctrl = HandoffResetTemporalController(snapshot, CloneActor(self.model, self.norm), self.r2_fn,
                                              ResidualBounds(alpha=R2_ALPHA))
        m = velocity_rollout(snapshot, ctrl, self.cfg)
        safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
        kinds = [r["kind"] for r in ctrl.clone_trace]
        return bool(delivery_success(m, self.cfg)), round(_min_dtz_mm(snapshot, m), 2), safe, kinds


def _theta_to_params(theta: np.ndarray, spec: CaptureSearchSpec) -> CaptureParams:
    """Map a flat CEM vector to bounded structural parameters (never raw per-step torques)."""
    return CaptureParams(n=3.0, s=float(spec.s_max * np.clip(theta[0], 0.0, 1.0)),
                         preload_start=float(np.clip(theta[1], 0.0, 1.0)), bmax=float(np.clip(theta[2], 0.0, 1.0)),
                         residual=theta[3:].reshape(spec.knots, 4), steps=spec.steps, kp=spec.kp, kd=spec.kd)


def _cost(outcome: CaptureOutcome) -> float:
    """Lexicographic-flavoured cost: unsafe is hard-penalised; else 0 on K6 or min_dtz."""
    if not outcome.safe:
        return 1e3
    return 0.0 if outcome.k6 else outcome.min_dtz_mm


def _evaluate(theta: np.ndarray, cap: PhaseShapeCapture, spec: CaptureSearchSpec,
              downstream: FrozenDownstream) -> CaptureOutcome:
    params = _theta_to_params(theta, spec)
    out = cap.roll(params)
    k6, md, safe = downstream.deliver(out.snapshot)
    return replace(out, k6=k6, min_dtz_mm=md, safe=safe)


@dataclass(frozen=True)
class CaptureResult:
    """A solved capture: the winning parameters, its outcome, and the seed (for deterministic replay + provenance)."""

    seed: int
    params: CaptureParams
    outcome: CaptureOutcome


def plan_capture(ready: Any, ref: HandoffReference, stack: Any, downstream: FrozenDownstream, *, seed: int,
                 spec: CaptureSearchSpec = CaptureSearchSpec()) -> CaptureResult:
    """Structured CEM over the bounded capture parameters, scored by the frozen downstream (stops on the first K6).

    # Preconditions: ``ready`` is the frozen analytic-transit READY snapshot; ``ref`` the cradle phase-point.
    # Postconditions: deterministic given ``seed``; the returned params reproduce the outcome via ``PhaseShapeCapture``.
    """
    cap = PhaseShapeCapture(ready, ref, stack)
    dim = 3 + spec.knots * 4
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    mean[:3] = [0.6, 0.5, 0.5]                       # priors: s_raw, preload_start, bmax
    std = np.full(dim, spec.init_std)
    best: "tuple[float, np.ndarray, CaptureOutcome] | None" = None
    for _ in range(spec.iters):
        pop = mean[None] + std[None] * rng.standard_normal((spec.population, dim))
        scored = []
        for theta in pop:
            out = _evaluate(theta, cap, spec, downstream)
            cost = _cost(out)
            scored.append((cost, theta))
            if best is None or cost < best[0]:
                best = (cost, theta, out)
        scored.sort(key=lambda z: z[0])
        elite = np.stack([theta for _, theta in scored[:spec.elite]])
        mean, std = elite.mean(0), elite.std(0) * 0.88 + 1e-3
        if best is not None and best[2].k6:
            break
    assert best is not None
    return CaptureResult(seed=seed, params=_theta_to_params(best[1], spec), outcome=best[2])
