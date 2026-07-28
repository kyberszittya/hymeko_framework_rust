"""R10.2 Stage 2 — structured-option torque-path action coordinate over the frozen phase-shaped scaffold.

The Stage-2 audit (``reports/2026-07-28-td3-capture-exploration-audit.md``, parent ``4d983d4f``) established a
negative-first result: the previous coordinate re-evaluated a per-step residual ``delta_theta(s)`` *each step*, so
vanilla TD3 explored with per-step IID Gaussian noise that integrates into the actuator preload and removed the delivery
at every scale, while a temporally-*coherent* correction recovered 8/8. This module implements the corrected coordinate:

  * The learned decision is a single **structured option** ``theta``, emitted **once at READY**, decoded to bounded
    structural offsets ``(ds, dpreload_start, dbmax)`` and an additive torque-path offset (two transient 4-D spline knots
    ``k1, k2`` and one explicit 4-D terminal preload offset ``dtau_T``).
  * The per-step physical action is a **deterministic tracker** of the resulting smooth desired-torque path, so
    exploration (a single Gaussian on ``theta`` per episode, added by the reused ``train_semi_mdp``) is temporally
    coherent by construction. No per-step residual noise is ever added.

Bit-exact realisation (the load-bearing float contract). A literal ``a_t = clip((tau_des - prev)/slew, -1, 1)`` would route
the scaffold's own increment through a ``/slew``-then-``*slew`` round-trip and lose the ``theta=0`` identity. Instead the
correction is injected relative to the scaffold's **already-executable, slew-governed increment** ``a_pi0``:

    a_t = clip( a_pi0^theta(phi) + o_theta(phi)/slew , -1, 1 )

``o_theta`` is therefore an offset relative to the *executed nominal torque increment* (the slew-clipped
``clip(target-prev, -slew, slew)``), NOT an unconstrained modification of the raw pre-slew scaffold target — near
saturation the two differ, and this definition is exactly what makes the ``theta=0`` identity bit-exact. At ``theta=0``,
``o_theta ≡ 0`` and the structural offsets are ``0.0`` (``x+0.0=x``), so ``a_t ≡ a_pi0`` on the identical code path and the
whole rollout reproduces ``PhaseShapeCapture.roll(pi0)`` bit-exact. The scaffold's ``scaffold_action`` / ``apply_step`` and
the governed ``step_ablation`` stack are reused unchanged, so every torque / slew / joint-speed cap and the "no state edit,
no hidden force" contract hold by reuse, not re-implementation.

This module is the pure action coordinate: no reward, no environment, no trainer (those live in ``torque_path_env`` and
the experiments, added only after the identity gates pass).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.capture_rl import capture_observation
from hymeko_rl.coin_delivery.theta_option.moving_precapture import (
    CaptureParams, HandoffReference, PhaseShapeCapture, quintic_coeffs)

# Structured-option layout (emitted once at READY):
#   [0]      ds              structural: terminal velocity scale offset
#   [1]      dpreload_start  structural: preload-ramp start-fraction offset
#   [2]      dbmax           structural: max preload-blend weight offset
#   [3:7]    k1 (4-D)        transient torque-path knot at phase 1/3   (torque units)
#   [7:11]   k2 (4-D)        transient torque-path knot at phase 2/3   (torque units)
#   [11:15]  dtau_T (4-D)    explicit terminal preload offset          (torque units)
THETA_DIM = 15
_S, _P, _B = slice(0, 1), slice(1, 2), slice(2, 3)
_K1, _K2, _DT = slice(3, 7), slice(7, 11), slice(11, 15)

TRANSIENT_KNOT_PHASES = (1.0 / 3.0, 2.0 / 3.0)     # interior phases of the two transient knots (endpoints pinned to 0)


@dataclass(frozen=True)
class ThetaScales:
    """Fixed, pre-registered decoder scales mapping the actor's tanh output ``z in [-1,1]^15`` to bounded offsets. These
    are part of the (frozen) action coordinate, distinct from the per-episode *exploration* scale (a later boundary). The
    torque-path scales are expressed as fractions of ``slew`` so a knot/terminal offset never exceeds the executable band
    by construction."""

    s: float = 0.15                # ds in [-0.15, 0.15]  (scaffold s ~ 0.38)
    preload_start: float = 0.20    # dpreload_start in [-0.20, 0.20]
    bmax: float = 0.20             # dbmax in [-0.20, 0.20]
    knot_frac: float = 0.50        # |k_i| <= 0.50 * slew   (transient torque-path knot)
    terminal_frac: float = 1.00    # |dtau_T| <= 1.00 * slew (explicit terminal preload offset)


@dataclass(frozen=True)
class StructuredOption:
    """The decoded structured option. ``k1, k2, dtau_T`` are per-joint torque offsets (4-D); the structural offsets are
    scalars. All-zero ``<=>`` the frozen medoid scaffold (the identity precondition)."""

    ds: float
    dpreload_start: float
    dbmax: float
    k1: np.ndarray = field(default_factory=lambda: np.zeros(4))
    k2: np.ndarray = field(default_factory=lambda: np.zeros(4))
    dtau_T: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def is_zero(self) -> bool:
        return bool(self.ds == 0.0 and self.dpreload_start == 0.0 and self.dbmax == 0.0
                    and not np.any(self.k1) and not np.any(self.k2) and not np.any(self.dtau_T))


def decode_theta(z_hat: np.ndarray, slew: float, scales: ThetaScales = ThetaScales()) -> StructuredOption:
    """Decode a tanh-bounded actor output ``z in [-1,1]^15`` to a :class:`StructuredOption`.

    # Preconditions: ``z_hat`` has length 15; values are clipped to [-1,1] defensively (a TD3 tanh head already respects
    #   this). # Postconditions: deterministic; ``z_hat == 0`` maps to every offset being *exactly* ``0.0`` (so the roll
    #   is bit-exact to the scaffold); ``k_i, dtau_T`` are bounded by their ``*_frac * slew`` band.
    """
    z = np.clip(np.asarray(z_hat, dtype=float).ravel(), -1.0, 1.0)
    assert z.shape == (THETA_DIM,), f"theta must be length {THETA_DIM}, got {z.shape}"
    return StructuredOption(
        ds=float(z[_S][0] * scales.s),
        dpreload_start=float(z[_P][0] * scales.preload_start),
        dbmax=float(z[_B][0] * scales.bmax),
        k1=z[_K1] * (scales.knot_frac * slew),
        k2=z[_K2] * (scales.knot_frac * slew),
        dtau_T=z[_DT] * (scales.terminal_frac * slew),
    )


# --------------------------------------------------------------------------------------------------------------------
# Phase bases: transient (pinned to 0 at both ends) + terminal ramp (0 -> 1)
# --------------------------------------------------------------------------------------------------------------------
_NODES = (0.0, TRANSIENT_KNOT_PHASES[0], TRANSIENT_KNOT_PHASES[1], 1.0)     # x0..x3 of the transient spline


def _catmull_rom_clamped(phase: float, y: "tuple[float, float, float, float]") -> float:
    """Clamped Catmull-Rom (cubic Hermite) through the 4 fixed nodes ``_NODES`` with node values ``y`` and **zero end
    tangents**. C1 everywhere (adjacent segments share the interior tangent) with zero slope at ``phase in {0,1}``; it
    interpolates every node exactly. Linear in ``y`` (so linear in the knots) and returns ``0`` for all-zero ``y``."""
    x = _NODES
    m = (0.0, (y[2] - y[0]) / (x[2] - x[0]), (y[3] - y[1]) / (x[3] - x[1]), 0.0)   # clamped Catmull-Rom tangents
    if phase <= x[0]:
        return y[0]
    if phase >= x[3]:
        return y[3]
    s = 2 if phase > x[2] else (1 if phase > x[1] else 0)                          # segment [x_s, x_{s+1}]
    h = x[s + 1] - x[s]
    t = (phase - x[s]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y[s] + (t3 - 2 * t2 + t) * h * m[s]
            + (-2 * t3 + 3 * t2) * y[s + 1] + (t3 - t2) * h * m[s + 1])


def transient_basis(phase: float, k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Per-joint transient offset: a **C1** clamped Catmull-Rom spline through ``(0,0), (1/3,k1), (2/3,k2), (1,0)`` — a
    physically smooth (continuous first derivative) mid-path correction, pinned to ``0`` with zero slope at ``phase in
    {0,1}`` (``psi_k(0)=psi_k(1)=0``) so there is no unnoticed endpoint drift and no slope kink at the handoff. It hits
    both knots exactly (``k1`` at 1/3, ``k2`` at 2/3). # Postconditions: returns ``0`` exactly at ``phase in {0,1}`` and
    when ``k1=k2=0``; C1 at the interior knots (tested)."""
    return np.array([_catmull_rom_clamped(phase, (0.0, float(k1[j]), float(k2[j]), 0.0)) for j in range(4)])


def terminal_basis(phase: float) -> float:
    """Terminal preload ramp ``psi_T``: smoothstep ``3p^2 - 2p^3`` with ``psi_T(0)=0, psi_T(1)=1`` (monotone, zero slope
    at the ends). The preload change is explicit and bounded — never a noise integral."""
    p = min(max(phase, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)


def torque_path_offset(phase: float, opt: StructuredOption) -> np.ndarray:
    """The additive desired-torque-path offset ``o_theta(phi) = transient(phi) + psi_T(phi)*dtau_T`` (torque units).
    # Postconditions: exactly ``0`` when ``opt`` is all-zero, and exactly ``0`` at ``phase=0`` for any ``opt`` (transient
    #   and terminal bases both vanish there) — so the option's first action equals the scaffold's first action."""
    return transient_basis(phase, opt.k1, opt.k2) + terminal_basis(phase) * opt.dtau_T


# --------------------------------------------------------------------------------------------------------------------
# Per-step saturation masks (computed WITHOUT modifying the frozen scaffold)
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StepMasks:
    """Per-step saturation record so a later terminal-offset audit can distinguish requested / slew-limited / clamped /
    physically-executed corrections. Inferred from quantities the roller already has — the frozen scaffold is untouched.

      * ``slew_limited``   — the scaffold's own increment saturated the slew band (``|a_pi0| >= 1``): the executed nominal
        increment was itself slew-limited before any correction.
      * ``action_clipped`` — the composed correction ``a_pi0 + o_theta/slew`` exceeded ``+/-1`` and was clipped by the
        normalised-action bound (always ``False`` for the zero policy, whose ``a_pi0`` already lies in ``[-1,1]``).
      * ``torque_clamped`` — the joint-torque ``[lo,hi]`` clamp in ``apply_step`` was active on the executed command.
    """

    slew_limited: np.ndarray       # (4,) bool
    action_clipped: np.ndarray     # (4,) bool
    torque_clamped: np.ndarray     # (4,) bool


_SLEW_EPS = 1e-9


def _step_masks(a_pi0: np.ndarray, offset_norm: np.ndarray, prev_before: np.ndarray, prev_after: np.ndarray,
                slew: float, lo: np.ndarray, hi: np.ndarray) -> StepMasks:
    composed = a_pi0 + offset_norm
    raw_cmd = prev_before + np.clip(composed, -1.0, 1.0) * slew        # pre-[lo,hi] executed command
    return StepMasks(
        slew_limited=np.abs(a_pi0) >= 1.0 - _SLEW_EPS,
        action_clipped=np.abs(composed) > 1.0 + _SLEW_EPS,
        torque_clamped=(raw_cmd < lo - _SLEW_EPS) | (raw_cmd > hi + _SLEW_EPS),
    )


# --------------------------------------------------------------------------------------------------------------------
# The structured-option torque-path roller (the SHARED training + deployment + identity code path)
# --------------------------------------------------------------------------------------------------------------------
class TorquePathCaptureRoll:
    """Roll the frozen scaffold under a single structured option ``theta`` via the deterministic torque-path tracker.

    The option is applied *once* (decoded at construction-time per ``rollout`` call); the per-step action is
    ``a_t = clip(a_pi0^theta(phi) + o_theta(phi)/slew, -1, 1)`` where ``a_pi0^theta`` is the scaffold increment under the
    structurally-reshaped params ``(s+ds, preload_start+dpreload_start, bmax+dbmax)`` and ``o_theta`` the additive
    torque-path offset. This is the *same* method used for training rollouts, deployment, and the identity gate — there is
    no ``theta==0`` special case.

    # Preconditions: ``ready`` is the frozen analytic-transit READY snapshot; ``ref`` the cradle phase-point; ``pi0`` the
    #   medoid scaffold. # Postconditions: deterministic given ``(ready, pi0, z_hat)``; ``z_hat=0`` reproduces
    #   ``PhaseShapeCapture.roll(pi0)`` bit-exact (asserted by the identity gate); every command passes the governed
    #   ``step_ablation`` + governor stack (all caps inherited); the coin is untouched by direct edit.
    """

    def __init__(self, ready: Any, ref: HandoffReference, stack: Any, pi0: CaptureParams, coin0: np.ndarray,
                 scales: ThetaScales = ThetaScales()) -> None:
        self.cap = PhaseShapeCapture(ready, ref, stack)
        self.ref, self.pi0, self.coin0, self.stack, self.scales = ref, pi0, coin0, stack, scales
        self.slew = self.cap.slew

    def structural_params(self, opt: StructuredOption) -> CaptureParams:
        """The scaffold params reshaped by the structural offsets, defensively clipped to valid bands. At ``opt=0`` this
        is bit-identical to ``pi0`` (``x + 0.0 = x``, and the clips are inactive at the nominal)."""
        s = float(np.clip(self.pi0.s + opt.ds, 0.05, 0.6))
        p = float(np.clip(self.pi0.preload_start + opt.dpreload_start, 0.0, 1.0))
        b = float(np.clip(self.pi0.bmax + opt.dbmax, 0.0, 1.0))
        return replace(self.pi0, s=s, preload_start=p, bmax=b)

    def rollout(self, z_hat: np.ndarray) -> dict:
        """Execute the structured-option torque-path capture; return the terminal snapshot + full per-step trace.

        Returned trace keys: ``snapshot`` (downstream input), ``obs`` (causal Markov states), ``acts`` (physical action
        trace ``a_t``), ``desired_path`` (``tau_des(phi_{t+1})`` = executed commanded torque), ``prev`` (terminal
        commanded torque), ``masks`` (per-step :class:`StepMasks`), ``contacts`` (per-step fingertip contact count),
        ``option`` (decoded :class:`StructuredOption`), ``params`` (structural params used).
        """
        opt = decode_theta(z_hat, self.slew, self.scales)
        params = self.structural_params(opt)
        coeffs = quintic_coeffs(self.cap.q0, self.cap.v0, self.ref.precursor(params.n),
                                params.s * self.ref.qvel_star, params.steps * self.ref.control_dt)
        knot_t = np.linspace(0, len(params.residual) - 1, params.steps)
        rl = self.cap.ready.branch()
        prev = self.cap.prev0.copy()
        obss, acts, desired, masks, contacts = [], [], [], [], []
        mujoco.set_mjcb_control(self.cap._governor())
        try:
            for i in range(params.steps):
                phase = i / max(1, params.steps - 1)
                obs = capture_observation(rl, prev, self.ref, self.coin0, i, params.steps)
                a_pi0 = self.cap.scaffold_action(rl, prev, i, coeffs, params, knot_t)
                offset_norm = torque_path_offset(phase, opt) / self.slew
                u = np.clip(a_pi0 + offset_norm, -1.0, 1.0)
                prev_before = prev
                prev = self.cap.apply_step(rl, prev, u)
                obss.append(obs)
                acts.append(u.astype(np.float32))
                desired.append(prev.copy())                    # executable torque path = executed commanded torque
                masks.append(_step_masks(a_pi0, offset_norm, prev_before, prev, self.slew, self.cap.lo, self.cap.hi))
                contacts.append(_fingertip_count(rl))
        finally:
            mujoco.set_mjcb_control(None)
        snap = kc.TransportSnapshot.from_live(copy.deepcopy(rl), self.stack, prev.copy())
        return {"snapshot": snap, "obs": obss, "acts": acts, "desired_path": desired, "prev": prev,
                "masks": masks, "contacts": contacts, "option": opt, "params": params}


def _fingertip_count(rl: Any) -> int:
    from hymeko_rl.coin_delivery.forward_displacement import primary_fingertip_contacts
    con = primary_fingertip_contacts(rl)
    return int(con["left"] is not None) + int(con["right"] is not None)


# --------------------------------------------------------------------------------------------------------------------
# Nominal phase tube (the ONLY shaping reference; no cradle tau_star) + terminal-offset audit infra
# --------------------------------------------------------------------------------------------------------------------
def record_phase_tube(roller: TorquePathCaptureRoll) -> dict:
    """The medoid scaffold's per-step reference trajectory ``(q0(phi), qvel0(phi), tau0(phi))`` — recorded as the
    ``z=0`` rollout (bit-exact to the scaffold by the identity), so the tube *is* the scaffold trajectory. This is the
    only reward-shaping reference; there is no cradle ``tau_star`` term anywhere.

    # Postconditions: ``q0/qvel0/tau0`` have shape ``(steps,4)``; ``tau0_terminal`` is the scaffold's terminal commanded
    #   torque (the reference for the terminal-offset audit)."""
    res = roller.rollout(np.zeros(THETA_DIM, dtype=np.float32))
    obs = np.asarray(res["obs"])                               # (steps, 31): [q(0:4), qvel(4:8), prev(8:12), ...]
    return {"q0": obs[:, 0:4].copy(), "qvel0": obs[:, 4:8].copy(), "tau0": obs[:, 8:12].copy(),
            "tau0_terminal": np.asarray(res["prev"], dtype=float).copy()}


def terminal_offset_report(roller: TorquePathCaptureRoll, z_hat: np.ndarray, tube: dict) -> dict:
    """Requested vs physically-executed terminal preload offset, with the saturation context that explains any gap.

    ``requested`` is the decoded ``dtau_T``; ``executed`` is ``prev_terminal - tau0_terminal`` (the actual terminal
    preload shift relative to the nominal tube). At ``z=0`` both are ``0`` (bit-exact). The masks let a later
    admissibility gate attribute a mismatch to slew-limiting, clamping, or contact rather than to a random walk.
    """
    res = roller.rollout(z_hat)
    requested = np.asarray(res["option"].dtau_T, dtype=float)
    executed = np.asarray(res["prev"], dtype=float) - np.asarray(tube["tau0_terminal"], dtype=float)
    last = res["masks"][-1]
    return {"requested": requested, "executed": executed, "abs_err": np.abs(executed - requested),
            "err_norm": float(np.linalg.norm(executed - requested)),
            "terminal_slew_limited": last.slew_limited.tolist(),
            "terminal_action_clipped": last.action_clipped.tolist(),
            "terminal_torque_clamped": last.torque_clamped.tolist(),
            "any_step_action_clipped": bool(np.any([np.any(m.action_clipped) for m in res["masks"]]))}
