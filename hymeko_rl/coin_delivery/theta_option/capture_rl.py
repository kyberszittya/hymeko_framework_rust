"""R10 Stage 2 — verified reusable infrastructure for a residual capture policy over the frozen scaffold.

This module holds ONLY the correct, reusable pieces established by the Stage-2 diagnostic audit:

  * ``freeze_scaffold`` — deterministic, outcome-independent selection of one scaffold ``pi_0`` (structural-parameter
    medoid) from the Stage-1B solutions;
  * ``capture_observation`` — the causal Markov state (no teacher / future information);
  * ``ResidualCaptureRoll`` — the action-preserving residual composition ``u = clip(u_pi0(phase) + alpha*delta(obs),
    -1, 1)``; zero residual reproduces ``pi_0`` bit-exact;
  * the deterministic perturbation panel + evaluation harness.

It does NOT contain a training loop, a reward, or an exploration policy. The Stage-2 audit showed that vanilla
per-step *IID* residual exploration is inadmissible on this slew-integrated capture interface (see the report
``reports/2026-07-28-td3-capture-exploration-audit.md``): because the action is a slew-normalised torque *change*,
per-step IID noise walks the preload out of the working region, while a temporally-coherent residual delivers. The
reward-driven TD3 policy is therefore NOT achieved yet, and the training path (structure-preserving episodic
exploration + phase-conditioned metric + critic-warm-up) is the *next* session's work — deliberately absent here so
nothing imports the refuted IID/cradle-centred path.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import mujoco
import numpy as np
import torch

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.moving_precapture import (
    CaptureParams, FrozenDownstream, HandoffReference, PhaseShapeCapture, quintic_coeffs)
from hymeko_rl.option_rl.agents import make_actor

ALPHA = 0.15                      # residual authority (R2/R3-C convention). The next session must re-derive it from the
#                                   structure-preserving MPC correction magnitudes, not import it from memory.
OBS_DIM = 31
ACT_DIM = 4


# --------------------------------------------------------------------------------------------------------------------
# Outcome-independent scaffold selection
# --------------------------------------------------------------------------------------------------------------------
def _param_vector(p: CaptureParams) -> np.ndarray:
    return np.concatenate([[p.n, p.s, p.preload_start, p.bmax], np.asarray(p.residual, float).ravel()])


def freeze_scaffold(solutions: "list[CaptureParams]") -> CaptureParams:
    """Select ``pi_0`` = the structural-parameter *medoid* of the K6 solutions (outcome-independent; not best min_dtz).

    # Preconditions: >= 1 solution. # Postconditions: deterministic; returns one of the input solutions unchanged.
    """
    assert solutions, "need >= 1 scaffold solution"
    vecs = [_param_vector(p) for p in solutions]
    dist = [sum(float(np.linalg.norm(vi - vj)) for vj in vecs) for vi in vecs]
    return solutions[int(np.argmin(dist))]


# --------------------------------------------------------------------------------------------------------------------
# Causal Markov observation (no teacher / future information)
# --------------------------------------------------------------------------------------------------------------------
def capture_observation(rl: Any, prev: np.ndarray, ref: HandoffReference, coin0: np.ndarray, i: int,
                        steps: int) -> np.ndarray:
    """31-D causal state: q, qvel, prev_tau, coin pose/vel, fingertip contact, handoff-relative deltas, capture phase."""
    d = rl.inner.data
    q, qv = d.qpos[:4].copy(), d.qvel[:4].copy()
    coin_rel = _coin_xy(rl) - coin0
    coin_vel = d.qvel[4:6].copy()
    con = primary_fingertip_contacts(rl)
    contact = np.array([float(con["left"] is not None), float(con["right"] is not None)])
    phase = np.array([i / max(1, steps - 1)])
    return np.concatenate([q, qv, prev, coin_rel, coin_vel, contact, q - ref.q_star, qv - ref.qvel_star,
                           prev - ref.tau_star, phase]).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------------
# Action-preserving residual composition over the frozen scaffold
# --------------------------------------------------------------------------------------------------------------------
class ResidualCaptureRoll:
    """Roll the frozen scaffold ``pi_0`` with a bounded residual: ``u = clip(u_pi0(phase) + alpha*delta(obs), -1, 1)``.

    ``delta`` is expected already bounded. Zero ``delta`` reproduces ``pi_0`` bit-exact (verified identity test).
    """

    def __init__(self, ready: Any, ref: HandoffReference, stack: Any, pi0: CaptureParams, coin0: np.ndarray,
                 alpha: float = ALPHA) -> None:
        self.cap = PhaseShapeCapture(ready, ref, stack)
        self.ref, self.pi0, self.coin0, self.alpha, self.stack = ref, pi0, coin0, alpha, stack
        self.coeffs = quintic_coeffs(self.cap.q0, self.cap.v0, ref.precursor(pi0.n),
                                     pi0.s * ref.qvel_star, pi0.steps * ref.control_dt)
        self.knot_t = np.linspace(0, len(pi0.residual) - 1, pi0.steps)

    def rollout(self, residual_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
        """Execute the residual capture; return the terminal snapshot + the per-step (obs, action) trace."""
        rl = self.cap.ready.branch()
        prev = self.cap.prev0.copy()
        obss, acts = [], []
        mujoco.set_mjcb_control(self.cap._governor())
        try:
            for i in range(self.pi0.steps):
                obs = capture_observation(rl, prev, self.ref, self.coin0, i, self.pi0.steps)
                a_pi0 = self.cap.scaffold_action(rl, prev, i, self.coeffs, self.pi0, self.knot_t)
                u = np.clip(a_pi0 + self.alpha * np.asarray(residual_fn(obs), float), -1.0, 1.0)
                prev = self.cap.apply_step(rl, prev, u)
                obss.append(obs)
                acts.append(u.astype(np.float32))
        finally:
            mujoco.set_mjcb_control(None)
        snap = kc.TransportSnapshot.from_live(copy.deepcopy(rl), self.stack, prev.copy())
        return {"snapshot": snap, "obs": obss, "acts": acts, "prev": prev}


def zero_residual(_obs: np.ndarray) -> np.ndarray:
    return np.zeros(ACT_DIM, dtype=np.float32)


# --------------------------------------------------------------------------------------------------------------------
# Deterministic perturbation panel + evaluation harness (structure-agnostic; used by any future policy)
# --------------------------------------------------------------------------------------------------------------------
class Perturbation:
    """A small admissible READY perturbation (arm q / qvel / prev_tau); coin stays canonical."""

    def __init__(self, dq: np.ndarray, dv: np.ndarray, dtau: np.ndarray) -> None:
        self.dq, self.dv, self.dtau = dq, dv, dtau


def perturbation_panel(*, n: int = 16, sig_q: float = 0.02, sig_v: float = 0.06, sig_tau: float = 0.05,
                       seed: int = 90210) -> "list[Perturbation]":
    """Pre-registered admissible READY-tracking / actuator-state perturbations (deterministic; nominal is member 0)."""
    rng = np.random.default_rng(seed)
    panel = [Perturbation(np.zeros(4), np.zeros(4), np.zeros(4))]
    for _ in range(n - 1):
        panel.append(Perturbation(rng.normal(0, sig_q, 4), rng.normal(0, sig_v, 4), rng.normal(0, sig_tau, 4)))
    return panel


def perturb_ready(ready: Any, stack: Any, pert: Perturbation) -> Any:
    """Apply a perturbation to READY's (q, qvel, prev_tau); coin stays canonical. Returns a branchable snapshot.

    Perturbing READY is a training/evaluation tool (like the basin characterisation) — the deployed policy runs from the
    true frozen-transit READY with no state edit.
    """
    rl = ready.branch()
    d = rl.inner.data
    d.qpos[:4] = d.qpos[:4] + pert.dq
    d.qvel[:4] = d.qvel[:4] + pert.dv
    mujoco.mj_forward(rl.inner.model, d)
    prev0 = np.asarray(ready.prev_tau, float) + pert.dtau
    return kc.TransportSnapshot.from_live(copy.deepcopy(rl), stack, prev0.copy())


def _wilson(k: int, n: int) -> "tuple[float, float]":
    if n == 0:
        return 0.0, 1.0
    p, z = k / n, 1.96
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre - half) / denom, 3), round((centre + half) / denom, 3)


def evaluate_on_panel(residual_fn: Callable[[np.ndarray], np.ndarray], ready: Any, ref: HandoffReference, stack: Any,
                      pi0: CaptureParams, coin0: np.ndarray, down: FrozenDownstream,
                      panel: "list[Perturbation]") -> dict:
    """Roll the residual capture from every panel perturbation through the frozen downstream; report robustness metrics."""
    k6, safe, dtz, resets = 0, 0, [], []
    for pert in panel:
        snap = perturb_ready(ready, stack, pert)
        out = ResidualCaptureRoll(snap, ref, stack, pi0, coin0).rollout(residual_fn)
        ok, md, sf, kinds = down.deliver_with_trace(out["snapshot"])
        k6 += int(ok)
        safe += int(sf)
        dtz.append(md)
        resets.append(kinds.count("HANDOFF_RESET"))
    n = len(panel)
    lo, hi = _wilson(k6, n)
    return {"n": n, "k6": k6, "k6_rate": round(k6 / n, 3), "wilson": [lo, hi], "safe_rate": round(safe / n, 3),
            "mean_min_dtz_mm": round(float(np.mean(dtz)), 2),
            "one_reset_frac": round(float(np.mean([r == 1 for r in resets])), 3)}


def policy_residual(actor: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a policy actor as a deterministic residual ``obs -> action in [-1,1]^4`` (no exploration)."""
    def fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = actor.mean_action(t) if hasattr(actor, "mean_action") else actor(t)
        return np.asarray(a[0].numpy(), float)
    return fn


def make_zero_actor(seed: int = 0, act_dim: int = ACT_DIM) -> Any:
    """A TD3 actor zero-initialised on its output head — its output is *exactly* zero (verified), so a composed/decoded
    action is bit-exact ``pi_0``. Used by the identity gates. ``act_dim`` defaults to the 4-D residual head; pass
    ``act_dim=THETA_DIM`` for the 15-D structured-option head (R10.2) — dimension-generic by reuse, not a near-copy."""
    from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import zero_init_detactor
    torch.manual_seed(seed)
    return zero_init_detactor(make_actor("td3", OBS_DIM, act_dim))
