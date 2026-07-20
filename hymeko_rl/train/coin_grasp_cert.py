"""COIN-GRASP-CERT-1 — bilateral clamp establishment + active grasp certification.

COIN-TRANSPORT-1 (F-COIN-TRANSPORT) closed the transport-controller direction and revealed that the previous "stable
acquisition" predicate is INSUFFICIENT for transport: the recovered 12/19 states are CONTACT-ACQUIRED (a marginal,
flickering, often one-sided contact), NOT a certified load-bearing bilateral clamp. This module distinguishes:
    CONTACT-ACQUIRED  <  BILATERAL-CONTACT  <  PRELOADED-CLAMP  <  CERTIFIED-GRASP  <  TRANSPORTABLE-GRASP
via an explicit contact-mode state machine μ ∈ {N, L, R, B_t, B_p, S, J}, bounded clamp-establishment actors (G0-G5),
active certification probes (P0-P4), and a micro-transport gate.

Signals are geometric/dynamic PROXIES (contact booleans, coin slip speed, coin-to-midpoint offset, aperture) — NEVER
called "force" or "force closure" (no force sensor exists). FROZEN: env/dynamics/reward/monitor; handoffs from the
frozen no-regrasp acquisition (reused from coin_transport). NO RL, NO transport, NO CORE change here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hymeko_rl.env.planar_snapshot import restore_planar
from hymeko_rl.experiments.coin_delivery1 import _dir_to_zone
from hymeko_rl.train.coin_delivery_acquisition import MID_TO_COIN
from hymeko_rl.train.coin_transport import obj_to_env

# thresholds (declared; all geometric/dynamic proxies)
_PRELOAD_DWELL = 6          # both-contact steps to count as PRELOADED (B_p)
_SLIP_STABLE = 0.04        # coin speed below which contact counts as non-slipping
_SLIP_HIGH = 0.15          # coin speed above which one-sided contact counts as SLIPPING (S)


class Mode(str, Enum):
    N = "N"        # no contact
    L = "L"        # left-only
    R = "R"        # right-only
    B_t = "B_t"    # bilateral transient
    B_p = "B_p"    # bilateral preloaded / stable
    S = "S"        # slipping / degrading
    J = "J"        # jammed / invalid collision


class ContactModeClassifier:
    """Deterministic stateful contact-mode classifier from available signals (dwell counter + slip + jam)."""

    def __init__(self, preload_dwell: int = _PRELOAD_DWELL) -> None:
        self.preload_dwell = preload_dwell
        self.reset()

    def reset(self) -> None:
        self.both_dwell = 0

    def classify(self, inner) -> Mode:
        m = inner._planar_metrics
        left, right = bool(m.left_contact), bool(m.right_contact)
        both = left and right
        slip = float(m.disk_speed)
        self.both_dwell = self.both_dwell + 1 if both else 0
        if bool(getattr(m, "arm_self_contact", False)) or bool(getattr(m, "fingers_self_contact", False)):
            return Mode.J
        if both:
            return Mode.B_p if (self.both_dwell >= self.preload_dwell and slip < _SLIP_STABLE) else Mode.B_t
        if left != right:                                   # exactly one fingertip
            return Mode.S if slip > _SLIP_HIGH else (Mode.L if left else Mode.R)
        return Mode.N


# ── clamp-establishment actors (G0-G5) ───────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GraspParams:
    close_aperture: float = -0.6   # aperture command (negative = close the clamp)
    squeeze: float = 0.9           # squeeze both tips toward the coin
    closure_velocity: float = 1.0  # scale of the aperture/squeeze command per step
    dwell: int = 6                 # BILATERAL_DWELL before declaring preloaded
    differential: float = 0.5      # asymmetric balance for contact-seeking (G2) / recentre (G3)
    settle: int = 4                # settle steps (G4)

    @staticmethod
    def from_unit(u: np.ndarray) -> "GraspParams":
        v = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
        return GraspParams(close_aperture=-(0.2 + v[0] * 0.8), squeeze=0.4 + v[1] * 0.6,
                           closure_velocity=0.3 + v[2] * 0.7, differential=v[3] * 0.8)


class GraspFamily(str, Enum):
    G0_HOLD = "G0_hold"
    G1_SYMMETRIC = "G1_symmetric_preload"
    G2_ASYMMETRIC = "G2_asymmetric_seek"
    G3_MICRO_RECENTER = "G3_micro_recenter"
    G4_SETTLE_FSM = "G4_settle_preload_fsm"
    G5_MODE_RECOVERY = "G5_mode_recovery"


def grasp_action(inner, obs: np.ndarray, mode: Mode, family: GraspFamily, p: GraspParams) -> np.ndarray:
    """Object-centric clamp-establishment command (no midpoint translation — establish the grasp in place)."""
    v = p.closure_velocity
    if family == GraspFamily.G0_HOLD:
        return obj_to_env(np.zeros(2), 0.0, 0.0, 0.8)
    if family == GraspFamily.G2_ASYMMETRIC and mode in (Mode.L, Mode.R):
        bal = p.differential * (1.0 if mode == Mode.L else -1.0)      # seek the missing-contact side
        return obj_to_env(np.zeros(2), v * p.close_aperture, bal, v * p.squeeze)
    if family == GraspFamily.G3_MICRO_RECENTER and mode in (Mode.N, Mode.L, Mode.R):
        off = obs[list(MID_TO_COIN)]
        return obj_to_env(0.3 * off, v * p.close_aperture, 0.0, v * p.squeeze)
    return obj_to_env(np.zeros(2), v * p.close_aperture, 0.0, v * p.squeeze)  # G1/G4/G5 symmetric close+squeeze


# ── certification probes (P0-P4) — active perturbations a real grasp must survive ─────────────────────────────────────
class Probe(str, Enum):
    P0_DWELL = "P0_dwell"
    P1_APERTURE_PULSE = "P1_aperture_pulse"
    P2_MIDPOINT_MICRO = "P2_midpoint_micro"
    P3_DIFF_MICRO = "P3_diff_micro"
    P4_COMBINED = "P4_combined"


def _probe_action(inner, obs, probe: Probe, t: int, d: np.ndarray) -> np.ndarray:
    """The active perturbation for a probe at step t (small, bounded, returns to hold)."""
    hold = obj_to_env(np.zeros(2), 0.0, 0.0, 0.85)
    if probe == Probe.P0_DWELL:
        return hold
    if probe == Probe.P1_APERTURE_PULSE:
        return obj_to_env(np.zeros(2), (-0.4 if t < 3 else 0.4) if t < 6 else 0.0, 0.0, 0.85)
    if probe == Probe.P2_MIDPOINT_MICRO:
        return obj_to_env((0.25 * d) if t < 4 else (-0.25 * d if t < 8 else np.zeros(2)), 0.0, 0.0, 0.85)
    if probe == Probe.P3_DIFF_MICRO:
        return obj_to_env(np.zeros(2), 0.0, (0.3 if t < 4 else -0.3 if t < 8 else 0.0), 0.85)
    # P4 combined: dwell → aperture pulse → target micro-translation
    if t < 4:
        return hold
    if t < 8:
        return obj_to_env(np.zeros(2), -0.4 if t < 6 else 0.4, 0.0, 0.85)
    return obj_to_env((0.25 * d) if t < 12 else (-0.25 * d if t < 16 else np.zeros(2)), 0.0, 0.0, 0.85)


# ── rollout: establish grasp, classify mode trajectory, then certify ─────────────────────────────────────────────────
def establish_grasp(env, handoff, family: GraspFamily, p: GraspParams, *, steps: int = 30,
                    mode_scramble=None) -> dict:
    """Restore the handoff, run a clamp-establishment actor, return the mode trajectory + reachability of B_p."""
    inner = env._env
    restore_planar(inner, handoff.snap)
    env._horizon = steps + 400
    clf = ContactModeClassifier(p.dwell)
    obs = handoff.obs.copy()
    modes = []
    bp_dwell = 0
    max_bp_dwell = 0
    bilateral_steps = 0
    slips = []
    for _t in range(steps):
        mode = clf.classify(inner)
        read_mode = mode if mode_scramble is None else mode_scramble(mode)
        a = np.clip(grasp_action(inner, obs, read_mode, family, p), -1, 1).astype(np.float32)
        obs, _r, _term, _trunc, _info = env.step(a)
        m = inner._planar_metrics
        modes.append(mode.value)
        bilateral_steps += int(bool(m.left_contact and m.right_contact))
        slips.append(float(m.disk_speed))
        bp_dwell = bp_dwell + 1 if mode == Mode.B_p else 0
        max_bp_dwell = max(max_bp_dwell, bp_dwell)
    reached_bp = Mode.B_p.value in modes
    return {"seed": handoff.seed, "modes": modes, "reached_B_p": reached_bp, "max_bp_dwell": max_bp_dwell,
            "bilateral_frac": round(bilateral_steps / max(1, steps), 4), "mean_slip": round(float(np.mean(slips)), 4),
            "final_mode": modes[-1] if modes else "N"}


def certify_grasp(env, handoff, family: GraspFamily, p: GraspParams, probe: Probe, *, establish: int = 20,
                  probe_steps: int = 18) -> dict:
    """Establish the grasp, then run an active probe; certification passes if the coin is retained + bilateral recovers
    + slip stays bounded + it is not merely static support (the coin must be actively re-clamped)."""
    inner = env._env
    restore_planar(inner, handoff.snap)
    env._horizon = establish + probe_steps + 400
    clf = ContactModeClassifier(p.dwell)
    obs = handoff.obs.copy()
    for _ in range(establish):                              # establish
        mode = clf.classify(inner)
        obs, _r, _t, _tr, _i = env.step(np.clip(grasp_action(inner, obs, mode, family, p), -1, 1).astype(np.float32))
    d, _n = _dir_to_zone(inner)
    both_before = bool(inner._planar_metrics.left_contact and inner._planar_metrics.right_contact)
    max_slip = 0.0
    both_after = False
    both_recovered = False
    for t in range(probe_steps):
        obs, _r, _tr2, _tr, _i = env.step(np.clip(_probe_action(inner, obs, probe, t, np.asarray(d)), -1, 1).astype(np.float32))
        m = inner._planar_metrics
        both = bool(m.left_contact and m.right_contact)
        max_slip = max(max_slip, float(m.disk_speed))
        both_after = both
        both_recovered = both_recovered or (t >= probe_steps // 2 and both)   # bilateral recovers after the perturbation
    retained = bool(inner._planar_metrics.left_contact or inner._planar_metrics.right_contact)
    certified = bool(both_before and retained and both_recovered and max_slip < _SLIP_HIGH)
    return {"seed": handoff.seed, "probe": probe.value, "both_before": both_before, "both_after": both_after,
            "both_recovered": both_recovered, "max_slip": round(max_slip, 4), "retained": retained,
            "certified": certified}


def micro_transport(env, handoff, family: GraspFamily, p: GraspParams, *, establish: int = 20,
                    epsilons=(0.05, 0.1, 0.2, 0.35), move_steps: int = 10) -> dict:
    """From an established grasp, translate the midpoint by increasing ε toward the target; report the largest ε that
    retains the coin (MICRO-TRANSPORT). Stronger than static retention, weaker than zone entry."""
    inner = env._env
    largest = 0.0
    for eps in epsilons:
        restore_planar(inner, handoff.snap)
        env._horizon = establish + move_steps + 400
        clf = ContactModeClassifier(p.dwell)
        obs = handoff.obs.copy()
        for _ in range(establish):
            mode = clf.classify(inner)
            obs, _r, _t, _tr, _i = env.step(np.clip(grasp_action(inner, obs, mode, family, p), -1, 1).astype(np.float32))
        d, _n = _dir_to_zone(inner)
        start = float(inner._planar_metrics.disk_to_zone)
        for _ in range(move_steps):
            obs, _r, _t, _tr, _i = env.step(obj_to_env(eps * np.asarray(d), 0.0, 0.0, 0.85))
        m = inner._planar_metrics
        retained = bool(m.left_contact or m.right_contact)
        moved = start - float(m.disk_to_zone)
        if retained and moved >= 0.01:                     # coin moved WITH the grasp (retained) by a threshold
            largest = eps
    return {"seed": handoff.seed, "largest_retaining_eps": largest, "micro_transport": largest > 0.0}
