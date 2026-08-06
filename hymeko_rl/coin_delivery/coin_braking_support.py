"""CONTACT_PRESERVING_BRAKING_PRIMITIVE_V2 — Part A: braking action-support discovery (no training, no primitive tuning).

Question: at the braking states the frozen pi_0 visits, does a bounded offset to the exact pi_0 action exist that
DECELERATES the target-directed coin velocity WHILE preserving required robot-attributed contact (relative to the pi_0
branch from the identical state)? If such SAFE_BENEFICIAL support exists across a meaningful fraction of braking states —
and not only at the two states V1 already delivered cleanly — Part B derives a state-feedback braking basis from these
offsets. If not, we stop rather than tune the primitive blindly.

Deterministic branch-search: from an exactly-restored braking state, apply a candidate first action, then frozen pi_0 for
a short diagnostic horizon; compare to the pi_0-only branch. No open-loop raw-action sequence is optimised or executed.
pi_0, the reward, the certifier, the task and the V1 primitive are all frozen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.coin_delivery.coin_primitive_mpc import (
    NEUTRAL_THETA,
    _default_restore,
    _default_snap,
    _pi0_action,
    detect_mode,
    unpack_theta,
)
from hymeko_rl.coin_delivery.coin_rl_env import HELD_DWELL

ENTRY_TOL = 0.05
ACTION_SCALE = 4.0

# ── frozen discovery configuration (frozen BEFORE inspecting aggregate results) ──
BRANCH_HORIZON = 4                 # candidate action at step 0 + frozen pi_0 for the remaining steps
OFFSET_BOUND = 1.2                 # bounded candidate offsets around pi_0 (action units)
N_CANDIDATES = 32                  # bounded candidate set size (+ the zero/pi_0 reference)
CANDIDATE_SEED = 0
CAPTURE_SPACING = 5                # snapshot every k-th BRAKE state after the first
CAPTURE_HORIZON = 60


@dataclass(frozen=True)
class SafeBeneficialConfig:
    """Frozen SAFE_BENEFICIAL tolerances. A candidate is safe-beneficial vs the pi_0 branch only if it (1) causes no new
    required-contact loss, (2) causes no new exit-before-K6, (3) reduces excessive target-directed coin velocity by
    ≥ ``radial_decel_eps`` OR improves strict/dwell, and (4) does not worsen target progress by > ``progress_tol``."""
    branch_horizon: int = BRANCH_HORIZON
    radial_decel_eps: float = 0.02     # m/s reduction in peak target-directed coin speed to count as deceleration
    progress_tol: float = 0.005        # m: candidate final distance-to-zone may be at most this much worse than pi_0
    meaningful_fraction: float = 0.5   # ≥ this fraction of braking states must have ≥1 safe-beneficial candidate
    min_failing_states_with_support: int = 3   # support must extend beyond the already-successful trajectories


def candidate_offsets(n: int = N_CANDIDATES, *, bound: float = OFFSET_BOUND, seed: int = CANDIDATE_SEED) -> np.ndarray:
    """Deterministic bounded candidate offset set; row 0 is the zero offset (the pi_0 reference branch)."""
    rng = np.random.default_rng(seed)
    off = rng.uniform(-bound, bound, size=(n, 4)).astype(np.float32)
    return np.concatenate([np.zeros((1, 4), np.float32), off], axis=0)


def _coin_kinematics(inner):
    """(distance_to_zone, radial coin velocity toward zone [+=approaching], tangential/lateral coin speed)."""
    direction, dtz = inner.direction_to_zone()                       # unit coin→zone vector + distance
    v = np.asarray(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2], np.float64)
    radial = float(np.dot(v, np.asarray(direction, np.float64)[:2]))
    tangential = float(np.linalg.norm(v - radial * np.asarray(direction, np.float64)[:2]))
    return float(dtz), radial, tangential


def _contact(inner):
    m = inner._planar_metrics
    return bool(m.left_contact or m.right_contact), bool(m.left_contact and m.right_contact)


def branch_eval(rl, first_action, pi0, *, horizon=BRANCH_HORIZON, snap=_default_snap, restore=_default_restore) -> dict:
    """Deterministic branch: restore is the CALLER's duty; this snapshots, applies ``first_action`` then frozen pi_0 for
    ``horizon-1`` steps, records the diagnostic metrics, and restores exactly. The real rollout is unchanged."""
    s = snap(rl)
    dtz0, radial0, _tan0 = _coin_kinematics(rl.inner); contact0, _bi0 = _contact(rl.inner)
    strict0 = int(rl._strict)
    entered = dtz0 <= ENTRY_TOL; exited_before_k6 = False; kept = contact0; bilat_kept = _contact(rl.inner)[1]
    first_loss = None; peak_radial = radial0; min_dtz = dtz0; tang_sum = 0.0; effort = 0.0; certified = False
    a = np.asarray(first_action, np.float32)
    for t in range(horizon):
        act = a if t == 0 else _pi0_action(pi0, rl.obs())
        rl.step(act); effort += float(np.abs(act).mean())
        dtz, radial, tang = _coin_kinematics(rl.inner); c, bi = _contact(rl.inner)
        kept = kept and c; bilat_kept = bilat_kept and bi
        if contact0 and not c and first_loss is None:
            first_loss = t
        peak_radial = max(peak_radial, radial); min_dtz = min(min_dtz, dtz); tang_sum += tang
        if dtz <= ENTRY_TOL:
            entered = True
        elif entered:
            exited_before_k6 = exited_before_k6 or (int(rl._strict) < HELD_DWELL)
        certified = certified or (int(rl._strict) >= HELD_DWELL and rl._touched)
        final_dtz = dtz
    restore(rl, s)
    return {"contact_kept": bool(kept), "bilateral_kept": bool(bilat_kept), "first_contact_loss": first_loss,
            "final_dtz": float(final_dtz), "min_dtz": float(min_dtz), "peak_radial_vel": float(peak_radial),
            "mean_tangential": float(tang_sum / horizon), "entered": bool(entered),
            "exit_before_k6": bool(exited_before_k6 and not certified), "strict_change": int(rl._strict) - strict0,
            "certified": bool(certified), "effort": round(effort / horizon, 4)}


def safe_beneficial(cand: dict, ref: dict, cfg: SafeBeneficialConfig) -> bool:
    """Is ``cand`` safe-beneficial vs the pi_0 branch ``ref`` (both from the identical state)?"""
    no_new_loss = cand["contact_kept"] or (not ref["contact_kept"])           # only illegal if pi_0 kept contact and cand didn't
    no_new_exit = (not cand["exit_before_k6"]) or ref["exit_before_k6"]
    decelerates = cand["peak_radial_vel"] < ref["peak_radial_vel"] - cfg.radial_decel_eps
    improves_task = cand["strict_change"] > ref["strict_change"]
    progress_ok = cand["final_dtz"] <= ref["final_dtz"] + cfg.progress_tol
    return bool(no_new_loss and no_new_exit and (decelerates or improves_task) and progress_ok)


def evaluate_state_support(rl, offsets, pi0, cfg: SafeBeneficialConfig, *, snap=_default_snap, restore=_default_restore):
    """Branch every candidate (offsets[0] = the pi_0 reference) from the current braking state; return per-state support."""
    a0 = _pi0_action(pi0, rl.obs())
    branches = [branch_eval(rl, np.clip(a0 + o, -ACTION_SCALE, ACTION_SCALE), pi0, horizon=cfg.branch_horizon,
                            snap=snap, restore=restore) for o in offsets]
    ref = branches[0]                                                          # zero offset = pi_0 branch
    safe = [(i, offsets[i], b) for i, b in enumerate(branches) if i > 0 and safe_beneficial(b, ref, cfg)]
    best_decel = max((ref["peak_radial_vel"] - b["peak_radial_vel"] for _i, _o, b in safe), default=0.0)
    return {"ref": ref, "n_candidates": len(offsets) - 1, "n_safe_beneficial": len(safe),
            "best_radial_decel": round(float(best_decel), 4),
            "safe_offsets": [np.asarray(o, np.float32).round(4).tolist() for _i, o, _b in safe],
            "pi0_peak_radial": round(ref["peak_radial_vel"], 4), "pi0_contact_kept": ref["contact_kept"]}


def support_gate(state_rows, cfg: SafeBeneficialConfig):
    """Aggregate support + the Part-A gate. FOUND only if a meaningful fraction of braking states have support AND that
    support is not isolated to the already-successful (contact-preserving-delivery) trajectories."""
    n = len(state_rows)
    with_support = [r for r in state_rows if r["support"]["n_safe_beneficial"] > 0]
    frac = len(with_support) / max(n, 1)
    failing = [r for r in state_rows if r["v1_outcome"] in ("contact_losing", "target_exit", "no_delivery")]
    failing_with_support = [r for r in failing if r["support"]["n_safe_beneficial"] > 0]
    found = (frac >= cfg.meaningful_fraction and len(failing_with_support) >= cfg.min_failing_states_with_support)
    return {"n_braking_states": n, "fraction_with_support": round(frac, 4),
            "n_failing_states": len(failing), "n_failing_states_with_support": len(failing_with_support),
            "mean_safe_beneficial_per_state": round(float(np.mean([r["support"]["n_safe_beneficial"] for r in state_rows])) if n else 0.0, 3),
            "best_radial_decel_overall": round(float(max((r["support"]["best_radial_decel"] for r in state_rows), default=0.0)), 4),
            "verdict": "BRAKING_SAFE_BENEFICIAL_SUPPORT_FOUND" if found else "BRAKING_SAFE_BENEFICIAL_SUPPORT_INSUFFICIENT"}


def capture_braking_states(pi0, starts, *, theta=None, spacing=CAPTURE_SPACING, horizon=CAPTURE_HORIZON):
    """Drive each reconstructed handoff with frozen pi_0 feedback; using V1 mode detection (fixed ``theta``), record the
    ABSOLUTE pi_0-from-reset step of the FIRST BRAKE-mode state and every ``spacing``-th BRAKE state thereafter. The
    absolute step (not a MuJoCo snapshot) is stored so each state is reconstructed EXACTLY via ``replay_pi0`` later —
    a raw snapshot would not reproduce the ``node_features`` velocity buffer (coin_late_start). Deterministic."""
    from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
    from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
    th = unpack_theta(NEUTRAL_THETA if theta is None else theta)
    captured = []
    for ls in starts:
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        ctrl_mode = "PUSH"; n_brake = 0
        for step in range(horizon):
            ctrl_mode = detect_mode(rl, th, ctrl_mode)
            if ctrl_mode == "BRAKE" and (n_brake == 0 or n_brake % spacing == 0):
                dtz, radial, tang = _coin_kinematics(rl.inner)
                captured.append({"seed": int(ls.seed), "family": ls.family, "handoff_prefix": int(ls.prefix_steps),
                                 "abs_step": int(ls.prefix_steps + step),
                                 "dtz": round(dtz, 4), "radial_vel": round(radial, 4), "tangential_vel": round(tang, 4)})
            if ctrl_mode == "BRAKE":
                n_brake += 1
            a = _pi0_action(pi0, rl.obs()); _o, _r, term, trunc, _ = rl.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            if term or trunc:
                break
    return captured
