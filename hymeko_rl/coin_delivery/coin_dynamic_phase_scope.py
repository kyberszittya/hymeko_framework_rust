"""DYNAMIC_PHASE_CURRICULUM_SCOPE_V1 — authoritative state = (control_phase, contact_flag), banks by the LIVE persistent
control-phase (not LateStart.family), Stage-1 actor masked to {target_entry, braking, settling_dwell}, dynamic-phase
balanced sampling, and episode scope that truncates when the control phase leaves the Stage-1 set.

KEY REWORK (audit §3/§4): ``contact_retention`` (unilateral contact) is demoted from a mutually-exclusive control phase
to an ORTHOGONAL ``contact_flag`` — in the completed Stage-1c detector it fired 321× and ERASED the task-progress phases
(target_entry/braking) it can coexist with. Control phase is now computed from task progress ONLY; contact is a flag.

    control_phase ∈ {transport, target_entry, braking, settling_dwell}
    contact_flag  ∈ {contact_present, contact_lost}
    state_onehot  = onehot4(control_phase) ++ onehot2(contact_flag)     (dim 6; actor/critic input = obs_48 ++ 6)
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.coin_late_start import LateStart
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, SETTLE_VEL, CoinRL4Dof
from hymeko_rl.coin_delivery.coin_stable_engagement import StableEngagementConfig, StableEngagementGate, stable_engagement_signals

ENTRY_TOL = 0.05
CONTROL_PHASES = ("transport", "target_entry", "braking", "settling_dwell")
CONTACT_FLAGS = ("contact_present", "contact_lost")
STAGE1_CONTROL = ("target_entry", "braking", "settling_dwell")
N_STATE = len(CONTROL_PHASES) + len(CONTACT_FLAGS)      # 6
PRECEDENCE = "settling_dwell > target_entry > braking > transport (task-progress only); contact is an ORTHOGONAL flag"


def control_phase(dtz, prev_dtz, min_dtz, speed, prev_speed, strict) -> str:
    """Task-progress control phase (contact-independent). Precedence: settling_dwell > target_entry > braking > transport."""
    if strict >= 1 or (dtz <= CENTER_TOL and speed < SETTLE_VEL):
        return "settling_dwell"
    if prev_dtz > ENTRY_TOL >= dtz:
        return "target_entry"
    if prev_speed - speed > 0.02 and dtz <= 2 * ENTRY_TOL:
        return "braking"
    return "transport"


def contact_flag(lc, rc) -> str:
    return "contact_present" if (lc or rc) else "contact_lost"


def matching_predicates(dtz, prev_dtz, min_dtz, speed, prev_speed, strict, lc, rc):
    """§3 overlap audit: ALL control-phase predicates that fire at a state + the contact predicates + the selected phase."""
    preds = []
    if strict >= 1 or (dtz <= CENTER_TOL and speed < SETTLE_VEL):
        preds.append("settling_dwell")
    if prev_dtz > ENTRY_TOL >= dtz:
        preds.append("target_entry")
    if prev_speed - speed > 0.02 and dtz <= 2 * ENTRY_TOL:
        preds.append("braking")
    if not preds:
        preds.append("transport")
    return {"control_predicates": preds, "selected": control_phase(dtz, prev_dtz, min_dtz, speed, prev_speed, strict),
            "contact_present": bool(lc or rc), "unilateral_contact": bool(lc != rc)}


class AuthPhaseDetector:
    """DYNAMIC (control_phase, contact_flag) from the current env state + running context. Call once per state."""

    def __init__(self):
        self.prev_dtz = None; self.min_dtz = None; self.prev_speed = None

    def _metrics(self, rl):
        m = rl.inner._planar_metrics
        return float(m.disk_to_zone), float(rl._speed()), bool(m.left_contact), bool(m.right_contact), int(rl._strict)

    def state_of(self, rl):
        dtz, speed, lc, rc, strict = self._metrics(rl)
        pdtz = dtz if self.prev_dtz is None else self.prev_dtz
        mdtz = dtz if self.min_dtz is None else self.min_dtz
        pspd = speed if self.prev_speed is None else self.prev_speed
        cp = control_phase(dtz, pdtz, mdtz, speed, pspd, strict); cf = contact_flag(lc, rc)
        self.min_dtz = min(mdtz, dtz); self.prev_dtz = dtz; self.prev_speed = speed
        return cp, cf

    def predicates_of(self, rl):
        dtz, speed, lc, rc, strict = self._metrics(rl)
        pdtz = dtz if self.prev_dtz is None else self.prev_dtz
        mdtz = dtz if self.min_dtz is None else self.min_dtz
        pspd = speed if self.prev_speed is None else self.prev_speed
        return matching_predicates(dtz, pdtz, mdtz, speed, pspd, strict, lc, rc)


def stage1_actor_trainable(gate_on: bool, control_phase: str) -> float:
    """§5 Stage-1 actor-update mask: 1.0 iff ``gate_t == 1`` AND ``control_phase ∈ {target_entry, braking,
    settling_dwell}``. Used as the per-transition weight in the masked actor loss, so excluded phases (e.g. transport)
    contribute EXACTLY zero actor gradient."""
    return 1.0 if (bool(gate_on) and control_phase in STAGE1_CONTROL) else 0.0


def state_onehot(cp: str, cf: str) -> np.ndarray:
    v = np.zeros(N_STATE, np.float32)
    v[CONTROL_PHASES.index(cp)] = 1.0
    v[len(CONTROL_PHASES) + CONTACT_FLAGS.index(cf)] = 1.0
    return v


def augment_state(obs, cp: str, cf: str) -> np.ndarray:
    return np.concatenate([np.asarray(obs, np.float32), state_onehot(cp, cf)]).astype(np.float32)


# ── §1/§2 rebuild banks by LIVE persistent control-phase ──
def rebuild_control_phase_bank(pi0, seeds, *, min_persist: int = 2, per_phase: int = 8, horizon: int = 360):
    """For each seed, roll frozen pi_0 from neutral; at each gate-active step, if the control_phase is a Stage-1 phase P
    AND it persists in P for >= ``min_persist`` consecutive gate-active steps, record a start (seed, prefix_steps, P).
    One start per (seed, phase). Returns {control_phase: [LateStart]} + per-phase counts."""
    from hymeko_rl.coin_delivery.coin_late_start import _base, _sha
    from hymeko_rl.coin_delivery.coin_residual_critic_state import ResidualCriticStateV2
    from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2
    banks = {p: [] for p in STAGE1_CONTROL}
    for s in seeds:
        if all(len(banks[p]) >= per_phase for p in STAGE1_CONTROL):
            break
        rl = CoinRL4Dof(horizon=horizon); o = rl.reset(int(s))
        gate = StableEngagementGate(StableEngagementConfig()); det = AuthPhaseDetector()
        hist = ResidualCriticStateV2(); hist.reset(o)
        cps = []                                                  # (step, control, gate, obs, base, causal, gate_dict)
        for k in range(horizon):
            cp, _cf = det.state_of(rl)
            gd = ReplayControllerStateV2.from_gate(gate).to_dict()
            cps.append((k, cp, gate.gate == 1.0, o.astype(np.float32), _base(pi0, o).astype(np.float32),
                        hist.feature(gd).astype(np.float32), gd))
            a = cps[-1][4]                                        # pi_0 prefix
            o2, _r, term, trunc, _ = rl.step(a); hist.push(o2, a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            o = o2
            if term or trunc:
                break
        # persistence: a start at index i qualifies for P if cps[i..i+min_persist-1] are gate-active AND control==P
        seen = set()
        for i in range(len(cps) - min_persist + 1):
            k, cp, g, obs, base, causal, gd = cps[i]
            if not g or cp not in STAGE1_CONTROL or cp in seen or len(banks[cp]) >= per_phase:
                continue
            if all(cps[i + j][2] and cps[i + j][1] == cp for j in range(min_persist)):
                seen.add(cp)
                banks[cp].append(LateStart(seed=int(s), prefix_steps=k, family=cp, obs_sha=_sha(obs),
                                           base_sha=_sha(base), causal_sha=_sha(causal), gate_state=gd))
    counts = {p: len(banks[p]) for p in STAGE1_CONTROL}
    return banks, counts
