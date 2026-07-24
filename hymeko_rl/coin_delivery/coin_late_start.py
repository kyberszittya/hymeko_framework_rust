"""Deterministic REPLAY-to-handoff late-start construction (§5) + mechanism-balanced late-start banks (§6).

A late-start is NOT an incomplete MuJoCo snapshot (which does not reproduce ``node_features``' velocity buffer — the
non-determinism found in the hold sweep). It is reconstructed deterministically:

    neutral reset(seed) → replay the frozen ``pi_0`` prefix for ``prefix_steps`` → (verify obs/gate/history/base) →
    begin the late-policy episode from the LIVE env at the handoff.

Because both the bank-build capture and the reconstruction are the identical ``pi_0``-from-``reset(seed)`` computation,
the reconstructed observation / gate state / history state / base action match the captured ones EXACTLY (diff 0).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_residual_critic_state import ResidualCriticStateV2
from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, SETTLE_VEL, CoinRL4Dof
from hymeko_rl.coin_delivery.coin_stable_engagement import (
    StableEngagementConfig,
    StableEngagementGate,
    stable_engagement_signals,
)

ENTRY_TOL = 0.05
ACTION_SCALE = 4.0
# PHASE_SWITCHED_TD3_BASELINE_V1 campaign families: settling+dwell MERGED (the detector enters dwell the instant strict
# settling begins, so a separate replay family is not load-bearing); contact_loss_reacquisition DEFERRED to a later
# recovery curriculum (rare under pi_0 replay).
LATE_FAMILIES = ("transport", "target_entry", "overshoot", "braking", "settling_dwell", "contact_retention")


def _base(pi0, obs):
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.as_tensor(np.asarray(obs, np.float32)[None]))[0].numpy(),
                       -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def _sha(a) -> str:
    return hashlib.sha256(np.asarray(a, np.float32).tobytes()).hexdigest()[:12]


def _classify(step, dtz, prev_dtz, min_dtz, speed, prev_speed, lc, rc, strict) -> str:
    """Primary late-phase family for a gate-active step (priority: most specific first). ``settling`` and ``dwell`` are
    MERGED into ``settling_dwell`` for the baseline campaign."""
    if strict >= 1 or (dtz <= CENTER_TOL and speed < SETTLE_VEL):
        return "settling_dwell"
    if lc != rc:
        return "contact_retention"
    if min_dtz <= ENTRY_TOL and dtz > min_dtz + 0.01:
        return "overshoot"
    if prev_dtz > ENTRY_TOL >= dtz:
        return "target_entry"
    if prev_speed - speed > 0.02 and dtz <= 2 * ENTRY_TOL:
        return "braking"
    return "transport"


@dataclass
class HandoffRecord:
    step: int
    gate_mult: float
    family: str
    obs: np.ndarray
    base: np.ndarray
    gate_state: dict
    causal_state: np.ndarray
    strict: int = 0            # dwell counter entering this handoff (for strict/boundary-conditioned panel selection)
    dtz: float = 0.0           # disk-to-zone at the handoff
    speed: float = 0.0         # disk speed at the handoff


def replay_pi0(pi0, seed: int, *, horizon: int = 360, stop_at: "int | None" = None, coin_shape: str = "cylinder",
               disk_radius_override=None, arm_mjcf_transform=None, geom: "str | None" = None):
    """Replay the frozen ``pi_0`` from neutral ``reset(seed)``. If ``stop_at`` is None, return the list of per-step
    :class:`HandoffRecord` (one per env step, BEFORE that step's action). Otherwise stop after ``stop_at`` steps and
    return ``(rl, gate, history, record_at_stop)`` — the LIVE env positioned at the handoff to begin the late episode.
    ``coin_shape`` / ``disk_radius_override`` / ``arm_mjcf_transform`` / ``geom`` drive OBJECT_TO_TARGET_VARIANTS and the
    ball-tip embodiment (defaults ⇒ the frozen canonical E0 clamp on the frozen coin)."""
    rl = CoinRL4Dof(horizon=horizon, geom=geom, arm_mjcf_transform=arm_mjcf_transform,
                    coin_shape=coin_shape, disk_radius_override=disk_radius_override)
    o = rl.reset(int(seed))
    gate = StableEngagementGate(StableEngagementConfig())
    hist = ResidualCriticStateV2(); hist.reset(o)
    records = []
    prev_dtz = float(rl.inner._planar_metrics.disk_to_zone); min_dtz = prev_dtz; prev_speed = rl._speed()
    for k in range(horizon):
        m = rl.inner._planar_metrics
        dtz = float(m.disk_to_zone); speed = rl._speed(); lc, rc = bool(m.left_contact), bool(m.right_contact)
        gd = ReplayControllerStateV2.from_gate(gate).to_dict()
        rec = HandoffRecord(step=k, gate_mult=float(gate.gate),
                            family=_classify(k, dtz, prev_dtz, min_dtz, speed, prev_speed, lc, rc, rl._strict),
                            obs=o.astype(np.float32), base=_base(pi0, o).astype(np.float32),
                            gate_state=gd, causal_state=hist.feature(gd).astype(np.float32),
                            strict=int(rl._strict), dtz=float(dtz), speed=float(speed))
        if stop_at is not None and k == stop_at:
            return rl, gate, hist, rec
        records.append(rec)
        a = rec.base                                            # frozen pi_0 prefix action
        o2, _r, term, trunc, _ = rl.step(a); hist.push(o2, a)
        lc2, rc2, coin, lt, rtp = stable_engagement_signals(rl.inner)
        gate.update(lc2, rc2, coin, lt, rtp, terminated=bool(term))
        min_dtz = min(min_dtz, dtz); prev_dtz, prev_speed = dtz, speed; o = o2
        if term or trunc:
            break
    if stop_at is not None:
        raise ValueError(f"seed {seed} terminated before stop_at={stop_at}")
    return records


@dataclass
class LateStart:
    """A deterministically reconstructable handoff: (seed, prefix_steps) + captured identity for verification."""

    seed: int
    prefix_steps: int
    family: str
    obs_sha: str
    base_sha: str
    causal_sha: str
    gate_state: dict = field(default_factory=dict)

    def to_row(self):
        return [self.seed, self.prefix_steps, self.family, self.obs_sha, self.base_sha, self.causal_sha]


def reconstruct_handoff(pi0, ls: "LateStart", *, horizon: int = 360, coin_shape: str = "cylinder",
                        disk_radius_override=None, arm_mjcf_transform=None, geom: "str | None" = None):
    """Replay to ``ls.prefix_steps`` and return ``(rl, gate, history, record)`` at the handoff (LIVE env). The object /
    embodiment overrides (defaults ⇒ frozen canonical) reconstruct the SAME (seed, prefix) handoff on a VARIANT coin/robot."""
    return replay_pi0(pi0, ls.seed, horizon=horizon, stop_at=ls.prefix_steps, coin_shape=coin_shape,
                      disk_radius_override=disk_radius_override, arm_mjcf_transform=arm_mjcf_transform, geom=geom)


def verify_reconstruction(pi0, ls: "LateStart", *, horizon: int = 360) -> dict:
    """Reconstruct and check the obs/base/causal/gate identity matches the captured late-start (exact, diff 0)."""
    _rl, _gate, _hist, rec = reconstruct_handoff(pi0, ls, horizon=horizon)
    return {"obs_ok": _sha(rec.obs) == ls.obs_sha, "base_ok": _sha(rec.base) == ls.base_sha,
            "causal_ok": _sha(rec.causal_state) == ls.causal_sha, "gate_ok": rec.gate_state == ls.gate_state,
            "gate_active": rec.gate_mult == 1.0, "family": rec.family}


def build_late_start_bank(pi0, seeds, *, per_family: int = 6, horizon: int = 360) -> "list[LateStart]":
    """Scan ``seeds``; at each gate-active handoff, record a :class:`LateStart`, balancing across LATE_FAMILIES
    (best-effort — some phases are rare). One late-start per (seed, family) to keep starts disjoint by seed-phase."""
    bank: list[LateStart] = []
    have = {f: 0 for f in LATE_FAMILIES}
    for s in seeds:
        if all(have[f] >= per_family for f in LATE_FAMILIES):
            break
        seen_fam = set()
        for rec in replay_pi0(pi0, int(s), horizon=horizon):
            if rec.gate_mult != 1.0 or rec.family in seen_fam or have[rec.family] >= per_family:
                continue
            seen_fam.add(rec.family); have[rec.family] += 1
            bank.append(LateStart(seed=int(s), prefix_steps=rec.step, family=rec.family,
                                  obs_sha=_sha(rec.obs), base_sha=_sha(rec.base), causal_sha=_sha(rec.causal_state),
                                  gate_state=rec.gate_state))
    return bank


def build_boundary_panel(pi0, seeds, held_out_forbidden, *, want,
                         families=("target_entry", "braking", "settling_dwell"),
                         strict_primary=(3, 4, 5), strict_fill=(2,), per_seed_cap=3):
    """Scan HELD-OUT seeds for gate-active, in-family handoffs at strict∈strict_primary (mid-dwell, at the
    CENTER_TOL/SETTLE_VEL boundary — where a small sustained action CAN flip the certifier); top up with strict_fill if
    short of ``want``. Returns (list[LateStart], family_composition, strict_histogram). Asserts held-out disjointness so a
    critic-train/dev seed can never leak into an evaluation panel."""
    picked, comp, strict_hist = [], Counter(), Counter()
    for pool in (strict_primary, strict_fill):
        for s in seeds:
            if len(picked) >= want:
                break
            assert s not in held_out_forbidden, f"seed {s} is a critic-train/dev seed — not held out"
            taken = 0
            for rec in replay_pi0(pi0, int(s), horizon=360):
                if taken >= per_seed_cap or len(picked) >= want:
                    break
                if rec.gate_mult != 1.0 or rec.family not in families or rec.strict not in pool:
                    continue
                picked.append(LateStart(seed=int(s), prefix_steps=rec.step, family=rec.family, obs_sha=_sha(rec.obs),
                                        base_sha=_sha(rec.base), causal_sha=_sha(rec.causal_state), gate_state=rec.gate_state))
                comp[rec.family] += 1; strict_hist[rec.strict] += 1; taken += 1
        if len(picked) >= want:
            break
    return picked, dict(comp), dict(sorted(strict_hist.items()))


def late_start_bank_manifest(bank) -> dict:
    rows = [ls.to_row() for ls in bank]
    have = {f: sum(ls.family == f for ls in bank) for f in LATE_FAMILIES}
    return {"n": len(rows), "coverage": have, "families": list(LATE_FAMILIES),
            "sha16": hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16], "rows": rows}
