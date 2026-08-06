"""GROUPED_COUNTERFACTUAL_LABELS + rich critic-transition collection (§6-§7, corrected V2 harness).

Fixes the four contract defects of the invalidated first-pass harness:

1. **Gated collection** — residual exploration is applied ONLY when ``gate==1``; gate-off steps are bit-identical to
   ``pi_0`` (the 9/9 grasp policy is never perturbed). The invalidated collector added full-action Gaussian noise in
   every phase.
2. **Both critic states** — every transition stores the instantaneous obs (48) AND the causal
   ``RESIDUAL_CRITIC_STATE_V2`` (163) so the §8 ablation trains both arms on identical data.
3. **``truncated`` kept** — ``terminated`` and ``truncated`` are stored separately; the Bellman ``done`` mask is
   ``terminated`` only (a horizon truncation bootstraps, it is not a real terminal).
4. **Within-group counterfactuals** — each captured physical state is one ``state_group_id``; candidate returns are the
   canonical *full-remaining-horizon* discounted return under the frozen ``pi_0`` continuation (not a 40-step stub),
   computed **twice** to certify deterministic restoration. Candidate directions are labeled by construction kind
   (± actuator basis, isotropic random) — never as task-space "toward/away"; the task-space effect is *measured*
   post-hoc from the continuation outcome.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_residual_critic import encode_controller_state
from hymeko_rl.coin_delivery.coin_residual_critic_state import ResidualCriticStateV2
from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, CoinRL4Dof
from hymeko_rl.coin_delivery.coin_stable_engagement import (
    StableEngagementConfig,
    StableEngagementGate,
    stable_engagement_signals,
)
from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar

ACTION_SCALE = 4.0
RESIDUAL_BOUND = 0.25
GAMMA = 0.99
ENTRY_TOL = 0.05
# fixed residual magnitudes (§7); directions are construction-kind labeled (no task-space naming)
MAGNITUDES = (0.0, 0.01, 0.025, 0.05, 0.10, 0.25)
FAMILIES = ("transport", "entry", "settling", "contact_retention")


def base_action(pi0, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.as_tensor(np.asarray(obs, np.float32)[None]))[0].numpy(),
                       -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def composite(base: np.ndarray, gate_mult: float, delta: np.ndarray) -> np.ndarray:
    d = np.clip(np.asarray(delta, np.float32), -RESIDUAL_BOUND, RESIDUAL_BOUND)
    b = np.clip(np.asarray(base, np.float32), -ACTION_SCALE, ACTION_SCALE)
    return np.clip(b + float(gate_mult) * d, -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def family_of(dtz: float, lc: bool, rc: bool) -> str:
    if dtz <= CENTER_TOL:
        return "settling"
    if lc != rc:
        return "contact_retention"
    return "transport" if dtz > ENTRY_TOL else "entry"


# ───────────────────────── §7 grouped counterfactual labels ─────────────────────────
def residual_candidates(base: np.ndarray, rng: np.random.Generator, *, n_random: int = 3):
    """(name, delta(4), meta) over MAGNITUDES × {±actuator basis, isotropic random}. delta==0 is the sole mag-0 entry.

    Directions carry only a construction *kind* (``basis+k``/``basis-k``/``rand``); NO task-space "toward/away" label —
    the physical effect is measured from the continuation, not presumed from a joint-space sign."""
    out = [("zero", np.zeros(4, np.float32), {"magnitude": 0.0, "kind": "zero", "dir": "none"})]
    dirs: list[tuple[str, np.ndarray]] = []
    for k in range(4):
        e = np.zeros(4, np.float32); e[k] = 1.0
        dirs.append((f"basis+{k}", e.copy()))
        dirs.append((f"basis-{k}", -e.copy()))
    for j in range(n_random):
        v = rng.standard_normal(4).astype(np.float32); v /= (np.linalg.norm(v) + 1e-9)
        dirs.append((f"rand{j}", v))
    for mag in MAGNITUDES[1:]:
        for name, u in dirs:
            out.append((f"{name}@{mag}", (RESIDUAL_BOUND * mag * u).astype(np.float32),
                        {"magnitude": float(mag), "kind": name.rstrip("0123456789+-"), "dir": name}))
    return out


@dataclass
class StateGroup:
    """One captured gate-active physical state = one ``state_group_id`` for within-group counterfactual ranking."""

    group_id: int
    seed: int
    family: str
    t: int
    obs: np.ndarray                 # instantaneous 48
    base: np.ndarray                # clip(pi_0(obs)) 4
    causal_state: np.ndarray        # RESIDUAL_CRITIC_STATE_V2 163
    cstate: dict                    # PHASE_GATE_CONTROLLER_STATE_V2 dict (for encode)
    snap: object                    # (PlanarSnapshot, _t, _strict, _touched)
    contact0: tuple                 # (lc, rc) at capture
    gate_snap: object = None        # deepcopy of the StableEngagementGate FSM at capture (controller-state restore)
    cand_names: list = field(default_factory=list)
    cand_delta: list = field(default_factory=list)
    cand_meta: list = field(default_factory=list)
    G: list = field(default_factory=list)          # canonical full-horizon discounted return per candidate
    G0: float = 0.0
    outcomes: list = field(default_factory=list)   # measured task-space effect per candidate


def _snap_rl(rl: CoinRL4Dof):
    return (snapshot_planar(rl.inner), rl._t, rl._strict, rl._touched)


def _restore_rl(rl: CoinRL4Dof, s) -> None:
    restore_planar(rl.inner, s[0]); rl._t, rl._strict, rl._touched = s[1], s[2], s[3]


def counterfactual_return(rl: CoinRL4Dof, pi0, snap, cand_action: np.ndarray):
    """Restore ``snap``; apply the single candidate composite action; continue with FROZEN ``pi_0`` base to term/trunc.

    Returns ``(discounted_return, outcome)``. Deterministic given ``snap`` (MuJoCo restore + mj_forward).
    """
    _restore_rl(rl, snap)
    o2, r, term, trunc, _ = rl.step(np.asarray(cand_action, np.float32))
    tot, disc = float(r), 1.0
    entered = rl._dtz() <= CENTER_TOL
    max_dwell = rl._strict
    m = rl.inner._planar_metrics
    contact_after = bool(m.left_contact or m.right_contact)
    contact_persist = contact_after
    o = o2
    while not (term or trunc):
        o2, r, term, trunc, _ = rl.step(base_action(pi0, o))
        disc *= GAMMA; tot += disc * r
        entered = entered or (rl._dtz() <= CENTER_TOL)
        max_dwell = max(max_dwell, rl._strict)
        mm = rl.inner._planar_metrics
        contact_persist = contact_persist and bool(mm.left_contact or mm.right_contact)
        o = o2
    outcome = {"contact_after": contact_after, "contact_persist": bool(contact_persist),
               "entered_zone": bool(entered), "max_dwell": int(max_dwell),
               "strict_success": bool(term and max_dwell >= HELD_DWELL)}
    return tot, outcome


def label_group(rl: CoinRL4Dof, pi0, g: StateGroup, cands, *, verify_determinism: bool = True):
    """Fill ``g.G``/``g.G0``/``g.outcomes`` for every candidate; certify determinism by a second restore+rollout."""
    g.cand_names = [n for n, _d, _m in cands]
    g.cand_delta = [d for _n, d, _m in cands]
    g.cand_meta = [m for _n, _d, m in cands]
    G, outs = [], []
    for _n, d, _m in cands:
        a = composite(g.base, 1.0, d)                          # captured state is gate-active (gate=1)
        ret, outcome = counterfactual_return(rl, pi0, g.snap, a)
        if verify_determinism:
            ret2, _ = counterfactual_return(rl, pi0, g.snap, a)
            if abs(ret - ret2) > 1e-9:
                raise AssertionError(f"non-deterministic counterfactual: {ret} vs {ret2} (group {g.group_id})")
        G.append(float(ret)); outs.append(outcome)
    g.G, g.outcomes, g.G0 = G, outs, G[0]
    return g


# ───────────────────────── §6 rich critic-transition collection ─────────────────────────
_FROZEN_BEHAVIOR = ("zero", "small", "medium", "boundary", "basis", "isotropic")


def frozen_behavior_delta(rng: np.random.Generator, gate_active: bool) -> np.ndarray:
    """FROZEN deployable behavior-residual distribution (§5): mixes zero / small / medium / rare boundary controls /
    ±actuator basis / isotropic random. Returns 0 when the gate is inactive (gate-off ⇒ pi_0 exact). NOT retuned by
    rollout success."""
    if not gate_active:
        return np.zeros(4, np.float32)
    kind = rng.choice(_FROZEN_BEHAVIOR, p=[0.15, 0.35, 0.25, 0.05, 0.12, 0.08])
    if kind == "zero":
        return np.zeros(4, np.float32)
    if kind == "small":
        return np.clip(rng.normal(0, 0.03, 4), -RESIDUAL_BOUND, RESIDUAL_BOUND).astype(np.float32)
    if kind == "medium":
        return np.clip(rng.normal(0, 0.10, 4), -RESIDUAL_BOUND, RESIDUAL_BOUND).astype(np.float32)
    if kind == "boundary":
        return (RESIDUAL_BOUND * rng.choice([-1.0, 1.0], 4)).astype(np.float32)
    if kind == "basis":
        e = np.zeros(4, np.float32); e[rng.integers(4)] = RESIDUAL_BOUND * rng.choice([-1.0, 1.0]); return e
    v = rng.standard_normal(4); v = v / (np.linalg.norm(v) + 1e-9)
    return (RESIDUAL_BOUND * rng.uniform(0.1, 1.0) * v).astype(np.float32)


def collect_critic_transitions(pi0, seeds, *, horizon: int = 360, seed: int = 0):
    """Roll the GATED residual controller on the frozen behavior distribution; store rich transitions for BOTH critic
    arms. Gate-off steps are bit-identical to ``pi_0``. ``terminated`` and ``truncated`` are stored separately.

    Each transition dict has: ``obs_t``(48), ``cs_t``(163), ``enc_t``(11), ``act``(4), ``reward``, ``obs_tp1``(48),
    ``cs_tp1``(163), ``enc_tp1``(11), ``gate_tp1``(scalar), ``terminated``, ``truncated``, ``traj``.
    """
    rng = np.random.default_rng(seed)
    rl = CoinRL4Dof(horizon=horizon)
    trs = []
    for ti, s in enumerate(seeds):
        o = rl.reset(int(s))
        gate = StableEngagementGate(StableEngagementConfig())
        cs = ResidualCriticStateV2(); cs.reset(o)
        for _t in range(horizon):
            gmult = gate.gate
            b = base_action(pi0, o)
            delta = frozen_behavior_delta(rng, gmult == 1.0)
            act = composite(b, gmult, delta)
            gate_t = ReplayControllerStateV2.from_gate(gate).to_dict()
            cs_t = cs.feature(gate_t).astype(np.float32)
            o2, r, term, trunc, _ = rl.step(act)
            cs.push(o2, act)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            gate_tp1 = ReplayControllerStateV2.from_gate(gate).to_dict()
            cs_tp1 = cs.feature(gate_tp1).astype(np.float32)
            trs.append({"obs_t": o.astype(np.float32), "cs_t": cs_t,
                        "enc_t": encode_controller_state(gate_t).astype(np.float32), "act": act,
                        "reward": float(r), "obs_tp1": o2.astype(np.float32), "cs_tp1": cs_tp1,
                        "enc_tp1": encode_controller_state(gate_tp1).astype(np.float32),
                        "gate_tp1": float(gate_tp1["gate"]), "terminated": bool(term),
                        "truncated": bool(trunc), "traj": ti})
            o = o2
            if term or trunc:
                break
    return trs


def capture_state_panel(pi0, seeds, *, per_family: int = 8, horizon: int = 360, n_random: int = 3,
                        gid_start: int = 0, label: bool = True) -> "list[StateGroup]":
    """Roll the gated composite (residual OFF) from neutral; capture gate-active states balanced across FAMILIES; label
    each with grouped counterfactual returns. Two capture offsets per family (early/settled inside the phase).

    ``label=False`` returns UNLABELED states (with the gate-FSM snapshot populated) for callers that impose their own
    labeling — e.g. the residual-hold-horizon sweep, which restores the controller state and re-advances the gate."""
    rl = CoinRL4Dof(horizon=horizon)
    groups: list[StateGroup] = []
    gid = gid_start
    for s in seeds:
        if all(len([g for g in groups if g.family == f]) >= per_family for f in FAMILIES):
            break
        o = rl.reset(int(s))
        gate = StableEngagementGate(StableEngagementConfig())
        cstate = ResidualCriticStateV2(); cstate.reset(o)
        seen = {f: 0 for f in FAMILIES}
        for _t in range(horizon):
            gmult = gate.gate
            b = base_action(pi0, o)
            m = rl.inner._planar_metrics
            dtz = float(m.disk_to_zone); lc, rc = bool(m.left_contact), bool(m.right_contact)
            gate_dict = ReplayControllerStateV2.from_gate(gate).to_dict()
            if gmult == 1.0:
                f = family_of(dtz, lc, rc); seen[f] += 1
                if len([g for g in groups if g.family == f]) < per_family and seen[f] in (2, 6):
                    groups.append(StateGroup(
                        group_id=gid, seed=int(s), family=f, t=rl._t,
                        obs=o.astype(np.float32), base=b.astype(np.float32),
                        causal_state=cstate.feature(gate_dict).astype(np.float32),
                        cstate=gate_dict, snap=_snap_rl(rl), contact0=(lc, rc),
                        gate_snap=copy.deepcopy(gate)))
                    gid += 1
            o2, r, term, trunc, _ = rl.step(b)                 # residual OFF during capture rollout
            cstate.push(o2, b)
            lc2, rc2, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc2, rc2, coin, lt, rtp, terminated=bool(term))
            o = o2
            if term or trunc:
                break
    # label each captured state with grouped one-step counterfactual returns (deterministic; verified x2)
    if label:
        for g in groups:
            cands = residual_candidates(g.base, np.random.default_rng(1000 + g.group_id), n_random=n_random)
            label_group(rl, pi0, g, cands)
    return groups
