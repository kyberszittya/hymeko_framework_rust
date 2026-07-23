"""FEEDBACK_CHUNK_WARMSTART_V2 — dense warm-start from the proven H-receding-horizon FEEDBACK planner (NOT isolated
open-loop CEM chunks). At EVERY replanning state of a receding-horizon feedback rollout we record a training example:
authoritative state, the planner's K-step chunk, the M-executed prefix, the resulting state, and outcomes. The label is
the planner chunk ONLY IF its executed M-prefix is SAFE-ADMISSIBLE (preserves robot contact vs pi_0, no target exit) AND
the planner continuation improves ≥1 task metric without degrading contact/exit; otherwise the pi_0-derived chunk is the
safe fallback. Planner is WARM_START_ONLY (never at learned-policy eval). Prefix-weighted regression + supervised DAgger.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_chunk_td3 import (
    ACT_DIM,
    ACTION_SCALE,
    K,
    M,
    ChunkActor,
    _pi0_action,
    execute_chunk,
    pi0_chunk_target,
    plan_chunk_target,
    state_vec,
)
from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.coin_delivery.coin_rl_env import HELD_DWELL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_transport_dwell import ENTRY_TOL, EventStateDetector
from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar

PREFIX_WEIGHTS = np.array([1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1], np.float32)   # frozen; first M=2 load-bearing
CONT_WINDOW = 15                                          # short continuation window for admissibility/improvement


def _snap(rl):
    return (snapshot_planar(rl.inner), rl._t, rl._strict, rl._touched)


def _restore(rl, s):
    restore_planar(rl.inner, s[0]); rl._t, rl._strict, rl._touched = s[1], s[2], s[3]


def _continuation(rl, pi0, snap, first_actions, *, window=CONT_WINDOW):
    """Restore ``snap``; execute ``first_actions`` (the M-prefix), then frozen pi_0 for ``window`` steps. Return the
    short-horizon outcome used for admissibility + improvement (all robot-attributed, matched start state)."""
    _restore(rl, snap)
    contact0 = bool(rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact)
    dtz0 = rl._dtz(); entered = dtz0 <= ENTRY_TOL; exited = False; contact_all = contact0
    max_dwell = rl._strict; brake = []; prev_speed = rl._speed(); term = trunc = False; prefix_contact = True
    seq = list(first_actions) + [None] * window
    for i, a in enumerate(seq):
        if a is None:
            a = _pi0_action(pi0, rl.obs())
        o2, r, term, trunc, _ = rl.step(a)
        sp = rl._speed(); m = rl.inner._planar_metrics; con = bool(m.left_contact or m.right_contact)
        contact_all = contact_all and con
        if i < len(first_actions):
            prefix_contact = prefix_contact and con           # contact preserved over the EXECUTED prefix
        brake.append(prev_speed - sp); prev_speed = sp
        dtz = rl._dtz(); max_dwell = max(max_dwell, rl._strict)
        if dtz <= ENTRY_TOL:
            entered = True
        elif entered:
            exited = True
        if term or trunc:
            break
    return {"prefix_contact": prefix_contact, "contact_all": contact_all, "entered": entered, "exited": exited,
            "max_dwell": max_dwell, "braking": float(np.mean(brake)) if brake else 0.0,
            "progress": float(dtz0 - rl._dtz()), "strict": int(max_dwell >= HELD_DWELL)}


def label_chunk(rl, pi0, snap, planner_chunk, pi0_chunk, *, m=M):
    """Return (label_chunk, provenance, flags). planner label ONLY if the executed M-prefix is safe-admissible AND the
    planner continuation improves ≥1 of {progress, braking, dwell, strict, entry} without degrading contact/exit."""
    pl = _continuation(rl, pi0, snap, planner_chunk[:m])
    p0 = _continuation(rl, pi0, snap, pi0_chunk[:m])
    safe = pl["prefix_contact"] >= p0["prefix_contact"] and (not pl["exited"] or p0["exited"])
    improve = (pl["progress"] > p0["progress"] + 1e-4 or pl["braking"] > p0["braking"] + 1e-4
               or pl["max_dwell"] > p0["max_dwell"] or pl["strict"] > p0["strict"] or (pl["entered"] and not p0["entered"]))
    no_degrade = pl["contact_all"] >= p0["contact_all"] and (not pl["exited"] or p0["exited"])
    if safe and improve and no_degrade:
        return planner_chunk, "planner", {"safe": True, "improving": True, "full_K_contact": pl["contact_all"]}
    return pi0_chunk, "pi0_fallback", {"safe": bool(safe), "improving": bool(improve), "full_K_contact": pl["contact_all"]}


def feedback_chunk_rollout(pi0, ls, *, horizon=60, plan_seed=0, pop=20, iters=4, record_teachers=False):
    """Receding-horizon FEEDBACK teacher rollout from a reconstructed late-start; one example per replanning state.
    Advances by executing the planner's M-prefix (the teacher); records the SAFE label per state. When
    ``record_teachers`` is set, each example also carries BOTH teacher first-actions (``pi0_first``/``planner_first``)
    and the label first-action, so the mixed-teacher-averaging audit can compare them (no effect on the label)."""
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    det = EventStateDetector(); pa = rec.base.astype(np.float32); ex = []
    steps = 0; term = trunc = False
    while steps < horizon and not (term or trunc):
        cm, cf, ev = det.state_of(rl)
        if gate.gate != 1.0:                                  # gate-off: pi_0 (not a replanning state)
            a = _pi0_action(pi0, rl.obs()); o2, r, term, trunc, _ = rl.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            pa = a; steps += 1; continue
        snap = _snap(rl)
        planner_chunk = plan_chunk_target(rl, rl.cf, plan_seed + steps, k=K, pop=pop, iters=iters)
        _restore(rl, snap); pi0_chunk = pi0_chunk_target(rl, pi0, k=K)
        _restore(rl, snap)
        label, prov, flags = label_chunk(rl, pi0, snap, planner_chunk, pi0_chunk)
        _restore(rl, snap)                                    # replay teacher (planner) prefix to advance
        st = state_vec(rl.obs(), cm, cf, ev, pa)
        rec_ex = {"state": st, "label_chunk": label.reshape(-1).astype(np.float32), "prov": prov, "cm": cm,
                  "flags": flags, "seed": int(ls.seed)}
        if record_teachers:
            rec_ex["pi0_first"] = pi0_chunk[0].astype(np.float32)
            rec_ex["planner_first"] = planner_chunk[0].astype(np.float32)
            rec_ex["label_first"] = label[0].astype(np.float32)
        ex.append(rec_ex)
        for j in range(M):
            gate_on = gate.gate == 1.0
            a = np.clip(planner_chunk[j], -ACTION_SCALE, ACTION_SCALE).astype(np.float32) if gate_on else _pi0_action(pi0, rl.obs())
            o2, r, term, trunc, _ = rl.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            pa = a; steps += 1
            if term or trunc:
                break
    return ex


def build_feedback_dataset(pi0, starts, *, horizon=60):
    from collections import Counter
    X, Y, prov, cmc = [], [], [], []
    for i, ls in enumerate(starts):
        for e in feedback_chunk_rollout(pi0, ls, horizon=horizon, plan_seed=1000 + 50 * i):
            X.append(e["state"]); Y.append(e["label_chunk"]); prov.append(e["prov"]); cmc.append(e["cm"])
    stats = {"n_examples": len(X), "by_provenance": dict(Counter(prov)), "by_phase": dict(Counter(cmc)),
             "n_planner_improving": int(sum(p == "planner" for p in prov)),
             "n_pi0_fallback": int(sum(p == "pi0_fallback" for p in prov))}
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), prov, stats


def build_teacher_annotated_dataset(pi0, starts, *, horizon=60):
    """Same states/labels as :func:`build_feedback_dataset` but each example also carries BOTH teacher first-actions
    (``pi0_first``, ``planner_first``) and the label first-action + admissibility ``flags``. For the mixed-teacher
    audit only — the label and its provenance are identical to the training dataset (record_teachers is additive)."""
    rows = []
    for i, ls in enumerate(starts):
        rows.extend(feedback_chunk_rollout(pi0, ls, horizon=horizon, plan_seed=1000 + 50 * i, record_teachers=True))
    X = np.stack([e["state"] for e in rows]).astype(np.float32)
    pi0_first = np.stack([e["pi0_first"] for e in rows]).astype(np.float32)
    planner_first = np.stack([e["planner_first"] for e in rows]).astype(np.float32)
    label_first = np.stack([e["label_first"] for e in rows]).astype(np.float32)
    prov = [e["prov"] for e in rows]; flags = [e["flags"] for e in rows]
    return X, pi0_first, planner_first, label_first, prov, flags


def reproduce_v2_actor(pi0, train, *, horizon=60, dagger_rounds=2, base_steps=4000, dagger_steps=3000, log=print):
    """Deterministically rebuild the FEEDBACK_CHUNK_WARMSTART_V2 supervised chunk actor (same dataset + DAgger + prefix-
    weighted regression, seed 0). Canonical single source so the m1 diagnostic and the mixed-teacher audit share the
    exact same actor rather than re-implementing the reproduction (§6.1)."""
    X, Y, _prov, stats = build_feedback_dataset(pi0, train, horizon=horizon)
    actor = train_supervised_v2(X, Y, steps=base_steps, log=lambda *a: None)
    for rd in range(dagger_rounds):
        dX, dY, _dp = dagger_collect(pi0, actor, train, horizon=horizon, plan_seed=7000 + 1000 * rd)
        if dX is not None:
            X = np.concatenate([X, dX]); Y = np.concatenate([Y, dY])
            actor = train_supervised_v2(X, Y, steps=dagger_steps, actor=actor, log=lambda *a: None)
    log(f"  reproduced V2 actor; dataset {len(X)}")
    return actor, X, Y, stats


# ── prefix-weighted supervised regression + DAgger ──
def prefix_weighted_loss(pred, tgt):
    w = torch.tensor(np.repeat(PREFIX_WEIGHTS, ACT_DIM), dtype=torch.float32)       # (K*4,)
    return (w * (pred - tgt) ** 2).mean()


def train_supervised_v2(X, Y, *, steps=4000, seed=0, lr=1e-3, actor=None, log=print):
    torch.manual_seed(seed)
    actor = actor or ChunkActor()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X), torch.tensor(Y); n = len(X); rng = np.random.default_rng(seed)
    for i in range(steps + 1):
        idx = rng.integers(0, n, min(256, n))
        loss = prefix_weighted_loss(actor(Xt[idx]), Yt[idx])
        if i == steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 1000 == 0:
            log(f"    sup step {i}/{steps} pw_mse {loss.item():.5f}")
    return actor


def dagger_collect(pi0, actor, starts, *, horizon=60, plan_seed=7000, pop=20, iters=4):
    """Roll the LEARNED chunk actor (receding-horizon); at each visited gate-active replanning state query the planner
    offline and add the safe label (or pi_0 fallback)."""
    X, Y, prov = [], [], []
    for i, ls in enumerate(starts):
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        det = EventStateDetector(); pa = rec.base.astype(np.float32); steps = 0; term = trunc = False
        while steps < horizon and not (term or trunc):
            cm, cf, ev = det.state_of(rl)
            if gate.gate != 1.0:
                a = _pi0_action(pi0, rl.obs()); o2, r, term, trunc, _ = rl.step(a)
                lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term)); pa = a; steps += 1; continue
            snap = _snap(rl); st = state_vec(rl.obs(), cm, cf, ev, pa)
            planner_chunk = plan_chunk_target(rl, rl.cf, plan_seed + 50 * i + steps, k=K, pop=pop, iters=iters); _restore(rl, snap)
            pi0_chunk = pi0_chunk_target(rl, pi0, k=K); _restore(rl, snap)
            label, pv, _fl = label_chunk(rl, pi0, snap, planner_chunk, pi0_chunk); _restore(rl, snap)
            X.append(st); Y.append(label.reshape(-1).astype(np.float32)); prov.append(pv)
            _chunk, executed, recs, _o = execute_chunk(rl, pi0, actor, gate, st, m=M)   # advance on the LEARNED actor
            pa = executed[-1]; steps += len(recs)
            term = recs[-1]["terminated"] if recs else term; trunc = recs[-1]["truncated"] if recs else trunc
    return (np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), prov) if X else (None, None, [])
