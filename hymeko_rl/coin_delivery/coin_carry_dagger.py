"""CARRY_DAGGER_ACTOR — Phase 4b: distil the validated structured carry expert into a deployable LOW-LEVEL 4D carry actor.

Teacher = structured-random receding-horizon: at each strict-0 step it searches macro-plans (push→brake→release), executes
ONLY the first 4D action, and replans from the next state. The full macro-plan is PLAN_ONLY; only the executed first action
is FEEDBACK_ADMISSIBLE (the label). The actor is low-level (obs48 → 4D action, same causal Markov contract), acts only at
strict-0 under the carry gate, and HANDS OFF to the FROZEN settling pi_0 at strict≥1. The teacher ABSTAINS on states it
cannot solve (no macro-plan reaches a handoff) — no fabricated labels.
"""
import copy

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_carry_structured import (
    T_MAX,
    T_MIN,
    first_action_of_theta,
    structured_carry_rollout,
    structured_random_around,
    structured_random_best,
)
from hymeko_rl.coin_delivery.coin_markov_ablation_train import ACTION_SCALE, _aug, _det
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.rl_clip_actor import ClipDeterministicActor, make_backbone

OBS48 = 48


def make_carry_actor(pi0):
    """A fresh low-level continuous carry actor: obs48 → 4D clipped action (NOT a perturbation of pi_0; NOT macro-params)."""
    feat = pi0.head.in_features
    return ClipDeterministicActor(make_backbone(OBS48, feat), feat, pi0.action_dim, pi0.action_scale)


def teacher_first_action(rl, gate, pi0, base, rng, *, shots, horizon):
    """Receding-horizon teacher query at the CURRENT state: best structured macro-plan's first action (push amplitude).
    Returns (action, admissible, outcome). admissible ⇔ some plan reaches a good handoff/K6 (else the teacher ABSTAINS)."""
    theta, out = structured_random_best(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, rng, shots=shots, horizon=horizon)
    admissible = out["reached_handoff"] == 1 or out["k6"] == 1
    return np.clip(theta[0:4], -ACTION_SCALE, ACTION_SCALE).astype(np.float32), bool(admissible), out


def _actor_action(actor, o48):
    with torch.no_grad():
        return torch.clamp(actor.action_mean(torch.as_tensor(o48[None]).float()), -ACTION_SCALE, ACTION_SCALE)[0].numpy()


def carry_actor_rollout(rl, gate, actor, pi0, base, *, horizon, collect=False):
    """The learned carry actor drives strict-0 (gate on); FROZEN pi_0 takes over at strict≥1. Returns the certifier
    outcome; with ``collect`` also the strict-0 obs48 the student actually visited (for DAgger relabeling)."""
    md = int(rl._strict); touched = rl._touched; max_strict = int(rl._strict)
    dtz = rl._dtz(); was_contained = dtz <= CENTER_TOL; contain_exit = 0; handed = False; visited = []
    for _t in range(horizon):
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        if not gate_on:
            a = _det(pi0, o48)
        elif handed or s >= 1:
            handed = True; a = _det(base, _aug(o48, s))
        else:
            a = _actor_action(actor, o48)
            if collect:
                visited.append((copy.deepcopy(rl), copy.deepcopy(gate), o48.copy()))   # snapshot for teacher relabeling
        _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        md = max(md, int(rl._strict)); touched = touched or rl._touched; max_strict = max(max_strict, int(rl._strict))
        dtz = rl._dtz()
        if int(rl._strict) >= 1:
            handed = True
        if was_contained and dtz > CENTER_TOL:
            contain_exit += 1
        was_contained = dtz <= CENTER_TOL
        if term or trunc:
            break
    out = {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "max_strict": max_strict,
           "reached_handoff": int(max_strict >= 1), "contain_exit_ct": contain_exit}
    return (out, visited) if collect else out


def teacher_receding_bank(rl0, gate0, pi0, base, rng, *, shots, teacher_h, roll_h):
    """Receding-horizon teacher trajectory from a carry state → (obs48 list, action list, stats). ABSTAINs (stops labeling)
    the moment the teacher can no longer reach a handoff; empty bank if it abstains at the start."""
    rl, gate = copy.deepcopy(rl0), copy.deepcopy(gate0)
    obs, act = [], []; md = int(rl._strict); touched = rl._touched; handed = False; abstained = False
    for _t in range(roll_h):
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        if not gate_on:
            a = _det(pi0, o48)
        elif handed or s >= 1:
            handed = True; a = _det(base, _aug(o48, s))
        else:
            a, adm, _o = teacher_first_action(rl, gate, pi0, base, rng, shots=shots, horizon=teacher_h)
            if not adm:
                abstained = True; break
            obs.append(o48.copy()); act.append(a.copy())
        _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        md = max(md, int(rl._strict)); touched = touched or rl._touched
        if int(rl._strict) >= 1:
            handed = True
        if term or trunc:
            break
    return obs, act, {"k6": int(md >= HELD_DWELL and touched), "handoff": int(handed), "n_labels": len(obs), "abstained": int(abstained)}


def teacher_openloop_plan(rl0, gate0, pi0, base, rng, *, shots, horizon):
    """PLAN_ONLY: the strong initial macro-plan + its full open-loop trajectory. The trajectory is a plan/provenance/warm-
    start artifact — its suffix actions are optimised for s0 and are NOT feedback-admissible, so they must NOT be fed
    directly as BC action labels (that reintroduces the quarantined open-loop-as-feedback error)."""
    theta, out = structured_random_best(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, rng, shots=shots, horizon=horizon)
    admissible = out["reached_handoff"] or out["k6"]
    roll = structured_carry_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, theta, horizon=horizon, capture=True) if admissible else {"obs": [], "act": []}
    return theta, {"plan_obs": roll["obs"], "plan_act": roll["act"], "k6": int(out["k6"]), "handoff": int(out["reached_handoff"]),
                   "admissible": int(bool(admissible)), "provenance": "OPEN_LOOP_PLAN_ONLY"}


def teacher_warmstart_bank(rl0, gate0, pi0, base, rng, *, strong_shots, warm_shots, teacher_h, roll_h,
                           wide_amp=1.5, wide_dur=6.0, warm_amp=0.4, warm_dur=2.0):
    """Warm-started RECEDING-horizon teacher: a strong initial solve from s0, then cheap warm-started replans (centred on
    the previous plan) at each strict-0 step. Every label is the first action REPLANNED from the CURRENT state (feedback-
    admissible); the plan is executed one step then re-solved. ABSTAINs (stops) when no warm plan reaches a handoff."""
    rl, gate = copy.deepcopy(rl0), copy.deepcopy(gate0)
    obs, act = [], []; theta = None; handed = False; md = int(rl._strict); touched = rl._touched
    center0 = np.concatenate([np.zeros(12, np.float32), np.full(3, (T_MIN + T_MAX) / 2.0, np.float32)]).astype(np.float32)
    for _t in range(roll_h):
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        if not gate_on:
            a = _det(pi0, o48)
        elif handed or s >= 1:
            handed = True; a = _det(base, _aug(o48, s))
        else:
            if theta is None:
                theta, out = structured_random_around(rl, gate, pi0, base, rng, shots=strong_shots, center=center0, std_amp=wide_amp, std_dur=wide_dur, horizon=teacher_h)
            else:
                theta, out = structured_random_around(rl, gate, pi0, base, rng, shots=warm_shots, center=theta, std_amp=warm_amp, std_dur=warm_dur, horizon=teacher_h)
            if not (out["reached_handoff"] or out["k6"]):
                break                                                     # ABSTAIN: no warm plan reaches a handoff here
            a = first_action_of_theta(theta, rl)                          # REPLANNED first action from the current state
            obs.append(o48.copy()); act.append(a)
        _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        md = max(md, int(rl._strict)); touched = touched or rl._touched
        if int(rl._strict) >= 1:
            handed = True
        if term or trunc:
            break
    return obs, act, {"k6": int(md >= HELD_DWELL and touched), "handoff": int(handed), "n_labels": len(obs), "abstained": int(len(obs) == 0)}


def train_bc(actor, obs, act, *, epochs, lr, batch, seed):
    """Behaviour-clone the low-level 4D action (MSE). Returns the final loss."""
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(obs, np.float32)); y = torch.as_tensor(np.asarray(act, np.float32))
    opt = torch.optim.Adam(actor.parameters(), lr=lr); loss = torch.tensor(0.0)
    for _ep in range(epochs):
        idx = torch.randperm(len(x))
        for j in range(0, len(x), batch):
            b = idx[j:j + batch]
            loss = ((actor.action_mean(x[b]) - y[b]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())
