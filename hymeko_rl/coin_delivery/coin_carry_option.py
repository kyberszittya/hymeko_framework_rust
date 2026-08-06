"""CARRY_OPTION_ACTOR — the deployable carry controller is an OPTION policy, not a myopic per-step actor. The teacher
diagnostic proved the carry task is commitment-dominated (open-loop macro K6 0.60 vs receding first-action 0.20), so the
learned unit is a temporally-extended option:

    strict-0 carry state → option actor: θ = push/brake/release params → stateful macro-controller (physical phase
    transitions, durations as upper bounds, safety abort) → robust handoff (strict≥1) → FROZEN settling pi_0 → K6.

The option label is a single CONSISTENT θ* (strong structured search + canonical tie-break) — not conflicting per-step 4D
actions. New decisions happen only at option completion / handoff / safety-abort (semi-MDP), never per timestep.
"""
import copy

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_carry_structured import (
    A_BOUND,
    DIM,
    PUSH_DTZ,
    T_MAX,
    T_MIN,
    _unpack,
    structured_carry_rollout,
    structured_random_best,
    structured_score,
)
from hymeko_rl.coin_delivery.coin_markov_ablation_train import ACTION_SCALE, _aug, _det
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation


class OptionActor(nn.Module):
    """obs48 → θ (12 amplitudes in ±A_BOUND via tanh, 3 durations in [T_MIN, T_MAX] via sigmoid). A committed macro, not a
    per-step action."""

    def __init__(self, obs_dim=48, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, DIM))

    def theta(self, o):
        raw = self.net(o)
        amp = torch.tanh(raw[..., :12]) * A_BOUND
        dur = torch.sigmoid(raw[..., 12:]) * (T_MAX - T_MIN) + T_MIN
        return torch.cat([amp, dur], -1)


def make_option_actor():
    return OptionActor()


def actor_theta(actor, o48):
    with torch.no_grad():
        return actor.theta(torch.as_tensor(o48[None]).float())[0].numpy().astype(np.float32)


def teacher_theta(rl0, gate0, pi0, base, rng, *, shots, horizon):
    """Strong structured search → the option label θ*. structured_random_best already ranks by the canonical lexicographic
    certificate (K6 ≻ dwell ≻ fewer exits ≻ contact ≻ less effort ≻ faster). Returns (θ*, admissible, outcome)."""
    theta, out = structured_random_best(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, rng, shots=shots, horizon=horizon)
    return theta.astype(np.float32), bool(out["reached_handoff"] or out["k6"]), out


def _safety_abort(rl, touched_before):
    """Commitment must not be suicidal: abort the option if contact is irreversibly lost after having had it, or the
    physics/support guard trips (NaN / gross divergence)."""
    m = rl.inner._planar_metrics
    lost_contact = touched_before and not (m.left_contact or m.right_contact) and rl._dtz() > 3 * CENTER_TOL
    diverged = not np.isfinite(rl._dtz()) or rl._dtz() > 1.0 or rl._speed() > 5.0
    return bool(lost_contact or diverged)


def option_controller_rollout(rl, gate, actor_or_theta, pi0, base, *, horizon, max_options=3):
    """Semi-MDP carry controller: pick θ (from the option actor, or a fixed θ), execute the committed macro with physical
    phase transitions until handoff / option-completion / safety-abort; re-decide only at option boundaries; frozen pi_0
    after a valid handoff. ``actor_or_theta`` is an OptionActor (re-decides per option) or a fixed θ array (single option)."""
    fixed = not isinstance(actor_or_theta, OptionActor)
    md = int(rl._strict); touched = rl._touched; max_strict = int(rl._strict)
    dtz = rl._dtz(); was_contained = dtz <= CENTER_TOL; contain_exit = 0; handed = False
    t_total = 0; opts = 0; aborts = 0
    while t_total < horizon and not handed:
        s = int(rl._strict)
        if s >= 1:
            handed = True; break
        if opts >= max_options and not fixed:
            break
        theta = np.asarray(actor_or_theta, np.float32) if fixed else actor_theta(actor_or_theta, rl.obs())
        a_push, T_push, a_brake, T_brake, a_release, T_release = _unpack(theta)
        phase, tph = "push", 0
        while t_total < horizon:
            gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
            if s >= 1:
                handed = True; break
            if not gate_on:
                a = _det(pi0, o48)
            elif phase == "push":
                a = a_push; tph += 1
                if tph >= T_push or rl._dtz() <= PUSH_DTZ:
                    phase, tph = "brake", 0
            elif phase == "brake":
                a = a_brake; tph += 1
                if tph >= T_brake or rl._dtz() <= CENTER_TOL or rl._speed() < 1.5 * SETTLE_VEL:
                    phase, tph = "release", 0
            else:
                a = a_release; tph += 1
            tb = touched
            _r, term, trunc = step_ablation(rl, np.clip(np.asarray(a, np.float32), -ACTION_SCALE, ACTION_SCALE), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict)); touched = touched or rl._touched; max_strict = max(max_strict, int(rl._strict))
            dtz = rl._dtz(); t_total += 1
            if was_contained and dtz > CENTER_TOL:
                contain_exit += 1
            was_contained = dtz <= CENTER_TOL
            if term or trunc:
                handed = handed or int(rl._strict) >= 1; t_total = horizon; break
            if _safety_abort(rl, tb):
                aborts += 1; break                                       # SAFETY abort → re-decide (or stop if fixed)
            if phase == "release" and tph >= T_release:
                break                                                    # option completed without handoff → re-decide
        opts += 1
        if fixed:
            break
    # frozen settling pi_0 to the horizon (delivers K6 from a valid handoff)
    while t_total < horizon:
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        a = _det(base, _aug(o48, s)) if (gate_on and s >= 1) else _det(pi0, o48)
        _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        md = max(md, int(rl._strict)); touched = touched or rl._touched; dtz = rl._dtz()
        if was_contained and dtz > CENTER_TOL:
            contain_exit += 1
        was_contained = dtz <= CENTER_TOL
        t_total += 1
        if term or trunc:
            break
    return {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "max_strict": max_strict,
            "reached_handoff": int(max_strict >= 1), "contain_exit_ct": contain_exit, "options": opts, "aborts": aborts}


def train_option_bc(actor, obs, theta, *, epochs, lr, batch, seed):
    """BC on the option label θ (normalised: amplitudes /A_BOUND, durations to [0,1]) — a single consistent target."""
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(obs, np.float32)); y = torch.as_tensor(np.asarray(theta, np.float32))
    yn = torch.cat([y[:, :12] / A_BOUND, (y[:, 12:] - T_MIN) / (T_MAX - T_MIN)], -1)
    opt = torch.optim.Adam(actor.parameters(), lr=lr); loss = torch.tensor(0.0)
    for _ep in range(epochs):
        idx = torch.randperm(len(x))
        for j in range(0, len(x), batch):
            b = idx[j:j + batch]; th = actor.theta(x[b])
            pn = torch.cat([th[:, :12] / A_BOUND, (th[:, 12:] - T_MIN) / (T_MAX - T_MIN)], -1)
            loss = ((pn - yn[b]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


def recovery_state_theta(rl0, gate0, actor, pi0, base, rng, *, shots, horizon, max_probe):
    """Roll the student option controller; if it fails to hand off within ``max_probe`` steps, return the recovery state's
    (obs48, teacher θ*) for DAgger — the option-initiation state the STUDENT actually reaches. None if it handed off."""
    rl, gate = copy.deepcopy(rl0), copy.deepcopy(gate0)
    theta = actor_theta(actor, rl.obs())
    a_push, T_push, a_brake, T_brake, a_release, T_release = _unpack(theta)
    phase, tph = "push", 0
    for _t in range(max_probe):
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        if s >= 1:
            return None                                                  # student handed off — no recovery label needed
        if not gate_on:
            a = _det(pi0, o48)
        elif phase == "push":
            a = a_push; tph += 1
            if tph >= T_push or rl._dtz() <= PUSH_DTZ:
                phase, tph = "brake", 0
        elif phase == "brake":
            a = a_brake; tph += 1
            if tph >= T_brake or rl._dtz() <= CENTER_TOL or rl._speed() < 1.5 * SETTLE_VEL:
                phase, tph = "release", 0
        else:
            a = a_release; tph += 1
        _r, term, trunc = step_ablation(rl, np.clip(np.asarray(a, np.float32), -ACTION_SCALE, ACTION_SCALE), "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        if term or trunc:
            break
    if int(rl._strict) >= 1:
        return None
    o48 = rl.obs()
    th, adm, _o = teacher_theta(rl, gate, pi0, base, rng, shots=shots, horizon=horizon)
    return (o48.copy(), th) if adm else None


def option_score(o):
    return structured_score(o)


def _jitter_theta(theta, rng, amp_std=0.2):
    t = np.asarray(theta, np.float32).copy()
    t[:12] = np.clip(t[:12] + rng.normal(0, amp_std, 12).astype(np.float32), -A_BOUND, A_BOUND)
    return t


def option_teacher_label(rl0, gate0, pi0, base, rng, *, shots, horizon, robust_checks=3):
    """Confident option label for one option-initiation state: strong θ* (K6-primary lexicographic canonical), CONFIDENT
    only when the option actually delivers K6 (never a merely-least-bad candidate). Robustness = fraction of small-jitter
    re-rolls of θ* that still reach K6. Returns (θ*, confident, provenance)."""
    theta, _adm, out = teacher_theta(rl0, gate0, pi0, base, rng, shots=shots, horizon=horizon)
    confident = int(out["k6"]) == 1
    reason = "K6" if out["k6"] else ("HANDOFF_ONLY" if out["reached_handoff"] else "NO_HANDOFF")
    robust_k6 = None
    if confident and robust_checks > 0:
        rk = [structured_carry_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, _jitter_theta(theta, rng), horizon=horizon)["k6"]
              for _ in range(robust_checks)]
        robust_k6 = round(float(np.mean(rk)), 3)
    prov = {"k6": int(out["k6"]), "handoff": int(out["reached_handoff"]), "any_exit": int(out["contain_exit_ct"] > 0),
            "effort": out["effort"], "completion": out["completion"], "handoff_step": out["handoff_step"],
            "termination_reason": reason, "robust_k6": robust_k6, "shots": shots, "confident": int(confident)}
    return theta.astype(np.float32), bool(confident), prov
