"""CARRY_STRUCTURED_EXPERT — a carry-specific action LANGUAGE (push → brake → release), not a perturbation of the
settling-tuned pi_0. The support-frontier proved that pi_0 + support-bounded offset plateaus at ~40% for contact_retention
and does not scale with support / search / budget — so the CEM was searching in the wrong coordinate system. Here the CEM
optimises a LOW-DIMENSIONAL, physically-meaningful macro-action (≈15 params) with closed-loop phase transitions; the frozen
pi_0 takes over only after a valid handoff (strict≥1).

θ = {a_push∈R⁴, T_push, a_brake∈R⁴, T_brake, a_release∈R⁴, T_release}  (12 amplitudes + 3 durations)

Phase transitions are state-triggered (feedback), not purely time-based:
  PUSH → BRAKE:    t ≥ T_push  OR  dtz ≤ PUSH_DTZ (coin near the entry)
  BRAKE → RELEASE: t ≥ T_brake OR  dtz ≤ CENTER_TOL  OR  speed < 1.5·SETTLE_VEL
  RELEASE → pi_0:  strict ≥ 1  OR  (t ≥ T_release and then wait for handoff)
"""
import copy

import numpy as np

from hymeko_rl.coin_delivery.coin_markov_ablation_train import ACTION_SCALE, _aug, _det
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation

A_BOUND, T_MIN, T_MAX = 3.0, 2, 18                        # amplitude bound + duration bounds for the macro-action
PUSH_DTZ = 0.06                                           # ENTRY_TOL — push until the coin is near the zone
DIM = 15                                                  # 3 phases × (4 amplitude + 1 duration) = 12 + 3


def _unpack(theta):
    t = np.asarray(theta, np.float32)
    dur = lambda x: int(np.clip(round(float(x)), T_MIN, T_MAX))
    return (np.clip(t[0:4], -A_BOUND, A_BOUND), dur(t[12]),
            np.clip(t[4:8], -A_BOUND, A_BOUND), dur(t[13]),
            np.clip(t[8:12], -A_BOUND, A_BOUND), dur(t[14]))


def structured_carry_rollout(rl, gate, pi0, base, theta, *, horizon):
    """Run the push→brake→release macro-action (closed-loop phases) until a valid handoff (strict≥1), then the FROZEN
    pi_0. Returns the certifier + handoff + lexicographic-score ingredients (contact kept, action effort, completion)."""
    a_push, T_push, a_brake, T_brake, a_release, T_release = _unpack(theta)
    phase, tph, handed = "push", 0, False
    md = int(rl._strict); touched = rl._touched; max_strict = int(rl._strict)
    dtz = rl._dtz(); was_contained = dtz <= CENTER_TOL; contain_exit = 0; handoff_step = None
    effort = 0.0; k6_step = None; nstep = 0
    for t in range(horizon):
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        if not gate_on:
            a = _det(pi0, o48)
        elif handed or s >= 1:
            handed = True; a = _det(base, _aug(o48, s))                # frozen settling pi_0 after valid handoff
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
        a = np.clip(np.asarray(a, np.float32), -ACTION_SCALE, ACTION_SCALE)
        effort += float(np.abs(a).sum())
        _r, term, trunc = step_ablation(rl, a, "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        md = max(md, int(rl._strict)); touched = touched or rl._touched; max_strict = max(max_strict, int(rl._strict)); nstep += 1
        dtz = rl._dtz()
        if handoff_step is None and int(rl._strict) >= 1:
            handoff_step = t; handed = True
        if k6_step is None and md >= HELD_DWELL and touched:
            k6_step = t
        if was_contained and dtz > CENTER_TOL:
            contain_exit += 1
        was_contained = dtz <= CENTER_TOL
        if term or trunc:
            break
    return {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "max_strict": max_strict,
            "reached_handoff": int(max_strict >= 1), "handoff_step": handoff_step, "contain_exit_ct": contain_exit,
            "touched": int(touched), "effort": round(effort, 2), "completion": k6_step if k6_step is not None else horizon}


def structured_score(o):
    """Lexicographic expert objective (higher better) — handoff alone is NOT enough (the frontier showed handoff↑ K6 flat):
    K6 ≻ reached-handoff ≻ dwell ≻ fewer full-containment exits ≻ contact kept ≻ less action effort ≻ faster completion."""
    return (o["k6"], o["reached_handoff"], o["max_dwell"], -o["contain_exit_ct"], o["touched"], -o["effort"], -o["completion"])


def _init_theta():
    mean = np.zeros(DIM, np.float32); mean[12:15] = (T_MIN + T_MAX) / 2.0
    std = np.concatenate([np.full(12, 1.0, np.float32), np.full(3, 5.0, np.float32)])
    return mean, std


def structured_cem(rl0, gate0, pi0, base, rng, *, shots, iters, elite_frac, horizon):
    """CEM over the ≈15-param macro-action. Returns the best outcome found (existence / coverage estimate)."""
    mean, std = _init_theta()
    best = {"k6": 0, "max_dwell": int(rl0._strict), "max_strict": int(rl0._strict), "reached_handoff": 0,
            "handoff_step": None, "contain_exit_ct": 0, "touched": int(rl0._touched), "effort": 0.0, "completion": horizon}
    for _it in range(iters):
        thetas = rng.normal(mean, std, size=(shots, DIM)).astype(np.float32)
        outs = [structured_carry_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, thetas[m], horizon=horizon) for m in range(shots)]
        order = sorted(range(shots), key=lambda m: structured_score(outs[m]), reverse=True)
        n_elite = max(2, int(elite_frac * shots)); elites = thetas[order[:n_elite]]
        mean = elites.mean(0); std = elites.std(0) + 1e-2
        if structured_score(outs[order[0]]) > structured_score(best):
            best = outs[order[0]]
    return best


def structured_random(rl0, gate0, pi0, base, rng, *, shots, horizon):
    """Budget-matched random control over the SAME structured parametrization (uniform amplitudes + durations)."""
    best = {"k6": 0, "max_dwell": int(rl0._strict), "max_strict": int(rl0._strict), "reached_handoff": 0,
            "handoff_step": None, "contain_exit_ct": 0, "touched": int(rl0._touched), "effort": 0.0, "completion": horizon}
    for _ in range(shots):
        theta = np.concatenate([rng.uniform(-A_BOUND, A_BOUND, 12), rng.uniform(T_MIN, T_MAX, 3)]).astype(np.float32)
        o = structured_carry_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, theta, horizon=horizon)
        if structured_score(o) > structured_score(best):
            best = o
    return best
