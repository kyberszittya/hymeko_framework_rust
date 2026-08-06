"""CARRY_TO_SETTLING_HANDOFF — hierarchical upstream extension of the frozen coin controller.

The frozen pi_0 is competent from strict≥1 (settling, K6 ≈ 0.95). The missing module is a CARRY controller that takes
strict-0 contact_retention/transport states to a good settling handoff (the coin enters the zone contained and slow, i.e.
strict reaches ≥1 and stays), after which the FROZEN pi_0 finishes the delivery. This module proves the *achievability*
(candidate coverage) prerequisite BEFORE any carry actor/critic is built: does a support-bounded carry action-sequence
exist that produces a good handoff (→ K6 via the frozen settling continuation)?

The exact simulator is used here only as an EXPERT (bounded random-shooting / CEM search) to demonstrate existence — the
question is "is there a good carry-prefix", not "can a learned policy find it" (that is Phase 4). All candidates are
support-bounded (pi_0 + clipped offset), never arbitrary.
"""
import copy

import numpy as np

from hymeko_rl.coin_delivery.coin_markov_ablation_train import ACTION_SCALE, _aug, _det
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL  # strict≥1 already encodes speed<SETTLE_VEL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation


def sequence_then_pi0(rl, gate, pi0, base, offset_seq, *, horizon):
    """Apply ``clip(pi_0(s_t) + offset_seq[t])`` at the first ``len(offset_seq)`` gate-on steps (the carry prefix), then
    the FROZEN pi_0 (the settling controller) for the rest. Returns the end-to-end certifier plus the handoff signal.

    Preconditions:  ``rl``/``gate`` are a fresh deepcopy of a reconstructed handoff (mutated here).
    Postconditions: reports k6 (full delivery via frozen settling), max_dwell, max_strict, ``handoff_step`` (first step
                    the coin is held in-zone, strict≥1 ⟺ dtz≤CENTER_TOL ∧ speed<SETTLE_VEL), and full-containment exits.
    """
    seq = np.asarray(offset_seq, np.float32); L = len(seq); applied = 0
    md = int(rl._strict); touched = rl._touched; max_strict = int(rl._strict)
    dtz = rl._dtz(); was_contained = dtz <= CENTER_TOL; contain_exit = 0; handoff_step = None
    for t in range(horizon):
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        if gate_on and applied < L:
            a = np.clip(_det(base, _aug(o48, s)) + seq[applied], -ACTION_SCALE, ACTION_SCALE); applied += 1
        elif gate_on:
            a = _det(base, _aug(o48, s))
        else:
            a = _det(pi0, o48)
        _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        md = max(md, int(rl._strict)); touched = touched or rl._touched; max_strict = max(max_strict, int(rl._strict))
        dtz = rl._dtz()
        if handoff_step is None and int(rl._strict) >= 1:
            handoff_step = t
        if was_contained and dtz > CENTER_TOL:
            contain_exit += 1
        was_contained = dtz <= CENTER_TOL
        if term or trunc:
            break
    return {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "max_strict": max_strict,
            "handoff_step": handoff_step, "reached_handoff": int(max_strict >= 1), "contain_exit_ct": contain_exit}


def _score(o):
    """Expert objective (higher better): full delivery ≻ deeper dwell ≻ reached handoff ≻ earlier handoff ≻ fewer exits."""
    hs = o["handoff_step"] if o["handoff_step"] is not None else 10 ** 6
    return (o["k6"], o["max_dwell"], o["reached_handoff"], -hs, -o["contain_exit_ct"])


def carry_cem(rl0, gate0, pi0, base, adim, rng, *, shots, length, iters, init_std, mag_max, elite_frac, horizon):
    """Bounded CEM over support-limited carry action-sequences (each ``length`` offsets, clipped to ±``mag_max``). Refines
    toward sequences that produce a good handoff → K6 (frozen settling continuation). Returns the best outcome found — an
    EXISTENCE (coverage) estimate, not a learned policy."""
    mean = np.zeros((length, adim), np.float32); std = np.full((length, adim), init_std, np.float32)
    best = {"k6": 0, "max_dwell": int(rl0._strict), "max_strict": int(rl0._strict), "reached_handoff": 0,
            "handoff_step": None, "contain_exit_ct": 0}
    for _it in range(iters):
        seqs = np.clip(rng.normal(mean, std, size=(shots, length, adim)), -mag_max, mag_max).astype(np.float32)
        outs = [sequence_then_pi0(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, seqs[m], horizon=horizon) for m in range(shots)]
        order = sorted(range(shots), key=lambda m: _score(outs[m]), reverse=True)
        n_elite = max(2, int(elite_frac * shots)); elites = seqs[order[:n_elite]]
        mean = elites.mean(0); std = elites.std(0) + 1e-3
        if _score(outs[order[0]]) > _score(best):
            best = outs[order[0]]
    return best


def carry_random(rl0, gate0, pi0, base, adim, rng, *, shots, length, mag_max, horizon):
    """RANDOM control: the best of ``shots`` uniformly-random support-bounded carry sequences (no refinement) — isolates
    whether reaching a handoff needs *search* or just *any* bounded perturbation."""
    best = {"k6": 0, "max_dwell": int(rl0._strict), "max_strict": int(rl0._strict), "reached_handoff": 0,
            "handoff_step": None, "contain_exit_ct": 0}
    for _ in range(shots):
        seq = rng.uniform(-mag_max, mag_max, size=(length, adim)).astype(np.float32)
        o = sequence_then_pi0(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, seq, horizon=horizon)
        if _score(o) > _score(best):
            best = o
    return best
