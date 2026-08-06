"""§1 gate — the executed carry-option control and the `.hymeko` automaton are the SAME object.

On fixed rollouts (fixed θ, deterministic reconstructed states) the automaton-driven executor (transitions sourced from
``data/robotics/coin_carry_option_v1.hymeko``) must produce a BIT-IDENTICAL outcome + phase trace to the legacy hard-coded
topology. If it does, the `.hymeko` is not a documentation shadow — it drives the running control; the hard-coded map is a
gated fallback."""
import copy
import json

import numpy as np

from hymeko_rl.coin_delivery.coin_carry_fsm import MACRO_PHASES, TERMINAL_MARKS, load_carry_automaton
from hymeko_rl.coin_delivery.coin_carry_option_rl import execute_one_option
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
THETAS = [
    np.array([2, -2, 1, -1, 0, 0, 1, -1, 0.5, -0.5, 0, 0, 6, 6, 6], np.float32),
    np.array([3, 3, -3, -3, -2, -2, 2, 2, 0, 0, 0, 0, 8, 4, 10], np.float32),
    np.array([-1, 1, 2, -2, 1, 1, -1, -1, 1, -1, 0.5, -0.5, 4, 12, 5], np.float32),
]


def _states(n=4):
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    out = []
    for r in cfg["banks"]["late_dev"]["rows"][:n]:
        ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
        rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        out.append((rl, gate))
    return pi0, base, out


def test_carry_automaton_parses_and_is_wellformed():
    spec = load_carry_automaton()
    assert spec.initial == "PUSH"
    names = {p.name for p in spec.phases}
    assert set(MACRO_PHASES) <= names and set(TERMINAL_MARKS) <= names
    # every macro phase routes handoff→HANDOFF and abort→ABORTED (checked from ANY macro phase)
    for ph in MACRO_PHASES:
        arcs = dict(spec.phase(ph).transitions)
        assert arcs.get("handoff") == "HANDOFF" and arcs.get("abort") == "ABORTED"
    assert dict(spec.phase("PUSH").transitions)["push_reached"] == "BRAKE"
    assert dict(spec.phase("BRAKE").transitions)["brake_centered"] == "RELEASE"
    assert dict(spec.phase("RELEASE").transitions)["release_done"] == "COMPLETED"


def test_hymeko_driven_equals_hardcoded_on_fixed_rollouts():
    pi0, base, states = _states()
    spec = load_carry_automaton()
    seen_terminals = set()
    for (rl, gate) in states:
        for theta in THETAS:
            tr_hc, tr_hy = [], []
            o_hc = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=140, spec=None, trace=tr_hc)
            o_hy = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=140, spec=spec, trace=tr_hy)
            for k in ("R_option", "tau", "done", "k6", "reached_handoff", "contain_exit_ct"):
                assert o_hc[k] == o_hy[k], f"{k}: hardcoded {o_hc[k]} != hymeko {o_hy[k]}"
            assert tr_hc == tr_hy                                          # identical phase trace
            assert tr_hc[0] == "PUSH" and tr_hc[-1] in TERMINAL_MARKS      # starts in PUSH, ends at a terminal marker
            assert all(p in set(MACRO_PHASES) | set(TERMINAL_MARKS) for p in tr_hc)
            seen_terminals.add(tr_hc[-1])
    # the fixed θ set exercises more than one terminal outcome (not a degenerate single path)
    assert len(seen_terminals) >= 2, f"only reached {seen_terminals}"


def test_default_auto_uses_hymeko_and_matches():
    pi0, base, states = _states(2)
    rl, gate = states[0]
    theta = THETAS[0]
    o_auto = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=120)      # spec="auto"
    o_none = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=120, spec=None)
    for k in ("R_option", "tau", "done", "k6", "reached_handoff", "contain_exit_ct"):
        assert o_auto[k] == o_none[k]                                     # the runtime default (.hymeko) == hard-coded fallback
