"""Regression tests for the competence-gated replay selection (experiments.coin_contact_replay.gate_step) between the
committed stratified and uniform samplers. The gate is driven ONLY by the run's own competence state (no CONTROL leak),
fires after GATE_CONFIRM consecutive certified evals, and is irreversible within the run."""
from __future__ import annotations

import inspect
import json

import numpy as np

from hymeko_rl.experiments.coin_contact_replay import GATE_CONFIRM, STRATA_WEIGHTS, gate_step, new_gate
from hymeko_rl.train.replay import ReplayBuffer


def _comp(consec):
    return {"progress_ok": True, "first_strict": consec >= 1, "consec_strict": consec}


def _seeded(strata: dict[int, int], online: int, *, dim=4, adim=2) -> ReplayBuffer:
    buf = ReplayBuffer(100_000, (dim,), adim)
    tags = np.concatenate([np.full(n, t, np.int16) for t, n in strata.items()])
    m = len(tags)
    z = np.zeros((m, dim), np.float32)
    buf.add_batch(z, np.zeros((m, adim), np.float32), np.zeros(m, np.float32), z, np.zeros(m, bool), tags=tags)
    for _ in range(online):
        buf.add(np.zeros(dim, np.float32), np.zeros(adim, np.float32), 0.0, np.zeros(dim, np.float32), False)
    return buf


def test_1_weak_competence_selects_stratified() -> None:
    g = new_gate()
    assert gate_step(g, _comp(0), eval_idx=1, step=2500, bc_coef=1.0) == "stratified"
    assert gate_step(g, _comp(1), eval_idx=2, step=5000, bc_coef=0.1) == "stratified"  # first_strict but not confirmed


def test_2_established_competence_selects_uniform() -> None:
    g = new_gate()
    gate_step(g, _comp(1), eval_idx=1, step=2500, bc_coef=0.1)
    assert gate_step(g, _comp(2), eval_idx=2, step=5000, bc_coef=0.1) == "uniform"


def test_3_gate_has_no_control_information() -> None:
    params = set(inspect.signature(gate_step).parameters)
    assert not (params & {"control", "uniform", "matched", "control_strict", "baseline"})   # own-run signal only
    assert params == {"gate", "comp", "eval_idx", "step", "bc_coef"}


def test_4_switch_after_two_consecutive_confirmations() -> None:
    assert GATE_CONFIRM == 2
    g = new_gate()
    assert gate_step(g, _comp(1), eval_idx=1, step=2500, bc_coef=0.1) == "stratified"       # 1 confirmation: no switch
    assert gate_step(g, _comp(2), eval_idx=2, step=5000, bc_coef=0.1) == "uniform"          # 2nd confirmation: switch


def test_5_switch_is_irreversible() -> None:
    g = new_gate()
    gate_step(g, _comp(2), eval_idx=1, step=2500, bc_coef=0.1)                               # switch
    assert gate_step(g, _comp(0), eval_idx=2, step=5000, bc_coef=1.0) == "uniform"           # competence lost -> stays uniform
    assert gate_step(g, _comp(5), eval_idx=3, step=7500, bc_coef=0.05) == "uniform"


def test_6_deterministic_mode_history() -> None:
    seq = [0, 0, 1, 2, 1, 0]
    hists = []
    for _ in range(2):
        g = new_gate()
        for i, c in enumerate(seq):
            gate_step(g, _comp(c), eval_idx=i, step=i * 2500, bc_coef=0.1)
        hists.append([h["mode"] for h in g["history"]])
    assert hists[0] == hists[1]
    assert hists[0] == ["stratified", "stratified", "stratified", "uniform", "uniform", "uniform"]


def test_7_preswitch_reproduces_committed_stratified() -> None:
    buf = _seeded({1: 300, 2: 300, 3: 300, 4: 300, 5: 300}, 1500)
    g = new_gate()
    use_strat = (gate_step(g, _comp(0), eval_idx=1, step=2500, bc_coef=1.0) == "stratified")
    assert use_strat
    # the batch a gated (pre-switch) step draws equals the committed stratified sampler for the same generator state
    b1 = buf.sample_stratified(64, demo_frac=0.5, strata_weights=STRATA_WEIGHTS, generator=np.random.default_rng(4))
    b2 = buf.sample_stratified(64, demo_frac=0.5, strata_weights=STRATA_WEIGHTS, generator=np.random.default_rng(4))
    assert np.array_equal(b1[1].numpy(), b2[1].numpy())


def test_8_postswitch_reproduces_committed_uniform() -> None:
    buf = _seeded({1: 300, 5: 300}, 1500)
    g = new_gate()
    gate_step(g, _comp(2), eval_idx=1, step=2500, bc_coef=0.1)                               # switch to uniform
    use_strat = (g["mode"] == "stratified")
    assert not use_strat
    b1 = buf.sample(64, generator=np.random.default_rng(6))                                  # committed uniform path
    b2 = buf.sample(64, generator=np.random.default_rng(6))
    assert np.array_equal(b1[0].numpy(), b2[0].numpy())


def test_9_empty_optional_strata_valid_before_switch() -> None:
    buf = _seeded({1: 200, 5: 200}, 800)                                                     # strata 2,3,4 empty
    sh = {}
    b = buf.sample_stratified(96, demo_frac=0.5, strata_weights=STRATA_WEIGHTS, generator=np.random.default_rng(1), shortage=sh)
    assert b[0].shape[0] == 96 and any(sh.get(t, {}).get("empty", 0) for t in (2, 3, 4))


def test_10_gate_state_serializes_and_restores() -> None:
    g = new_gate()
    for i, c in enumerate([0, 1, 2, 0]):
        gate_step(g, _comp(c), eval_idx=i, step=i * 2500, bc_coef=0.1)
    restored = json.loads(json.dumps(g))                                                     # checkpoint round-trip
    assert restored["mode"] == g["mode"] == "uniform"
    assert restored["switch_step"] == g["switch_step"]
    assert [h["mode"] for h in restored["history"]] == [h["mode"] for h in g["history"]]
