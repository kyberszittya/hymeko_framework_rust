"""Stage 0 tests — spec_bench deterministic core (synthetic labels, HTL-F1, the propose→gate loop)."""
from __future__ import annotations

from hymeko_rl.eval.spec_bench.spec_bench import (
    ScriptedModel,
    _extract_formula,
    calibrate_thresholds,
    evaluate_formula,
    f1_score,
    formula_f1,
    propose_and_gate,
    score_raw,
    synth_rollouts,
)


def test_synth_is_balanced_and_deterministic() -> None:
    a = synth_rollouts(40, seed=0)
    labels = [r.success for r in a]
    assert sum(labels) == 20 and len(labels) - sum(labels) == 20        # balanced
    b = synth_rollouts(40, seed=0)
    assert [r.success for r in b] == labels                             # deterministic
    assert a[0].trace == b[0].trace


def test_formal_beats_distractor_beats_malformed() -> None:
    rolls = synth_rollouts(60, seed=2)
    formal = formula_f1("F(in_place >= 0.9)", rolls)
    distractor = formula_f1("F(near_object >= 0.9)", rolls)
    malformed = formula_f1("eventually(in_place>=0.9)", rolls)          # wrong syntax → unusable
    assert formal > distractor > malformed
    assert malformed == 0.0 and formal > 0.85


def test_f1_score_hand_computed() -> None:
    # pred/labels: tp=2, fp=1, fn=1 → prec=2/3, rec=2/3, f1=2/3
    s = f1_score([True, True, True, False], [True, True, False, True])   # scores rounded to 4 decimals
    assert abs(s["precision"] - 2 / 3) < 1e-3 and abs(s["recall"] - 2 / 3) < 1e-3
    assert abs(s["f1"] - 2 / 3) < 1e-3


def test_evaluate_formula_shapes() -> None:
    rolls = synth_rollouts(10, seed=3)
    acc = evaluate_formula("F(in_place >= 0.9)", rolls)
    assert len(acc) == 10 and all(isinstance(x, bool) for x in acc)


def test_extract_formula_strips_fences_and_prose() -> None:
    assert _extract_formula("```htl\nF(in_place >= 0.9)\n```") == "F(in_place >= 0.9)"
    assert _extract_formula("Here is the formula:\nF(in_place >= 0.9)") == "F(in_place >= 0.9)"


def test_gate_error_loop_rescues_malformed() -> None:
    # first reply malformed → parse-gate feeds the error back → second reply valid (one candidate, 2 attempts).
    # calibrate=False isolates the error-loop (calibration would otherwise tune the returned threshold).
    m = ScriptedModel(replies=["eventually(x>=1)", "F(in_place >= 0.9)"])
    g = propose_and_gate(m, "propose", synth_rollouts(20, seed=4), k=1, retries=2, calibrate=False)
    assert g.parsed_any and g.formula == "F(in_place >= 0.9)" and g.n_attempts == 2


def test_gate_faithfulness_selects_best_and_beats_raw() -> None:
    rolls = synth_rollouts(60, seed=5)
    verif = synth_rollouts(40, seed=6)
    # proposer emits a distractor first, the faithful one later; raw takes the distractor, gate selects the best.
    gate_model = ScriptedModel(replies=["F(near_object >= 0.9)", "F(in_place >= 0.9)"])
    g = propose_and_gate(gate_model, "propose", verif, k=2, retries=1, calibrate=False)
    raw_model = ScriptedModel(replies=["F(near_object >= 0.9)", "F(in_place >= 0.9)"])
    raw, n_valid, _ = score_raw(raw_model, "propose", k=2)
    assert g.formula == "F(in_place >= 0.9)"
    assert formula_f1(g.formula, rolls) > formula_f1(raw, rolls)        # H4: gate ≫ raw
    assert n_valid == 2


# ── threshold calibration (the arbiter's "refine") ──────────────────────────────────────────────────────
def test_calibration_lifts_miscalibrated_structure_to_ceiling() -> None:
    verif, test = synth_rollouts(40, seed=100), synth_rollouts(80, seed=200)
    bad = "F(in_place >= 1 AND obj_to_target <= 0)"     # right structure, blind-guessed (wrong) thresholds
    cal = calibrate_thresholds(bad, verif)
    assert formula_f1(bad, test) < 0.1                  # raw is unfaithful
    assert formula_f1(cal, test) > 0.85                 # calibrated reaches ~ceiling
    assert "in_place" in cal and "obj_to_target" in cal  # structure (signals) preserved


def test_calibration_preserves_structure() -> None:
    cal = calibrate_thresholds("F(near_object >= 0.5 AND grasp_success == 1)", synth_rollouts(20, seed=1))
    assert cal.startswith("F(") and "AND" in cal and "near_object" in cal and "grasp_success" in cal


def test_gate_calibration_reaches_ceiling_from_bad_thresholds() -> None:
    verif, test = synth_rollouts(40, seed=100), synth_rollouts(80, seed=200)
    # model emits the right structure with wrong numbers every time; the gate's calibration must rescue it.
    m = ScriptedModel(replies=["F(in_place >= 1 AND obj_to_target <= 0)"] * 3)
    g = propose_and_gate(m, "propose", verif, k=3, retries=1, calibrate=True)
    assert g.formula is not None and formula_f1(g.formula, test) > 0.85


def test_gate_returns_none_when_nothing_parses() -> None:
    m = ScriptedModel(replies=["not htl at all", "still bad"])
    g = propose_and_gate(m, "propose", synth_rollouts(10, seed=7), k=2, retries=0)
    assert not g.parsed_any and g.formula is None


def test_raw_returns_none_when_no_valid() -> None:
    m = ScriptedModel(replies=["garbage", "more garbage"])
    raw, n_valid, n_att = score_raw(m, "propose", k=2)
    assert raw is None and n_valid == 0 and n_att == 2
