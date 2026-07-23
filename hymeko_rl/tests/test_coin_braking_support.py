"""CONTACT_PRESERVING_BRAKING_PRIMITIVE_V2 Part-A contract tests (pure, deterministic; no env). Candidate-offset
determinism/bounds, the frozen SAFE_BENEFICIAL predicate, and the support gate."""
import numpy as np

from hymeko_rl.coin_delivery.coin_braking_support import (
    OFFSET_BOUND,
    SafeBeneficialConfig,
    candidate_offsets,
    safe_beneficial,
    support_gate,
)

CFG = SafeBeneficialConfig()


def test_candidate_offsets_deterministic_bounded_and_zero_reference():
    o1 = candidate_offsets(32); o2 = candidate_offsets(32)
    assert o1.shape == (33, 4) and np.allclose(o1, o2)              # deterministic, n+1 (row0 = pi_0 reference)
    assert np.allclose(o1[0], 0.0) and np.abs(o1[1:]).max() <= OFFSET_BOUND + 1e-6


def _b(*, kept=True, exit=False, peak=0.30, dtz=0.04, strict=0):
    return {"contact_kept": kept, "exit_before_k6": exit, "peak_radial_vel": peak, "final_dtz": dtz, "strict_change": strict}


def test_safe_beneficial_decelerating_contact_preserving_is_true():
    ref = _b(peak=0.40, dtz=0.04); cand = _b(peak=0.30, dtz=0.041)     # decelerates 0.4→0.3, keeps contact, progress ok
    assert safe_beneficial(cand, ref, CFG)


def test_safe_beneficial_false_on_new_contact_loss():
    ref = _b(kept=True, peak=0.40); cand = _b(kept=False, peak=0.20)   # decelerates but drops contact pi_0 kept
    assert not safe_beneficial(cand, ref, CFG)


def test_safe_beneficial_false_on_worsened_progress():
    ref = _b(peak=0.40, dtz=0.04); cand = _b(peak=0.20, dtz=0.06)      # decelerates but coin ends much farther
    assert not safe_beneficial(cand, ref, CFG)


def test_safe_beneficial_false_on_new_exit():
    ref = _b(exit=False, peak=0.40); cand = _b(exit=True, peak=0.20)
    assert not safe_beneficial(cand, ref, CFG)


def test_zero_offset_reference_is_not_beneficial():
    ref = _b(peak=0.40); cand = dict(ref)                              # identical to pi_0 ⇒ no deceleration, no task gain
    assert not safe_beneficial(cand, ref, CFG)


def test_strict_improvement_counts_even_without_deceleration():
    ref = _b(peak=0.30, strict=0); cand = _b(peak=0.31, strict=1)      # no decel but delivers more ⇒ beneficial
    assert safe_beneficial(cand, ref, CFG)


def _row(outcome, n_safe, decel=0.05):
    return {"v1_outcome": outcome, "support": {"n_safe_beneficial": n_safe, "best_radial_decel": decel}}


def test_gate_found_when_broad_support_incl_failing_states():
    rows = [_row("contact_losing", 3), _row("contact_losing", 2), _row("target_exit", 1),
            _row("no_delivery", 2), _row("delivered_contact_preserving", 4), _row("no_delivery", 1)]
    g = support_gate(rows, CFG)
    assert g["verdict"] == "BRAKING_SAFE_BENEFICIAL_SUPPORT_FOUND" and g["fraction_with_support"] == 1.0


def test_gate_insufficient_when_support_isolated_to_successful_states():
    rows = [_row("delivered_contact_preserving", 5), _row("delivered_contact_preserving", 3),
            _row("contact_losing", 0), _row("target_exit", 0), _row("no_delivery", 0), _row("no_delivery", 0)]
    g = support_gate(rows, CFG)
    assert g["verdict"] == "BRAKING_SAFE_BENEFICIAL_SUPPORT_INSUFFICIENT"      # <3 failing states with support
    assert g["n_failing_states_with_support"] == 0


def test_gate_insufficient_when_fraction_below_threshold():
    rows = [_row("contact_losing", 1)] + [_row("no_delivery", 0) for _ in range(9)]
    g = support_gate(rows, CFG)
    assert g["fraction_with_support"] < 0.5 and g["verdict"] == "BRAKING_SAFE_BENEFICIAL_SUPPORT_INSUFFICIENT"
