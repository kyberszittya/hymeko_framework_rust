"""Tests for the DirectLiNGAM-estimated-B robustness layer."""
from __future__ import annotations

import numpy as np

from hymeko_rl.eval.causal.lingam_estimated_b_robustness import (
    _verdict,
    estimate_b,
    recovery_metrics,
    run_condition_estimated,
    run_robustness,
)
from hymeko_rl.eval.causal.lingam_operator_harness import HarnessConfig, generate_signed_dag, sample_sem

_SMOKE = dict(seeds=1, n=8, samples=300, epochs=30)


def test_estimate_b_acyclic_zero_diagonal() -> None:
    b = generate_signed_dag(10, seed=0)
    x = sample_sem(b, 400, kind="linear", seed=0)
    b_hat = estimate_b(x)
    assert b_hat.shape == (10, 10)
    assert np.all(np.diag(b_hat) == 0.0)                 # DirectLiNGAM is acyclic → no self-loops


def test_estimate_b_recovers_linear_sem() -> None:
    b = generate_signed_dag(12, density=0.4, seed=1)
    x = sample_sem(b, 800, kind="linear", seed=1)
    rec = recovery_metrics(b, estimate_b(x))
    assert rec["support_recall"] >= 0.8                  # linear SEM is recoverable
    assert rec["sign_agreement"] >= 0.9


def test_recovery_metrics_perfect_on_identical() -> None:
    b = generate_signed_dag(10, seed=2)
    rec = recovery_metrics(b, b.copy())
    for key in ("support_precision", "support_recall", "support_f1", "sign_agreement", "pearson", "spearman",
                "topk_overlap"):
        assert abs(rec[key] - 1.0) < 1e-6


def test_recovery_metrics_detects_wrong_structure() -> None:
    b = generate_signed_dag(12, density=0.4, seed=3)
    wrong = -b[::-1, ::-1]                                # flipped + sign-negated → poor support + sign match
    rec = recovery_metrics(b, wrong)
    assert rec["support_precision"] < 0.9 or rec["sign_agreement"] < 0.9


def test_run_condition_has_six_models_and_recovery() -> None:
    r = run_condition_estimated("nonlinear", HarnessConfig(**_SMOKE))
    assert set(r["models"]) == {"linear", "mlp", "deepsets", "hsikan_gt", "hsikan_est", "hsikan_est_scrambled"}
    assert "support_recall" in r["recovery"]
    assert r["degradation_est_vs_gt_frac"] is not None


def test_run_robustness_smoke() -> None:
    r = run_robustness(HarnessConfig(**_SMOKE))
    assert set(r["conditions"]) == {"linear", "nonlinear", "flat"}
    assert r["verdict"]["case"].startswith(("A", "B", "C", "D"))
    assert r["verdict"]["coffee_push_gate"] in ("PASS", "HOLD")


# ── verdict / case classification (pure) ────────────────────────────────────────────────────────────────
def _nl(*, hk_est: float, hk_gt: float, mlp: float, collapse: float, degr: float) -> dict:
    return {"models": {"hsikan_est": {"test_median": hk_est}, "hsikan_gt": {"test_median": hk_gt}},
            "gap_est_vs_mlp": mlp - hk_est, "collapse_est_frac": collapse, "degradation_est_vs_gt_frac": degr}


def _flat(collapse: float) -> dict:
    return {"models": {"hsikan_est": {"test_median": 2.0}}, "gap_est_vs_mlp": 0.0,
            "collapse_est_frac": collapse, "degradation_est_vs_gt_frac": 0.0}


def test_case_A_strong_and_gate_pass() -> None:
    v = _verdict({"nonlinear": _nl(hk_est=0.8, hk_gt=0.65, mlp=1.5, collapse=0.7, degr=0.24), "flat": _flat(0.0)})
    assert v["case"] == "A_strong" and v["coffee_push_gate"] == "PASS"


def test_case_B_partial_when_degraded_far_from_gt() -> None:
    v = _verdict({"nonlinear": _nl(hk_est=1.2, hk_gt=0.5, mlp=1.5, collapse=0.7, degr=1.4), "flat": _flat(0.0)})
    assert v["case"] == "B_partial"                      # beats MLP but far from ground-truth


def test_case_C_negative_when_no_mlp_win() -> None:
    v = _verdict({"nonlinear": _nl(hk_est=1.5, hk_gt=0.5, mlp=1.52, collapse=0.02, degr=2.0), "flat": _flat(0.0)})
    assert v["case"] == "C_negative" and v["coffee_push_gate"] == "HOLD"


def test_case_D_flat_false_positive_holds_gate() -> None:
    v = _verdict({"nonlinear": _nl(hk_est=0.8, hk_gt=0.65, mlp=1.5, collapse=0.7, degr=0.24), "flat": _flat(0.5)})
    assert v["case"] == "D_flat_false_positive" and v["coffee_push_gate"] == "HOLD"
