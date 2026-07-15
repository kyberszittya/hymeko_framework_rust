"""Tests for SignedHyperLiNGAM — native signed-hyperedge causal discovery.

Covers: the reused DirectLiNGAM ordering, the additive-SEM **tie** (no false win), the joint-interaction **win**
(recovers the hyperedge pairwise LiNGAM misses), the joint-SEM generator's non-additivity, sign correctness on
additive edges, native CausalHypergraph output, input validation, and the head-to-head runner."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from hymeko_rl.eval.causal.lingam import DirectLiNGAM
from hymeko_rl.eval.causal.lingam_estimated_b_robustness import estimate_b, recovery_metrics
from hymeko_rl.eval.causal.lingam_operator_harness import generate_signed_dag, sample_sem
from hymeko_rl.eval.causal.signed_hyper_lingam import (
    SHLConfig,
    SignedHyperLiNGAM,
    SignedHyperResult,
    sample_interaction_sem,
)


def test_ordering_is_reused_from_directlingam() -> None:
    b = generate_signed_dag(6, density=0.4, seed=0)
    x = sample_sem(b, 3000, kind="linear", seed=1)
    assert SignedHyperLiNGAM().fit(x).order == DirectLiNGAM().fit(x).order


def test_fit_validates_shape() -> None:
    with pytest.raises(ValueError, match="n > d"):
        SignedHyperLiNGAM().fit(np.zeros((3, 5)))                       # n <= d
    with pytest.raises(ValueError, match="non-finite"):
        SignedHyperLiNGAM().fit(np.array([[1.0, np.inf, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9, 1, 2]]))


def test_result_is_signed_hyper_result_with_triangular_adjacency() -> None:
    b = generate_signed_dag(5, density=0.5, seed=2)
    r = SignedHyperLiNGAM().fit(sample_sem(b, 2500, kind="linear", seed=3))
    assert isinstance(r, SignedHyperResult)
    # every mechanism tail is a set of causal-order predecessors → strictly triangular in `order`
    pos = {v: i for i, v in enumerate(r.order)}
    for effect, tail in r.mechanisms.items():
        assert all(pos[c] < pos[effect] for c, _s, _w, _j in tail)


def test_additive_sems_tie_no_false_win() -> None:
    """On additive (linear / per-edge-nonlinear) SEMs DirectLiNGAM already recovers support — must not claim a win."""
    for kind in ("linear", "nonlinear"):
        dl_rec, sh_rec = [], []
        for s in range(4):
            b = generate_signed_dag(6, density=0.4, seed=s)
            x = sample_sem(b, 3000, kind=kind, seed=10 + s)
            dl_rec.append(recovery_metrics(b, estimate_b(x))["support_recall"])
            sh_rec.append(recovery_metrics(b, SignedHyperLiNGAM().fit(x).adjacency)["support_recall"])
        assert np.median(dl_rec) >= 0.9 and np.median(sh_rec) >= 0.9          # both recover; no headroom


def test_joint_interaction_win_recovers_hyperedge_pairwise_misses() -> None:
    """The core claim: on joint-interaction SEMs SignedHyperLiNGAM recall > DirectLiNGAM (which structurally misses)."""
    dl_rec, sh_rec = [], []
    for s in range(6):
        b = generate_signed_dag(6, density=0.4, seed=s)
        x = sample_interaction_sem(b, 4000, seed=100 + s)
        dl_rec.append(recovery_metrics(b, estimate_b(x))["support_recall"])
        sh_rec.append(recovery_metrics(b, SignedHyperLiNGAM().fit(x).adjacency)["support_recall"])
    assert np.median(sh_rec) > np.median(dl_rec) + 0.1                        # a real, not marginal, win
    assert np.median(sh_rec) >= 0.9                                           # SHL recovers the joint tails


def test_interaction_sem_is_genuinely_non_additive() -> None:
    """A joint-product parent has ~0 marginal correlation with its effect (that's why pairwise fails)."""
    # a fixed 3-var support where x2 has two parents {x0,x1} -> forced joint (joint_frac=1)
    b = np.zeros((3, 3))
    b[2, 0], b[2, 1] = 0.8, 0.7
    x = sample_interaction_sem(b, 5000, seed=1, joint_frac=1.0)
    assert abs(float(np.corrcoef(x[:, 0], x[:, 2])[0, 1])) < 0.1               # marginal effect vanishes
    assert abs(float(np.corrcoef(x[:, 0] * x[:, 1], x[:, 2])[0, 1])) > 0.5     # the product predicts


def test_sign_correct_on_additive_edge() -> None:
    b = np.zeros((3, 3))
    b[2, 0], b[2, 1] = 0.9, -0.8                                              # x2 = 0.9 x0 - 0.8 x1
    x = sample_interaction_sem(b, 5000, seed=2, joint_frac=0.0)               # additive → clean signs
    tail = {c: s for c, s, _w, _j in SignedHyperLiNGAM().fit(x).mechanisms.get(2, [])}
    assert tail.get(0) == 1 and tail.get(1) == -1


def test_to_causal_hypergraph_is_acyclic() -> None:
    b = generate_signed_dag(5, density=0.5, seed=1)
    r = SignedHyperLiNGAM().fit(sample_interaction_sem(b, 3000, seed=7))
    chg = r.to_causal_hypergraph()
    assert chg.check_acyclicity().acyclic
    assert all(len(h) >= 1 for h, _head, _signs in r.hyperedges())


def test_config_is_immutable_and_defaulted() -> None:
    cfg = SHLConfig()
    assert cfg.max_predecessors == 8 and cfg.parsimony == 0.03
    with pytest.raises(Exception):
        cfg.parsimony = 0.1                                                  # frozen dataclass


def test_head_to_head_runner_smoke(tmp_path: "object") -> None:
    from pathlib import Path

    from hymeko_rl.experiments.exp_signed_hyper_lingam import run_head_to_head
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = run_head_to_head(n_vars=6, density=0.4, n_samples=3000, seeds=(0, 1, 2, 3), out_dir=Path(str(tmp_path)))
    assert set(s["verdicts"]) == {"linear", "nonlinear", "joint"}
    assert s["verdicts"]["joint"] == "SignedHyperLiNGAM_WINS"                 # the headline
    assert s["verdicts"]["linear"] == "TIE"
    # on additive-nonlinear SHL must NOT falsely win (it ties or slightly under-recovers — a parsimony cost)
    assert s["verdicts"]["nonlinear"] in ("TIE", "DirectLiNGAM_WINS")
    assert (Path(str(tmp_path)) / "signed_hyper_lingam.json").exists()
