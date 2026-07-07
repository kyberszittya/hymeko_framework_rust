"""Unit + guarded-integration tests for the Q-term collapse root-cause probe (hymeko_rl/eval/critic_probe.py).

The scientifically load-bearing part is the *classification* (`_classify`): a bug there yields a wrong root-
cause verdict. Every hypothesis branch is unit-tested. The class methods (fit / M1 / M3 / M2) are exercised
end-to-end by a smoke integration test, guarded on the presence of the measured BC clone checkpoint so CI
without the artifact skips rather than fails.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from hymeko_rl.eval.critic_probe import (
    ProbeConfig, ProbeVerdict, QTermCollapseProbe, _classify, _diverged, _soft_update, _spearman, _tied_ranks,
    make_deliver_env,
)

_CLONE = Path("experiments/2026_07_05_03_29_galambos_coord_ab_deliver/policies/"
              "galambos_coord_ab_deliver_s0.pt")


# ── _spearman ────────────────────────────────────────────────────────────────
def test_spearman_perfect_monotone() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _spearman(x, 2.0 * x + 1.0) == pytest.approx(1.0)          # monotone increasing → +1
    assert _spearman(x, -x) == pytest.approx(-1.0)                    # monotone decreasing → -1


def test_spearman_degenerate() -> None:
    assert _spearman(np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0   # constant → 0 (no variance)
    assert _spearman(np.array([1.0]), np.array([1.0])) == 0.0                        # len < 2 → 0


def test_tied_ranks_average() -> None:
    # ties share their mean rank (1-based, scipy convention): [10,10,20,30] → [1.5,1.5,3,4].
    assert np.allclose(_tied_ranks(np.array([10.0, 10.0, 20.0, 30.0])), [1.5, 1.5, 3.0, 4.0])
    assert np.allclose(_tied_ranks(np.array([5.0, 5.0, 5.0])), [2.0, 2.0, 2.0])   # all tied → mean of 1,2,3


def test_spearman_rank_not_linear() -> None:
    # Spearman is rank-based: a monotone-nonlinear map still gives +1 (unlike Pearson).
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert _spearman(x, np.exp(x)) == pytest.approx(1.0)


# ── _classify (each hypothesis branch) ───────────────────────────────────────
def test_classify_h_fit_low_rho() -> None:
    hyp, diag = _classify(clone_deliv=0.3, crit_loss=0.1, rho=0.1, q_rise=1.0,
                          ret_delta=0.0, deliv_delta=0.0, ood=0.0)
    assert hyp == "H_fit" and "cannot rank" in diag


def test_classify_h_fit_nonfinite_loss() -> None:
    hyp, _ = _classify(0.3, float("inf"), 0.9, 1.0, 0.0, 0.0, 0.0)     # diverged fit → H_fit even if rho high
    assert hyp == "H_fit"


def test_classify_h_fit_divergence_beats_downstream() -> None:
    # a diverged fit → H_fit even when rho is high and the deltas look like some other hypothesis: a diverged
    # critic's gradient is meaningless, so the downstream verdict must not be trusted.
    hyp, diag = _classify(0.3, 6.29, rho=0.9, q_rise=2.0, ret_delta=-3.0, deliv_delta=-0.2, ood=1.0,
                          crit_diverged=True)
    assert hyp == "H_fit" and "DIVERGED" in diag and "FRAMEWORK" in diag


def test_diverged_flags_marching_loss_and_q() -> None:
    # the primary-run signature: loss rising 0.39→6.29, |Q| marching 16→60 → diverged.
    traj = [{"u": 4000, "q": -16.4, "loss": 0.39}, {"u": 8000, "q": -29.9, "loss": 0.61},
            {"u": 12000, "q": -41.6, "loss": 1.28}, {"u": 16000, "q": -52.0, "loss": 1.33},
            {"u": 20000, "q": -59.6, "loss": 6.29}]
    assert _diverged(traj) is True


def test_diverged_accepts_converged_trajectory() -> None:
    traj = [{"u": 2000, "q": -8.0, "loss": 0.9}, {"u": 4000, "q": -8.2, "loss": 0.3},
            {"u": 6000, "q": -8.1, "loss": 0.12}, {"u": 8000, "q": -8.15, "loss": 0.10}]
    assert _diverged(traj) is False
    assert _diverged([{"u": 1, "q": 1.0, "loss": 1.0}]) is False        # too short → not flagged


def test_classify_h_ood_phantom() -> None:
    # critic ranks well (rho high) but ascending Q drove the TRUE return down → phantom.
    hyp, diag = _classify(0.3, 0.1, 0.8, q_rise=2.0, ret_delta=-3.0, deliv_delta=-0.2, ood=1.5)
    assert hyp == "H_ood" and "phantom" in diag


def test_classify_h_reward_misalignment() -> None:
    # faithful critic, true return ROSE, but delivery fell → reward locally misaligned with the task.
    hyp, diag = _classify(0.3, 0.1, 0.8, q_rise=2.0, ret_delta=+1.0, deliv_delta=-0.2, ood=0.0)
    assert hyp == "H_reward" and "mismatch" in diag


def test_classify_h_shift_operator_sound() -> None:
    # clean improvement did not degrade the clone → online loop (H_shift), not the critic/reward.
    hyp, diag = _classify(0.3, 0.1, 0.8, q_rise=2.0, ret_delta=+1.0, deliv_delta=+0.05, ood=0.0)
    assert hyp == "H_shift" and "online" in diag


def test_classify_boundary_delivery_tol() -> None:
    # a delivery drop smaller than tol does NOT trip H_reward (guards against eval-noise false positives).
    hyp, _ = _classify(0.3, 0.1, 0.8, 2.0, ret_delta=+1.0, deliv_delta=-0.04, ood=0.0)
    assert hyp == "H_shift"


# ── ProbeConfig.resolved ─────────────────────────────────────────────────────
def test_config_smoke_caps_budgets() -> None:
    r = ProbeConfig(smoke=True).resolved()
    assert r.buffer_steps <= 1_500 and r.fit_updates <= 500 and r.improve_steps <= 40 and r.n_eval <= 6


def test_config_nonsmoke_is_identity() -> None:
    c = ProbeConfig(buffer_steps=20_000, fit_updates=20_000)
    assert c.resolved() is c                                           # non-smoke: unchanged, same object


# ── _soft_update ─────────────────────────────────────────────────────────────
def test_soft_update_moves_target_toward_source() -> None:
    torch.manual_seed(0)
    tgt, src = nn.Linear(3, 2), nn.Linear(3, 2)
    with torch.no_grad():
        tgt.weight.zero_()
        src.weight.fill_(1.0)
    _soft_update(tgt, src, tau=0.1)
    assert torch.allclose(tgt.weight, torch.full_like(tgt.weight, 0.1))   # 0.9*0 + 0.1*1


# ── make_deliver_env ─────────────────────────────────────────────────────────
def test_make_deliver_env_sets_deliver_reward() -> None:
    env = make_deliver_env(difficulty=0.3)
    assert env.reward_spec is not None and env.max_steps == 300


# ── guarded integration smoke (exercises fit / M1 / M3 / M2 on the real path) ─
@pytest.mark.skipif(not _CLONE.exists(), reason="measured BC clone checkpoint not present")
def test_probe_smoke_runs_and_preserves_clone() -> None:
    probe = QTermCollapseProbe(_CLONE, ProbeConfig(smoke=True, seed=0))
    before = probe.mu0.action_heads[0].weight.detach().clone()
    verdict = probe.run()
    assert isinstance(verdict, ProbeVerdict)
    assert -1.0 <= verdict.rank_spearman <= 1.0
    assert verdict.hypothesis in {"H_fit", "H_ood", "H_reward", "H_shift"}
    assert verdict.diagnosis
    # M3 must not mutate the frozen clone (mu1 is a deep copy).
    assert torch.allclose(before, probe.mu0.action_heads[0].weight)


@pytest.mark.skipif(not _CLONE.exists(), reason="measured BC clone checkpoint not present")
def test_stability_scan_runs() -> None:
    probe = QTermCollapseProbe(_CLONE, ProbeConfig(smoke=True, seed=0))
    buf = probe._collect_on_clone()
    recs = probe.stability_scan(buf, [{"label": "t_clip0", "max_grad_norm": 0.0, "fit_updates": 120},
                                      {"label": "t_clip10", "max_grad_norm": 10.0, "fit_updates": 120}])
    assert len(recs) == 2
    for r in recs:
        assert isinstance(r["diverged"], bool) and np.isfinite(r["final_loss"]) and r["traj"]
