"""Unit tests for the critic ranking benchmark (pure diagnostics) + the critic-repair loss strategies.
Env-free; the full env-driven benchmark is ``scratchpad/critic_benchmark_run.py``."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hymeko_rl.eval.critic_benchmark import (
    ClassMetrics,
    acceptance,
    classify_critic,
    counterfactual_ranking,
    kendall,
    ood_overestimation,
    phase_label,
    phasewise_ranking,
    q_vs_return_calibration,
    spearman,
)
from hymeko_rl.train.critic_repair import (
    CriticRepairConfig,
    build_critic_loss,
    cql_regularizer,
    train_critic_only,
)


# --- rank correlations --------------------------------------------------------------------------------------
def test_rank_correlations():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0            # degenerate constant → 0
    assert kendall([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([3, 1, 2], [3, 1, 2]) == pytest.approx(1.0)   # ties-free permutation


def test_phase_label():
    c = dict(near_coin=0.06, progress_eps=0.002)
    assert phase_label(0.5, False, 0.0, False, **c) == "APPROACH"
    assert phase_label(0.02, True, 0.0, False, **c) == "CONTACT"
    assert phase_label(0.02, True, 0.02, False, **c) == "PUSH"
    assert phase_label(0.01, True, 0.02, True, **c) == "DELIVERY"


# --- diagnostics --------------------------------------------------------------------------------------------
def _classes(q_aligned: bool):
    """Six classes with monitor score ordered good→bad; Q either aligned or inverted vs return/monitor."""
    names = ["scripted", "mlp_dagger_selected", "mlp_bc0", "body_shove_exploit", "one_fingertip", "failed_rl"]
    mon = [0.35, 0.28, 0.09, -0.20, -0.27, -0.31]
    mc = [-99, -115, -209, -111, -330, -343]
    q = list(mon) if q_aligned else list(reversed(mon))   # inverted critic ranks bad above good
    return [ClassMetrics(n, mc_return=m, reward=m, ft_dom=0.0, monitor_pass=0.0, monitor_score=s, violation="",
                         q_onpolicy=qq) for n, m, s, qq in zip(names, mc, mon, q)]


def test_q_vs_return_calibration_aligned_vs_inverted():
    good = q_vs_return_calibration(_classes(q_aligned=True))
    assert good["spearman_q_monitor"] == pytest.approx(1.0)
    bad = q_vs_return_calibration(_classes(q_aligned=False))
    assert bad["spearman_q_monitor"] < 0.0


def test_counterfactual_ranking_flags_the_inversion():
    inverted = counterfactual_ranking(
        {"mlp_dagger_selected": -6.55, "body_shove_exploit": -5.70, "one_fingertip": -6.76},
        {"mlp_dagger_selected": 0.278, "body_shove_exploit": -0.202, "one_fingertip": -0.273})
    assert not inverted["Q_dagger_gt_exploit"] and inverted["Q_dagger_gt_one_fingertip"]
    assert inverted["ranking_best_to_worst"][0] == "body_shove_exploit"
    repaired = counterfactual_ranking(
        {"mlp_dagger_selected": -4.0, "body_shove_exploit": -6.0, "one_fingertip": -6.5},
        {"mlp_dagger_selected": 0.278, "body_shove_exploit": -0.202, "one_fingertip": -0.273})
    assert repaired["Q_dagger_gt_exploit"] and repaired["Q_dagger_gt_one_fingertip"]
    assert repaired["spearman_q_monitor_counterfactual"] == pytest.approx(1.0)


def test_ood_overestimation():
    bad = ood_overestimation(q_dagger=-4.0, q_random=-3.0, q_perturbed_by_mag={0.1: -4.1, 0.5: -3.5, 1.0: -3.0})
    assert not bad["Q_random_below_dagger"] and not bad["Q_decreases_with_perturbation"]
    good = ood_overestimation(q_dagger=-4.0, q_random=-7.0, q_perturbed_by_mag={0.1: -4.3, 0.5: -5.5, 1.0: -7.0})
    assert good["Q_random_below_dagger"] and good["Q_decreases_with_perturbation"] and good["Q_largest_perturb_below_dagger"]


def test_phasewise_ranking_flags_contact_push():
    r = phasewise_ranking(
        q_dagger_by_phase={"APPROACH": -4.0, "CONTACT": -6.0, "PUSH": -6.5, "DELIVERY": -3.0},
        q_exploit_by_phase={"APPROACH": -5.0, "CONTACT": -5.5, "PUSH": -5.0, "DELIVERY": -4.0})
    assert set(r["misranked_phases"]) == {"CONTACT", "PUSH"}
    assert r["by_phase"]["APPROACH"]["dagger_gt_exploit"]


def _diag(q_dagger, q_exploit, q_one, q_random, *, sp_mon=0.7, sp_mc=0.5, pert_ok=True):
    cf = {"Q_by_action": {"mlp_dagger_selected": q_dagger, "body_shove_exploit": q_exploit, "one_fingertip": q_one},
          "Q_dagger_gt_exploit": q_dagger > q_exploit, "Q_dagger_gt_one_fingertip": q_dagger > q_one}
    ood = {"Q_dagger": q_dagger, "Q_random": q_random, "Q_random_below_dagger": q_random < q_dagger,
           "Q_decreases_with_perturbation": pert_ok, "Q_largest_perturb_below_dagger": pert_ok}
    cal = {"spearman_q_monitor": sp_mon, "spearman_q_mc": sp_mc}
    return cf, ood, cal


def test_classify_critic_three_tiers():
    # STRONG: exploit margin +11.6, one-finger +7.3, ood gap +16.8 (CQL-like)
    cf, ood, cal = _diag(-12.3, -23.9, -19.6, -29.1)
    strong = classify_critic(cf, ood, cal, tensor_contract_pass=True, policy_provenance_pass=True)
    assert strong.tier == "STRONG_PASS" and strong.margin_exploit >= 3.0 and not strong.reasons
    # WEAK: ranks right but knife-edge (+0.05) — baseline-like
    cf, ood, cal = _diag(-10.855, -10.908, -12.134, -13.126)
    weak = classify_critic(cf, ood, cal, tensor_contract_pass=True, policy_provenance_pass=True)
    assert weak.tier == "WEAK_PASS" and any("margin_exploit" in r for r in weak.reasons)
    # WEAK: exploit margin fine but one-finger margin < 3 (behavior-support-like)
    cf, ood, cal = _diag(-11.125, -16.95, -14.004, -20.162)
    weakb = classify_critic(cf, ood, cal, tensor_contract_pass=True, policy_provenance_pass=True)
    assert weakb.tier == "WEAK_PASS" and any("one_finger" in r for r in weakb.reasons)
    # FAIL: exploit ranked above dagger (expectile-like)
    cf, ood, cal = _diag(-11.669, -11.177, -13.236, -12.061, pert_ok=False)
    fail = classify_critic(cf, ood, cal, tensor_contract_pass=True, policy_provenance_pass=True)
    assert fail.tier == "FAIL"
    # FAIL: STRONG diagnostics but a guard fails
    cf, ood, cal = _diag(-12.3, -23.9, -19.6, -29.1)
    guardfail = classify_critic(cf, ood, cal, tensor_contract_pass=False, policy_provenance_pass=True)
    assert guardfail.tier == "FAIL" and "guard_failed" in guardfail.reasons


def test_cql_regularizer_is_finite_scalar():
    reg = cql_regularizer(-np.ones(4, np.float32), np.ones(4, np.float32), alpha=1.0, n_samples=4)
    critics = [TinyCritic(), TinyCritic()]
    actor = TinyActor()
    s, a, z = torch.randn(8, 8), torch.rand(8, 4) * 2 - 1, torch.randn(8, 5)
    out = reg(critics, s, a, z, actor)
    assert out.ndim == 0 and torch.isfinite(out) and out.requires_grad


def test_acceptance_gate():
    cf_ok = {"Q_dagger_gt_exploit": True, "Q_dagger_gt_one_fingertip": True}
    ood_ok = {"Q_random_below_dagger": True, "Q_largest_perturb_below_dagger": True}
    cal_ok = {"spearman_q_monitor": 0.8, "spearman_q_mc": 0.6}
    a = acceptance(cf_ok, ood_ok, cal_ok, tensor_contract_pass=True, policy_provenance_pass=True)
    assert a.passed
    cf_bad = {"Q_dagger_gt_exploit": False, "Q_dagger_gt_one_fingertip": True}
    assert not acceptance(cf_bad, ood_ok, cal_ok, tensor_contract_pass=True, policy_provenance_pass=True).passed
    assert not acceptance(cf_ok, ood_ok, cal_ok, tensor_contract_pass=False, policy_provenance_pass=True).passed


# --- critic-repair loss strategies (shape + finiteness) -----------------------------------------------------
class TinyCritic(nn.Module):
    def __init__(self, obs=8, act=4, priv=5):
        super().__init__()
        self.lin = nn.Linear(obs + act + priv, 1)

    def forward(self, s, a, z):
        x = torch.cat([s.reshape(s.shape[0], -1), a, z], -1)
        return self.lin(x).squeeze(-1)


class TinyActor(nn.Module):
    def __init__(self, obs=8, act=4):
        super().__init__()
        self.lin = nn.Linear(obs, act)

    def forward(self, s):
        return torch.tanh(self.lin(s.reshape(s.shape[0], -1)))


@pytest.mark.parametrize("variant", ["A", "B", "C", "E"])
def test_critic_losses_finite_scalar(variant):
    torch.manual_seed(0)
    critics = [TinyCritic(), TinyCritic()]
    actor = TinyActor()
    b = 16
    s = torch.randn(b, 8)
    a = torch.rand(b, 4) * 2 - 1
    z = torch.randn(b, 5)
    y = torch.randn(b)
    lo, hi = -torch.ones(4), torch.ones(4)
    cfg = CriticRepairConfig(variant=variant)
    loss = build_critic_loss(cfg).compute(critics, y, s, a, z, actor, lo, hi, cfg)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="unknown critic-repair variant"):
        build_critic_loss(CriticRepairConfig(variant="Z"))


def test_train_critic_only_runs_and_keeps_actor_frozen():
    """Smoke: the trainer runs a few steps with a real ReplayBuffer and does NOT touch the frozen actor."""
    from hymeko_rl.eval.task_monitor import param_hash
    from hymeko_rl.train.replay import ReplayBuffer

    torch.manual_seed(0)
    buf = ReplayBuffer(200, (8,), 4, priv_dim=5)
    rng = np.random.default_rng(0)
    for _ in range(64):
        buf.add(rng.standard_normal(8).astype(np.float32), (rng.random(4) * 2 - 1).astype(np.float32),
                float(rng.standard_normal()), rng.standard_normal(8).astype(np.float32), False,
                priv=rng.standard_normal(5).astype(np.float32), priv_next=rng.standard_normal(5).astype(np.float32))
    critics = [TinyCritic(), TinyCritic()]
    actor = TinyActor()
    h0 = param_hash(actor)
    cfg = CriticRepairConfig(variant="C", steps=20, batch_size=16, log_every=0, cql_n_samples=4)
    train_critic_only(critics, actor, buf, cfg, action_lo=-np.ones(4, np.float32), action_hi=np.ones(4, np.float32))
    assert param_hash(actor) == h0          # actor frozen
