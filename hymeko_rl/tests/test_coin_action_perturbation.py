"""Gate tests for LOCAL_ACTION_RANKING_FIDELITY_V1 — perturbation wrapper, direction geometry, critic-gradient
sign, the additive dtz/speed eval keys, and the scipy-free Spearman."""
import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_action_perturbation import (
    PerturbedActor,
    actuator_basis_delta,
    bootstrap_ci,
    classify_vs_chance,
    critic_grad_delta,
    eps_from_drifts,
    hierarchical_bootstrap_ci,
    lex_better,
    lex_key,
    nstep_return,
    primary_divergence,
    spearman,
)


class _Base:
    """Trivial constant policy: action_mean(obs) = zeros(action_dim), clamped-domain like the real actor."""

    def __init__(self, adim=4):
        self.adim = adim

    def action_mean(self, obs):
        return torch.zeros(obs.shape[0], self.adim)


class _Critic:
    """Q(obs, a) = w·a with a fixed preference vector w — so dQ/da = w everywhere (a known gradient direction)."""

    def __init__(self, w):
        self.w = torch.as_tensor(w, dtype=torch.float32)

    def min_q(self, obs, a):
        return (a * self.w).sum(-1)


def test_delta_zero_reproduces_base():
    base = _Base(4)
    obs = torch.randn(5, 55)
    pert = PerturbedActor(base, torch.zeros(4))
    assert torch.allclose(pert.action_mean(obs), base.action_mean(obs), atol=0.0)


def test_actuator_basis_shifts_only_its_axis():
    base = _Base(4)
    obs = torch.randn(3, 55)
    eps = 0.02
    for axis in range(4):
        for sign in (+1, -1):
            d = actuator_basis_delta(4, axis, sign, eps)
            assert abs(float(d.norm()) - eps) < 1e-7                    # ||delta|| == eps
            out = PerturbedActor(base, d).action_mean(obs)
            base_out = base.action_mean(obs)
            diff = (out - base_out).numpy()
            assert np.allclose(diff[:, axis], sign * eps, atol=1e-6)    # only its axis moved
            other = [j for j in range(4) if j != axis]
            assert np.allclose(diff[:, other], 0.0, atol=1e-6)


def test_critic_grad_direction_has_norm_eps_and_raises_q():
    base = _Base(4)
    w = [1.0, -2.0, 0.5, 0.0]
    critic = _Critic(w)
    obs = torch.randn(6, 55)
    eps = 0.01
    delta = critic_grad_delta(base, critic, eps)(obs)
    # norm == eps per state
    assert np.allclose(delta.norm(dim=-1).numpy(), eps, atol=1e-6)
    # points along +w (the critic gradient) → local directional derivative dot(delta, w) > 0
    wt = torch.as_tensor(w)
    assert (delta @ wt > 0).all()
    # the perturbed action raises Q relative to base (small-eps first order)
    q_base = critic.min_q(obs, base.action_mean(obs))
    q_pert = critic.min_q(obs, PerturbedActor(base, critic_grad_delta(base, critic, eps)).action_mean(obs))
    assert (q_pert - q_base > 0).all()


def test_clamp_at_action_scale():
    obs = torch.zeros(1, 55)

    class _NearMax(_Base):
        def action_mean(self, obs):
            return torch.full((obs.shape[0], self.adim), 3.99)

    pert = PerturbedActor(_NearMax(2), actuator_basis_delta(2, 0, +1, 0.5), scale=4.0)
    out = pert.action_mean(obs)
    assert float(out.max()) <= 4.0 + 1e-6                              # clamped, not 4.49


def test_spearman_monotone_ties_and_degenerate():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9      # monotone → +1
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9      # anti-monotone → -1
    assert spearman([1, 2, 3], [5, 5, 5]) is None                          # degenerate → None (NOT 0), caller excludes
    assert spearman([7], [7]) is None                                      # <2 points → None
    r = spearman([1, 1, 2, 3, 3], [2, 1, 3, 5, 4])                         # ties via average ranks
    assert r is not None and -1.0 <= r <= 1.0 and r > 0.5


def test_bootstrap_ci_over_states_drops_none():
    ci = bootstrap_ci([0.0, 0.5, 1.0, None, 0.5], stat=np.median)          # None (degenerate states) dropped
    assert ci["n"] == 4 and ci["lo"] <= ci["stat"] <= ci["hi"]
    assert bootstrap_ci([], stat=np.median)["n"] == 0                      # empty → n=0, no crash


def test_one_step_candidate_outcome_applies_candidate_only_at_t0():
    """Real integration test of the corrected rollout: candidate at t=0 then frozen pi_0; keys + determinism."""
    import copy
    import json

    from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
    from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0, one_step_candidate_outcome
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

    d = "experiments/2026_07_22_coin_v3_learning/rl_entry"
    cfg = json.load(open(f"{d}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{d}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = cfg["banks"]["late_dev"]["rows"][0]
    ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    o55 = np.concatenate([rec.obs.astype(np.float32), np.zeros(7, np.float32)])
    a0 = base.action_mean(torch.as_tensor(o55)[None])[0].numpy()

    o1 = one_step_candidate_outcome(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, a0, arm="A", horizon=12)
    o2 = one_step_candidate_outcome(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, a0, arm="A", horizon=12)
    for k in ("max_dwell", "k6", "contain_exit_ct", "exit_ct", "final_dtz", "mean_speed", "applied"):
        assert k in o1
    assert o1["applied"] is True                                          # candidate reached a gate-on first step
    assert o1 == o2                                                       # deterministic (deepcopy templates)
    assert o1["final_dtz"] >= 0.0 and o1["mean_speed"] >= 0.0
    assert isinstance(o1["contain_exit_ct"], int) and o1["contain_exit_ct"] >= 0   # true full-containment (CENTER_TOL) exit
    # arm B physical outcome equals arm A (certifier metrics are arm-independent — only reward differs)
    oB = one_step_candidate_outcome(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, a0, arm="B", horizon=12)
    assert oB["max_dwell"] == o1["max_dwell"] and oB["k6"] == o1["k6"]
    assert oB["contain_exit_ct"] == o1["contain_exit_ct"] and oB["exit_ct"] == o1["exit_ct"]
    # a large one-step nudge is allowed to change the outcome, but must still apply exactly once (no crash, applied True)
    big = np.clip(a0 + np.array([3.9, -3.9, 3.9, -3.9], np.float32), -4, 4)
    ob = one_step_candidate_outcome(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, big, arm="A", horizon=12)
    assert ob["applied"] is True


def _traces_equal(t1, t2):
    if len(t1) != len(t2):
        return False
    for (o1, a1, s1, d1, v1, tm1, tr1), (o2, a2, s2, d2, v2, tm2, tr2) in zip(t1, t2):
        if not (np.array_equal(o1, o2) and np.array_equal(a1, a2) and s1 == s2 and d1 == d2 and v1 == v2 and tm1 == tm2 and tr1 == tr2):
            return False
    return True


def test_deepcopy_and_reconstruct_fidelity_and_order_independence():
    """Safeguard 4: a deepcopy'd handoff reproduces a fresh reconstruct bit-for-bit (obs/action/strict/dtz/speed/
    termination + final simulator state), the zero-δ candidate rollout == the frozen pi_0 rollout, and clone execution
    order does not couple clones."""
    import copy
    import json

    from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
    from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0, one_step_candidate_outcome
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

    d = "experiments/2026_07_22_coin_v3_learning/rl_entry"
    cfg = json.load(open(f"{d}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{d}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = cfg["banks"]["late_dev"]["rows"][0]
    ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])

    def a0_of(rec):
        o55 = np.concatenate([rec.obs.astype(np.float32), np.zeros(7, np.float32)])
        return base.action_mean(torch.as_tensor(o55)[None])[0].numpy()

    def roll(rl, gate, first, hz=15):
        return one_step_candidate_outcome(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, first, arm="A", horizon=hz, capture=True)

    rl0, g0, _h0, rec0 = reconstruct_handoff(pi0, ls, horizon=360); a0 = a0_of(rec0)
    rl1, g1, _h1, rec1 = reconstruct_handoff(pi0, ls, horizon=360)               # independent fresh reconstruct

    o_dc = roll(rl0, g0, a0)                                                     # deepcopy of rl0
    o_fresh = roll(rl1, g1, a0)                                                  # fresh reconstruct
    assert _traces_equal(o_dc["trace"], o_fresh["trace"])                        # deepcopy ≡ fresh reconstruct
    assert np.array_equal(o_dc["final_qpos"], o_fresh["final_qpos"])             # + final simulator state
    assert np.array_equal(o_dc["final_qvel"], o_fresh["final_qvel"])
    assert o_dc["k6"] == o_fresh["k6"] and o_dc["max_dwell"] == o_fresh["max_dwell"]

    # order independence: roll a big-perturbation clone FIRST, then a baseline clone — baseline must match a lone clone
    big = np.clip(a0 + np.array([3.5, -3.5, 3.5, -3.5], np.float32), -4, 4)
    _first = roll(rl0, g0, big)                                                  # consume a clone before A
    o_A = roll(rl0, g0, a0)
    o_C = roll(rl0, g0, a0)                                                      # lone clone, no prior roll
    assert _traces_equal(o_A["trace"], o_C["trace"])                            # prior clone's roll did not couple
    # template itself untouched (one_step_candidate_outcome only mutates the deepcopies it is handed)
    o_again = roll(rl0, g0, a0)
    assert _traces_equal(o_again["trace"], o_dc["trace"])


def test_hierarchical_bootstrap_over_seeds_and_states():
    # per-seed lists of per-state values; None (degenerate) dropped; no single seed dominates
    ci = hierarchical_bootstrap_ci([[0.1, 0.2, None, 0.3], [0.15, 0.25], [0.05, None, 0.2]], stat=np.median)
    assert ci["n_seeds"] == 3 and ci["n_states"] == 7                    # 4→3 (one None) + 2 + 3→2 (one None) = 7
    assert ci["lo"] <= ci["stat"] <= ci["hi"]
    assert hierarchical_bootstrap_ci([[None], []])["n_seeds"] == 0        # all-empty → n_seeds 0, no crash


def test_eps_from_drifts_pre_registered():
    eps, info = eps_from_drifts([0.001, 0.002, 0.004, 0.006, 0.009], cap=0.010)
    assert info["source"] == "empirical-accepted-drift" and info["n_accepted"] == 5
    assert eps == sorted(eps) and len(set(eps)) == len(eps)              # sorted, unique
    assert 0.010 in eps                                                  # trust cap always included (largest safe)
    assert all(e > 0 for e in eps) and max(eps) <= 0.010 + 1e-9          # bounded by the cap
    eps0, info0 = eps_from_drifts([], cap=0.010)                         # no accepted steps → cap fallback
    assert eps0 == [0.01] and info0["source"] == "trust-cap-fallback"


def test_handoff_record_exposes_strict_dtz_speed():
    import json

    from hymeko_rl.coin_delivery.coin_late_start import replay_pi0
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

    d = "experiments/2026_07_22_coin_v3_learning/rl_entry"
    _cfg = json.load(open(f"{d}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{d}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    recs = replay_pi0(pi0, 6200, horizon=120)
    assert all(hasattr(r, "strict") and hasattr(r, "dtz") and hasattr(r, "speed") for r in recs)
    assert all(isinstance(r.strict, int) and r.strict >= 0 for r in recs)
    assert all(r.dtz >= 0.0 and r.speed >= 0.0 for r in recs)
    assert max(r.strict for r in recs) >= 1                              # strict counter climbs during the replay


def test_boundary_panel_is_held_out_and_in_family():
    import importlib.util

    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

    d = "experiments/2026_07_22_coin_v3_learning/rl_entry"
    spec = importlib.util.spec_from_file_location("lrf", f"{d}/coin_local_ranking_fidelity.py")
    lrf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lrf)
    pi0 = load_frozen_clip_actor(f"{d}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    forbidden = set(range(6000, 6089)) | set(range(6100, 6149))
    panel, comp, strict_hist = lrf.build_boundary_panel(pi0, range(6200, 6320), forbidden, want=6)
    assert len(panel) >= 1
    assert all(ls.seed not in forbidden for ls in panel)                 # exclusively held-out
    assert all(ls.family in lrf.FAMS for ls in panel)                    # in the three ID families
    assert all(s in (2, 3, 4, 5) for s in strict_hist)                   # boundary strict only


# ── DIVERGENT_K6_PAIR_RANKING_V1 primitives ──
def test_nstep_return_bootstrap_and_terminal():
    # G = Σ γ^t r_t + γ^K q_boot
    g = nstep_return([1.0, 1.0], q_boot=10.0, gamma=0.5, terminated=False)
    assert abs(g - (1.0 + 0.5 * 1.0 + 0.25 * 10.0)) < 1e-9              # 1 + 0.5 + 2.5 = 4.0
    # terminated during prefix → no bootstrap
    gt = nstep_return([1.0, 1.0], q_boot=10.0, gamma=0.5, terminated=True)
    assert abs(gt - 1.5) < 1e-9
    assert nstep_return([], q_boot=7.0, gamma=0.9) == 7.0               # empty prefix → pure bootstrap


def test_classify_vs_chance_equivalence_aware():
    # CI that merely CONTAINS 0.5 is INCONCLUSIVE, never "defective"
    assert classify_vs_chance({"stat": 0.7, "lo": 0.6, "hi": 0.8}) == "ABOVE"
    assert classify_vs_chance({"stat": 0.3, "lo": 0.2, "hi": 0.45}) == "ANTI"
    assert classify_vs_chance({"stat": 0.55, "lo": 0.3, "hi": 0.8}) == "INCONCLUSIVE"   # wide, straddles 0.5
    assert classify_vs_chance({"stat": 0.5, "lo": 0.47, "hi": 0.53}) == "EQUIVALENT_TO_CHANCE"  # whole CI in band
    assert classify_vs_chance({"stat": None, "lo": None, "hi": None}) == "NO_DATA"


def test_primary_divergence_ignores_fine_tiebreakers():
    base = {"k6": 0, "max_dwell": 3, "contain_exit_ct": 0, "mean_speed": 0.05, "final_dtz": 0.015}
    same_primary = {**base, "mean_speed": 0.09, "final_dtz": 0.019}     # only fine tiebreakers differ
    assert not primary_divergence(base, same_primary)
    assert primary_divergence(base, {**base, "k6": 1})                  # K6 differs
    assert primary_divergence(base, {**base, "max_dwell": 4})           # dwell by ≥1
    assert not primary_divergence(base, {**base, "max_dwell": 3})       # same dwell
    assert primary_divergence(base, {**base, "contain_exit_ct": 2})     # containment-exit flip


def test_lex_key_certificate_priority():
    # K6 dominates everything; then dwell; then fewer containment exits
    k6 = {"k6": 1, "max_dwell": 3, "contain_exit_ct": 5, "mean_speed": 9.0, "final_dtz": 9.0}
    nok6 = {"k6": 0, "max_dwell": 6, "contain_exit_ct": 0, "mean_speed": 0.0, "final_dtz": 0.0}
    assert lex_better(k6, nok6)                                         # K6 wins despite worse everything else
    a = {"k6": 0, "max_dwell": 4, "contain_exit_ct": 0, "mean_speed": 0.1, "final_dtz": 0.02}
    b = {"k6": 0, "max_dwell": 3, "contain_exit_ct": 0, "mean_speed": 0.0, "final_dtz": 0.0}
    assert lex_better(a, b)                                             # more dwell wins over slower/closer
    c = {**a, "max_dwell": 4, "contain_exit_ct": 1}
    assert lex_better(a, c) and lex_key(a) > lex_key(c)                 # same dwell, fewer containment exits wins


def test_prefix_candidate_rollout_captures_both_rewards_and_bootstrap():
    import copy
    import json

    from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
    from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0, prefix_candidate_rollout
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

    d = "experiments/2026_07_22_coin_v3_learning/rl_entry"
    cfg = json.load(open(f"{d}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{d}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = cfg["banks"]["late_dev"]["rows"][0]
    ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
    rl, gate, _h, _rec = reconstruct_handoff(pi0, ls, horizon=360)
    off = np.array([0.05, -0.05, 0.05, -0.05], np.float32)
    o = prefix_candidate_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, off, 4, horizon=20)
    assert o["k_applied"] <= 4 and len(o["rewardsA"]) == o["k_applied"] == len(o["rewardsB"])   # both arms, len ≤ K
    for k in ("max_dwell", "k6", "contain_exit_ct", "final_dtz", "mean_speed"):
        assert k in o["outcome"]
    if not o["terminated_in_prefix"]:
        assert o["obs55_K"] is not None and o["obs55_K"].shape == (55,)                         # s_K for bootstrap
    # zero offset ⇒ the prefix is exactly pi_0, deterministic across deepcopies
    z1 = prefix_candidate_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, np.zeros(4, np.float32), 4, horizon=20)
    z2 = prefix_candidate_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, np.zeros(4, np.float32), 4, horizon=20)
    assert z1["outcome"] == z2["outcome"] and z1["rewardsA"] == z2["rewardsA"]
