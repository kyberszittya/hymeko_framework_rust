"""§4 gates — the option-RL run is described in HyMeKo: every load-bearing fact is answerable from the parsed graph
(query gate); the parsed description regenerates the engine runtime config without duplicate topology (round-trip gate); an
existing Stage-5 checkpoint loads and reproduces the update-0 evaluation contract (compat gate); load-bearing invariants are
validated fail-closed."""
import dataclasses
import json

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0
from hymeko_rl.coin_delivery.coin_option_rl_spec import _validate, load_carry_option_rl_run
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
from hymeko_rl.option_rl import smdp_target

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"


def test_query_gate_every_fact_from_the_graph():
    s = load_carry_option_rl_run()
    q = s.query_dump()
    assert q["proposal_checkpoint"] == "carry_proposal_refined"
    assert q["search_budget"] == 8
    assert q["candidate_generator"] == "structured_random_around" and q["candidate_scorer"] == "structured_score"
    assert q["trainable_skill"] == "carry_option" and q["frozen_skill"] == "settling_pi0"
    assert q["handoff_certificate"] == "stable_entry_v1"
    assert q["bellman_action"] == "theta_center" and q["selected_action_role"] == "provenance"
    assert q["uses_gamma_tau"] is True and q["terminal_bootstraps"] is False
    assert q["gamma"] == 0.99
    assert q["trainer_primary"] == "sac" and q["trainer_control"] == "td3"
    assert q["physical_metric"] == "held_out_k6_paired" and q["selects_on_critic_q_alone"] is False
    assert q["manifests"] == {"train": (9000, 10800), "dev": (11000, 12200), "final": (12200, 13600)}
    assert q["certificate_tolerances"]["center_tol"] == 0.02 and q["certificate_tolerances"]["held_dwell"] == 6


def test_round_trip_regenerates_engine_config():
    s = load_carry_option_rl_run()
    cfg = s.to_runtime_config()
    # the runtime config the existing hymeko_rl/option_rl engine consumes — b and γ come from the description
    assert cfg.b == 8 and cfg.gamma == 0.99
    # the declared γ^τ target is exactly what the engine's smdp_target computes (non-terminal bootstraps by γ^τ, terminal not)
    assert s.uses_gamma_tau() and not s.terminal_bootstraps()
    assert abs(smdp_target(1.0, cfg.gamma, 4.0, 0.0, 5.0) - (1 + cfg.gamma ** 4 * 5)) < 1e-6
    assert abs(smdp_target(1.0, cfg.gamma, 4.0, 1.0, 5.0) - 1.0) < 1e-6


def test_checkpoint_compat_reproduces_update0_eval_contract():
    s = load_carry_option_rl_run()
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    prop = load_proposal(s.resolve_checkpoint(D))                         # the Stage-5 proposal checkpoint named in the .hymeko
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    torch.manual_seed(0)
    for row in cfg["banks"]["late_dev"]["rows"][:2]:
        ls = LateStart(seed=row[0], prefix_steps=row[1], family=row[2], obs_sha=row[3], base_sha=row[4], causal_sha=row[5])
        rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        # the update-0 eval contract: proposal center → FIXED b-search (budget from the .hymeko) → K6 verdict
        _th, out = search_select(rl, gate, prop.theta(rl.obs()), pi0, base, np.random.default_rng(0), b=s.search_budget(), horizon=160)
        assert "k6" in out and out["k6"] in (0, 1)


def test_fail_closed_on_broken_load_bearing_semantics():
    import pytest
    s = load_carry_option_rl_run()

    def _mutated(member, key, val):
        m = {k: dict(v) for k, v in s.members.items()}; m[member][key] = val
        return dataclasses.replace(s, members=m)

    with pytest.raises(ValueError):                                      # Bellman action must be the proposal center
        _validate(_mutated("option", "bellman_action", "theta_selected"), "<t>")
    with pytest.raises(ValueError):                                      # selected candidate must not be the trained action
        _validate(_mutated("provenance", "is_trained_action", "1"), "<t>")
    with pytest.raises(ValueError):                                      # terminal transitions must not bootstrap
        _validate(_mutated("target", "terminal_bootstrap", "1"), "<t>")
    with pytest.raises(ValueError):                                      # selection must be physical, not critic Q alone
        _validate(_mutated("eval", "critic_q_alone", "1"), "<t>")
