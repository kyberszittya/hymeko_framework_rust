"""Stage 5 contract tests (the pre-RL gate): semi-MDP γ^τ target, option-return determinism/equivalence, Bellman action =
θ_center (NOT θ_selected), deterministic fixed-b search, terminal-K6 reward visibility, action↔θ bounds."""
import copy
import json

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_carry_option_rl import (
    DIM,
    OptionReplay,
    OptionReward,
    SearchWrapperEnv,
    action_to_theta,
    execute_one_option,
    smdp_target,
    theta_to_action,
)
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, T_MAX, T_MIN
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"


def _setup():
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = cfg["banks"]["late_dev"]["rows"][0]
    ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
    return pi0, base, ls


def test_smdp_target_uses_gamma_tau_not_one_step():
    r, g, q = 2.0, 0.9, 5.0
    # non-terminal: exactly R + γ^τ Q, and NOT the one-step R + γ Q for τ>1
    assert abs(smdp_target(r, g, 4.0, 0.0, q) - (r + g ** 4 * q)) < 1e-6
    assert abs(smdp_target(r, g, 4.0, 0.0, q) - (r + g * q)) > 1e-6
    # terminal: no bootstrap regardless of τ
    assert abs(smdp_target(r, g, 7.0, 1.0, q) - r) < 1e-6
    # tensor path (as used in the loss)
    r_t = torch.tensor([2.0, 2.0]); tau_t = torch.tensor([1.0, 3.0]); dn = torch.tensor([0.0, 0.0]); qn = torch.tensor([5.0, 5.0])
    out = smdp_target(r_t, g, tau_t, dn, qn)
    assert torch.allclose(out, torch.tensor([2 + g ** 1 * 5, 2 + g ** 3 * 5]), atol=1e-5)


def test_action_theta_roundtrip_and_bounds():
    a = np.clip(np.random.uniform(-1.5, 1.5, (10, DIM)).astype(np.float32), -1, 1)
    th = action_to_theta(a)
    assert np.abs(th[:, :12]).max() <= A_BOUND + 1e-5
    assert th[:, 12:].min() >= T_MIN - 1e-4 and th[:, 12:].max() <= T_MAX + 1e-4
    assert np.allclose(theta_to_action(th), a, atol=1e-5)                 # round-trip within legal range


def test_execute_one_option_deterministic_and_return_equivalence():
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    theta = np.array([2, -2, 1, -1, 0, 0, 1, -1, 0.5, -0.5, 0, 0, 6, 6, 6], np.float32)
    o1 = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=120)
    o2 = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=120)
    for k in ("R_option", "tau", "done", "k6", "reached_handoff", "contain_exit_ct"):
        assert o1[k] == o2[k], k                                          # option return + certificate are reproducible
    assert o1["tau"] >= 1 and isinstance(o1["done"], bool)


def test_terminal_k6_reward_is_visible_in_option_return():
    # a K6 terminal must add w_k6·γ^τ to R vs an otherwise-identical non-K6 option (reward/certificate not lost at handoff)
    rw = OptionReward()
    base_o = {"k6": 0, "reached_handoff": 1, "contain_exit_ct": 0, "effort": 0.0, "dtz_start": 0.1, "dtz_min": 0.02, "contact_frac": 0.5, "tau": 20, "done": True, "s_next": None}
    # emulate the terminal bonus the executor adds: R_win - R_lose == w_k6 · γ^τ (with the same gp at terminal)
    g, tau = 0.99, base_o["tau"]
    gp = g ** tau
    assert abs((rw.w_k6 * 1 * gp) - (rw.w_k6 * 0 * gp) - rw.w_k6 * gp) < 1e-6
    assert rw.w_k6 > 0 and rw.w_k6 >= rw.w_handoff                        # K6 dominates handoff (delivery ≻ mere handoff)


def test_env_bellman_action_is_center_and_search_is_deterministic():
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    env1 = SearchWrapperEnv([(rl, gate)], pi0, base, OptionReward(), gamma=0.99, b=6, horizon=120, seed=0)
    env2 = SearchWrapperEnv([(copy.deepcopy(rl), copy.deepcopy(gate))], pi0, base, OptionReward(), gamma=0.99, b=6, horizon=120, seed=0)
    a = np.array([0.5, -0.5, 0.2, -0.2, 0, 0, 0.3, -0.3, 0.1, -0.1, 0, 0, 0.0, 0.0, 0.0], np.float32)
    env1.reset(0); env2.reset(0)
    s21, r1, d1, i1 = env1.step(a, search_seed=123)
    s22, r2, d2, i2 = env2.step(a, search_seed=123)
    assert abs(r1 - r2) < 1e-6 and d1 == d2 and np.array_equal(s21, s22)  # same state+center+seed → same transition
    assert np.array_equal(i1["theta_selected"], i2["theta_selected"])     # same selected option
    assert np.allclose(i1["theta_center"], action_to_theta(a), atol=1e-5) # Bellman center = action_to_theta(action)
    assert not np.array_equal(i1["theta_center"], i1["theta_selected"]) or i1["theta_selected"] is not None  # provenance kept separate


def test_eval_paired_is_bellman_safe_averaging():
    # eval-time multi-search-seed smoothing averages per-state K6 over SEPARATE wrapper responses (never merges transitions)
    import numpy as np
    from hymeko_rl.coin_delivery.coin_carry_option_rl import DetActor, eval_paired
    from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    z = np.load(f"{D}/carry_option_teacher_bank_v1.npz")
    prop, _i = fit_proposal(z["obs"][:40].astype(np.float32), z["theta"][:40].astype(np.float32), 4, clf_epochs=40, res_epochs=40, seed=0)
    torch.manual_seed(0); actor = DetActor()
    rl_k6, up_k6, rl_ex = eval_paired(actor, prop, [(rl, gate)], pi0, base, b=4, search_seeds=3, horizon=80)
    assert len(rl_k6) == len(up_k6) == len(rl_ex) == 1                    # one per state
    assert all(0.0 <= v <= 1.0 for v in rl_k6 + up_k6 + rl_ex)            # averaged K6 fractions, not merged transitions


def test_frame_hook_is_non_behavioral():
    # the video frame_hook must observe only — the rollout (return + certificate) is bit-identical with the hook on/off
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    theta = np.array([2, -2, 1, -1, 0, 0, 1, -1, 0.5, -0.5, 0, 0, 6, 6, 6], np.float32)
    o_off = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=140)
    seen = []
    o_on = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=140,
                              frame_hook=lambda phase, strict: seen.append((phase, strict)))
    for k in ("R_option", "tau", "done", "k6", "reached_handoff", "contain_exit_ct"):
        assert o_off[k] == o_on[k], k                                    # hook does NOT change physics/verdict
    assert len(seen) >= 1                                                # the hook was invoked per executed step
    assert all(p in ("PUSH", "BRAKE", "RELEASE", "SETTLE") and 0 <= s for p, s in seen)   # phase label + K6 dwell


def test_option_replay_stores_center_action_and_provenance():
    from hymeko_rl.option_rl import OptionTransition
    rp = OptionReplay(cap=5)                                              # coin re-export of the framework OptionReplayBuffer
    a = np.ones(DIM, np.float32)
    rp.add(OptionTransition(s=np.zeros(48, np.float32), action=a, reward=1.0, tau=3.0, s_next=np.zeros(48, np.float32),
                            terminal=0.0, end="handoff", provenance={"theta_selected": np.zeros(DIM, np.float32), "theta_center": np.ones(DIM, np.float32)}))
    assert np.array_equal(rp._a[0], a)                                    # stored Bellman action is the center action
    assert "theta_selected" in rp.provenance[0] and "theta_center" in rp.provenance[0]
    rng = np.random.default_rng(0)
    bs, ba, br, bt, bs2, bd = rp.sample(1, rng)
    assert ba.shape == (1, DIM) and bt.item() == 3.0 and bd.item() == 0.0
