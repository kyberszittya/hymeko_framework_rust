"""Framework option-RL engine tests: task-independent unit contracts + a SYNTHETIC (non-coin) OptionEnv trained end-to-end
by `train_semi_mdp` — proving the engine carries no task specifics (no coin import anywhere in this file)."""
import numpy as np
import torch

from hymeko_rl.option_rl import (
    FixedBudgetSearch,
    OptionReplayBuffer,
    OptionTransition,
    SelectedActionProvenance,
    SkillRoute,
    across_seed_summary,
    make_actor,
    paired_final_score,
    preregistered_select,
    smdp_target,
    train_semi_mdp,
)
from hymeko_rl.option_rl.agents import SemiMDPConfig


# ---------------- unit contracts ----------------
def test_smdp_target_gamma_tau():
    assert abs(smdp_target(2.0, 0.9, 4.0, 0.0, 5.0) - (2 + 0.9 ** 4 * 5)) < 1e-6
    assert abs(smdp_target(2.0, 0.9, 4.0, 0.0, 5.0) - (2 + 0.9 * 5)) > 1e-6      # NOT one-step
    assert abs(smdp_target(2.0, 0.9, 9.0, 1.0, 5.0) - 2.0) < 1e-6                # terminal: no bootstrap
    r = torch.tensor([2.0, 2.0]); tau = torch.tensor([1.0, 3.0]); dn = torch.tensor([0.0, 0.0]); qn = torch.tensor([5.0, 5.0])
    assert torch.allclose(smdp_target(r, 0.9, tau, dn, qn), torch.tensor([2 + 0.9 * 5, 2 + 0.9 ** 3 * 5]), atol=1e-5)


def test_replay_evict_and_sample():
    rb = OptionReplayBuffer(cap=3)
    for i in range(6):
        rb.add(OptionTransition(np.full(4, i, np.float32), np.ones(2, np.float32), float(i), 2.0, np.zeros(4, np.float32), 0.0, "handoff", {"k6": i}))
    assert len(rb) == 3 and rb.provenance[-1]["k6"] == 5                          # FIFO eviction, provenance kept
    s, a, r, tau, s2, term = rb.sample(4, np.random.default_rng(0))
    assert s.shape == (4, 4) and a.shape == (4, 2) and tau.shape == (4,)


def test_fixed_budget_search_picks_argmax_and_keeps_provenance():
    class Gen:
        def sample(self, center, n, rng):
            return np.stack([center + i for i in range(n)]).astype(np.float32)

    class Scorer:
        def score(self, cand, rng):
            return float(-abs(cand[0] - 2.0)), {"cand0": float(cand[0])}          # best when cand[0] closest to 2

    fbs = FixedBudgetSearch(Gen(), Scorer(), budget=5)
    prov = fbs.select(np.zeros(2, np.float32), np.random.default_rng(0))
    assert isinstance(prov, SelectedActionProvenance)
    assert abs(prov.selected[0] - 2.0) < 1e-6                                     # argmax-score candidate
    assert not np.array_equal(prov.center, prov.selected)                        # center (Bellman action) ≠ selected (provenance)
    fbs0 = FixedBudgetSearch(Gen(), Scorer(), budget=0)
    assert np.array_equal(fbs0.select(np.ones(2, np.float32), np.random.default_rng(0)).selected, np.ones(2))  # b=0 = center direct


def test_preregistered_select_and_paired_final():
    win, scored = preregistered_select([
        {"name": "up", "rl_dev": [0, 0, 0, 0], "base_dev": [0, 0, 0, 0], "exit_dev": 0.0},
        {"name": "rl", "rl_dev": [1, 1, 1, 0], "base_dev": [0, 0, 0, 0], "exit_dev": 0.1},
    ])
    assert win["name"] == "rl" and win["dev_delta"] > 0
    fin = paired_final_score([1, 1, 0], [0, 0, 0])
    assert fin["final_delta"] > 0 and fin["solved_rl"] == [0, 1]
    summ = across_seed_summary([0.2, 0.1, -0.05], [0.05, -0.1, -0.2])
    assert summ["n_seeds"] == 3 and summ["seeds_ci_lower_gt0"] == 1


def test_skill_route_downstream_frozen_on_handoff():
    r = SkillRoute("carry_to_settle", handed_off=lambda o: o["reached_handoff"] == 1, downstream=object())
    assert r.route({"reached_handoff": 1}) == "downstream_frozen_skill"
    assert r.route({"reached_handoff": 0}) == "upstream_option_redecide"


# ---------------- SYNTHETIC non-coin env: the task-independence proof ----------------
class ToyReachEnv:
    """A trivial option env with NO task specifics: obs=[pos2,target2]; action∈[-1,1]^2 is the committed move (one option);
    reward = −dist_after + 10·reached; terminal on reach or budget. Solvable by action≈target−pos."""

    OBS, ACT = 4, 2

    def __init__(self, seed=0, max_options=3):
        self.rng = np.random.default_rng(seed); self.max_options = max_options

    def reset(self, idx=None):
        self.pos = self.rng.uniform(-0.5, 0.5, 2).astype(np.float32)
        self.target = self.rng.uniform(-0.5, 0.5, 2).astype(np.float32); self.n = 0
        return np.concatenate([self.pos, self.target])

    def step(self, action, *, search_seed=None):
        self.pos = np.clip(self.pos + np.clip(action, -1, 1).astype(np.float32), -1.5, 1.5)
        dist = float(np.linalg.norm(self.pos - self.target)); reached = dist < 0.3; self.n += 1
        done = reached or self.n >= self.max_options
        reward = -dist + (10.0 if reached else 0.0)
        info = {"tau": 1.0, "k6": int(reached), "reached_handoff": int(reached), "terminal": float(reached), "contain_exit_ct": 0}
        return np.concatenate([self.pos, self.target]), reward, done, info


def _toy_dev_eval(actor, n=32, seed=999):
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(n):
        pos = rng.uniform(-0.5, 0.5, 2).astype(np.float32); target = rng.uniform(-0.5, 0.5, 2).astype(np.float32)
        with torch.no_grad():
            a = actor.mean_action(torch.as_tensor(np.concatenate([pos, target])[None]).float())[0].numpy()
        ok += int(np.linalg.norm(np.clip(pos + np.clip(a, -1, 1), -1.5, 1.5) - target) < 0.3)
    return ok / n, {}


def test_engine_trains_a_synthetic_non_coin_task():
    torch.manual_seed(0)
    env = ToyReachEnv(seed=1)
    actor = make_actor("sac", ToyReachEnv.OBS, ToyReachEnv.ACT)
    cfg = SemiMDPConfig(warmup_options=30, total_options=400, eval_every=100, batch=64, reward_scale=1.0)
    ckpts, hist = train_semi_mdp("sac", env, actor, _toy_dev_eval, cfg, obs_dim=ToyReachEnv.OBS, act_dim=ToyReachEnv.ACT, log=lambda *a: None, seed=0)
    assert set(ckpts) >= {"update0", "early", "mid", "best_val", "final"} and len(hist) >= 3
    best = make_actor("sac", ToyReachEnv.OBS, ToyReachEnv.ACT); best.load_state_dict(ckpts["best_val"])
    final_score = _toy_dev_eval(best)[0]
    assert final_score > 0.6                                                      # the engine LEARNS a non-coin task end-to-end
