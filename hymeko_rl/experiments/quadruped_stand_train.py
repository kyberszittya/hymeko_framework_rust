"""Aibo STANDING campaign — BC(PD-hold-q0) -> TD3+BC, driven by ``data/robotics/quadruped_stand.hymeko``.

The single-source witness on a real 22-DOF robot: the env (plant + tuning + reward), the demos (PD-hold
demonstrator), and the training strategy (algorithm/budget/off-policy knobs) all come from the one ``.hymeko``.
Reuses the shared :class:`hymeko_rl.train.campaign.Campaign` (BC -> TD3+BC -> best-checkpoint eval -> results.json
+ gif + run.log) — a new experiment is a config plus four closures, not a re-implemented loop (avoids the
`§6.5 #3` scaffold duplication).

Pre-launch REWARD-ALIGNMENT gate (the standing analogue of ``reward_oracle.certify``, which is galambos-specific):
the campaign refuses to launch unless the PD-hold demonstrator (which stands) scores the training reward strictly
above a collapsing zero-policy — i.e. the reward *delivers* standing, it cannot be farmed by not-standing. The
certificate is printed at launch and stored in the run summary.

    python -m hymeko_rl.experiments.quadruped_stand_train --smoke   # 1 seed, short, local sanity
    python -m hymeko_rl.experiments.quadruped_stand_train           # the declared 3-seed x 150k run
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.env.quadruped_env import _DEFAULT_STAND_SCENARIO, QuadrupedGoalEnv
from hymeko_rl.eval.evaluate import DwellMetric, eval_metric, greedy_action_fn
from hymeko_rl.experiments.training_spec import TrainingSpec
from hymeko_rl.train.campaign import Campaign, CampaignConfig
from hymeko_rl.train.ddpg import build_offpolicy

_ARCH = "sa_hsikan"          # the recommended structural backbone for quadruped_stand (tasks.best_arch)
_N_EVAL = 24                 # eval episodes per checkpoint (standing dwell rate)


def _make_env() -> QuadrupedGoalEnv:
    return QuadrupedGoalEnv.from_hymeko()


def _build(env: QuadrupedGoalEnv) -> "tuple[Any, list[Any]]":
    """SA-HSiKAN actor + twin LayerNorm critics over the 22-DOF Aibo's per-vertex obs (12-D leg action)."""
    return build_offpolicy(_ARCH, obs_dim=2, flat_dim=env.hg.n_vertices * 2,
                           action_dim=env.n_actions, action_scale=1.0, n_critics=2,
                           hg_state=env.hg, critic_layernorm=True)


def _measure(make_env: "Any", actor: Any) -> "dict[str, float]":
    """One evaluation -> the standing dwell rate (held upright-at-height >=200 consecutive steps)."""
    vals = eval_metric(make_env(), greedy_action_fn(actor), DwellMetric("standing", 200),
                       n_episodes=_N_EVAL, seed0=7000)
    return {"standing": float(np.mean(vals))}


_DEMO_INIT_NOISE = 0.10    # wider start distribution than the eval env's 0.02 → the expert demonstrates recovery
_DEMO_ACT_NOISE = 0.22     # DART: perturb the APPLIED action so the dog visits off-nominal states, but LABEL


def _demos(env: QuadrupedGoalEnv, n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """DART-style noisy PD-hold demos (Laskey 2017): the standing holding-torques are tiny (near-vertical legs
    bear weight axially), so a clean clone can't reproduce them precisely and drifts. Perturbing the *applied*
    action pushes the dog into deviated states where the expert's correction is large and learnable, and labelling
    with the *clean* :attr:`QuadrupedGoalEnv.expert_action` teaches the recovery feedback law over the off-nominal
    region — fixing both the poor signal-to-noise and the covariate shift of naive BC on a reactive stabiliser.

    # Postconditions ``(obs (M, N, feat), acts (M, n_act))``; labels are the CLEAN expert at each visited state.
    """
    rng = np.random.default_rng(seed)
    prev_noise = env.init_noise
    env.init_noise = _DEMO_INIT_NOISE
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    try:
        for ep in range(n):
            obs, _ = env.reset(seed=seed + ep)
            for _ in range(env.max_steps):
                clean = env.expert_action                       # the LABEL: state -> correct holding torque
                obs_list.append(obs)
                act_list.append(clean)
                applied = np.clip(clean + rng.normal(0.0, _DEMO_ACT_NOISE, clean.shape),
                                  -1.0, 1.0).astype(np.float32)  # explore off-nominal states
                obs, _r, terminated, truncated, _i = env.step(applied)
                if terminated or truncated:
                    break
    finally:
        env.init_noise = prev_noise
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def certify_reward_delivers_standing(n_ep: int = 4) -> "dict[str, Any]":
    """Standing analogue of ``reward_oracle.certify``: roll the PD-hold demonstrator (stands) and a zero-policy
    (collapses); the training reward *delivers* standing iff the demonstrator both stands and out-scores the
    collapse. Empirical (on the real env), not a stub — the reward cannot be farmed by not-standing.

    # Postconditions returns ``{delivers, demo_reward, zero_reward, demo_standing, zero_standing}``.
    """
    env = _make_env()

    def roll(policy: "Any") -> "tuple[float, float]":
        rewards: list[float] = []
        stood = 0
        for ep in range(n_ep):
            env.reset(seed=1000 + ep)
            for _ in range(env.max_steps):
                _o, r, term, trunc, info = env.step(policy(env))
                rewards.append(float(r))
                stood += int(info["standing"])
                if term or trunc:
                    break
        return float(np.mean(rewards)), stood / (n_ep * env.max_steps)

    demo_r, demo_s = roll(lambda e: e.expert_action)
    zero_r, zero_s = roll(lambda e: np.zeros(e.n_actions, np.float32))
    return {"delivers": bool(demo_s > 0.8 and demo_r > zero_r),
            "demo_reward": round(demo_r, 3), "zero_reward": round(zero_r, 3),
            "demo_standing": round(demo_s, 3), "zero_standing": round(zero_s, 3)}


def run_stand_campaign(scenario: "str | Path" = _DEFAULT_STAND_SCENARIO, *,
                       smoke: bool = False, allow_uncertified: bool = False) -> "dict[str, Any]":
    """Read the declared strategy from ``scenario`` and run the BC->TD3+BC standing campaign.

    Refuses to launch on a reward that does not deliver standing (the pre-queue semantic gate) unless
    ``allow_uncertified`` waives it. ``smoke`` shrinks to 1 seed / short budget / few demos for a local sanity run.
    """
    t = TrainingSpec.from_hymeko(scenario)
    cert = certify_reward_delivers_standing()
    print(f"[oracle] standing-reward certificate: {cert}", flush=True)
    if not cert["delivers"] and not allow_uncertified:
        raise RuntimeError(f"reward does not deliver standing (certificate {cert}); refusing to launch "
                           f"(pass allow_uncertified=True to override)")

    seeds = tuple(int(s) for s in t.budget.get("seeds", (0, 1, 2)))
    total = int(t.budget.get("total_steps", 150_000))
    n_demos = int(t.budget.get("n_demos", 200))
    bc_epochs = int(t.budget.get("bc_epochs", 200))
    if smoke:
        seeds, total, n_demos, bc_epochs = (0,), 6_000, 24, 40

    cfg = CampaignConfig(
        name="quadruped_stand", select="standing", seeds=seeds, total_steps=total,
        eval_every=max(1_000, total // 6), n_demos=n_demos, bc_epochs=bc_epochs,
        n_eval=_N_EVAL, n_envs=8, device="auto", offpolicy=dict(t.strategy))
    summary = Campaign(cfg, _make_env, _build, _measure, demos=_demos).run()
    summary["reward_certificate"] = cert
    return summary


if __name__ == "__main__":
    import sys

    run_stand_campaign(smoke=("--smoke" in sys.argv), allow_uncertified=("--force" in sys.argv))
