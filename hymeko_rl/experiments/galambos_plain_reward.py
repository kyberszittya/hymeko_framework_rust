"""Galambos coin-toss isolation path with no HyMeKo robot/reward profile.

Purpose: distinguish "the .hymeko reward/spec translation is causing the TD3+BC collapse" from "the
off-policy actor update leaves the contact manifold even when the reward is plain Python." The env still uses
the same MuJoCo planar physics class, demonstrator, BC, and trainer, but:

* ``robot=None`` selects the hand-authored planar arm MJCF instead of emitting a robot from ``.hymeko``;
* ``reward_spec`` is a small Python object with ``evaluate(env, dist, action)``;
* environment geometry uses ``DEFAULT_ENV`` from code, not ``EnvSpec.from_hymeko``.

This module is diagnostic only. Do not cite its numbers as the main Galambos result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.env.env_spec import DEFAULT_ENV
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.eval.evaluate import DwellMetric, eval_metric, greedy_action_fn
from hymeko_rl.experiments.galambos_bc import collect_galambos_demos
from hymeko_rl.train.campaign import Campaign, CampaignConfig


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


@dataclass
class PlainPythonDeliverReward:
    """Plain Python reward for the coin-toss isolation run.

    ``kind="terminal"`` is sparse: reward only on the step that completes the dwell requirement.
    ``kind="dense"`` adds simple geometry/contact shaping with no HyMeKo term parser:
    closer-to-zone is better, in-zone/contact are useful, and large actions are lightly penalized.
    """

    kind: str = "dense"
    terminal: float = 30.0
    zone_weight: float = 2.0
    contact_weight: float = 0.5
    distance_weight: float = 1.0
    action_weight: float = 0.001

    def evaluate(self, env: Any, dist: float, action: np.ndarray) -> float:
        m = env._planar_metrics
        success_steps = int(getattr(env, "success_steps", 1))
        terminal_hit = bool(m.in_zone and int(getattr(env, "_success", 0)) + 1 == success_steps)
        reward = self.terminal if terminal_hit else 0.0
        if self.kind == "terminal":
            return float(reward)
        if self.kind != "dense":
            raise ValueError(f"unknown plain reward kind {self.kind!r}")
        both = bool(m.left_contact and m.right_contact)
        reward += self.zone_weight * float(m.in_zone)
        reward += self.contact_weight * float(both)
        reward -= self.distance_weight * float(dist)
        reward -= self.action_weight * float(np.square(action).mean())
        return float(reward)


def make_plain_env(*, reward_kind: str = "dense", difficulty: float = 0.3,
                   max_steps: int = 300) -> PlanarGraspEnv:
    """Build a coin-toss env without reading HyMeKo robot/reward/env profiles."""
    return PlanarGraspEnv(robot=None, reward_spec=PlainPythonDeliverReward(kind=reward_kind),
                          env=DEFAULT_ENV, max_steps=max_steps, difficulty=difficulty)


def measure_plain_factory(*, reward_kind: str, difficulty: float, n_eval: int) -> Any:
    """Reward-independent delivery/contact measurement for the plain-reward env."""

    def _measure(_make_env: Any, actor: Any) -> dict[str, float]:
        env = make_plain_env(reward_kind=reward_kind, difficulty=difficulty)
        try:
            dwell = int(getattr(env, "success_steps", 1))
            delivery = eval_metric(env, greedy_action_fn(actor), DwellMetric("in_zone", dwell),
                                   n_episodes=n_eval, seed0=9_000)
            both = 0
            steps = 0
            act_fn = greedy_action_fn(actor)
            for ep in range(n_eval):
                obs, _ = env.reset(seed=12_000 + ep)
                for _ in range(env.max_steps):
                    obs, _r, term, trunc, info = env.step(act_fn(env, obs))
                    both += int(bool(info.get("both_contact", False)))
                    steps += 1
                    if term or trunc:
                        break
            return {"delivery": float(sum(delivery)) / max(1, n_eval),
                    "both_contact": float(both) / max(1, steps)}
        finally:
            _close_env(env)

    return _measure


def run_plain_isolation(*, reward_kind: str = "dense", total_steps: int = 5_000,
                        n_demos: int = 40, bc_epochs: int = 40,
                        n_eval: int = 10, seed: int = 0) -> dict[str, Any]:
    """One bounded TD3+BC isolation cell. Defaults are smoke-sized."""

    def env_factory() -> PlanarGraspEnv:
        return make_plain_env(reward_kind=reward_kind)

    cfg = CampaignConfig(name=f"galambos_plain_{reward_kind}_td3bc_{total_steps}_s{seed}",
                         select="delivery", seeds=(seed,), total_steps=total_steps,
                         eval_every=max(1_000, total_steps // 2), n_demos=n_demos,
                         bc_epochs=bc_epochs, n_eval=n_eval, n_envs=8, device="auto")
    return Campaign(
        cfg,
        make_env=env_factory,
        build=lambda env: build_collaborative_offpolicy(env, kind="sa_hsikan", hidden=64),
        measure=measure_plain_factory(reward_kind=reward_kind, difficulty=0.3, n_eval=n_eval),
        demos=lambda env, n, s: collect_galambos_demos(env, n, s),
        gif=False,
    ).run()


def main() -> int:
    res = run_plain_isolation()
    print("PLAIN_REWARD_RESULT", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
