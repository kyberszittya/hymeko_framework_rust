"""The RL training STRATEGY as a declarative spec — the in-memory form of the
``meta_strategy.hymeko`` vocabulary (the algorithm half of the description).

A :class:`StrategySpec` is read from a ``.hymeko`` strategy profile's ``strategy_spec`` bundle —
the PPO **exploration** tactic (entropy bonus, action-noise scale, start-state curriculum) and
**exploitation** knobs (gamma, lam, clip, lr, epochs, rollout/iteration counts). So tuning how the
agent explores is an edit to a ``.hymeko``, not the Python. Same reader (`read_bundle`) and shape as
:class:`hymeko_rl.env.env_spec.EnvSpec`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hymeko_rl.env._profile import read_bundle
from hymeko_rl.ppo import PPOConfig


def _num(body: str, name: str, default: float) -> float:
    m = re.search(rf"\b{name}\s+(-?[\d.]+(?:[eE][-+]?\d+)?)", body)
    return float(m.group(1)) if m else default


@dataclass(frozen=True)
class StrategySpec:
    """Declarative PPO explore/exploit strategy (defaults = the planar-grasp strategy).

    # Preconditions (``from_hymeko``) the profile declares one ``strategy_spec`` bundling an
    ``exploration`` and an ``exploitation`` config term.
    # Postconditions all fields finite; ``ppo_config`` yields a valid :class:`PPOConfig`.
    """

    # exploration
    ent_coef: float = 0.005
    log_std_init: float = -1.6
    curriculum_iters: int = 0
    # exploitation
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 8
    value_warmup: int = 0
    n_steps: int = 512
    n_iters: int = 150

    def ppo_config(self, *, seed: int = 0) -> PPOConfig:
        """The :class:`PPOConfig` for the exploitation/optimisation half (+ the entropy bonus)."""
        return PPOConfig(
            n_iters=self.n_iters, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
            clip=self.clip, lr=self.lr, update_epochs=self.update_epochs,
            ent_coef=self.ent_coef, value_warmup=self.value_warmup, seed=seed)

    @classmethod
    def from_hymeko(cls, profile: str | Path) -> "StrategySpec":
        """Build the spec from a ``.hymeko`` strategy profile's ``strategy_spec`` bundle.

        # Errors ``FileNotFoundError``; ``ValueError`` (no/multiple ``strategy_spec``, missing term).
        """
        terms = {kind: body for _name, kind, body in read_bundle(profile, "strategy_spec")}
        missing = {"exploration", "exploitation"} - terms.keys()
        if missing:
            raise ValueError(f"{profile}: strategy_spec missing config term(s) {sorted(missing)}")
        ex, xp = terms["exploration"], terms["exploitation"]
        return cls(
            ent_coef=_num(ex, "ent_coef", 0.005),
            log_std_init=_num(ex, "log_std_init", -1.6),
            curriculum_iters=int(_num(ex, "curriculum_iters", 0.0)),
            gamma=_num(xp, "gamma", 0.99),
            lam=_num(xp, "lam", 0.95),
            clip=_num(xp, "clip", 0.2),
            lr=_num(xp, "lr", 3e-4),
            update_epochs=int(_num(xp, "update_epochs", 8.0)),
            value_warmup=int(_num(xp, "value_warmup", 0.0)),
            n_steps=int(_num(xp, "n_steps", 512.0)),
            n_iters=int(_num(xp, "n_iters", 150.0)),
        )
