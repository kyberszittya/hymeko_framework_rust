"""Tests for the declarative explore/exploit strategy (hymeko_rl/strategy_spec.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

from hymeko_rl.agents.strategy_spec import StrategySpec

_STRATEGY = "data/robotics/galambos_strategy.hymeko"


def test_strategy_spec_parses_galambos_strategy() -> None:
    s = StrategySpec.from_hymeko(_STRATEGY)
    # exploration tactic
    assert s.ent_coef == 0.01 and s.log_std_init == -0.5 and s.curriculum_iters == 200
    # exploitation
    assert (s.gamma, s.lam, s.clip, s.lr) == (0.99, 0.95, 0.2, 0.0003)
    assert (s.update_epochs, s.n_steps, s.n_iters) == (8, 512, 300)


def test_strategy_spec_builds_ppo_config() -> None:
    s = StrategySpec.from_hymeko(_STRATEGY)
    cfg = s.ppo_config(seed=7)
    assert cfg.n_iters == 300 and cfg.n_steps == 512 and cfg.ent_coef == 0.01
    assert cfg.gamma == 0.99 and cfg.lr == 0.0003 and cfg.seed == 7


def test_strategy_spec_missing_term_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad_strategy.hymeko"
    bad.write_text(
        "bad_description {\n  @\"../../data/robotics/meta_strategy.hymeko\";\n  using strategy as s;\n}\n"
        "bad: s {\n  @strategy_spec: s.strategy_spec { (+ nope); }\n}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        StrategySpec.from_hymeko(bad)
