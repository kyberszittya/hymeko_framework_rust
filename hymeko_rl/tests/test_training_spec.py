"""Config-parse tests for the declarative training-strategy loader (`TrainingSpec.from_hymeko`).

Pure/fast (no MuJoCo): asserts the DAgger strategy parses from its `.hymeko`, the additive `@dagger` vocab does
not disturb the existing galambos experiment (regression), and an unknown algorithm is rejected.
"""
from __future__ import annotations

import pytest

from hymeko_rl.experiments.training_spec import TrainingSpec


def test_dagger_spec_parses() -> None:
    spec = TrainingSpec.from_hymeko("data/robotics/pick_place_dagger.hymeko")
    assert spec.algorithm == "dagger"
    assert spec.budget["n_demos"] == 18 and spec.budget["bc_epochs"] == 80
    assert spec.budget["seeds"] == (0, 1, 2)
    assert spec.strategy["dagger_iters"] == 4 and spec.strategy["rollouts_per_iter"] == 12
    assert spec.strategy["beta"] == 0.5 and spec.strategy["beta_decay"] == 0.5
    assert spec.strategy["expert_replay_ratio"] == 1.0 and spec.strategy["n_eval"] == 24
    assert spec.strategy["warm_start"] == 1.0
    assert isinstance(spec.budget["n_demos"], int) and isinstance(spec.strategy["dagger_iters"], int)


def test_galambos_experiment_still_parses() -> None:
    """The additive `@dagger` vocab must not disturb the existing galambos td3_bc experiment (regression)."""
    spec = TrainingSpec.from_hymeko("data/robotics/galambos_ab_deliver.hymeko")
    assert spec.algorithm == "bc"                 # galambos declares no `algorithm` field → default
    assert "seeds" in spec.budget
    assert spec.strategy.get("critic_huber") == 1.0   # from its @offpolicy block


def test_unknown_algorithm_raises(tmp_path) -> None:
    p = tmp_path / "bad.hymeko"
    p.write_text(
        'bad: xp, x {\n'
        '    @budget: xp.budget { algorithm "sac"; n_demos 4; }\n'
        '    @experiment_spec: x.experiment_spec { (+ budget); }\n'
        '}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        TrainingSpec.from_hymeko(p)
