from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.global_pool_entropy_toys import TrainConfig, make_dataset, run_suite  # noqa: E402


def test_generated_toy_shapes():
    x_train, y_train, x_test, y_test = make_dataset(
        "moons",
        n_train=12,
        n_test=8,
        n_points=16,
        seed=7,
    )
    assert x_train.shape == (12, 16, 2)
    assert y_train.shape == (12,)
    assert x_test.shape == (8, 16, 2)
    assert y_test.shape == (8,)


def test_global_pool_entropy_suite_smoke():
    result = run_suite(
        tasks=["moons"],
        n_train=32,
        n_test=16,
        n_points=16,
        cfg=TrainConfig(epochs=2, hidden=8, batch_size=16),
        seed=5,
    )
    row = result["models"]["moons"]
    assert set(row) == {"baseline", "entropy_feedback"}
    for model_row in row.values():
        assert 0.0 <= model_row["test"]["acc"] <= 1.0
        assert model_row["forward_us"]["median_us_per_sample"] > 0.0
        assert model_row["n_params"] > 0
