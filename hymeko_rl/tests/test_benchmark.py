"""The RL benchmark harness — a cell trains + evals, the sweep builds a leaderboard."""
import numpy as np

from hymeko_rl.eval.benchmark import BenchCell, run_benchmark, train_and_eval


def test_cell_trains_and_scores() -> None:
    score = train_and_eval(BenchCell("cartpole", "td3", "mlp", steps=1000), seed=0, n_eval=2)
    assert isinstance(score, float) and np.isfinite(score)


def test_run_benchmark_leaderboard_rows() -> None:
    cells = [BenchCell("cartpole", "td3", "mlp", steps=800),
             BenchCell("cartpole", "ddpg", "mlp", steps=800)]
    rows = run_benchmark(cells, seeds=[0])
    assert len(rows) == 2
    assert all(set(r) >= {"scenario", "algo", "backbone", "median", "iqr", "scores"} for r in rows)


def test_rejects_bad_algo() -> None:
    try:
        train_and_eval(BenchCell("cartpole", "nope", "mlp"), seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown algo")
