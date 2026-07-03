"""Graph-based explicit-planning pre-trainer: A* search, obstacle routing, and the BC pre-train end-to-end."""
from collections.abc import Iterable
from typing import Any

import numpy as np

from hymeko_rl.control.graph_planner import GridAStarPlanner, astar, pretrain_from_planner


def _grid4(r: int) -> "Any":
    def nb(c: "tuple[int, int]") -> "Iterable[tuple[tuple[int, int], float]]":
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (c[0] + di, c[1] + dj)
            if 0 <= n[0] < r and 0 <= n[1] < r:
                yield n, 1.0
    return nb


def test_astar_finds_shortest_path() -> None:
    goal = (4, 4)
    path = astar((0, 0), lambda c: c == goal, _grid4(5),
                 lambda c: float(abs(c[0] - 4) + abs(c[1] - 4)))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == goal
    assert len(path) == 9                                   # Manhattan distance 8 -> 9 nodes (optimal)


def test_astar_returns_none_when_unreachable() -> None:
    assert astar((0, 0), lambda c: c == (1, 1), lambda _c: iter(()), lambda _c: 0.0) is None


class _FakeEnv:
    max_steps = 100

    def __init__(self, start: "tuple[float, float]", goal: "tuple[float, float]") -> None:
        self.pos = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)


def test_grid_planner_routes_around_a_wall() -> None:
    # A vertical wall x in [0.45, 0.55], y <= 0.7 — the plan must go OVER the top, not through.
    def blocked(xy: np.ndarray) -> bool:
        return bool(0.45 <= xy[0] <= 0.55 and xy[1] <= 0.7)

    p = GridAStarPlanner(goal_xy=lambda e: e.goal, pos_xy=lambda e: e.pos,
                         to_action=lambda e, t: t - e.pos, bounds=(0.0, 1.0, 0.0, 1.0),
                         resolution=20, blocked=blocked)
    p.reset(_FakeEnv((0.1, 0.5), (0.9, 0.5)))
    assert len(p._path) > 2                                 # a real detour, not a straight line
    assert all(not blocked(wp) for wp in p._path)           # never plans through the wall
    assert float(np.hypot(*(p._path[-1] - np.array([0.9, 0.5], np.float32)))) < 0.2   # reaches near the goal


def test_grid_planner_rejects_bad_bounds() -> None:
    base: "dict[str, Any]" = {"goal_xy": lambda e: e.goal, "pos_xy": lambda e: e.pos,
                              "to_action": lambda e, t: t}
    bad_cases: "list[dict[str, Any]]" = [
        {"bounds": (0.0, 0.0, 0.0, 1.0), "resolution": 20},   # degenerate x-extent
        {"bounds": (0.0, 1.0, 0.0, 1.0), "resolution": 1},    # resolution < 2
    ]
    for bad in bad_cases:
        try:
            GridAStarPlanner(**base, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


class _PointEnv:
    """A point-mass: the action is a velocity, so to_action=identity and the planner drives it to the goal."""
    max_steps = 40

    def reset(self, *, seed: int = 0, options: Any = None) -> "tuple[np.ndarray, dict[str, Any]]":
        self.pos = np.array([0.1, 0.1], dtype=np.float32)
        self.goal = np.array([0.8, 0.8], dtype=np.float32)
        return self.pos.copy(), {}

    def step(self, a: np.ndarray) -> "tuple[np.ndarray, float, bool, bool, dict[str, Any]]":
        self.pos = np.clip(self.pos + 0.05 * np.asarray(a, dtype=np.float32), 0.0, 1.0).astype(np.float32)
        done = bool(np.hypot(*(self.pos - self.goal)) < 0.05)
        return self.pos.copy(), 0.0, done, False, {}


def test_pretrain_from_planner_collects_and_clones() -> None:
    from hymeko_rl.agents.policy import build_policy
    env = _PointEnv()
    env.reset()
    ac: Any = build_policy("mlp", obs_dim=2, action_dim=2)
    planner = GridAStarPlanner(goal_xy=lambda e: e.goal, pos_xy=lambda e: e.pos,
                               to_action=lambda e, t: t - e.pos, bounds=(0.0, 1.0, 0.0, 1.0), resolution=16)
    stats = pretrain_from_planner(ac, env, planner, n_episodes=4, bc_epochs=40, seed=0)
    assert stats["demo_steps"] > 0 and stats["episodes"] == 4
    import torch
    with torch.no_grad():
        a = ac.action_mean(torch.as_tensor(np.array([[0.1, 0.1]], np.float32))).squeeze(0).numpy()
    assert np.all(np.isfinite(a))                           # the cloned policy produces valid actions
