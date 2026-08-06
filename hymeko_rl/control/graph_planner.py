"""Graph-based explicit-planning pre-trainer — plan the solution by search, then clone the policy onto it.

The *plan-then-amortise* line ([[project-planner-demos-imitation]]): a planner solves the (kinematic) task by
**explicit search over a graph** — here an 8-connected workspace-occupancy grid, solved with A* — producing a
waypoint path a proportional controller follows. The resulting trajectory **pre-trains the policy via behaviour
cloning**, replacing (or seeding) the hand-scripted demonstrator with a *derived* plan that is obstacle-aware and
re-planned per episode.

Honest limit (measured precedent): the planner is **kinematic** — it solves reach/transport, not the contact-rich
grasp. So the graph plan lays the highway; RL (TD3+BC, per the registry) handles the last inch. This module is the
engine; it plugs into the simulator ecosystem (:mod:`hymeko_rl.eval.tasks`) via per-env adapters, and into the off-policy
trainers as the warm-start that actually holds (off-policy, unlike PPO).
"""
from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

Cell = "tuple[int, int]"


def astar(start: Any, is_goal: "Callable[[Any], bool]",
          neighbours: "Callable[[Any], Iterable[tuple[Any, float]]]",
          heuristic: "Callable[[Any], float]", *, max_expansions: int = 200_000) -> "list[Any] | None":
    """A* shortest path over an *implicit* graph. ``neighbours(node)`` yields ``(next, edge_cost)``; ``heuristic``
    must be admissible (never over-estimate) for optimality. Returns the node path ``start..goal`` or ``None`` if
    no goal is reachable within ``max_expansions``.

    # Preconditions edge costs ``>= 0``; ``heuristic(node) >= 0``. # Postconditions the returned path starts at
    ``start`` and ends at a node satisfying ``is_goal``; consecutive nodes are ``neighbours``-adjacent."""
    counter = 0                                                  # tie-break so heapq never compares the nodes
    frontier: "list[tuple[float, int, Any]]" = [(heuristic(start), counter, start)]
    came: "dict[Any, Any]" = {}
    g: "dict[Any, float]" = {start: 0.0}
    expansions = 0
    closed: "set[Any]" = set()
    while frontier and expansions < max_expansions:
        _, _, cur = heapq.heappop(frontier)
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if is_goal(cur):
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        cur_g = g[cur]
        for nxt, cost in neighbours(cur):
            ng = cur_g + cost
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                counter += 1
                heapq.heappush(frontier, (ng + heuristic(nxt), counter, nxt))
    return None


class GraphPlanner(Protocol):
    """A planner used as a deterministic policy: ``reset`` re-plans for the new episode; ``act`` returns the next
    action that follows the explicit plan. Drop-in as the demonstrator in :func:`pretrain_from_planner`."""

    def reset(self, env: Any) -> None: ...
    def act(self, env: Any, obs: np.ndarray) -> np.ndarray: ...


_DIAG = 2.0 ** 0.5


@dataclass
class GridAStarPlanner:
    """A* over a 2-D workspace occupancy grid to a goal point; a proportional controller drives the controllable
    point along the planned waypoints. **Graph-based explicit planning** (8-connected grid, Euclidean heuristic),
    re-planned each episode — distinct from a straight-line scripted reach because it *searches around obstacles*.

    Task-agnostic via injected adapters: ``goal_xy(env)`` the target, ``pos_xy(env)`` the controllable point,
    ``to_action(env, target_xy)`` mapping a workspace target to the env action (identity for a Cartesian env, the
    arm IK for a jointed one). ``blocked(xy)`` marks occupied cells (default: free space).

    # Preconditions ``bounds`` is ``(x_lo, x_hi, y_lo, y_hi)`` with ``x_hi > x_lo``, ``y_hi > y_lo``;
      ``resolution >= 2``. # Invariants ``act`` advances through the path monotonically; never indexes past the end.
    """

    goal_xy: "Callable[[Any], np.ndarray]"
    pos_xy: "Callable[[Any], np.ndarray]"
    to_action: "Callable[[Any, np.ndarray], np.ndarray]"
    bounds: "tuple[float, float, float, float]"
    resolution: int = 24
    blocked: "Callable[[np.ndarray], bool] | None" = None
    _path: "list[np.ndarray]" = field(default_factory=list, init=False, repr=False)
    _wp: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        x_lo, x_hi, y_lo, y_hi = self.bounds
        if not (x_hi > x_lo and y_hi > y_lo) or self.resolution < 2:
            raise ValueError("require x_hi>x_lo, y_hi>y_lo, resolution>=2")
        self._cell_w = (x_hi - x_lo) / self.resolution
        self._cell_h = (y_hi - y_lo) / self.resolution

    def _to_cell(self, xy: np.ndarray) -> "tuple[int, int]":
        x_lo, _, y_lo, _ = self.bounds
        i = int(np.clip((xy[0] - x_lo) / self._cell_w, 0, self.resolution - 1))
        j = int(np.clip((xy[1] - y_lo) / self._cell_h, 0, self.resolution - 1))
        return i, j

    def _to_xy(self, cell: "tuple[int, int]") -> np.ndarray:
        x_lo, _, y_lo, _ = self.bounds
        return np.array([x_lo + (cell[0] + 0.5) * self._cell_w, y_lo + (cell[1] + 0.5) * self._cell_h],
                        dtype=np.float32)

    def reset(self, env: Any) -> None:
        goal_cell = self._to_cell(self.goal_xy(env))
        start_cell = self._to_cell(self.pos_xy(env))
        res = self.resolution

        def neighbours(c: "tuple[int, int]") -> "Iterable[tuple[tuple[int, int], float]]":
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nb = (c[0] + di, c[1] + dj)
                if 0 <= nb[0] < res and 0 <= nb[1] < res:
                    if self.blocked is None or not self.blocked(self._to_xy(nb)):
                        yield nb, (1.0 if di == 0 or dj == 0 else _DIAG)

        def h(c: "tuple[int, int]") -> float:
            return float(np.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1]))

        cells = astar(start_cell, lambda c: c == goal_cell, neighbours, h)
        self._path = [self._to_xy(c) for c in cells] if cells else [self.goal_xy(env).astype(np.float32)]
        self._wp = 0

    def act(self, env: Any, _obs: np.ndarray) -> np.ndarray:
        pos = self.pos_xy(env)
        reach = 0.6 * float(np.hypot(self._cell_w, self._cell_h))   # advance once within ~one cell of the waypoint
        while self._wp < len(self._path) - 1 and float(np.hypot(*(pos - self._path[self._wp]))) < reach:
            self._wp += 1
        return np.asarray(self.to_action(env, self._path[self._wp]), dtype=np.float32)


def pretrain_from_planner(policy: Any, env: Any, planner: GraphPlanner, *, n_episodes: int = 16,
                          bc_epochs: int = 120, seed: int = 0) -> "dict[str, int]":
    """The graph-planning PRE-TRAINER: roll the explicit ``planner`` to collect demonstrations, then behaviour-clone
    ``policy`` onto them. The off-policy-friendly warm-start (clone first, then let TD3+BC refine the contact).

    # Preconditions ``policy`` exposes ``action_mean`` (cloned via :func:`hymeko_rl.train.bc.behaviour_clone`); ``env`` is
    a gym 5-tuple env with ``max_steps``. # Postconditions ``policy`` is BC-fit to the planner; returns demo stats.
    """
    from hymeko_rl.train.bc import behaviour_clone
    obs_all: "list[np.ndarray]" = []
    act_all: "list[np.ndarray]" = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        planner.reset(env)
        for _ in range(env.max_steps):
            action = planner.act(env, obs)
            obs_all.append(np.asarray(obs, dtype=np.float32))
            act_all.append(action)
            obs, _r, term, trunc, _info = env.step(action)
            if term or trunc:
                break
    obs_arr = np.asarray(obs_all, dtype=np.float32)
    act_arr = np.asarray(act_all, dtype=np.float32)
    behaviour_clone(policy, obs_arr, act_arr, n_epochs=bc_epochs, seed=seed)
    return {"demo_steps": int(obs_arr.shape[0]), "episodes": n_episodes}
