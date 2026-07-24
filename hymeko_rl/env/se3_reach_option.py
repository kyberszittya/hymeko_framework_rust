"""6D-0 — bind the SE(3) pose-reach task to the FROZEN ``OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1``.

This is the runtime's INTEGRATION test on a real MuJoCo SE(3) task (the ToyReach analogue, but genuine physics):

    SE3ReachEnv → SE3ReachAdapter → StructuredState
                → LSTMTemporalEncoder.update (streaming, hidden = runtime state)
                → fuse_state
                → SE3WaypointProposal (K=1 via SingleModeProposal) → MultimodalBudgetSearch
                → committed option (servo to a joint waypoint for H steps)
                → pose certificate (pos ∧ ang)

The "option" for reach is a committed joint waypoint ``q_des`` held for H steps under position control; the proposal
estimates ``q_des`` by a pure-kinematic damped-least-squares IK plan (no physics stepping), and the search jitters it in
joint space. Everything below `option_rl` is the frozen runtime — this module only supplies the five task interfaces
(adapter / proposal / generator / scorer / certificate), exactly the assimilation contract.

# Preconditions the env is in a position/velocity control mode (torque diverges on a large 6-D IK step — measured);
``budget ≥ 1``. # Invariants the scorer restores env state after every candidate (snapshot qpos/qvel), so scoring is
side-effect-free on the shared env.
"""
from __future__ import annotations

import mujoco
import numpy as np

from hymeko_rl.env.se3_reach_env import SE3ReachEnv
from hymeko_rl.option_rl import (
    MultimodalBudgetSearch, ProposalMode, SingleModeProposal, StructuredState)


class SE3ReachAdapter:
    """`StructuredStateAdapter`: the SE(3) reach env → a `StructuredState` on the kinematic hypergraph. Node features
    are the POSE_OBSERVATION channels (18/vertex); the geometry channel is the 6-D pose error ``[target−ee ; rotvec]``
    — the task-relevant relational quantity the runtime's downstream heads consume."""

    def structured(self, env: SE3ReachEnv) -> StructuredState:
        n = env.hg.n_vertices
        edges = [tuple(int(v) for v in e) for e in np.atleast_2d(np.asarray(env.hg.edges)).tolist()
                 if all(0 <= int(v) < n for v in e)]
        geom = np.concatenate([env._target - env._ee_pos(), env.orientation_error()]).astype(np.float32)
        return StructuredState(env.node_features(), edges=edges or [(i, i + 1) for i in range(n - 1)],
                               geometry=geom, phase=0, metadata={"task": "se3_reach"})


def ik_estimate(env: SE3ReachEnv, *, iters: int = 25, gain: float = 0.5, damp: float = 0.08) -> np.ndarray:
    """Pure-kinematic 6-D damped-least-squares IK plan: iterate ``q ← clip(q + Jᵀ(JJᵀ+λ²I)⁻¹ e)`` on a state snapshot
    (no physics stepping) to estimate the joint config achieving the target pose. # Postconditions returns a legal
    ``(n_actions,)`` config; env state is restored."""
    n = env.n_actions
    saved = env.data.qpos.copy()
    q = env.data.qpos.copy()
    for _ in range(iters):
        env.data.qpos[:] = q
        mujoco.mj_forward(env.model, env.data)
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jac(env.model, env.data, jacp, jacr, env._ee_pos(), env._ee)
        jac = np.vstack([jacp[:, :n], jacr[:, :n]])
        err = np.concatenate([env._target - env._ee_pos(), env.orientation_error()]).astype(np.float64)
        dq = gain * jac.T @ np.linalg.solve(jac @ jac.T + damp ** 2 * np.eye(6), err)
        q[:n] = np.clip(q[:n] + dq, env._q_lo[:n], env._q_hi[:n])
    env.data.qpos[:] = saved
    mujoco.mj_forward(env.model, env.data)
    return q[:n].astype(np.float32)


class SE3WaypointProposal:
    """`ProposalPolicy`: the proposal centre is the IK-estimated target config (the joint waypoint the committed option
    servos to). Wrap in `SingleModeProposal` for the K=1 multimodal interface (6D-1 will add avoidance modes)."""

    def __init__(self, env: SE3ReachEnv):
        self.env = env

    def center(self, obs: np.ndarray) -> np.ndarray:
        return ik_estimate(self.env)


class SE3JitterGenerator:
    """`CandidateGenerator`: Gaussian jitter of the joint waypoint, clipped to joint limits."""

    def __init__(self, std: float = 0.05):
        self.std = std

    def sample(self, center: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        c = np.asarray(center, np.float64)
        return c[None, :] if n == 1 else c + rng.normal(0, self.std, (int(n), len(c)))


class SE3OptionScorer:
    """`CandidateScorer` bound to ONE reach state: execute the committed option (servo the env to the joint waypoint
    for ``horizon`` steps under position control), grade by the negated pose error, certificate = reached. Snapshots
    and RESTORES (qpos, qvel) around every candidate so scoring never perturbs the shared env."""

    def __init__(self, env: SE3ReachEnv, horizon: int = 120):
        self.env, self.horizon = env, horizon
        self._q, self._v = env.data.qpos.copy(), env.data.qvel.copy()

    def _restore(self) -> None:
        self.env.data.qpos[:] = self._q
        self.env.data.qvel[:] = self._v
        self.env.data.time = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env._step = 0

    def score(self, cand: np.ndarray, rng: np.random.Generator) -> tuple[float, dict]:
        self._restore()
        q_des = np.clip(np.asarray(cand, np.float32), self.env._ctrl_lo, self.env._ctrl_hi)
        reached, info = False, {"dist": float("nan"), "ang_err": float("nan")}
        for _ in range(self.horizon):
            _obs, _r, term, trunc, info = self.env.step(q_des)
            if term and not info.get("death", False):
                reached = True
                break
            if term or trunc:
                break
        self._restore()
        pose_err = float(info["dist"]) + float(info["ang_err"])
        return -pose_err, {"reached": int(reached), "k6": int(reached),
                           "pos_err": float(info["dist"]), "ang_err": float(info["ang_err"])}


def solve_reach(env: SE3ReachEnv, rng: np.random.Generator, *, budget: int = 12, horizon: int = 120):
    """One deploy decision through the frozen runtime: K=1 proposal + `MultimodalBudgetSearch` over joint waypoints →
    the best committed option's provenance (selected q_des + reached certificate). Does NOT mutate ``env`` (the scorer
    restores state); the caller executes ``provenance.selected`` to actually move."""
    proposal = SingleModeProposal(SE3WaypointProposal(env))
    search = MultimodalBudgetSearch(SE3JitterGenerator(), SE3OptionScorer(env, horizon), budget=budget)
    return search.select(proposal, env.node_features().reshape(-1), rng)


def encode_state(env: SE3ReachEnv, adapter: SE3ReachAdapter, encoder, hidden):
    """One representation step of the full runtime front-end: structured state → its FlatStateView, LSTM streaming over
    the flattened obs (hidden threaded), fuse. Returns (fused_vector, new_hidden). Import-local torch to keep the env
    module torch-free at import time."""
    import torch

    from hymeko_rl.option_rl import FlatStateView, fuse_state
    s = adapter.structured(env)
    struct_emb = FlatStateView().view(s)
    with torch.no_grad():
        temporal, hidden = encoder.update(torch.as_tensor(env.node_features().reshape(-1)), hidden)
    return fuse_state(struct_emb, temporal.numpy()[0]), hidden


def make_pose_mode(env: SE3ReachEnv) -> ProposalMode:
    """Expose the IK waypoint as a single explicit `ProposalMode` (for callers assembling a multi-mode proposal in
    6D-1; here it documents the K=1 mode the reach uses)."""
    return ProposalMode(1.0, ik_estimate(env), None, 0)


# ─────────────────────────── 6D-1: obstacle reach via disjoint route modes ───────────────────────────
# The committed option is an EE-waypoint PATH (start → via → goal): a candidate is the VIA point (3-D world position).
# A route mode = a via in one direction {left/right/over/under}; two vias on opposite sides are different homotopy
# classes, so local jitter within a mode cannot cross to another route — the disjoint-mode structure the test needs.
ROUTE_DIRS = {"left": (0.0, -1.0, 0.0), "right": (0.0, 1.0, 0.0), "over": (0.0, 0.0, 1.0), "under": (0.0, 0.0, -1.0)}


def ik_position(env, target_pos, *, iters: int = 20, gain: float = 0.5, damp: float = 0.08) -> np.ndarray:
    """Pure-kinematic position-only (3-D) DLS-IK plan to ``target_pos`` — for intermediate via waypoints (orientation
    is only gated at the goal). Snapshot/restore; returns a legal ``(n_actions,)`` config."""
    n = env.n_actions
    saved = env.data.qpos.copy()
    q = env.data.qpos.copy()
    tgt = np.asarray(target_pos, np.float64)
    for _ in range(iters):
        env.data.qpos[:] = q
        mujoco.mj_forward(env.model, env.data)
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jac(env.model, env.data, jacp, jacr, env._ee_pos(), env._ee)
        jp = jacp[:, :n]
        err = tgt - env._ee_pos()
        dq = gain * jp.T @ np.linalg.solve(jp @ jp.T + damp ** 2 * np.eye(3), err)
        q[:n] = np.clip(q[:n] + dq, env._q_lo[:n], env._q_hi[:n])
    env.data.qpos[:] = saved
    mujoco.mj_forward(env.model, env.data)
    return q[:n].astype(np.float32)


def execute_route_option(env, via_ee, *, via_steps: int = 60, goal_steps: int = 150) -> dict:
    """Execute the committed EE-path option: open-loop servo to the via (routes around the obstacle), then a CLOSED-LOOP
    goal phase (re-planned expert DLS-IK each step) that reliably closes the goal pose from the far side — the open-loop
    ``ik_estimate`` goal phase does not close a large post-detour 6-D error (measured; the 6D-0 closability wall). EE-path
    obstacle collision is tracked throughout. Grade = reached goal pose AND collision-free. Caller snapshots/restores."""
    q_via = ik_position(env, via_ee)
    collided = bool(env.ee_in_obstacle(env._ee_pos()))
    for _ in range(via_steps):
        env.step(q_via)
        collided = collided or env.ee_in_obstacle(env._ee_pos())
    reached, info = False, {"dist": float("nan"), "ang_err": float("nan")}
    for _ in range(goal_steps):
        _o, _r, term, trunc, info = env.step(env.expert_action)      # closed-loop goal phase (reliable pose closing)
        collided = collided or env.ee_in_obstacle(env._ee_pos())
        if term and not info.get("death", False):
            reached = True
            break
        if term or trunc:
            break
    success = bool(reached and not collided)
    return {"reached": int(reached), "collided": int(collided), "success": int(success), "k6": int(success),
            "pos_err": float(info["dist"]), "ang_err": float(info["ang_err"])}


def route_execution_feasible(env, direction, *, offset: float = 0.14, budget: int = 3, seed: int = 0) -> bool:
    """EXECUTION-based route feasibility (NOT the geometric polyline — which is unreliable: an AABB-clean via can be
    kinematically unreachable or the servo cuts the corner). Try the committed option for this route a few times
    (small via jitter) from a state SNAPSHOT; feasible iff any trial reaches the goal pose collision-free. Restores the
    env. Controller-independent (a property of state+executor, not of the tested proposal)."""
    q, v = env.data.qpos.copy(), env.data.qvel.copy()
    rng = np.random.default_rng(seed)
    ok = False
    base = route_via(env, direction, offset=offset)
    for _ in range(budget):
        env.data.qpos[:], env.data.qvel[:] = q, v
        mujoco.mj_forward(env.model, env.data)
        env._step = 0
        via = base + rng.normal(0, 0.02, 3).astype(np.float32)
        if execute_route_option(env, via)["success"]:
            ok = True
            break
    env.data.qpos[:], env.data.qvel[:] = q, v
    mujoco.mj_forward(env.model, env.data)
    env._step = 0
    return ok


def route_via(env, direction, *, offset: float = 0.14) -> np.ndarray:
    """The via point for a route family: the start↔goal midpoint offset laterally by ``offset`` along ``direction``."""
    mid = 0.5 * (env._start_ee + np.asarray(env._target, np.float32))
    return (mid + offset * np.asarray(direction, np.float32)).astype(np.float32)


class RouteModeProposal:
    """`MultimodalProposalPolicy`: K route modes = K via points (one per direction). The ALLOCATION strategy is encoded
    ENTIRELY in the mode probabilities (the frozen `MultimodalBudgetSearch`/`allocate_budget` are untouched):
      A 'prob'      — modes carry the geometric route prior (argmax-weighted split);
      B 'equal'     — uniform probs ⇒ equal minimum budget per mode;
      C 'top_probe' — one dominant + tiny alternates ⇒ top mode refined + one probe per alternate.
    K=1 keeps only the top-prior mode (the single-head baseline)."""

    def __init__(self, env, directions, allocation: str = "prob", offset: float = 0.14, k: int | None = None):
        self.env, self.directions, self.allocation, self.offset = env, list(directions), allocation, offset
        self.k = len(self.directions) if k is None else int(k)   # keep only the top-k routes by prior (K=1 = single-head)

    def _prior(self):
        """Geometric route prior (imperfect — it does NOT know per-arm reachability/collision): prefer directions most
        perpendicular to the start→goal line (they route around a between-obstacle better)."""
        seg = np.asarray(self.env._target, np.float32) - self.env._start_ee
        seg = seg / (np.linalg.norm(seg) + 1e-9)
        perp = np.array([1.0 - abs(float(np.dot(np.asarray(d, np.float32) / (np.linalg.norm(d) + 1e-9), seg))) for d in self.directions])
        w = np.exp(3.0 * perp)
        return w / w.sum()

    def modes(self, _obs):
        prior = self._prior()
        order = np.argsort(-prior)[: self.k]                     # top-k routes by prior
        k = len(order)
        if self.allocation == "equal":
            probs = {i: 1.0 / k for i in order}
        elif self.allocation == "top_probe":
            probs = {i: (0.97 if i == order[0] else 0.01) for i in order}
        else:  # 'prob'
            probs = {i: float(prior[i]) for i in order}
        return [ProposalMode(probs[i], route_via(self.env, self.directions[i], offset=self.offset), None, int(i))
                for i in order]


class RouteOptionScorer:
    """`CandidateScorer` for 6D-1: a candidate is a via point; execute the EE-path option and grade by collision-free
    goal reach. Snapshots/restores (qpos, qvel) around every candidate. Score is lexicographic success ≻ reached ≻
    −collided ≻ −pos_err (as a float via a monotone encoding)."""

    def __init__(self, env, via_steps: int = 60, goal_steps: int = 140):
        self.env, self.via_steps, self.goal_steps = env, via_steps, goal_steps
        self._q, self._v = env.data.qpos.copy(), env.data.qvel.copy()

    def _restore(self):
        self.env.data.qpos[:] = self._q
        self.env.data.qvel[:] = self._v
        self.env.data.time = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env._step = 0

    def score(self, cand, rng):
        self._restore()
        out = execute_route_option(self.env, cand, via_steps=self.via_steps, goal_steps=self.goal_steps)
        self._restore()
        pe = out["pos_err"] if np.isfinite(out["pos_err"]) else 9.9
        sc = 1e6 * out["success"] + 1e3 * out["reached"] - 1e2 * out["collided"] - pe   # monotone success≻reached≻¬collide
        return float(sc), out
