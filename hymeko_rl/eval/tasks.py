"""Scenario registry — the common simulator ecosystem.

One declarative :class:`TaskSpec` per robot/scenario: its env factory, its success metric, its eval budget.
Every experiment, eval, and demo run dispatches through here, so adding a robot is **one registration**, not
another hand-rolled ``_env()`` + ``eval_*`` + driver (CLAUDE.md §6.1, §6.5 #2/#3). The success criterion is a
:class:`hymeko_rl.eval.evaluate.RolloutMetric` — data, not a duplicated loop — so a scenario *declares* what success
means and the shared :func:`hymeko_rl.eval.evaluate.eval_metric` runs it.

    from hymeko_rl.eval.tasks import get_task, evaluate_task, task_names
    env = get_task("pick_place").make_env()
    lift_place = evaluate_task("pick_place", actor, n_episodes=8)   # uses the declared LiftPlaceMetric
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hymeko_rl.eval.evaluate import (
    DwellMetric,
    FinalScalarMetric,
    FlagSeenMetric,
    LiftPlaceMetric,
    ReturnMetric,
    RolloutMetric,
    StepCountMetric,
    eval_metric,
    greedy_action_fn,
)


@dataclass(frozen=True)
class ArchRecommendation:
    """The best-known ``(backbone, algorithm, head)`` for a scenario + the EVIDENCE. ``confidence`` is honest
    provenance (CLAUDE.md): ``"measured"`` = an A/B on this task; ``"inferred"`` = from a sibling task or a
    mechanism; ``"recommended"`` = a motivated but un-validated lever. Read it before assuming a config wins."""

    backbone: str        # policy.POLICY_KINDS: mlp / hsikan / sa_hsikan / signedkan / mixture
    algorithm: str       # ppo / ddpg / td3 / td3_bc / sac / bc / bc_ppo (BC warm-start then PPO refine)
    head: str            # pooled (collapse-then-expand) / per_node (policy.PerNodeActionHead)
    confidence: str      # measured / inferred / recommended
    basis: str           # the one-line evidence


@dataclass(frozen=True)
class TaskSpec:
    """A scenario: env factory + success metric + eval budget + the best-known architecture. ``make_env`` builds
    a FRESH env each call; ``metric`` builds a fresh :class:`RolloutMetric` each eval. # Invariants ``name`` is
    the registry key."""

    name: str
    make_env: Callable[[], Any]
    metric: Callable[[], RolloutMetric]
    description: str
    recommended: "ArchRecommendation | None" = None
    default_steps: "dict[str, int]" = field(default_factory=lambda: {"smoke": 2_000, "full": 30_000})


_REGISTRY: "dict[str, TaskSpec]" = {}


def register_task(spec: TaskSpec) -> None:
    """Register a scenario. # Preconditions ``spec.name`` is not already registered (no silent shadowing)."""
    if spec.name in _REGISTRY:
        raise ValueError(f"task {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec


def get_task(name: str) -> TaskSpec:
    """# Postconditions returns the registered spec; raises ``ValueError`` naming the known tasks otherwise."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown task {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def task_names() -> "list[str]":
    return sorted(_REGISTRY)


def evaluate_task(name: str, ac: Any, *, n_episodes: int = 8, seed: int = 20_000) -> "list[Any]":
    """Build the scenario env + roll ``ac`` greedily under the scenario's declared metric — the one-line eval."""
    spec = get_task(name)
    return eval_metric(spec.make_env(), greedy_action_fn(ac), spec.metric(), n_episodes=n_episodes, seed0=seed)


def best_arch(name: str) -> ArchRecommendation:
    """The best-known architecture (backbone, algorithm, head + evidence) for ``name`` — what to reach for first
    on a new run. # Postconditions raises ``ValueError`` if no recommendation is recorded for the scenario."""
    spec = get_task(name)
    if spec.recommended is None:
        raise ValueError(f"task {name!r} has no architecture recommendation recorded")
    return spec.recommended


# --- built-in scenarios. Lazy factories: a scenario's (heavy) env import happens only when it is built, so
#     importing this registry is cheap and free of env-package import cycles. ---

def _cartpole_env() -> Any:
    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
    return InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())


def _galambos_env() -> Any:
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    return PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3, task_graph=True)


def _pick_place_env() -> Any:
    from hymeko_rl.viz.render_pick_place import fanuc_pick_env
    return fanuc_pick_env()


def _arm_reach_env() -> Any:
    from hymeko_rl.env.arm_reach_env import ArmReachEnv
    return ArmReachEnv()


def _quadruped_env() -> Any:
    from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
    return QuadrupedGoalEnv()


def _quadruped_stand_env() -> Any:
    # SINGLE SOURCE OF TRUTH: the whole balance MDP (plant + tuning + reward + strategy) is declared in
    # data/robotics/quadruped_stand.hymeko; from_hymeko reads the scene tuning + reward_spec (base="free" and
    # task="stand" are structural to a standing task). The env holds the 10 expressive DOF and RL drives the 12 legs.
    from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
    return QuadrupedGoalEnv.from_hymeko()


def _register_builtins() -> None:
    register_task(TaskSpec(
        "cartpole", _cartpole_env, StepCountMetric,
        "Inverted pendulum; upright-steps — the control floor.",
        recommended=ArchRecommendation(
            "hsikan", "ddpg", "pooled", "measured",
            "HSiKAN actor-critic learns upright (28->144/200); DDPG ~250x more sample-efficient than PPO here.")))
    register_task(TaskSpec(
        "galambos", _galambos_env, lambda: FlagSeenMetric("in_zone"),
        "Planar two-arm coin delivery; in-zone rate.",
        recommended=ArchRecommendation(
            "hsikan", "bc_ppo", "pooled", "measured",
            "BC teaches the grasp structure then PPO refines (pure RL never delivers); single-agent >= 2-agent "
            "CTDE — no collaborative win on this non-cyclic task."),
        default_steps={"smoke": 2_000, "full": 20_000}))
    register_task(TaskSpec(
        "pick_place", _pick_place_env, LiftPlaceMetric,
        "FANUC arm pick-and-place; (lift, place) rate with a divergence guard.",
        recommended=ArchRecommendation(
            "hsikan", "td3_bc", "per_node", "recommended",
            "BC works (0.75-0.875, measured); PPO COLLAPSES it (measured; the known offline-to-online problem); "
            "TD3+BC holds at smoke scale. per_node head is the un-validated lever (3723x on structured targets).")))
    register_task(TaskSpec(
        "arm_reach", _arm_reach_env, lambda: FinalScalarMetric("dist"),
        "6-DOF arm reach; final end-effector distance (m).",
        recommended=ArchRecommendation(
            "hsikan", "bc", "pooled", "measured",
            "HSiKAN ~= MLP on the serial arm (small structural gap); BC imitation suffices, no RL refine needed.")))
    register_task(TaskSpec(
        "quadruped", _quadruped_env, ReturnMetric,
        "Quadruped goal-reach; episode return.",
        recommended=ArchRecommendation(
            "hsikan", "ppo", "pooled", "measured",
            "FLAT HSiKAN beats 4-leg CTDE on goal-reach (-33 vs -84); the collaborative framing loses on a "
            "non-cyclic reach — gait (cyclic=holonomy) is the untested place CTDE+rotor might win.")))
    register_task(TaskSpec(
        "quadruped_stand", _quadruped_stand_env, lambda: DwellMetric("standing", 200),
        "Quadruped balance on a free base; rate of holding upright-at-height for >=200 consecutive steps.",
        recommended=ArchRecommendation(
            "sa_hsikan", "td3", "pooled", "recommended",
            "Rung-2 POSTURAL task (exp_designed_control) — balance is coordinative/cyclic, the regime where "
            "declared topology (kinematic vs Steiner/sunflower) is hypothesised to pay, unlike the non-cyclic "
            "goal-reach where flat HSiKAN wins. Off-policy TD3 from a dense shaped reward (no demonstrator); "
            "PD-hold-q0 BC warm-start is the fallback lever if from-scratch balance is too unstable."),
        default_steps={"smoke": 2_000, "full": 100_000}))


_register_builtins()
