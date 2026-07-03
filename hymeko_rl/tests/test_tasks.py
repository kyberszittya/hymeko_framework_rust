"""The scenario registry (the common simulator ecosystem)."""
from typing import Any

from hymeko_rl.eval.evaluate import LiftPlaceMetric, StepCountMetric
from hymeko_rl.eval.tasks import TaskSpec, evaluate_task, get_task, register_task, task_names


def test_registry_has_builtins() -> None:
    assert {"cartpole", "galambos", "pick_place", "arm_reach", "quadruped"} <= set(task_names())


def test_spec_metric_factories_typed() -> None:
    assert isinstance(get_task("pick_place").metric(), LiftPlaceMetric)
    assert isinstance(get_task("cartpole").metric(), StepCountMetric)
    assert callable(get_task("galambos").make_env)


def test_unknown_and_duplicate_raise() -> None:
    for bad in ("nope", ""):
        try:
            get_task(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")
    try:
        register_task(get_task("cartpole"))     # re-register = duplicate
    except ValueError:
        return
    raise AssertionError("expected duplicate ValueError")


def test_evaluate_task_runs_on_cartpole() -> None:
    from hymeko_rl.agents.policy import build_policy
    spec = get_task("cartpole")
    env = spec.make_env()
    ss = env.observation_space.shape
    ac: Any = build_policy("mlp", obs_dim=int(ss[0]) * int(ss[1]), action_dim=1)
    res = evaluate_task("cartpole", ac, n_episodes=2, seed=0)
    assert len(res) == 2 and all(isinstance(r, int) for r in res)


def test_custom_task_registration() -> None:
    spec = TaskSpec("dummy_test_task", lambda: None, StepCountMetric, "test-only")
    register_task(spec)
    assert get_task("dummy_test_task").description == "test-only"


def test_every_builtin_has_a_valid_recommendation() -> None:
    from hymeko_rl.agents.policy import POLICY_KINDS
    from hymeko_rl.eval.tasks import best_arch
    for name in ("cartpole", "galambos", "pick_place", "arm_reach", "quadruped"):
        r = best_arch(name)
        assert r.backbone in POLICY_KINDS, f"{name}: bad backbone {r.backbone}"
        assert r.confidence in {"measured", "inferred", "recommended"}
        assert r.head in {"pooled", "per_node"} and r.basis    # honest provenance + non-empty evidence


def test_best_arch_requires_a_recommendation() -> None:
    from hymeko_rl.eval.evaluate import StepCountMetric as _SC
    from hymeko_rl.eval.tasks import best_arch
    register_task(TaskSpec("norec_task", lambda: None, _SC, "no rec recorded"))
    try:
        best_arch("norec_task")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a task with no recommendation")
