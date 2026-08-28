"""Conformance tests for the generic task-execution + planning contracts (addendum §2, §4-6, §8-9).

These pin the embodiment-independent surface: the TaskResult envelope, the generic event/trajectory model, and the
role/planner/registry decoupling. Scenario specializations (coin) are tested in their own repos; here we prove the
GENERIC contracts stand alone (torch-free, no scenario import) and enforce their invariants.
"""
from __future__ import annotations

import pytest

from hymeko_control.cip.certificate import CertificateResult
from hymeko_control.execution import (
    Planner,
    PlannerRegistry,
    PlanningRequest,
    PlanningResult,
    PlanningRole,
    PlanningStatus,
    TaskEvent,
    TaskEventKind,
    TaskResult,
    TaskStatus,
    TrajectorySample,
)


def _ok_result(**kw) -> TaskResult:
    base = {"task_id": "t", "status": TaskStatus.SUCCEEDED, "succeeded": True}
    base.update(kw)
    return TaskResult(**base)


class TestTaskResult:
    def test_minimal_construction(self) -> None:
        r = _ok_result()
        assert r.task_id == "t" and r.succeeded and r.status is TaskStatus.SUCCEEDED
        assert r.events == () and r.trajectory == ()

    def test_succeeded_must_agree_with_status(self) -> None:
        with pytest.raises(ValueError):
            TaskResult(task_id="t", status=TaskStatus.FAILED, succeeded=True)
        with pytest.raises(ValueError):
            TaskResult(task_id="t", status=TaskStatus.SUCCEEDED, succeeded=False)

    def test_failed_result_ok(self) -> None:
        r = TaskResult(task_id="t", status=TaskStatus.FAILED, succeeded=False, failure_class="DELIVERY")
        assert not r.succeeded and r.failure_class == "DELIVERY"

    def test_mappings_are_readonly(self) -> None:
        r = _ok_result(metrics={"a": 1.0})
        with pytest.raises(TypeError):
            r.metrics["a"] = 2.0  # type: ignore[index]

    def test_certificate_reuses_cip_contract(self) -> None:
        cert = CertificateResult(passed=True, success_passed=True, safety_passed=True)
        r = _ok_result(certificate=cert)
        assert r.certificate is cert  # no parallel certificate abstraction (addendum §1)

    def test_events_of_filters_by_generic_kind(self) -> None:
        evs = (TaskEvent(0, TaskEventKind.TASK_STARTED),
               TaskEvent(1, TaskEventKind.PLAN_COMPUTED, label="ReachPlanned"),
               TaskEvent(2, TaskEventKind.TASK_COMPLETED))
        r = _ok_result(events=evs)
        got = r.events_of(TaskEventKind.PLAN_COMPUTED)
        assert len(got) == 1 and got[0].label == "ReachPlanned"

    def test_timestamps_monotonic_guard(self) -> None:
        good = _ok_result(events=(TaskEvent(0, TaskEventKind.TASK_STARTED),
                                  TaskEvent(1, TaskEventKind.TASK_COMPLETED)),
                          trajectory=(TrajectorySample(0, "MOVE"), TrajectorySample(1, "CERTIFY")))
        assert good.timestamps_monotonic()
        bad = _ok_result(trajectory=(TrajectorySample(3, "MOVE"), TrajectorySample(1, "CERTIFY")))
        assert not bad.timestamps_monotonic()

    def test_to_public_is_json_safe_generic_surface(self) -> None:
        import json

        cert = CertificateResult(passed=True, success_passed=True, safety_passed=True,
                                 per_certificate={"strict_k6": True}, details={"dtz_end_mm": 16.5})
        r = _ok_result(certificate=cert, metrics={"q_task": -0.018}, provenance={"reach_planner_binding": "RRT_CONNECT"},
                       events=(TaskEvent(0, TaskEventKind.PLAN_COMPUTED, "ReachPlanned", {"planner": "RRT_CONNECT"}),),
                       trajectory=(TrajectorySample(0, "MOVE", {"coin_x": 0.1}),))
        pub = r.to_public()
        s = json.dumps(pub)  # no custom default: no mappingproxy, no enum objects leak through
        back = json.loads(s)
        assert back["status"] == "succeeded" and back["succeeded"] is True   # enum -> value
        assert back["certificate"]["per_certificate"]["strict_k6"] is True
        assert back["events"][0]["kind"] == "plan_computed" and back["events"][0]["payload"]["planner"] == "RRT_CONNECT"
        assert back["trajectory"][0]["payload"]["coin_x"] == 0.1
        assert "coin" not in back   # generic base carries no scenario extension


class TestTaskEventAndTrajectory:
    def test_event_payload_frozen(self) -> None:
        e = TaskEvent(0, TaskEventKind.PLAN_COMPUTED, "ReachPlanned", {"role": "reach", "planner": "RRT_CONNECT"})
        with pytest.raises(TypeError):
            e.payload["planner"] = "x"  # type: ignore[index]

    def test_trajectory_sample_generic_payload(self) -> None:
        s = TrajectorySample(5, "CERTIFY", payload={"coin_x": 0.1, "coin_y": 0.2}, provenance={"hook": "read_only"})
        assert s.phase == "CERTIFY" and s.payload["coin_x"] == 0.1
        with pytest.raises(TypeError):
            s.payload["coin_x"] = 9.0  # type: ignore[index]


class _StubPlanner:
    """A minimal Planner impl (test double) — proves any object with role/name/plan satisfies the Protocol."""

    role = PlanningRole.REACH
    name = "STUB"

    def plan(self, request: PlanningRequest) -> PlanningResult:
        return PlanningResult(role=request.role, planner=self.name, status=PlanningStatus.FEASIBLE, feasible=True)


class TestPlanning:
    def test_role_is_not_hardcoded_to_an_implementation(self) -> None:
        # REACH is a role; the planner name is separate and swappable.
        reg = PlannerRegistry().register(_StubPlanner())
        assert reg.binding(PlanningRole.REACH) == "STUB"

    def test_registry_resolve_and_swap(self) -> None:
        reg = PlannerRegistry().register(_StubPlanner())
        assert isinstance(reg.resolve(PlanningRole.REACH), Planner)

        class _Other:
            role = PlanningRole.REACH
            name = "GRAPH_SEARCH"

            def plan(self, request: PlanningRequest) -> PlanningResult:
                return PlanningResult(role=request.role, planner=self.name, status=PlanningStatus.UNKNOWN)

        reg.register(_Other())  # swap the binding — the ONLY change a replacement needs
        assert reg.binding(PlanningRole.REACH) == "GRAPH_SEARCH"

    def test_resolve_missing_role_raises(self) -> None:
        with pytest.raises(KeyError):
            PlannerRegistry().resolve(PlanningRole.TRANSPORT)

    def test_planning_result_quarantines_impl_diagnostics(self) -> None:
        res = PlanningResult(role=PlanningRole.REACH, planner="RRT_CONNECT", status=PlanningStatus.FEASIBLE,
                             feasible=True, diagnostics={"nodes": 812, "shortcut_ratio": 0.4},
                             provenance={"owner": "pipeline._do_reach_and_capture", "postprocess": "SHORTCUT+DENSIFY"})
        assert res.provenance["postprocess"] == "SHORTCUT+DENSIFY"
        with pytest.raises(TypeError):
            res.diagnostics["nodes"] = 0  # type: ignore[index]

    def test_stub_planner_runtime_checkable(self) -> None:
        assert isinstance(_StubPlanner(), Planner)
