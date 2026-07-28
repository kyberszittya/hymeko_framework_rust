"""The task-contract initial condition and its certificate, plus the certificate-filtered initial distribution.

``InitialCondition`` is a *specification* of a legal rollout start (expected configuration, zero velocities, empty
memory/contacts, no snapshot/teacher injection, step 0). ``certify`` checks a :class:`RolloutState` against it and returns
an :class:`InitialConditionCertificate` — the first-class, verifiable object a ``ZERO_HOME -> K6`` claim must carry.

``InitialDistribution`` is deliberately *not* a raw box: a candidate pose is admissible only if an injected predicate says
so (collision-free start, minimum clearance, non-empty goal set, certified reach). Inadmissible samples are
``INVALID_INITIAL_CONDITION`` and are accounted in a :class:`RejectionLedger` — they never enter a policy-success
denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from hymeko_rl.ir.rollout import RolloutState, Vec


class InitialConditionViolation(RuntimeError):
    """Raised when a rollout state fails its declared initial condition. The caller passed an illegal start."""


@dataclass(frozen=True)
class InitialConditionCertificate:
    """Result of certifying a state against an :class:`InitialCondition`. ``checks`` is per-clause pass/fail."""

    condition_name: str
    valid: bool
    violations: tuple[str, ...]
    checks: Mapping[str, bool]

    def raise_if_invalid(self) -> "InitialConditionCertificate":
        """Postcondition: returns self iff ``valid``; otherwise raises with the violated clauses."""
        if not self.valid:
            raise InitialConditionViolation(f"{self.condition_name}: violated {list(self.violations)}")
        return self


@dataclass(frozen=True)
class InitialCondition:
    """Specification of a legal rollout start.

    Preconditions (of :meth:`certify`): ``state.q.shape == q_expected.shape``.
    Postconditions: returns a certificate whose ``valid`` is the conjunction of all enabled clauses.
    """

    name: str
    q_expected: Vec
    q_atol: float = 1e-9
    vel_atol: float = 1e-9
    tau_atol: float = 1e-9
    require_zero_prev_tau: bool = True
    require_empty_contacts: bool = True
    require_empty_memory: bool = True
    require_step_zero: bool = True
    require_no_snapshot: bool = True
    require_no_teacher: bool = True

    def certify(self, state: RolloutState) -> InitialConditionCertificate:
        assert state.q.shape == self.q_expected.shape, "state.q and q_expected must share shape"
        clauses = self._clause_results(state)
        violations = tuple(name for name, ok in clauses.items() if not ok)
        return InitialConditionCertificate(self.name, len(violations) == 0, violations, clauses)

    def _clause_results(self, state: RolloutState) -> dict[str, bool]:
        """Per-clause pass/fail. A clause with its requirement disabled passes vacuously (``not enabled or passed``)."""
        passed = {
            "q_at_expected": bool(np.allclose(state.q, self.q_expected, atol=self.q_atol, rtol=0.0)),
            "qdot_zero": float(np.abs(state.qdot).max(initial=0.0)) <= self.vel_atol,
            "object_at_rest": float(np.abs(state.object_vel).max(initial=0.0)) <= self.vel_atol,
            "prev_tau_zero": float(np.abs(state.prev_tau).max(initial=0.0)) <= self.tau_atol,
            "empty_contacts": state.n_task_contacts == 0,
            "empty_memory": state.controller_memory_empty,
            "step_zero": state.step == 0,
            "no_snapshot": not state.has_snapshot_parent,
            "no_teacher": not state.has_teacher_state,
        }
        enabled = {
            "prev_tau_zero": self.require_zero_prev_tau, "empty_contacts": self.require_empty_contacts,
            "empty_memory": self.require_empty_memory, "step_zero": self.require_step_zero,
            "no_snapshot": self.require_no_snapshot, "no_teacher": self.require_no_teacher,
        }
        return {name: (not enabled.get(name, True)) or ok for name, ok in passed.items()}


@dataclass(frozen=True)
class AdmissibilityResult:
    """Whether a sampled pose is a valid initial condition, with a machine-readable ``reason`` when it is not."""

    admissible: bool
    reason: str  # "ADMISSIBLE" or an INVALID_INITIAL_CONDITION cause, e.g. "start_in_collision"


@dataclass
class RejectionLedger:
    """Accounting for admissibility filtering: admissible poses vs. rejected ones by reason. Rejections are excluded from
    any success denominator (they are ``INVALID_INITIAL_CONDITION``), and reported separately as a rejection rate."""

    admitted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def record(self, result: AdmissibilityResult) -> None:
        if result.admissible:
            self.admitted += 1
        else:
            self.rejected[result.reason] = self.rejected.get(result.reason, 0) + 1

    @property
    def total(self) -> int:
        return self.admitted + sum(self.rejected.values())

    @property
    def rejection_rate(self) -> float:
        """Postcondition: 0.0 when nothing has been recorded; else rejected / total in [0, 1]."""
        return 0.0 if self.total == 0 else sum(self.rejected.values()) / self.total


@dataclass(frozen=True)
class InitialDistribution:
    """A named sampling region gated by a certificate predicate. ``admits`` delegates to the injected predicate so the
    generic IR never hard-codes domain geometry; the coin adapter supplies collision/clearance/reach admissibility."""

    name: str
    lo: Vec
    hi: Vec
    predicate: Callable[[NDArray[np.float64]], AdmissibilityResult]

    def __post_init__(self) -> None:
        assert self.lo.shape == self.hi.shape and self.lo.ndim == 1, "lo/hi must be matching 1-D bounds"
        assert bool(np.all(self.lo <= self.hi)), "each lower bound must not exceed its upper bound"

    def admits(self, pose: NDArray[np.float64]) -> AdmissibilityResult:
        """Preconditions: ``pose.shape == self.lo.shape``. Out-of-box poses are rejected before the predicate runs."""
        assert pose.shape == self.lo.shape, "pose must match the distribution dimensionality"
        if not bool(np.all((pose >= self.lo) & (pose <= self.hi))):
            return AdmissibilityResult(False, "out_of_bounds")
        return self.predicate(pose)
