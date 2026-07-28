"""Domain-generic snapshot of a rollout's dynamical state at a single instant.

This is the observation the R11 HyMeKo IR reasons over: the configuration, its rate, the last applied actuation, the
manipulated object's velocity, the task-relevant contact count, and the provenance flags (memory emptiness, step index,
whether the state was injected from a snapshot / teacher). It carries **no** simulator types — an adapter (e.g. the coin
adapter) reads a MuJoCo live state into this struct, and every certificate/invariant/guard downstream is expressed purely
over these fields. Keeping the boundary here is what lets the initial-condition certificate be verified without a physics
engine in the loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]


@dataclass(frozen=True)
class RolloutState:
    """A read-only dynamical snapshot.

    Preconditions:
        ``q``, ``qdot``, ``prev_tau`` are 1-D and share the actuated-DOF length; ``object_vel`` is 1-D.
    Invariants:
        Frozen — never mutated after construction; all array fields are stored as ``float64``.
    """

    q: Vec
    qdot: Vec
    prev_tau: Vec
    object_vel: Vec
    n_task_contacts: int
    controller_memory_empty: bool
    step: int
    has_snapshot_parent: bool
    has_teacher_state: bool

    def __post_init__(self) -> None:
        assert self.q.ndim == self.qdot.ndim == self.prev_tau.ndim == 1, "q/qdot/prev_tau must be 1-D"
        assert self.q.shape == self.qdot.shape == self.prev_tau.shape, "actuated-DOF arrays must share length"
        assert self.object_vel.ndim == 1, "object_vel must be 1-D"
        assert self.n_task_contacts >= 0 and self.step >= 0, "counts are non-negative"

    @staticmethod
    def at_rest(q: Vec, *, object_dof: int = 2, step: int = 0, memory_empty: bool = True,
                snapshot_parent: bool = False, teacher_state: bool = False,
                n_task_contacts: int = 0) -> "RolloutState":
        """A zero-velocity, zero-torque state at configuration ``q`` (the nominal fresh-reset shape)."""
        q = np.asarray(q, dtype=np.float64)
        return RolloutState(q=q, qdot=np.zeros_like(q), prev_tau=np.zeros_like(q),
                            object_vel=np.zeros(object_dof, dtype=np.float64), n_task_contacts=n_task_contacts,
                            controller_memory_empty=memory_empty, step=step, has_snapshot_parent=snapshot_parent,
                            has_teacher_state=teacher_state)
