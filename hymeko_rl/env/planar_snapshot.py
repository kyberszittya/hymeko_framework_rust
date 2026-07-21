"""State snapshot / restore + exact branch-from-state component returns for :class:`PlanarGraspEnv`.

The fair vector-critic retest needs two things the env does not provide out of the box:

* **restore to an exact visited state** so the gradient-alignment probe can roll *different first actions from the
  same fixed CONTACT/PUSH state* (a counterfactual the base env cannot express — ``reset`` only samples fresh);
* **measured Monte-Carlo component returns** ``Q^\\pi_c(s, a)`` = the discounted component return of *taking action
  ``a`` at ``s`` then following the frozen policy* — the exact, action-conditioned target that replaces the prior
  run's near-deterministic-DAgger bootstrap.

A ``PlanarGraspEnv`` state is MuJoCo (``qpos``/``qvel``/``qacc_warmstart``/``time``) **plus** six Python latches
(``_disk_out``, ``_arm_body_ever``, ``_arm_body_steps``, ``_arm_body_impulse_sum``, ``_prev_disk_to_zone``,
``_ever_grasped``) and the per-episode zone ``(_zone_x, _zone_y)`` — all of which feed ``privileged_state`` /
``_planar_metrics`` / the delivery predicate, so all must be captured. This module is non-core; it reads and writes
only public MuJoCo buffers + those documented latches, never the env's internals beyond them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import torch

from hymeko_rl.train.search_objective import COMPONENTS, SearchObjective

_LATCH_ATTRS = ("_disk_out", "_arm_body_ever", "_arm_body_steps", "_arm_body_impulse_sum",
                "_prev_disk_to_zone", "_ever_grasped")


@dataclass
class PlanarSnapshot:
    """A complete, restorable ``PlanarGraspEnv`` state (MuJoCo buffers + Python latches + zone)."""

    qpos: np.ndarray
    qvel: np.ndarray
    qacc_warmstart: np.ndarray
    time: float
    zone: tuple[float, float]
    latches: dict[str, Any]
    disk_to_zone: float  # = env._prev_disk_to_zone at this state (the current coin→zone distance)


def snapshot_planar(env: Any) -> PlanarSnapshot:
    """Capture the current state of ``env``.

    # Preconditions: ``env`` has been ``reset`` (MuJoCo data + latches populated).
    # Postconditions: the returned snapshot restores ``env`` to a byte-identical dynamical state (up to the
      solver warmstart) via :func:`restore_planar`; ``env`` itself is left unmodified.
    """
    d = env.data
    prev = getattr(env, "_prev_disk_to_zone", None)
    return PlanarSnapshot(
        qpos=d.qpos.copy(),
        qvel=d.qvel.copy(),
        qacc_warmstart=d.qacc_warmstart.copy(),
        time=float(d.time),
        zone=(float(env._zone_x), float(env._zone_y)),
        latches={a: getattr(env, a) for a in _LATCH_ATTRS},
        disk_to_zone=float(prev) if prev is not None else float("nan"),
    )


def restore_planar(env: Any, snap: PlanarSnapshot) -> None:
    """Restore ``env`` to the state captured in ``snap`` (in place).

    # Preconditions: ``snap`` was taken from an env with the same model (identical ``nq``/``nv``).
    # Postconditions: ``env.data`` + the six latches + zone equal the snapshot; ``mj_forward`` has refreshed all
      derived quantities so the next ``env.step`` evolves from exactly this state.
    """
    d = env.data
    d.qpos[:] = snap.qpos
    d.qvel[:] = snap.qvel
    d.qacc_warmstart[:] = snap.qacc_warmstart
    d.time = snap.time
    env._zone_x, env._zone_y = snap.zone
    for a, v in snap.latches.items():
        setattr(env, a, v)
    mujoco.mj_forward(env.model, env.data)


def _greedy(actor: Any, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy().astype(np.float32)


def branch_component_returns(
    env: Any, frozen_actor: Any, snap: PlanarSnapshot, first_action: np.ndarray, *,
    objective: SearchObjective, gamma: float, horizon: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Exact per-component discounted return of **taking ``first_action`` at ``snap``, then following
    ``frozen_actor``** for up to ``horizon`` steps — the measured MC target ``Q^\\pi_c(s, a)``.

    Returns ``(returns, first_step)`` where ``returns[c]`` is the discounted component return-to-go from the first
    action and ``first_step`` carries the first transition's ``next_obs``/``done``/``z_next`` (for the critic
    dataset) plus the realised branch length.

    # Preconditions: ``0 < gamma <= 1``, ``horizon >= 1``, ``first_action`` shape ``(nu,)``.
    # Postconditions: ``env`` is left at the terminal branch state (caller re-restores before the next branch);
      ``returns[c]`` uses the same monitor-physical component signals as the replay (no reward leakage).
    """
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    restore_planar(env, snap)
    seq = {c: [] for c in COMPONENTS}
    prev_dist = snap.disk_to_zone
    a = np.asarray(first_action, dtype=np.float32)
    first_step: dict[str, Any] = {}
    steps = 0
    for t in range(horizon):
        nobs, _r, term, trunc, info = env.step(a)
        m = env._planar_metrics
        sig = objective.step_signals(
            prev_dist=prev_dist, dist=float(info["disk_to_zone"]),
            min_tip=min(float(m.left_tip_dist), float(m.right_tip_dist)),
            both_contact=bool(info["both_contact"]), fingertip_contact=bool(info["fingertip_contact"]),
            arm_body_contact=bool(info["arm_body_contact_this_step"]), in_zone=bool(info["in_zone"]))
        for c in COMPONENTS:
            seq[c].append(float(sig[c]))
        steps = t + 1
        if t == 0:
            first_step = {
                "next_obs": np.asarray(nobs, dtype=np.float32),
                "done": np.float32(1.0 if term else 0.0),
                "z_next": env.privileged_state().astype(np.float32),
            }
        prev_dist = float(info["disk_to_zone"])
        if term or trunc:
            break
        a = _greedy(frozen_actor, nobs)
    disc = np.power(gamma, np.arange(steps, dtype=np.float64))
    returns = {c: float(np.dot(disc, np.asarray(seq[c][:steps], dtype=np.float64))) for c in COMPONENTS}
    first_step["branch_len"] = steps
    # undiscounted trajectory sums for the fingertip-progress ratio (a trajectory-level objective, not a critic)
    first_step["ft_prog_sum"] = float(np.sum(seq["progress"][:steps]))
    first_step["body_prog_sum"] = float(np.sum(seq["body_progress"][:steps]))
    return returns, first_step
