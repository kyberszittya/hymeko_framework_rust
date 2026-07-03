"""Runtime rollout health monitor — catch the silent failures that waste long runs.

Two physics failure modes, both measured 2026-06-30 and both expensive when undetected:

  * **blow-up** — a contact penetration detonates the solver (``qacc`` explodes); bodies are ejected, and a
    metric that trusts post-blow-up state reads a *false success* (the FANUC gripper's 0.875 "lift" was this).
  * **stall** — the policy freezes the joints and burns the whole episode going nowhere; a multi-seed/overnight
    run then grinds for hours on a dead policy.

This watches the **physics** (``qacc`` divergence + actuated-joint motion) and aborts the rollout at the first
failure, so a campaign can skip the dead cell instead of paying for it. It is complementary to the HTL/STL
*task-logic* monitor (:mod:`hymeko_rl.control.htl_reward`): that one judges whether the reward predicate is satisfied;
this one judges whether the simulation is even valid. Reuse this — do not re-roll an ad-hoc qacc check per script.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco  # type: ignore[import-untyped]  # mujoco ships no py.typed stubs
import numpy as np

ActionFn = Callable[[object, np.ndarray], np.ndarray]


def actuated_dofs(model: object) -> tuple[int, ...]:
    """The DOF addresses driven by joint actuators — the joints a policy actually controls. Excludes passive
    bodies (e.g. a free-joint object), so stall is judged on the *arm*, not on a jittering box."""
    m = model
    return tuple(sorted({int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])])      # type: ignore[attr-defined]
                         for a in range(int(m.nu))                            # type: ignore[attr-defined]
                         if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)}))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class RolloutHealth:
    """Verdict for one monitored rollout. ``event_step`` is where the first failure was flagged (else ``None``)."""
    steps: int
    diverged: bool
    stalled: bool
    event_step: "int | None"
    max_qacc: float
    moved: float            # cumulative actuated-joint motion (rad); ~0 over the window = frozen

    @property
    def healthy(self) -> bool:
        return not (self.diverged or self.stalled)

    def summary(self) -> str:
        if self.diverged:
            return f"DIVERGED @ step {self.event_step} (|qacc|max={self.max_qacc:.0f}) — blow-up, result invalid"
        if self.stalled:
            return f"STALLED @ step {self.event_step} — joints frozen, no progress"
        return f"healthy ({self.steps} steps, moved {self.moved:.2f} rad)"


class RolloutMonitor:
    """Watch a rollout for a physics blow-up and a joint stall.

    # Preconditions ``env`` is a MuJoCo gym env (exposes ``data.qacc``/``data.qpos`` + the gym 5-tuple ``step``);
      ``action_fn(env, obs) -> action``. # Postconditions ``run`` returns a :class:`RolloutHealth`; with
      ``abort`` (default) it stops at the first failure rather than finishing the episode.
    """

    def __init__(self, *, diverge_qacc: float = 5e3, stall_window: int = 60, stall_eps: float = 1e-3,
                 abort: bool = True) -> None:
        if stall_window < 1 or diverge_qacc <= 0.0 or stall_eps < 0.0:
            raise ValueError("require stall_window>=1, diverge_qacc>0, stall_eps>=0")
        self.diverge_qacc = float(diverge_qacc)
        self.stall_window = int(stall_window)
        self.stall_eps = float(stall_eps)
        self.abort = bool(abort)

    def run(self, env: object, action_fn: ActionFn, *, max_steps: int, seed: int = 0,
            motion_dofs: "list[int] | tuple[int, ...] | None" = None) -> RolloutHealth:
        """Roll ``action_fn`` on ``env`` for up to ``max_steps``, monitoring health. ``motion_dofs`` selects the
        joints whose stillness counts as a stall (default: the env's actuated DOFs if it exposes them, else all)."""
        obs, _ = env.reset(seed=seed)                                          # type: ignore[attr-defined]
        data = env.data                                                        # type: ignore[attr-defined]
        if motion_dofs is None:
            auto = getattr(env, "_act_dofs", None)
            if auto is None:
                auto = getattr(env, "_arm_dofs", None)
            motion_dofs = tuple(int(i) for i in auto) if auto is not None else actuated_dofs(env.model)  # type: ignore[attr-defined]
        idx = list(motion_dofs) if len(motion_dofs) else None
        prev_q = np.asarray(data.qpos).copy()
        diverged = stalled = False
        event: "int | None" = None
        still = 0
        moved = 0.0
        max_qa = 0.0
        steps = 0
        for t in range(int(max_steps)):
            steps = t + 1
            a = np.asarray(action_fn(env, obs), dtype=np.float32)
            obs, _r, term, trunc, _info = env.step(a)                          # type: ignore[attr-defined]
            qa = float(np.abs(data.qacc).max())
            max_qa = max(max_qa, qa)
            if (not np.isfinite(qa) or qa > self.diverge_qacc) and not diverged:
                diverged, event = True, (event if event is not None else t)
                if self.abort:
                    return RolloutHealth(steps, True, stalled, event, max_qa, moved)
            dq = np.asarray(data.qpos) - prev_q
            prev_q = np.asarray(data.qpos).copy()
            step_motion = float(np.abs(dq[idx]).sum()) if idx is not None else float(np.abs(dq).sum())
            moved += step_motion
            still = still + 1 if step_motion < self.stall_eps else 0
            if still >= self.stall_window and not stalled:
                stalled, event = True, (event if event is not None else t)
                if self.abort:
                    return RolloutHealth(steps, diverged, True, event, max_qa, moved)
            if term or trunc:
                break
        return RolloutHealth(steps, diverged, stalled, event, max_qa, moved)
