"""Sustained-PUSH audit — quantify the contact-coverage gap between the scripted expert and the DAgger clone.

The prior fair-retest closed per-step refinement and measured PUSH states at 1/14,280 for the perturbed DAgger
policy. This module is the diagnostic that makes that gap first-class: for any policy it rolls monitored episodes
and reports the two-fingertip **contact windows** (contiguous runs of `both_contact`), their duration histogram,
coin/fingertip/body progress *inside* those windows, delivery conditioned on the longest window, phase
distribution, and exploit/arm-body rates.

**Sustained PUSH window** := a run of ≥ K consecutive `both_contact` steps with positive coin toward-progress in
the window, negligible body-only progress, and no arm-body contact — i.e. a genuine two-finger push, not a graze
or a body shove. All window logic reuses `contiguous_runs` + `MonitorContext` (no new primitives).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from hymeko_rl.eval.critic_benchmark import phase_label
from hymeko_rl.eval.task_monitor.contract import MonitorContract
from hymeko_rl.eval.task_monitor.context import MonitorContext, contiguous_runs, record_trajectory

_PHASES = ("APPROACH", "CONTACT", "PUSH", "DELIVERY")
_DURATION_BINS = (0, 1, 3, 5, 10, 20, 10_000)   # both-contact run-length bins


@dataclass
class PushAuditResult:
    name: str
    n_episodes: int
    delivery_rate: float
    phase_dist: dict[str, float]                       # fraction of steps per monitor-physical phase
    both_contact_frac: float                           # fraction of all steps with two-finger contact
    sustained_push_per_ep: float                       # mean # of sustained-PUSH windows per episode
    mean_push_window_len: float                        # mean length of qualifying windows (steps)
    max_push_window_len: float                         # longest qualifying window seen
    duration_histogram: dict[str, int]                 # all both-contact run lengths, binned
    delivery_by_max_window: dict[str, float]           # delivery rate conditioned on longest both-contact run bin
    ft_progress_in_contact: float                      # summed fingertip-attributed progress inside windows / ep
    body_progress_in_contact: float                    # summed body-only progress inside windows / ep
    arm_body_rate: float                               # fraction of episodes with any arm-body contact
    exploit_rate: float                                # fraction of episodes graded body-driven (delivered w/ body>ft)
    violation_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def sustained_windows_raw(both_contact: np.ndarray, toward: np.ndarray, body_prog_step: np.ndarray,
                          body_contact: np.ndarray, *, progress_eps: float, body_eps: float,
                          k: int) -> list[tuple[int, int]]:
    """The single definition of a sustained-PUSH window, on raw per-step arrays: a run of ≥ k consecutive
    ``both_contact`` steps with positive coin toward-progress, negligible body-only progress, and no arm-body
    contact in the run. Used by the audit (via a MonitorContext) and by demo tagging (via a live rollout)."""
    out: list[tuple[int, int]] = []
    for s, e in contiguous_runs(both_contact):
        if (e - s) < k:
            continue
        if (float(toward[s:e].sum()) > progress_eps and float(body_prog_step[s:e].sum()) <= body_eps
                and not bool(body_contact[s:e].any())):
            out.append((s, e))
    return out


def sustained_push_windows(ctx: MonitorContext, contract: Any, k: int) -> list[tuple[int, int]]:
    """Sustained-PUSH windows for a built :class:`MonitorContext` (thin wrapper over
    :func:`sustained_windows_raw`; single definition of 'sustained PUSH')."""
    return sustained_windows_raw(ctx.both_contact, ctx.toward, ctx.body_prog_step, ctx.body_contact,
                                 progress_eps=contract.progress_eps, body_eps=contract.body_eps, k=k)


def _bin_label(x: int, bins: tuple[int, ...]) -> str:
    for i in range(len(bins) - 1):
        if bins[i] <= x < bins[i + 1]:
            hi = bins[i + 1]
            return f"[{bins[i]},{hi if hi < 10_000 else '∞'})"
    return f"[{bins[-2]},∞)"


def audit_policy(make_env: Callable[[], Any], make_action_fn: Callable[[Any], Any], *,
                 name: str, n_episodes: int, seed0: int = 9000, k_sustained: int = 5,
                 progress_eps: float = 0.002, near_coin: float = 0.06) -> PushAuditResult:
    """Roll ``n_episodes`` monitored episodes of the policy and compute the sustained-PUSH audit.

    # Preconditions: ``make_action_fn(env)`` returns a fresh ``action_fn(env, obs)`` per episode (so a stateful
      demonstrator's FSM is reset); ``make_env()`` builds a PlanarGraspEnv.
    # Postconditions: a :class:`PushAuditResult`; no training, no side effects on any shared object."""
    env0 = make_env()
    contract = MonitorContract.from_env(env0)
    phase_steps = {p: 0 for p in _PHASES}
    total_steps = 0
    both_steps = 0
    delivered_n = arm_body_n = exploit_n = 0
    n_windows = 0
    win_lens: list[int] = []
    all_run_lens: list[int] = []
    ft_in = body_in = 0.0
    deliver_by_bin: dict[str, list[int]] = {}
    for ep in range(n_episodes):
        env = make_env()
        traj = record_trajectory(env, make_action_fn(env), seed0 + ep)
        if not traj:
            continue
        ctx = MonitorContext.build(traj, contract)
        n = ctx.n
        total_steps += n
        both_steps += int(ctx.both_contact.sum())
        delivered = bool(ctx.max_dwell >= contract.success_steps)
        delivered_n += int(delivered)
        arm_body_n += int(bool(ctx.body_contact.any()))
        exploit_n += int(delivered and ctx.body_prog > ctx.ft_prog)
        for t in range(n):
            phase_steps[phase_label(min_tip=float(ctx.min_tip[t]), both_contact=bool(ctx.both_contact[t]),
                                    toward=float(ctx.toward[t]), in_zone=bool(ctx.in_zone[t]),
                                    near_coin=near_coin, progress_eps=progress_eps)] += 1
        runs = contiguous_runs(ctx.both_contact)
        all_run_lens.extend(e - s for s, e in runs)
        max_run = max((e - s for s, e in runs), default=0)
        deliver_by_bin.setdefault(_bin_label(max_run, _DURATION_BINS), []).append(int(delivered))
        for s, e in sustained_push_windows(ctx, contract, k_sustained):
            n_windows += 1
            win_lens.append(e - s)
            ft_in += float(ctx.ft_prog_step[s:e].sum())
            body_in += float(ctx.body_prog_step[s:e].sum())
    ne = max(1, n_episodes)
    ts = max(1, total_steps)
    hist: dict[str, int] = {}
    for length in all_run_lens:
        hist[_bin_label(length, _DURATION_BINS)] = hist.get(_bin_label(length, _DURATION_BINS), 0) + 1
    return PushAuditResult(
        name=name, n_episodes=n_episodes, delivery_rate=delivered_n / ne,
        phase_dist={p: phase_steps[p] / ts for p in _PHASES},
        both_contact_frac=both_steps / ts,
        sustained_push_per_ep=n_windows / ne,
        mean_push_window_len=float(np.mean(win_lens)) if win_lens else 0.0,
        max_push_window_len=float(max(win_lens)) if win_lens else 0.0,
        duration_histogram=hist,
        delivery_by_max_window={b: float(np.mean(v)) for b, v in sorted(deliver_by_bin.items())},
        ft_progress_in_contact=ft_in / ne, body_progress_in_contact=body_in / ne,
        arm_body_rate=arm_body_n / ne, exploit_rate=exploit_n / ne)
