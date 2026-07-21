"""The position-based trajectory tensor + its precomputed struct-of-arrays view.

``record_trajectory`` rolls a policy for one episode and records ONLY positions / velocities / contacts /
distances / role labels (a top-down camera view — no policy internals). ``MonitorContext.build`` precomputes,
once per trajectory, the derived per-step arrays every submonitor reads (toward-zone progress, fingertip- vs
body-attributed progress, dwell, engagement timing). Data-oriented: the submonitors read slices of shared
contiguous arrays rather than re-walking the trajectory each.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsDistanceSeries(Protocol):
    """Minimal reward-independent progress interface: a step count + a distance-to-target series.

    Any monitor that needs nothing beyond progress-over-distance (e.g. :class:`StagnationMonitor`) accepts this
    instead of the coin-specific :class:`MonitorContext`, so it is task-agnostic. Both :class:`MonitorContext`
    (coin-delivery) and :class:`~hymeko_rl.eval.task_monitor.metaworld.CoffeePushContext` (MetaWorld) satisfy it
    structurally — no inheritance required.
    """

    n: int
    dist: np.ndarray


def record_trajectory(env: Any, action_fn: Any, seed: int) -> list[dict[str, Any]]:
    """Roll ``action_fn(env, obs)`` for one episode; record the position-based monitor tensor per step
    (positions, velocities, contacts, distances, role labels) — NO policy internals, NO reward."""
    obs, _ = env.reset(seed=seed)
    traj: list[dict[str, Any]] = []
    done = False
    while not done:
        obs, _r, term, trunc, info = env.step(np.asarray(action_fn(env, obs), np.float32))
        m = env._planar_metrics
        lg = m.legality
        traj.append({
            "coin_xy": np.array([float(m.disk_pos[0]), float(m.disk_pos[1])]),
            "coin_vel": np.array([float(m.disk_vel[0]), float(m.disk_vel[1])]),
            "dist_to_zone": float(m.disk_to_zone),
            "left_tip_contact": bool(m.left_contact), "right_tip_contact": bool(m.right_contact),
            "arm_body_contact": bool(lg.arm_body_contact) if lg is not None else False,
            "left_tip_dist": float(m.left_tip_dist), "right_tip_dist": float(m.right_tip_dist),
            "in_zone": bool(info.get("in_zone", m.in_zone)),
        })
        done = bool(term or trunc)
    return traj


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index runs where ``mask`` is true (for violation trajectory slices)."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


@dataclass
class MonitorContext:
    """Precomputed struct-of-arrays view of one trajectory, shared by every submonitor."""

    contract: Any
    n: int
    # --- raw per-step arrays (the top-down camera) ---
    dist: np.ndarray            # [N] coin distance to zone centre
    coin_xy: np.ndarray         # [N, 2]
    coin_vel: np.ndarray        # [N, 2]
    left_tip_dist: np.ndarray   # [N]
    right_tip_dist: np.ndarray  # [N]
    min_tip: np.ndarray         # [N] min(left, right) tip-coin distance
    left_contact: np.ndarray    # [N] bool
    right_contact: np.ndarray   # [N] bool
    both_contact: np.ndarray    # [N] bool
    body_contact: np.ndarray    # [N] bool (arm-body ↔ coin)
    in_zone: np.ndarray         # [N] bool
    # --- derived per-step arrays ---
    toward: np.ndarray          # [N] max(0, dist[t-1]-dist[t]) toward-zone displacement
    ft_prog_step: np.ndarray    # [N] toward under fingertip contact
    body_prog_step: np.ndarray  # [N] toward under body-only contact
    dwell: np.ndarray           # [N] consecutive in-zone step count
    # --- scalars ---
    d0: float
    dfin: float
    ft_prog: float
    body_prog: float
    total_prog: float
    max_dwell: int
    two_finger_steps: int
    first_two_finger: int       # 1-indexed step of first both-fingertip contact (0 = never)
    first_delivery: int         # 1-indexed step dwell first reached k (0 = never)
    first_approach: int         # 1-indexed step min_tip first within near_coin (0 = never)

    @classmethod
    def build(cls, traj: list[dict[str, Any]], contract: Any) -> "MonitorContext":
        n = len(traj)
        dist = np.array([s["dist_to_zone"] for s in traj], np.float64)
        coin_xy = np.array([s["coin_xy"] for s in traj], np.float64)
        coin_vel = np.array([s["coin_vel"] for s in traj], np.float64)
        lt = np.array([s["left_tip_dist"] for s in traj], np.float64)
        rt = np.array([s["right_tip_dist"] for s in traj], np.float64)
        lc = np.array([s["left_tip_contact"] for s in traj], bool)
        rc = np.array([s["right_tip_contact"] for s in traj], bool)
        bc = np.array([s["arm_body_contact"] for s in traj], bool)
        iz = np.array([s["in_zone"] for s in traj], bool)
        both = lc & rc
        fingertip = lc | rc
        toward = np.zeros(n)
        toward[1:] = np.maximum(0.0, dist[:-1] - dist[1:])
        ft_prog_step = np.where(fingertip, toward, 0.0)
        body_prog_step = np.where(bc & ~fingertip, toward, 0.0)
        dwell = np.zeros(n, int)
        run = 0
        for t in range(n):
            run = run + 1 if iz[t] else 0
            dwell[t] = run
        k = int(contract.success_steps)
        first_both = int(np.argmax(both) + 1) if both.any() else 0
        deliv_mask = dwell >= k
        first_deliv = int(np.argmax(deliv_mask) + 1) if deliv_mask.any() else 0
        appr_mask = np.minimum(lt, rt) <= contract.near_coin
        first_appr = int(np.argmax(appr_mask) + 1) if appr_mask.any() else 0
        ft_prog = float(ft_prog_step.sum())
        body_prog = float(body_prog_step.sum())
        return cls(
            contract=contract, n=n, dist=dist, coin_xy=coin_xy, coin_vel=coin_vel,
            left_tip_dist=lt, right_tip_dist=rt, min_tip=np.minimum(lt, rt),
            left_contact=lc, right_contact=rc, both_contact=both, body_contact=bc, in_zone=iz,
            toward=toward, ft_prog_step=ft_prog_step, body_prog_step=body_prog_step, dwell=dwell,
            d0=float(dist[0]), dfin=float(dist[-1]), ft_prog=ft_prog, body_prog=body_prog,
            total_prog=ft_prog + body_prog, max_dwell=int(dwell.max()),
            two_finger_steps=int(both.sum()), first_two_finger=first_both,
            first_delivery=first_deliv, first_approach=first_appr,
        )
