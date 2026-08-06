"""Live-state handoff oracle + earliest-solvable-frontier mapping.

The option-chain result localized the wall to CAPTURE: the empirical TRANSPORT_READY bank was dominated by LATE
near-goal states (the frozen policy's own late trajectory), so reaching it demanded transport-like motion. This module
tests the deploy-matched question directly: during a real APPROACH→CAPTURE rollout, at each post-contact timestep, hand
control to the frozen TRANSPORT policy IN-ROLLOUT (no reset — the live MuJoCo + wrapper state is preserved, contact
history and phase intact) under the sticky ownership contract, and record strict delivery. The EARLIEST timestep from
which frozen transport still finishes is the earliest deploy-solvable handoff frontier — which may be far earlier
(farther from goal) than the near-goal bank. Frozen transport is allowed to leave the handoff region while transporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.env.planar_snapshot import PlanarSnapshot
from hymeko_rl.experiments.coin_bridge_relay import _I_BODY, _I_LEFT, _I_RIGHT, _restore_generated, greedy_fn
from hymeko_rl.experiments.coin_clearance_curriculum import _clearance
from hymeko_rl.experiments.coin_option_chain import _dtz
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import rollout

_STALL = 8


class _ForcedHandoffController:
    """APPROACH → (first contact) → CAPTURE → at absolute step ``handoff_step`` (given a contact) FORCE the handoff to
    the frozen transport policy, then STICKY (fall back only on body-shove / stall). Used to sweep the handoff time."""

    def __init__(self, approach: Any, capture: Any, transport: Any, handoff_step: int) -> None:
        self._ap, self._cap, self._tp = greedy_fn(approach), greedy_fn(capture), greedy_fn(transport)
        self._hstep = int(handoff_step)
        self.handoff_t = -1
        self.handoff_dtz = float("nan")

    def act_fn(self):
        st = {"opt": "APPROACH", "best_dtz": None, "no_prog": 0, "contacted": False}

        def act(inner: Any, t: int, obs: np.ndarray) -> np.ndarray:
            o = np.asarray(obs)
            left, right, body = o[_I_LEFT] > 0.5, o[_I_RIGHT] > 0.5, o[_I_BODY] > 0.5
            st["contacted"] = st["contacted"] or bool(left or right)
            opt = st["opt"]
            if opt == "APPROACH":
                if (left or right) and not body:
                    st["opt"] = "CAPTURE"
            elif opt == "CAPTURE":
                if st["contacted"] and t >= self._hstep:           # FORCED live handoff at the swept step
                    st["opt"] = "TRANSPORT"
                    self.handoff_t = t
                    self.handoff_dtz = _dtz(o)
                    st["best_dtz"] = _dtz(o)
                    st["no_prog"] = 0
            else:
                d = _dtz(o)
                if d < st["best_dtz"] - 1e-4:
                    st["best_dtz"] = d
                    st["no_prog"] = 0
                else:
                    st["no_prog"] += 1
                if body or st["no_prog"] >= _STALL:
                    st["opt"] = "CAPTURE"
            opt = st["opt"]
            return (self._ap if opt == "APPROACH" else self._cap if opt == "CAPTURE" else self._tp)(inner, t, o)
        return act


@dataclass
class FrontierResult:
    solvable: bool
    earliest_step: int
    handoff_dtz: float
    handoff_clearance: float
    transport_horizon: int          # steps from handoff to episode end
    snapshot: PlanarSnapshot


def earliest_frontier(env: Any, approach: Any, capture: Any, transport: Any, snap: PlanarSnapshot, *,
                      max_steps: int = 60, sweep: range | None = None) -> FrontierResult:
    """Sweep the forced-handoff step from early→late; return the EARLIEST live handoff that yields strict delivery."""
    clr0 = (_restore_generated(env, snap), _clearance(env.inner))[1]
    for hstep in (sweep or range(2, max_steps, 2)):
        _restore_generated(env, snap)
        ctrl = _ForcedHandoffController(approach, capture, transport, hstep)
        tr = rollout(env, ctrl.act_fn(), max_steps=max_steps)
        if ctrl.handoff_t >= 0 and bool(policy_strict(tr)):
            return FrontierResult(True, ctrl.handoff_t, round(float(ctrl.handoff_dtz), 4), round(float(clr0), 4),
                                  len(tr.steps) - ctrl.handoff_t, snapshot_at(env, snap, ctrl.handoff_t, approach,
                                                                             capture, transport, hstep))
    return FrontierResult(False, -1, float("nan"), round(float(clr0), 4), -1, snap)


def snapshot_at(env: Any, snap: PlanarSnapshot, handoff_t: int, approach: Any, capture: Any, transport: Any,
                hstep: int) -> PlanarSnapshot:
    """Re-run APPROACH→CAPTURE to the handoff step and snapshot the live physical state (the frontier state)."""
    from hymeko_rl.env.planar_snapshot import snapshot_planar
    _restore_generated(env, snap)
    ctrl = _ForcedHandoffController(approach, capture, transport, 10 ** 9)   # never hand off — just drive CAPTURE
    fn = ctrl.act_fn()
    obs = env._last_obs
    for t in range(handoff_t):
        a = np.clip(fn(env.inner, t, obs), -1, 1).astype(np.float32)
        obs, _r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return snapshot_planar(env.inner)


def map_frontier(env: Any, approach: Any, capture: Any, transport: Any,
                 snaps: list[PlanarSnapshot]) -> "tuple[list[FrontierResult], dict]":
    """Map the earliest live-solvable frontier over ``snaps``; return the results + a geometry summary."""
    results = [earliest_frontier(env, approach, capture, transport, s) for s in snaps]
    solv = [r for r in results if r.solvable]
    summary = dict(
        n=len(snaps), n_solvable=len(solv),
        earliest_step=sorted(r.earliest_step for r in solv)[:12],
        handoff_dtz=[r.handoff_dtz for r in solv],
        handoff_clearance=[r.handoff_clearance for r in solv],
        transport_horizon=[r.transport_horizon for r in solv],
        median_earliest=(float(np.median([r.earliest_step for r in solv])) if solv else None),
        median_handoff_dtz=(round(float(np.median([r.handoff_dtz for r in solv])), 4) if solv else None),
        median_horizon=(float(np.median([r.transport_horizon for r in solv])) if solv else None))
    return results, summary
