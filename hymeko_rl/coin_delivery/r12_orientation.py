"""R12.2 — orientation-aware geometric model: non-invasive yaw-varied handoff adapter.

The frozen delivery pipeline places every object at yaw=0 — `_home_with_coin` sets only `qpos[4:6]=(x,y)` and never the
planar yaw `qpos[6]` (the `disk_rz` hinge). That is exactly why the R12.1/T2 probe found object orientation pinned to
≤1.9°: the benchmark never varies it. This adapter places the object at a COMMANDED yaw and runs the SAME
reach/capture/certify pipeline (every acquisition primitive reused unchanged), so R12.2 can ask whether a certified
straddle grasp *preserves* varied orientation. Nothing in the yaw=0 pipeline is modified — the frozen R11.6C/R11.7
results are untouched by construction.

The coin/box body is a 3-DOF planar joint (`disk_x` slide @ qpos4, `disk_y` slide @ qpos5, `disk_rz` hinge @ qpos6);
verified against the live model, not the (misleading) "6-DoF freejoint" docstring in `fixed_position.py`.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import mujoco  # type: ignore[import-untyped]  # mujoco ships no stubs
import numpy as np

from hymeko_rl.coin_delivery import ir_adapter as A
from hymeko_rl.coin_delivery.delivery_bc.dataset import descriptor
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.exact_zero_composition import (
    _DRIFT_EPS, CompositionOutcomeClass, CompositionRecord, HandoffResult)
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option.home_states import HomeState, build_home_snapshot

_COIN_YAW_QADR = 6              # disk_rz hinge — the planar yaw (verified: qpos = arms[0:4] ++ disk_x,disk_y,disk_rz)
_ = _DRIFT_EPS                  # re-exported for callers that reproduce the frozen drift gate


def home_with_coin_yaw(rig: dict[str, Any], coin_xy: np.ndarray, yaw: float) -> "tuple[Any, np.ndarray]":
    """Exact-zero home ``q=[0,0,0,0]`` with the object placed at ``(coin_xy, yaw)``. Mirrors ``_home_with_coin`` but
    ALSO sets the planar yaw ``qpos[6]`` (the frozen builder leaves it 0, so ``yaw=0`` reproduces it exactly).

    # Preconditions: ``yaw`` finite; ``coin_xy`` a 2-vector; ``rig`` has the 3-DOF planar coin joint (qpos[6]=disk_rz).
    # Postconditions: the returned snapshot's coin ``qpos[6] == yaw``; identical to ``_home_with_coin`` when ``yaw==0``.
    """
    if not math.isfinite(yaw):
        raise ValueError(f"yaw must be finite, got {yaw}")
    stack = rig["cradle"].stack
    home = build_home_snapshot(rig["cradle"], HomeState(name="ZERO", q=np.zeros(4), mode="ZERO", description="true zero"))
    rl = home.branch()
    rl.inner.data.qpos[4:6] = np.asarray(coin_xy, float)
    rl.inner.data.qpos[_COIN_YAW_QADR] = float(yaw)
    mujoco.mj_forward(rl.inner.model, rl.inner.data)
    return kc.TransportSnapshot.from_live(copy.deepcopy(rl), stack, np.zeros(4)), np.asarray(coin_xy, float)


def object_yaw(snap: Any) -> float:
    """World-frame yaw (rad) of the object geom from its rotation matrix — joint-agnostic (works pre/post grasp and
    regardless of the joint parameterization). Same measure as the R12.1/T2 probe, so spreads are comparable."""
    inner = snap._rl.inner
    m, d = inner.model, inner.data
    gid = next((g for g in range(m.ngeom)
                if any(k in (m.geom(g).name or "") for k in ("disk", "coin", "box"))), None)
    if gid is None:
        return float("nan")
    r = np.asarray(d.geom_xmat[gid], np.float64).reshape(3, 3)
    return math.atan2(r[1, 0], r[0, 0])


def reach_capture_at_yaw(rig: dict[str, Any], scen: Any, seed: int, yaw: float, cfg: Any, conf: Any, obj: Any) -> HandoffResult:
    """``reach_capture_descriptor`` from a yaw-varied home: exact-zero IC → RRT reach → certified straddle capture →
    live 30-D descriptor. Reuses every acquisition primitive (``A``/``P``/``mp``/``descriptor``) UNCHANGED — only the
    home carries the yaw — so it inherits the same contracts (admissibility, certified-grasp gate) as the frozen path.

    # Preconditions: ``scen`` a valid coin/target scenario; ``yaw`` finite. # Postconditions: ``record`` set iff the
    chain failed before delivery, else ``snap``/``x`` carry the yaw-varied handoff (drift check is the caller's, as in
    the frozen path — the placed yaw is intentional drift from the stored yaw=0 descriptor, so it is NOT gated here).
    """
    sid, split = scen.scenario_id, scen.split.value

    def fail(klass: CompositionOutcomeClass, **kw: Any) -> HandoffResult:
        return HandoffResult(CompositionRecord(sid, split, seed, klass.value, **kw))

    home, coin = home_with_coin_yaw(rig, scen.coin_xy, yaw)
    ic = A.EXACT_ZERO_HOME_V1.certify(A.read_rollout_state(home.branch()))
    if not ic.valid or not A.coin_admissibility(rig, scen.coin_xy, cfg).admissible:
        return fail(CompositionOutcomeClass.INVALID_INITIAL_CONDITION)
    reason, rcap = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
    if rcap is None:
        return fail(CompositionOutcomeClass.REACH_FAILURE, reach_reason=reason)
    if not rcap.handoff_admissible:
        return fail(CompositionOutcomeClass.PRECONTACT_HANDOFF_INVALID)
    if not mp.is_certified_grasp(rcap.result.outcome, obj):
        return fail(CompositionOutcomeClass.CAPTURE_NO_CERTIFIED_GRASP)
    snap = rcap.result.outcome.snapshot
    return HandoffResult(None, snap=snap, x=descriptor(scen, rcap, snap))
