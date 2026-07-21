"""Fixed-position Coin Delivery replay execution — reachability analysis, fail-loud gates, and the traced two-phase
rollout shared by the deterministic-problem and exact-state replay paths.

The rollout mirrors the canonical ``experiments.coin_neutral_start.eval_composed`` two-phase loop (E-approach until a
grasp/contact handoff, then the learned transport until strict delivery) but records the full per-step trace (coin
trajectory, contact, actions, phase transitions) so a replay can be certified, hashed, videoed and compared.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from hymeko_rl.coin_delivery.fixed_position import CoinInitialState

_MAX_APPROACH = 160
_MAX_TRANSPORT = 200


# ── reachability / validity analysis (§6) ────────────────────────────────────────────────────────────────────────────
@dataclass
class Reachability:
    signed_clearance: float                 # disk_to_zone - (disk_r + zone_half); > 0 ⇔ coin outside the zone
    fingertip_distances: dict[str, float]   # coin-centre distance to each spherical fingertip
    coin_reachable: bool                    # coin within both arms' reach annulus
    collision_free: bool                    # no MuJoCo contacts at the initial state
    initial_contact: dict[str, bool]        # left/right fingertip contact at t0
    disk_to_zone: float
    disk_radius: float
    zone_half: float

    def to_dict(self) -> dict[str, Any]:
        return {"signed_clearance": round(self.signed_clearance, 6), "disk_to_zone": round(self.disk_to_zone, 6),
                "fingertip_distances": {k: round(v, 6) for k, v in self.fingertip_distances.items()},
                "coin_reachable": self.coin_reachable, "collision_free": self.collision_free,
                "initial_contact": self.initial_contact, "disk_radius": round(self.disk_radius, 6),
                "zone_half": round(self.zone_half, 6)}


def analyze_reachability(env: Any, cf: Any) -> Reachability:
    """Compute the initial-state geometry report (already-applied env). # Postconditions no mutation of the env."""
    from hymeko_rl.env.planar_arm_kinematics import extract_arms
    inner = cf._env
    d, m = inner.data, inner._planar_metrics
    coin = np.asarray(m.disk_pos[:2], dtype=np.float64)
    disk_r = float(inner.model.geom_size[inner._disk_geom][0])
    dtz = float(m.disk_to_zone)
    signed_clr = dtz - (disk_r + float(inner._zone_half))
    arms = extract_arms(inner.model)
    ft_dist, reach_any = {}, False
    for side, arm in arms.items():
        tip = np.asarray(d.site_xpos[arm.tip_site][:2], dtype=np.float64)
        ft_dist[side] = float(np.linalg.norm(tip - coin))
        base = np.asarray(arm.base_xy, dtype=np.float64)
        r = float(np.linalg.norm(coin - base))
        reach_any = reach_any or (r <= arm.l1 + arm.l2 + 1e-3)   # a fingertip of at least one arm can touch the coin
    # "collision-free" = the arms are not already touching the coin (the coin resting on the table is benign, so we
    # do NOT gate on raw ncon which counts table contacts); use the contact-legality state.
    lg = getattr(m, "legality", None)
    arm_body = bool(lg.arm_body_contact) if lg is not None else False
    collision_free = not (bool(m.left_contact) or bool(m.right_contact) or arm_body)
    return Reachability(signed_clearance=signed_clr, fingertip_distances=ft_dist, coin_reachable=reach_any,
                        collision_free=collision_free,
                        initial_contact={"left": bool(m.left_contact), "right": bool(m.right_contact)},
                        disk_to_zone=dtz, disk_radius=disk_r, zone_half=float(inner._zone_half))


# ── fail-loud replay gates (§2) ──────────────────────────────────────────────────────────────────────────────────────
def assert_replayable(env: Any, cf: Any, state: CoinInitialState, *, embodiment: str, neutral_start: bool,
                      require_checkpoints: bool = True) -> dict[str, Any]:
    """Fail loud (no fallback to a generated seed / contact bank) if the applied state is not a legal replay target.

    Rejects: embodiment mismatch, obs/action schema mismatch, checkpoint-hash mismatch, initial contact when a neutral
    start is required, coin overlapping the target, or a non-finite / unreachable state. # Errors
    :class:`~hymeko_rl.coin_delivery.fixed_position.InvalidInitialState` / ``FileNotFoundError``.
    """
    from hymeko_rl.coin_delivery.fixed_position import InvalidInitialState, env_fingerprint, verify_checkpoint_hashes
    if state.embodiment != embodiment:
        raise InvalidInitialState(f"embodiment mismatch: state={state.embodiment!r} requested={embodiment!r}")
    fp = env_fingerprint(cf)
    if fp["obs_space"] != [6, 8] or fp["action_space"] != [4]:
        raise InvalidInitialState(f"observation/action schema mismatch: {fp} (expected obs [6,8], action [4])")
    reach = analyze_reachability(env, cf)
    if not np.all(np.isfinite(cf._env.data.qpos)) or not np.all(np.isfinite(cf._env.data.qvel)):
        raise InvalidInitialState("applied state has non-finite qpos/qvel")
    if neutral_start and (reach.initial_contact["left"] or reach.initial_contact["right"]):
        raise InvalidInitialState(f"neutral start requested but the state begins in contact: {reach.initial_contact}")
    if reach.signed_clearance <= 0.0:
        raise InvalidInitialState(f"coin already overlaps the target (signed clearance "
                                  f"{reach.signed_clearance:+.4f} <= 0)")
    ckpt = verify_checkpoint_hashes() if require_checkpoints else {}
    return {"fingerprint": fp, "reachability": reach.to_dict(), "checkpoint_hashes": ckpt}


# ── traced two-phase rollout (mirrors eval_composed) ─────────────────────────────────────────────────────────────────
@dataclass
class ReplayTrace:
    strict_delivered: bool
    first_contact: bool
    bilateral_contact: bool
    targetward_motion: bool                 # coin got monotonically closer to the zone at some point
    zone_entry: bool
    handoff_step: int
    completion_step: int                    # step at which strict delivery certified (or -1)
    completion_time_s: float
    n_steps: int
    coin_xy: list = field(default_factory=list)          # per control step
    contact: list = field(default_factory=list)          # per step [left, right]
    actions: list = field(default_factory=list)          # per step executed action
    phase: list = field(default_factory=list)            # per step "approach"|"transport"
    failure_reason: str = ""

    def trajectory_hash(self) -> str:
        """A hash of the deterministic trajectory (coin xy + actions + contact) — the §5 bit-identity handle.

        Actions are ragged across phases (4-DoF approach vs 6-DoF transport), so they are hashed element-wise.
        """
        h = hashlib.sha256()
        h.update(np.asarray(self.coin_xy, dtype=np.float64).round(9).tobytes())
        for a in self.actions:
            h.update(np.asarray(a, dtype=np.float64).round(9).tobytes())
        h.update(np.asarray(self.contact, dtype=np.int8).tobytes())
        return h.hexdigest()[:16]

    def to_summary(self) -> dict[str, Any]:
        return {"strict_delivered": self.strict_delivered, "first_contact": self.first_contact,
                "bilateral_contact": self.bilateral_contact, "targetward_motion": self.targetward_motion,
                "zone_entry": self.zone_entry, "handoff_step": self.handoff_step,
                "completion_step": self.completion_step, "completion_time_s": round(self.completion_time_s, 4),
                "n_steps": self.n_steps, "trajectory_hash": self.trajectory_hash(),
                "failure_reason": self.failure_reason}


def composed_rollout(env: Any, cf: Any, approach: Any, transport_fn: Callable | None, *, grasp_hold: int = 1,
                     contact_window: int | None = 20, policy: str = "P4_E_APPROACH_HANDOFF",
                     on_step: Callable | None = None) -> ReplayTrace:
    """Run the two-phase composed chain FROM THE CURRENT env state and record the full trace.

    ``policy`` selects the causal control: ``P4_E_APPROACH_HANDOFF`` (E → learned transport), ``P1_FROZEN_TRANSPORT``
    (E → frozen transport ``transport_fn``), ``P0_ZERO_ACTION`` (no actions at all). Mirrors ``eval_composed`` exactly
    (same 160/200 caps, same handoff rule) but starts from the pre-applied state instead of ``reset(seed)``.
    ``on_step(phase, step_index, delivered)`` is invoked after every control step (for video capture).
    """
    import torch

    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance
    inner = cf._env
    tr = ReplayTrace(strict_delivered=False, first_contact=False, bilateral_contact=False, targetward_motion=False,
                     zone_entry=False, handoff_step=-1, completion_step=-1, completion_time_s=0.0, n_steps=0)
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    start_dtz = float(inner._planar_metrics.disk_to_zone)
    best_dtz = start_dtz

    def _record(action: np.ndarray, phase: str) -> None:
        m = inner._planar_metrics
        tr.coin_xy.append([float(m.disk_pos[0]), float(m.disk_pos[1])])
        tr.contact.append([bool(m.left_contact), bool(m.right_contact)])
        tr.actions.append([float(x) for x in np.asarray(action, np.float64).ravel()])
        tr.phase.append(phase)

    if policy == "P0_ZERO_ACTION":
        # zero-action control: step the OUTER env (6-DoF transport action space) with zeros, like eval_neutral
        for _t in range(_MAX_TRANSPORT):
            cert.update(_cert_step(inner, cf))
            m = inner._planar_metrics
            tr.first_contact |= bool(m.left_contact or m.right_contact)
            tr.bilateral_contact |= bool(m.left_contact and m.right_contact)
            best_dtz = min(best_dtz, float(m.disk_to_zone))
            tr.zone_entry |= float(m.disk_to_zone) <= 0.02
            if cert.delivery_certified:
                tr.strict_delivered = True
                tr.completion_step = _t
                break
            _record(np.zeros(6, np.float32), "zero")
            env.step(np.zeros(6, np.float32))
            if on_step is not None:
                on_step("zero", _t, False)
        tr.n_steps = len(tr.coin_xy)
        tr.targetward_motion = (start_dtz - best_dtz) > 1e-3
        _finalize(tr, best_dtz, start_dtz)
        return tr

    # phase 1 — E-approach until handoff (running bi/cw counters, byte-for-byte the eval_composed rule)
    bi = cw = 0
    for _k in range(_MAX_APPROACH):
        m = inner._planar_metrics
        cert.update(_cert_step(inner, cf))
        tr.first_contact |= bool(m.left_contact or m.right_contact)
        tr.bilateral_contact |= bool(m.left_contact and m.right_contact)
        best_dtz = min(best_dtz, float(m.disk_to_zone))
        bi = bi + 1 if (m.left_contact and m.right_contact) else 0
        cw = cw + 1 if (m.left_contact or m.right_contact) else 0
        if bi >= grasp_hold:
            tr.handoff_step = _k
            break
        if contact_window is not None and cw >= contact_window:
            tr.handoff_step = _k
            break
        with torch.no_grad():
            a = approach.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].numpy()
        _record(a, "approach")
        inner.step(np.asarray(a, np.float32))
        if on_step is not None:
            on_step("approach", _k, False)
    if tr.handoff_step < 0:
        tr.handoff_step = len(tr.coin_xy)

    # handoff bookkeeping (identical to eval_composed)
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    cf._t = 0
    cf._both_hist = []
    env._suffix_t = 0
    env._prev_dtz = env._dtz()
    env._prev_both = env._both()
    o = cf._obs(np.zeros(4, np.float32))

    # phase 2 — transport until strict delivery
    for _t in range(_MAX_TRANSPORT):
        cert.update(_cert_step(inner, cf))
        m = inner._planar_metrics
        best_dtz = min(best_dtz, float(m.disk_to_zone))
        tr.zone_entry |= float(m.disk_to_zone) <= 0.02
        if cert.delivery_certified:
            tr.strict_delivered = True
            tr.completion_step = tr.handoff_step + _t
            break
        act = transport_fn(env, o, None)
        _record(act, "transport")
        o = env.step(np.asarray(act, np.float32))[0]
        if on_step is not None:
            on_step("transport", tr.handoff_step + _t, False)
    tr.n_steps = len(tr.coin_xy)
    tr.targetward_motion = (start_dtz - best_dtz) > 1e-3
    _finalize(tr, best_dtz, start_dtz)
    return tr


def _finalize(tr: ReplayTrace, best_dtz: float, start_dtz: float) -> None:
    from hymeko_rl.coin_delivery.fixed_position import CONTROL_DT
    tr.completion_time_s = (tr.completion_step if tr.completion_step >= 0 else tr.n_steps) * CONTROL_DT
    if tr.strict_delivered:
        tr.failure_reason = ""
    elif not tr.first_contact:
        tr.failure_reason = "no_first_contact"
    elif not tr.bilateral_contact:
        tr.failure_reason = "no_bilateral_contact"
    elif not tr.targetward_motion:
        tr.failure_reason = "no_targetward_motion"
    elif not tr.zone_entry:
        tr.failure_reason = "reached_but_no_zone_entry"
    else:
        tr.failure_reason = "zone_entry_but_not_strict_held"


def build_actors(policy: str):
    """Load the (approach, transport_fn) for a causal-control policy from the verified checkpoints."""
    import torch

    from hymeko_rl.coin_delivery.e_approach import load_e_approach_policy
    from hymeko_rl.coin_delivery.fixed_position import _CKPT_MANIFEST
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_delivery_e0_stabilize import build_sac
    if policy == "P0_ZERO_ACTION":
        return None, None
    approach = load_e_approach_policy()
    ck = _CKPT_MANIFEST["handoff_transport" if policy == "P4_E_APPROACH_HANDOFF" else "frozen_transport"][0]
    actor, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(torch.load(ck, weights_only=True))
    return approach, _greedy_action_fn(actor)
