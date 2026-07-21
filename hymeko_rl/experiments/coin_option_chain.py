"""APPROACH → CAPTURE → frozen-TRANSPORT option chain — the decomposition the dwell-relay result motivated.

The bridge-dwell iteration proved the relay is causally alive once the handoff contract is sticky (frozen transport
runs to completion; fall back only on genuine physical failure, never on loss of readiness) — but a monolithic bridge
could not reach AND hold a ready state from clear-start (approach-to-basin wall). This module splits the learned work
into two options with explicit named-field boundaries: APPROACH (clear-start → first useful fingertip contact) and
CAPTURE (contact → bilateral bracket → one-step TRANSPORT_READY), then the frozen TRANSPORT_POLICY owns the rest under
the sticky contract. Each option is trained separately on its own reachable-state bank; delivery-v2b / strict predicate /
frozen transport / detector labels are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.env.planar_snapshot import PlanarSnapshot
from hymeko_rl.eval.team_tensor import field_index
from hymeko_rl.experiments.coin_bridge_relay import (
    ReadinessDetector, ReadyLabel, _I_BODY, _I_BOTH, _I_C2T_X, _I_C2T_Y, _I_LEFT, _I_RIGHT,
    _restore_generated, greedy_fn,
)

_ACT = 6
_I_L2C_X, _I_L2C_Y = field_index("l_to_coin_x"), field_index("l_to_coin_y")
_I_R2C_X, _I_R2C_Y = field_index("r_to_coin_x"), field_index("r_to_coin_y")


def _min_ft_dist(o: np.ndarray) -> float:
    """Closest fingertip→coin distance (named fields) — the APPROACH potential."""
    return float(min(np.hypot(o[_I_L2C_X], o[_I_L2C_Y]), np.hypot(o[_I_R2C_X], o[_I_R2C_Y])))


def _dtz(o: np.ndarray) -> float:
    return float(np.hypot(o[_I_C2T_X], o[_I_C2T_Y]))


# ── Phase 4: reachable intermediate state banks (bucketed from labelled candidates by contact state) ───────────────
def option_banks(labels: list[ReadyLabel]) -> "dict[str, list[PlanarSnapshot]]":
    """Bucket the labelled candidate states into the option banks by named contact fields. A1/C0 = exactly one
    fingertip; C1 = bilateral without shove; T0 = the empirical TRANSPORT_READY basin. (A0 precontact = the clear-start
    corpora, passed separately.)"""
    banks: dict[str, list[PlanarSnapshot]] = {"A1_first_contact": [], "C1_bilateral": [], "T0_ready": []}
    for lab in labels:
        o = lab_feats_contact(lab)
        if lab.label == "TRANSPORT_READY":
            banks["T0_ready"].append(lab.snapshot)
        elif o["both"] and not o["body"]:
            banks["C1_bilateral"].append(lab.snapshot)
        elif o["left"] ^ o["right"]:
            banks["A1_first_contact"].append(lab.snapshot)
    return banks


def lab_feats_contact(lab: ReadyLabel) -> dict:
    f = lab.features
    # features order: l2c_x,l2c_y,r2c_x,r2c_y,left,right,both,body,aperture,... (see coin_bridge_relay._FEAT_NAMES)
    return dict(left=bool(f[4] > 0.5), right=bool(f[5] > 0.5), both=bool(f[6] > 0.5), body=bool(f[7] > 0.5))


# ── Phase 5: APPROACH reward env (terminate on FIRST valid fingertip contact) ─────────────────────────────────────
@dataclass
class ApproachReward:
    w_approach: float = 12.0       # potential: decrease in closest fingertip→coin distance
    w_corridor: float = 0.5        # per-step while closing (both fingertips getting closer)
    w_body: float = 3.0            # penalty: arm-body shove
    w_away: float = 4.0            # penalty: coin pushed away from target while approaching
    w_action: float = 0.01
    r_contact: float = 20.0        # terminal: first valid fingertip contact (dominant)


class ApproachRewardEnv:
    """Terminate on FIRST valid fingertip contact (left or right, no body). Reward closing fingertip→coin distance."""

    def __init__(self, inner_env: Any, pool: list[PlanarSnapshot], rng: np.random.Generator,
                 reward: ApproachReward | None = None) -> None:
        if not pool:
            raise ValueError("ApproachRewardEnv needs a non-empty clear-start pool")
        self.env, self._pool, self._rng = inner_env, pool, rng
        self._rw = reward or ApproachReward()
        self.observation_space, self.action_space = inner_env.observation_space, inner_env.action_space
        self.max_steps = getattr(inner_env, "max_steps", 60)
        self._prev_d = self._prev_dtz = 0.0

    def reset_to(self, snap: PlanarSnapshot) -> np.ndarray:
        o = _restore_generated(self.env, snap)
        self._prev_d, self._prev_dtz = _min_ft_dist(o), _dtz(o)
        return o

    def reset(self, *, seed: int | None = None):
        return self.reset_to(self._pool[self._rng.integers(len(self._pool))]), {}

    def step(self, action: np.ndarray):
        o, _rv, _term, trunc, info = self.env.step(action)
        rw = self._rw
        d = _min_ft_dist(o)
        r = rw.w_approach * (self._prev_d - d)
        if d < self._prev_d:
            r += rw.w_corridor
        self._prev_d = d
        left, right, body = o[_I_LEFT] > 0.5, o[_I_RIGHT] > 0.5, o[_I_BODY] > 0.5
        if body:
            r -= rw.w_body
        dtz = _dtz(o)
        if dtz > self._prev_dtz + 1e-4:
            r -= rw.w_away * (dtz - self._prev_dtz)
        self._prev_dtz = dtz
        r -= rw.w_action * float(np.mean(np.square(action)))
        terminated = bool((left or right) and not body)            # first valid fingertip contact
        if terminated:
            r += rw.r_contact
        return o, float(r), terminated, bool(trunc), info


# ── Phase 6: CAPTURE reward env (terminate on FIRST TRANSPORT_READY entry) ────────────────────────────────────────
@dataclass
class CaptureReward:
    w_potential: float = 12.0      # potential: decrease in distance to the nearest empirical ready state
    w_bilateral: float = 3.0       # one-time: one-sided → bilateral
    w_symmetry: float = 1.0        # per-step: both fingertips in contact (symmetry)
    w_body: float = 3.0
    w_lose_all: float = 3.0        # penalty: lost ALL contact
    w_oscillate: float = 1.0       # penalty: flip between left-only and right-only
    w_action: float = 0.01
    r_ready: float = 25.0          # terminal: first TRANSPORT_READY (dominant)


class CaptureRewardEnv:
    """Start from contact banks (A1/C1). Terminate on FIRST TRANSPORT_READY (one step — no dwell)."""

    def __init__(self, inner_env: Any, detector: ReadinessDetector, pool: list[PlanarSnapshot],
                 rng: np.random.Generator, reward: CaptureReward | None = None) -> None:
        if not pool:
            raise ValueError("CaptureRewardEnv needs a non-empty contact pool")
        self.env, self._det, self._pool, self._rng = inner_env, detector, pool, rng
        self._rw = reward or CaptureReward()
        self.observation_space, self.action_space = inner_env.observation_space, inner_env.action_space
        self.max_steps = getattr(inner_env, "max_steps", 60)
        self._prev_dist = 0.0
        self._had_bilateral = False
        self._prev_side = 0                                         # -1 left-only, +1 right-only, 0 else

    def reset_to(self, snap: PlanarSnapshot) -> np.ndarray:
        o = _restore_generated(self.env, snap)
        self._prev_dist = self._det.distance(o)
        self._had_bilateral = bool(o[_I_BOTH] > 0.5)
        self._prev_side = 0
        return o

    def reset(self, *, seed: int | None = None):
        return self.reset_to(self._pool[self._rng.integers(len(self._pool))]), {}

    def step(self, action: np.ndarray):
        o, _rv, _term, trunc, info = self.env.step(action)
        rw = self._rw
        dist = self._det.distance(o)
        r = rw.w_potential * (self._prev_dist - dist)
        self._prev_dist = dist
        left, right, both, body = o[_I_LEFT] > 0.5, o[_I_RIGHT] > 0.5, o[_I_BOTH] > 0.5, o[_I_BODY] > 0.5
        if both and not self._had_bilateral:
            r += rw.w_bilateral
            self._had_bilateral = True
        if both:
            r += rw.w_symmetry
        if body:
            r -= rw.w_body
        if not (left or right):
            r -= rw.w_lose_all
        side = -1 if (left and not right) else (1 if (right and not left) else 0)
        if side != 0 and self._prev_side != 0 and side != self._prev_side:
            r -= rw.w_oscillate                                    # flip left-only ↔ right-only
        if side != 0:
            self._prev_side = side
        r -= rw.w_action * float(np.mean(np.square(action)))
        terminated = bool(self._det.is_ready(o, currently_transport=False))   # first TRANSPORT_READY
        if terminated:
            r += rw.r_ready
        return o, float(r), terminated, bool(trunc), info


# ── Phase 3: the option-chain controller (APPROACH → CAPTURE → sticky frozen TRANSPORT) ────────────────────────────
@dataclass
class ChainLog:
    first_contact_step: int = -1
    capture_step: int = -1
    handoff_step: int = -1
    handoffs: int = 0
    fallbacks: int = 0
    handoff_hash: str = ""
    opt_trace: list[str] = field(default_factory=list)


class OptionChainController:
    """APPROACH until first valid fingertip contact → CAPTURE until one-step TRANSPORT_READY → sticky frozen TRANSPORT
    (owns the rest; falls back to CAPTURE only on body shove / sustained genuine stall — NEVER on loss of readiness)."""

    def __init__(self, approach: Any, capture: Any, transport: Any, detector: ReadinessDetector, *,
                 stall_window: int = 8) -> None:
        self._ap, self._cap, self._tp = greedy_fn(approach), greedy_fn(capture), greedy_fn(transport)
        self._det, self._stall = detector, int(stall_window)

    def act_fn(self, log: ChainLog) -> Callable[[Any, int, np.ndarray], np.ndarray]:
        st = {"opt": "APPROACH", "best_dtz": None, "no_prog": 0}

        def act(inner: Any, t: int, obs: np.ndarray) -> np.ndarray:
            o = np.asarray(obs)
            left, right, body = o[_I_LEFT] > 0.5, o[_I_RIGHT] > 0.5, o[_I_BODY] > 0.5
            if (left or right) and log.first_contact_step < 0:
                log.first_contact_step = t
            opt = st["opt"]
            if opt == "APPROACH":
                if (left or right) and not body:
                    st["opt"] = "CAPTURE"
                    log.capture_step = t
            elif opt == "CAPTURE":
                if self._det.is_ready(o, currently_transport=False):
                    st["opt"] = "TRANSPORT"                         # irreversible one-step handoff
                    log.handoffs += 1
                    log.handoff_step = t
                    st["best_dtz"] = _dtz(o)
                    st["no_prog"] = 0
            else:                                                  # TRANSPORT — sticky
                d = _dtz(o)
                if d < st["best_dtz"] - 1e-4:
                    st["best_dtz"] = d
                    st["no_prog"] = 0
                else:
                    st["no_prog"] += 1
                if body or st["no_prog"] >= self._stall:           # genuine physical failure only
                    st["opt"] = "CAPTURE"
                    log.fallbacks += 1
            opt = st["opt"]
            log.opt_trace.append(opt[0])
            return (self._ap if opt == "APPROACH" else self._cap if opt == "CAPTURE" else self._tp)(inner, t, o)
        return act


def chain_rollout(env: Any, approach: Any, capture: Any, transport: Any, detector: ReadinessDetector,
                  snap: PlanarSnapshot, *, max_steps: int = 60) -> "tuple[Any, ChainLog]":
    from hymeko_rl.train.coin_delivery_actor import rollout
    _restore_generated(env, snap)
    log = ChainLog()
    ctrl = OptionChainController(approach, capture, transport, detector)
    trace = rollout(env, ctrl.act_fn(log), max_steps=max_steps)
    return trace, log


def train_option(env_maker: Callable[[Any, np.random.Generator], Any], *, steps: int, seed: int,
                 warm_from: str | None = None, init_actor: Any = None, log_every: int = 10_000) -> Any:
    """Train one option (SAC) on its reward env. Warm-started from the transport policy (a sensible motion init) or a
    previous band's checkpoint. Returns (actor, critics)."""
    from hymeko_rl.experiments.coin_generator_exp import direct_env
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    inner = direct_env()
    inner._base_override = lambda _i, _t: np.zeros(_ACT, np.float32)
    inner._delta_override = 1.0
    rng = np.random.default_rng(seed)
    renv = env_maker(inner, rng)
    if init_actor is not None:
        actor, critics = init_actor
    else:
        actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=_ACT, action_scale=1.0)
        if warm_from:
            actor.load_state_dict(torch.load(warm_from, map_location="cpu"))
    train_sac(actor, critics, renv, SACConfig.stable(total_steps=steps, seed=seed, bc_coef=0.0,
              log_every=log_every, eval_every=max(steps, 1) + 1))
    return actor, critics
