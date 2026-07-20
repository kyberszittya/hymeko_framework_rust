"""COIN-DELIVERY-RL-1 — post-handoff transport & centering via a PPO residual around ``grasp_carry``.

DELIVERY-ENV-0 (F-COIN-DELIVERY-SEMANTICS) established that the coin task is DELIVERY to the zone (handoff is a phase
event, not success) and is feasible under corrected semantics: scripted ``grasp_carry`` reaches ~0.73 zone-entry / ~0.52
center-reach at horizon 120. This module asks whether reward-driven RL can improve the *post-handoff transport and
centering* beyond that scripted controller — phase-conditioned control:

    1. Scripted acquisition PREFIX: run ``grasp_carry`` (zero residual) until the handoff phase event (or a cap).
    2. RL transport SUFFIX: hand control to a PPO **residual** policy around ``grasp_carry`` until center-reach /
       zone-entry (metric) / safety / corrected horizon.

Executed action (the residual abstraction; zero residual == scripted ``grasp_carry`` byte-identically, the hard
invariant tested in Stage 0):

    a_exec = clip( a_grasp_carry(inner) + delta * tanh(policy(obs)),  lo, hi )

Reward is a monitor-aligned, POTENTIAL-BASED delivery reward (Ng-Harada-Russell shaping with Φ = −disk_to_zone, hence
policy-invariant / non-farmable): progress toward the zone centre + one-shot zone-entry + the strongest center-reach
event − stall − contact-drop. NO per-step holding/grasp bonus, NO handoff bonus, NO accumulating lift/contact reward.

Reuses ``train.ppo.train_ppo`` (unchanged), ``agents.policy.build_policy`` (the residual policy), and
``coin_delivery1.p_grasp_carry`` (the base primitive). NO env-dynamics/reward/CORE change — this is a non-invasive
TRAINING harness over the unchanged ``ContactFormationEnv`` (the training sibling of ``delivery_env0.CoinDeliveryEvalWrapper``,
which owns the eval-only, no-reward version of these semantics).

disk_to_zone is coin-centre → zone-centre (planar Euclidean; verified ``planar_grasp_env.py:293-295``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from hymeko_rl.experiments.coin_delivery1 import p_grasp_carry
from hymeko_rl.experiments.pedc_selection import _env
from hymeko_rl.train.coin_delivery_actor import rollout


@dataclass(frozen=True)
class DeliveryRLConfig:
    """Phase-conditioned residual-transport RL configuration.

    ``delta`` bounds the residual (``|a_exec − base| <= delta`` per dim before clipping); ``prefix_cap`` bounds the
    scripted acquisition prefix; ``horizon`` is the post-handoff transport-suffix budget (DELIVERY-ENV-0's adequate
    horizon); ``center_tol`` is the center-reach terminal (coin-centre within it of the zone-centre); ``zone_half`` is
    the looser zone-entry radius (a secondary metric only). Reward weights follow the potential-based delivery form.

    # Invariants ``delta > 0``; ``prefix_cap >= 1``; ``horizon >= 1``; ``0 < center_tol <= zone_half``; ``lo < hi``.
    """

    delta: float = 0.3
    prefix_cap: int = 40
    horizon: int = 120
    center_tol: float = 0.02
    zone_half: float = 0.04
    lo: float = -1.0
    hi: float = 1.0
    # potential-based delivery reward weights
    w_progress: float = 10.0      # per metre closed toward the zone centre (Φ = −disk_to_zone shaping)
    w_zone: float = 1.0           # one-shot on first zone entry (disk_to_zone <= zone_half)
    w_center: float = 3.0         # the STRONGEST single event: center reach (disk_to_zone <= center_tol)
    w_stall: float = 0.02         # per-step penalty while GENUINELY stuck (a stagnation RUN, not merely slow)
    w_drop: float = 0.5           # ONE-TIME penalty for the first both-contact loss before delivery (an EVENT)
    stall_eps: float = 1e-3       # |Δ disk_to_zone| below this = "no transport" this step
    stall_window: int = 8         # a stall penalty applies only after this many CONSECUTIVE no-transport steps
    #                               (so a slow-but-progressing carry is NOT penalised — only a stuck one)

    def __post_init__(self) -> None:
        if self.delta <= 0 or self.prefix_cap < 1 or self.horizon < 1:
            raise ValueError("delta>0, prefix_cap>=1, horizon>=1 required")
        if not (0.0 < self.center_tol <= self.zone_half) or self.lo >= self.hi:
            raise ValueError("require 0 < center_tol <= zone_half and lo < hi")


def residual_action(base: np.ndarray, raw: np.ndarray, delta: float, lo: float, hi: float) -> np.ndarray:
    """Compose the executed action ``clip(base + delta*tanh(raw), lo, hi)``.

    # Preconditions ``base``, ``raw`` broadcast to the same shape; ``delta > 0``; ``lo < hi``.
    # Postconditions returns float32 in ``[lo, hi]``; ``raw == 0`` (or ``tanh(raw)==0``) yields ``clip(base, lo, hi)``
      — the hard zero-residual invariant (with ``base`` already in ``[lo, hi]`` this is exactly ``base``)."""
    b = np.asarray(base, dtype=np.float32)
    a = b + float(delta) * np.tanh(np.asarray(raw, dtype=np.float32))
    return np.clip(a, lo, hi).astype(np.float32)


def delivery_reward(prev_dtz: float, dtz: float, *, entered_zone_now: bool, center_now: bool,
                    stalled: bool, dropped: bool, cfg: DeliveryRLConfig) -> float:
    """Potential-based delivery reward for one transport step (Φ = −disk_to_zone).

    # Postconditions transport progress (prev>dtz) contributes positively; holding still contributes ``<= 0``; moving
      away contributes ``< 0``; ``center_now`` is the largest single positive event; no term rewards holding/handoff."""
    r = cfg.w_progress * (prev_dtz - dtz)                 # potential shaping: γΦ(s')−Φ(s), γ≈1 → policy-invariant
    if entered_zone_now:
        r += cfg.w_zone
    if center_now:
        r += cfg.w_center
    if stalled:
        r -= cfg.w_stall
    if dropped:
        r -= cfg.w_drop
    return float(r)


def zero_residual_head(ac: Any) -> None:
    """Zero-init the residual policy's action head (``actor_mean``) so the deterministic policy starts at the scripted
    ``grasp_carry`` (zero residual) — the ``exp_v14d._zero_head`` pattern, adapted to :class:`ActorCritic`."""
    import torch
    head = getattr(ac, "actor_mean", None)
    if head is None:
        raise AttributeError("ActorCritic has no actor_mean head to zero-init")
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)


class CoinDeliveryTrainEnv:
    """Phase-conditioned delivery-transport RL env (satisfies ``train.ppo.RolloutEnv``).

    ``reset`` runs the scripted ``grasp_carry`` acquisition prefix (zero residual) until the handoff phase event (or a
    cap / safety), then returns the handover obs. ``step`` applies the residual transport action and the potential-based
    delivery reward, terminating on center-reach (main success) / safety, truncating at the transport horizon. Handoff
    NEVER terminates (delivery semantics); the underlying env's own reward is unused. Dynamics/CORE unchanged.

    # Invariants the wrapped ``ContactFormationEnv`` is stepped with actions in ``[lo, hi]``; ``disk_to_zone`` is
      coin-centre → zone-centre; a zero raw action reproduces the scripted ``grasp_carry`` transport."""

    def __init__(self, env: Any, cfg: DeliveryRLConfig | None = None) -> None:
        self.env = env
        self.inner = env._env                                    # the planar MuJoCo env (metrics live here)
        self.cfg = cfg or DeliveryRLConfig()
        self._delta_override: float | None = None               # per-episode residual scale (Stage-1 oracle / per-class δ)
        self._base_override: "Callable[[Any, int], np.ndarray] | None" = None  # replace the grasp_carry SUFFIX base
        self._last_obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        self._start_obs = self._last_obs.copy()                 # pre-acquisition obs (for hard-state geometry features)
        self._reset_state()

    @property
    def delta(self) -> float:
        """The active residual scale — ``_delta_override`` if set, else the config default."""
        return self.cfg.delta if self._delta_override is None else float(self._delta_override)

    # ── contract accessors ────────────────────────────────────────────────────────────────────────────────────────
    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def _reset_state(self) -> None:
        self._suffix_t = 0
        self._had_both = False
        self._prev_both = False           # both-contact on the previous step (for the drop falling-edge)
        self._dropped_once = False        # the one-time drop event already fired this episode
        self._stag_run = 0                # consecutive no-transport steps (windowed stall)
        self._zone_entered = False
        self._center_reached = False
        self._handoff = False
        self._prefix_safety = False
        self._prev_dtz = self._start_dtz = float(self.inner._planar_metrics.disk_to_zone)
        self._resid_abs: list[float] = []
        self._sat_hits = 0
        self._sat_total = 0

    def _dtz(self) -> float:
        return float(self.inner._planar_metrics.disk_to_zone)

    def _both(self) -> bool:
        m = self.inner._planar_metrics
        return bool(m.left_contact and m.right_contact)

    def _base(self) -> np.ndarray:
        """The SUFFIX base action at the current state, clipped into the action box: ``grasp_carry`` by default, or a
        ``base_override`` primitive (e.g. the centering primitive) when set. The acquisition PREFIX always uses
        ``grasp_carry`` (it steps ``p_grasp_carry`` directly in :meth:`reset`), so an override affects only transport."""
        prim = self._base_override(self.inner, self._suffix_t) if self._base_override is not None \
            else p_grasp_carry(self.inner, self._suffix_t)
        return np.clip(prim, self.cfg.lo, self.cfg.hi).astype(np.float32)

    # ── RolloutEnv API ────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        # let the underlying env run past its own handoff-termination for the whole prefix+suffix (delivery semantics)
        self.env._horizon = self.cfg.prefix_cap + self.cfg.horizon + 8
        self._reset_state()
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._start_obs = self._last_obs.copy()                # pre-acquisition geometry (coin/zone/vector obs fields)
        for _ in range(self.cfg.prefix_cap):                    # scripted acquisition prefix (grasp_carry, zero residual)
            acquire = np.clip(p_grasp_carry(self.inner, 0), self.cfg.lo, self.cfg.hi).astype(np.float32)
            obs, _r, _term, _trunc, sinfo = self.env.step(acquire)
            self._last_obs = np.asarray(obs, dtype=np.float32)
            self._had_both = self._had_both or self._both()
            if sinfo.get("safety_violation"):
                self._prefix_safety = True
                break
            if sinfo.get("handoff_ready"):
                self._handoff = True
                break
        self._prev_dtz = self._start_dtz = self._dtz()
        self._prev_both = self._both()                          # seed the drop falling-edge at handover
        return self._last_obs, {"handoff_event": self._handoff, "high_coin_y": bool(info.get("high_coin_y", False))}

    def step(self, raw: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._prefix_safety:                                 # acquisition already failed → degenerate terminal
            return self._last_obs, -self.cfg.w_drop, True, False, self._info(safety=True)
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        base = self._base()
        a_exec = residual_action(base, raw, self.delta, self.cfg.lo, self.cfg.hi)
        self._record_telemetry(base, raw, a_exec)
        obs, _r, _term, _trunc, sinfo = self.env.step(a_exec)   # dynamics + env reward UNUSED (delivery semantics)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._suffix_t += 1
        reward, terminated, truncated, safety, dtz = self._transition(sinfo)
        return self._last_obs, reward, terminated, truncated, self._info(safety=safety, dtz=dtz)

    # ── transition / reward bookkeeping (kept small for the complexity gate) ─────────────────────────────────────────
    def _event_flags(self, dtz: float, both: bool, zone_before: bool) -> tuple[bool, bool, bool, bool]:
        """The four reward events this step: (first zone entry, center reached, windowed stall, drop edge)."""
        entered_now = (not zone_before) and (dtz <= self.cfg.zone_half)
        center_now = dtz <= self.cfg.center_tol
        # windowed stall: penalise only a genuine stagnation RUN, not an inherently-slow (but progressing) carry
        self._stag_run = self._stag_run + 1 if abs(self._prev_dtz - dtz) < self.cfg.stall_eps else 0
        stalled = (self._stag_run >= self.cfg.stall_window) and (not zone_before)
        # drop: a ONE-TIME event on the first both-contact FALLING EDGE before zone entry (not a sustained state)
        dropped = (not self._dropped_once) and self._prev_both and (not both) and (not zone_before)
        return entered_now, center_now, stalled, dropped

    def _transition(self, sinfo: dict) -> tuple[float, bool, bool, bool, float]:
        dtz, both = self._dtz(), self._both()
        zone_before = self._zone_entered
        entered_now, center_now, stalled, dropped = self._event_flags(dtz, both, zone_before)
        reward = delivery_reward(self._prev_dtz, dtz, entered_zone_now=entered_now, center_now=center_now,
                                 stalled=stalled, dropped=dropped, cfg=self.cfg)
        self._dropped_once = self._dropped_once or dropped
        self._zone_entered = zone_before or (dtz <= self.cfg.zone_half)
        self._center_reached = self._center_reached or center_now
        self._had_both = self._had_both or both
        self._prev_both = both
        self._prev_dtz = dtz
        safety = bool(sinfo.get("safety_violation"))
        terminated = bool(center_now or safety)
        truncated = bool(self._suffix_t >= self.cfg.horizon)
        return reward, terminated, truncated, safety, dtz

    def _record_telemetry(self, base: np.ndarray, raw: np.ndarray, a_exec: np.ndarray) -> None:
        resid = a_exec - np.clip(base, self.cfg.lo, self.cfg.hi)
        self._resid_abs.append(float(np.abs(resid).mean()))
        th = np.abs(np.tanh(raw))
        self._sat_hits += int(np.sum(th > 0.99))
        self._sat_total += int(th.size)

    def _info(self, *, safety: bool, dtz: float | None = None) -> dict[str, Any]:
        return {"handoff_event": self._handoff, "delivery_success": self._zone_entered,
                "center_reached": self._center_reached, "contact_lost": self._dropped_once,
                "safety_violation": safety, "disk_to_zone": self._dtz() if dtz is None else dtz}

    # ── telemetry summary ─────────────────────────────────────────────────────────────────────────────────────────
    def start_geometry(self) -> dict[str, float]:
        """Pre-acquisition geometry from the start obs (ACTOR_FIELDS): coin/zone positions + coin→zone vector."""
        o = self._start_obs
        return {"coin_x": float(o[0]), "coin_y": float(o[1]), "zone_x": float(o[4]), "zone_y": float(o[5]),
                "coin_to_zone_x": float(o[18]), "coin_to_zone_y": float(o[19]),
                "start_dist": float(np.hypot(o[18], o[19]))}

    def residual_norm(self) -> float:
        return float(np.mean(self._resid_abs)) if self._resid_abs else 0.0

    def saturation_rate(self) -> float:
        return float(self._sat_hits / self._sat_total) if self._sat_total else 0.0


def make_delivery_rl_env(cfg: DeliveryRLConfig | None = None) -> CoinDeliveryTrainEnv:
    """Build a fresh delivery-transport RL env over the held ``ContactFormationEnv`` (reuses ``pedc_selection._env``)."""
    return CoinDeliveryTrainEnv(_env(), cfg or DeliveryRLConfig())


# ── deterministic evaluation (Strategy over the action source) ─────────────────────────────────────────────────────
ActionFn = Callable[[np.ndarray], np.ndarray]


def scripted_action_fn() -> ActionFn:
    """Zero residual → the scripted ``grasp_carry`` baseline (the zero-residual invariant, as an action source)."""
    return lambda _obs: np.zeros(6, dtype=np.float32)


def greedy_action_fn(ac: Any) -> ActionFn:
    """The trained policy's deterministic mean action (raw; the env applies ``delta*tanh``)."""
    import torch

    def fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            raw = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0)
        return raw.detach().numpy().astype(np.float32)
    return fn


def roll_delivery(env: CoinDeliveryTrainEnv, seed: int, action_fn: ActionFn) -> dict[str, Any]:
    """Roll one deterministic delivery episode through the canonical :func:`rollout`; derive the delivery/precision/
    telemetry bundle from the PUBLIC trace (per-step ``info`` + reward) rather than a private re-implemented loop."""
    env.reset(seed=int(seed))
    start_dtz = env._start_dtz
    trace = rollout(env, lambda _inner, _t, obs: action_fn(obs), max_steps=env.cfg.horizon)
    min_dtz = start_dtz
    t_zone = t_center = None
    for t, s in enumerate(trace.steps):
        min_dtz = min(min_dtz, float(s.info["disk_to_zone"]))
        if s.info["delivery_success"] and t_zone is None:
            t_zone = t
        if s.info["center_reached"] and t_center is None:
            t_center = t
    info = trace.steps[-1].info if trace.steps else {
        "delivery_success": False, "center_reached": False, "handoff_event": env._handoff,
        "contact_lost": False, "disk_to_zone": start_dtz}
    ret = sum(float(s.reward) for s in trace.steps)
    return {"zone_entry": bool(info["delivery_success"]), "center_reach": bool(info["center_reached"]),
            "handoff_event": bool(info["handoff_event"]), "final_dtz": round(float(info["disk_to_zone"]), 4),
            "min_dtz": round(min_dtz, 4), "start_dtz": round(start_dtz, 4), "time_to_zone": t_zone,
            "time_to_center": t_center, "contact_lost": bool(info["contact_lost"]), "return": round(ret, 4),
            "residual_norm": round(env.residual_norm(), 4), "saturation": round(env.saturation_rate(), 4)}


def _rate(rows: list[dict], key: str) -> float:
    return round(sum(bool(r[key]) for r in rows) / max(1, len(rows)), 4)


def _med(rows: list[dict], key: str, *, only: str | None = None) -> float | None:
    vals = [r[key] for r in rows if (only is None or r[only]) and r[key] is not None]
    return round(float(np.median(vals)), 4) if vals else None


def eval_delivery(action_fn: ActionFn, seeds, cfg: DeliveryRLConfig | None = None,
                  *, env: CoinDeliveryTrainEnv | None = None) -> dict[str, Any]:
    """Aggregate deterministic delivery metrics over ``seeds`` for one action source (scripted or trained)."""
    cfg = cfg or DeliveryRLConfig()
    env = env or make_delivery_rl_env(cfg)
    rows = [roll_delivery(env, s, action_fn) for s in seeds]
    return {
        "n": len(rows),
        "zone_entry": _rate(rows, "zone_entry"),
        "center_reach": _rate(rows, "center_reach"),
        "handoff_event": _rate(rows, "handoff_event"),
        "contact_lost": _rate(rows, "contact_lost"),
        "grasp_no_delivery": round(sum(r["handoff_event"] and not r["zone_entry"] for r in rows) / max(1, len(rows)), 4),
        "final_dtz_med": _med(rows, "final_dtz"),
        "min_dtz_med": _med(rows, "min_dtz"),
        "time_to_zone_med": _med(rows, "time_to_zone", only="zone_entry"),
        "time_to_center_med": _med(rows, "time_to_center", only="center_reach"),
        "return_med": _med(rows, "return"),
        "residual_norm_med": _med(rows, "residual_norm"),
        "saturation_med": _med(rows, "saturation"),
        "rows": rows,
    }
