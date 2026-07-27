"""R9 — bounded CAUSAL per-step residual correction over the FROZEN R8 champion.

R8 emitted a CONSTANT residual `a_R8` per option (the dev-selected TD3 champion's `mean(init_obs)`). R9 keeps that frozen
base and learns only a bounded, temporally-coherent INCREMENT `Δa_t` on top of it, at a short fixed control interval:

    a_exec_t = clip(a_R8 + Δa_t, −1, 1)          (Δa held between control decisions; slew- and magnitude-bounded)

The three residual roles are unchanged (forward-velocity / squeeze / stop-gain). There is NO direct release action — the R6
release certificate remains the SOLE release authority. The Bellman action is `Δa_t` ONLY; the frozen base `a_R8`, the
combined `a_exec`, the scaffold targets, torque and certificate internals are provenance, never the label. At `Δa ≡ 0` the
adapter reproduces the R8 champion trajectory bit-for-bit (a_exec = clip(a_R8) = a_R8) — the R9 update-zero identity.

Reuses the R8 `_base_targets` / `_observe` / `_servo` seams unchanged; only the action computation (a_R8 + causal Δa) and the
provenance (Δa is the Bellman action) differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.forward_displacement import _coin_speed
from hymeko_rl.coin_delivery.theta_option.residual_adapter import (
    ResidualBounds, ResidualTipAdapter, _coin_spin)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams

DELTA_ROLES = ("dd_fwd_vel", "dd_squeeze", "dd_stop_gain")   # the causal INCREMENT over the frozen R8 residual
HISTORY_LEN = 4                                              # short causal window (recent control decisions)
_HIST_FEAT = 6                                               # per past decision: dtz, v_par, coin_speed, Δa[0..2]
CAUSAL_DIM = 11 + 3 + 3 + HISTORY_LEN * _HIST_FEAT           # obs(11) + a_R8(3) + prev Δa(3) + history


@dataclass(frozen=True)
class DeltaBounds:
    """Bounds on the causal INCREMENT (deliberately smaller than the R8 residual bounds — Δa nudges, a_R8 carries the base)."""

    d_fwd_vel: float = 0.25
    d_squeeze: float = 0.04
    d_stop_gain: float = 0.25
    slew: float = 0.15           # max |Δa_t − Δa_{t-1}| per role per control decision (anti-chatter)

    def vec(self) -> np.ndarray:
        return np.array([self.d_fwd_vel, self.d_squeeze, self.d_stop_gain], np.float64)


class DeltaActor(Protocol):
    def __call__(self, causal_state: np.ndarray) -> np.ndarray: ...


@dataclass
class ZeroDeltaActor:
    """Δa = 0 for any causal state — the update-zero actor (reproduces the frozen R8 champion). # Post: length-3 zeros."""

    def __call__(self, causal_state: np.ndarray) -> np.ndarray:
        return np.zeros(3, np.float64)


@dataclass
class ConstantDeltaActor:
    """Emits a FIXED (already unit-scaled) Δa for every decision — for the update-nonzero sensitivity / identity tests."""

    d: np.ndarray

    def __call__(self, causal_state: np.ndarray) -> np.ndarray:
        return np.asarray(self.d, np.float64).ravel()[:3]


class R9CausalResidualAdapter(ResidualTipAdapter):
    """Frozen R8 base residual `a_R8` + a bounded causal Δa from `delta_actor`, emitted every `control_interval` frames and
    held between. `a_exec = clip(a_R8 + Δa·bounds_scale)` drives the SAME servo. Bellman action = the (unit) Δa emission.
    # Preconditions: `a_R8 ∈ [−1,1]^3` (the frozen champion emission). # Postconditions: at Δa≡0, identical to the base
      ResidualTipAdapter(ConstantResidualActor(a_R8)); the executed residual stays within the frozen R8 bounds; release
      stays certificate-gated."""

    def __init__(self, snap: CradleSnapshot, a_r8: np.ndarray, delta_actor: DeltaActor,
                 params: TipTransportParams = TipTransportParams(), bounds: ResidualBounds = ResidualBounds(),
                 dbounds: DeltaBounds = DeltaBounds(), control_interval: int = 4, cfg: Any = DELIVERY_CFG) -> None:
        self.a_r8 = np.clip(np.asarray(a_r8, np.float64).ravel()[:3], -1.0, 1.0)
        self.delta_actor = delta_actor
        self.dbounds = dbounds
        self.control_interval = max(1, int(control_interval))
        self.horizon = float(cfg.horizon)
        # actor slot kept for the parent's reset() hook (unused: R9 overrides the action computation)
        self.actor = delta_actor
        self.bounds = bounds
        super().__init__(snap, delta_actor, params, bounds, cfg)

    def reset(self) -> None:
        super().reset()
        self._held_delta = np.zeros(3, np.float64)     # current held (unit) Δa
        self._prev_delta = np.zeros(3, np.float64)     # previous decision's Δa (for slew + causal state)
        self._history: list[list[float]] = []
        if hasattr(self.delta_actor, "reset"):
            self.delta_actor.reset()

    def _causal_state(self, rl: Any, bt: "dict[str, Any]", obs: np.ndarray) -> np.ndarray:
        hist = np.zeros((HISTORY_LEN, _HIST_FEAT), np.float64)
        for i, row in enumerate(self._history[-HISTORY_LEN:]):
            hist[HISTORY_LEN - min(len(self._history), HISTORY_LEN) + i] = row
        return np.concatenate([obs, self.a_r8, self._prev_delta, hist.ravel()]).astype(np.float64)

    def _decide(self, rl: Any, t: int, bt: "dict[str, Any]", obs: np.ndarray) -> np.ndarray:
        """Causal Δa (unit, in [−1,1]^3): re-decide every `control_interval` frames, hold between; slew- and magnitude-clip."""
        if t % self.control_interval == 0:
            raw = np.clip(np.asarray(self.delta_actor(self._causal_state(rl, bt, obs)), np.float64).ravel()[:3], -1.0, 1.0)
            slewed = np.clip(raw, self._prev_delta - self.dbounds.slew, self._prev_delta + self.dbounds.slew)
            self._held_delta = slewed
            self._prev_delta = slewed
        return self._held_delta

    def dtau_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        bt = self._base_targets(rl, t)
        obs = self._observe(rl, t, bt)
        delta_unit = self._decide(rl, t, bt, obs)                      # BELLMAN ACTION (the causal increment, unit)
        a_exec = np.clip(self.a_r8 + delta_unit * self.dbounds.vec() / self.bounds.vec(), -1.0, 1.0)  # in R8 residual units
        d = a_exec * self.bounds.vec()                                 # scaled residual added to the base targets
        qref = float(np.clip(bt["base_qref"] + d[0], 0.0, self.p.qdot_max))
        sqz = float(np.clip(bt["base_sqz"] + d[1], self.p.squeeze_min, self.p.squeeze_hold))
        kv = self.p.k_v * float(np.clip(1.0 + d[2], self.bounds.kv_lo, self.bounds.kv_hi))
        dtau = self._servo(rl, bt, qref, sqz, kv)
        self.last_bellman_action = delta_unit
        self._history.append([bt["dtz"], bt["v_par"], _coin_speed(rl), *[float(x) for x in delta_unit]])
        self.provenance.append({
            "t": int(t), "bellman_action": [round(float(x), 6) for x in delta_unit],   # Δa (the ONLY Bellman action)
            "a_r8_base": [round(float(x), 6) for x in self.a_r8], "a_exec": [round(float(x), 6) for x in a_exec],
            "residual": [round(float(x), 6) for x in d], "base_qref": round(bt["base_qref"], 6),
            "corrected_qref": round(qref, 6), "base_sqz": round(bt["base_sqz"], 6), "corrected_sqz": round(sqz, 6),
            "kv": round(kv, 6), "in_release": bool(bt["in_release"]), "coin_spin": round(_coin_spin(rl), 6),
            "obs": [round(float(x), 6) for x in obs]})
        self.trace.append({"t": int(t), "phase": int(self.phase), "dtz_mm": round(bt["dtz"] * 1000, 2),
                           "v_par": round(bt["v_par"], 5), "qdot_max": round(float(np.max(np.abs(rl.inner.data.qvel[:4]))), 4),
                           "sqz": round(self._sqz, 4)})
        return dtau
