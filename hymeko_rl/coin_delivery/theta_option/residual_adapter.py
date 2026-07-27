"""R8 S2 — the bounded residual adapter over the frozen tip-referenced scaffold.

The actor never replaces raw torque. It emits a compact residual `a ∈ [-1,1]^3` over three directly-interpretable scaffold
targets — the forward tip/joint-velocity reference, the squeeze, and the deceleration/stop aggressiveness (servo gain) — which
is scaled to physical bounds and ADDED to the frozen scaffold's base targets before the same servo, safety clip and R6
release certificate run. The release event is NOT a free actor bit: the actor shapes squeeze / stop / release-readiness, but
the certificate remains the final release shield.

    causal state/history  →  frozen scaffold base target  +  bounded actor residual  →  clip/anti-windup  →  physics  →  R6 cert

Bellman action = the actor emission `a` ONLY. Everything else (base target, corrected target, clipped residual, torque
target, executed torque, certificate/controller internals) is provenance, never the label. At `a = 0` the adapter reproduces
the scaffold exactly (the residual is `+0`, every clip is identity) — the S2 update-zero requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.forward_displacement import _coin_speed
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipReferencedController, TipTransportParams

# residual roles + observation layout (frozen)
RESIDUAL_ROLES = ("d_fwd_vel", "d_squeeze", "d_stop_gain")
OBS_KEYS = ("dtz", "v_par", "coin_speed", "coin_spin", "qdot_max", "fn_l", "fn_r", "both", "sqz", "elapsed_frac", "in_release")


class Actor(Protocol):
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


@dataclass
class ZeroActor:
    """The update-zero actor: emits a = 0 for any observation. # Postconditions: returns a length-3 zero vector."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(3, np.float64)


@dataclass
class ConstantResidualActor:
    """Emits a FIXED residual a for every step (S3 sensitivity + constant candidates). # Postconditions: length-3."""

    a: np.ndarray

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.a, np.float64).ravel()[:3]


class SequenceResidualActor:
    """Emits a per-step residual from a precomputed sequence (S3 piecewise / temporally-coherent / noise candidates); holds
    the last value past the end. Deterministic given the sequence. # Postconditions: each call returns a length-3 vector."""

    def __init__(self, seq: np.ndarray) -> None:
        self.seq = np.asarray(seq, np.float64).reshape(-1, 3)
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return a


@dataclass(frozen=True)
class ResidualBounds:
    """Physical bounds the tanh actor emission `a ∈ [-1,1]^3` scales into (added to the base targets). Deliberately small —
    the residual SHAPES the safe scaffold, it does not replace it. kv is a multiplicative gain window."""

    d_fwd_vel: float = 0.4                   # rad/s — ± on the base joint-velocity reference
    d_squeeze: float = 0.06                  # N·m — ± on the base squeeze
    d_stop_gain: float = 0.4                 # ± fraction on the servo gain k_v (deceleration aggressiveness)
    kv_lo: float = 0.5                       # multiplicative gain clamp (1 + d_stop_gain·a) ∈ [kv_lo, kv_hi]
    kv_hi: float = 1.6

    def vec(self) -> np.ndarray:
        return np.array([self.d_fwd_vel, self.d_squeeze, self.d_stop_gain], np.float64)


def _coin_spin(rl: Any) -> float:
    d = rl.inner.data
    idx = int(rl.inner._disk_dofy) + 1
    return float(d.qvel[idx]) if idx < d.qvel.shape[0] else 0.0


class ResidualTipAdapter(TipReferencedController):
    """The frozen scaffold + a bounded residual actor. `dtau_for_step` computes the scaffold base targets, adds the
    clipped/scaled actor residual, and runs the SAME servo. Records full provenance; `last_bellman_action` is the actor
    emission `a` (the ONLY Bellman action). # Postconditions: at `a = 0`, identical to the base scaffold; every executed
    residual is within the frozen bounds; release stays certificate-gated."""

    def __init__(self, snap: CradleSnapshot, actor: Actor, params: TipTransportParams = TipTransportParams(),
                 bounds: ResidualBounds = ResidualBounds(), cfg: Any = DELIVERY_CFG) -> None:
        self.actor = actor                                     # set BEFORE super().__init__ (it calls self.reset())
        self.bounds = bounds
        self.horizon = float(cfg.horizon)
        super().__init__(snap, params, cfg)

    def reset(self) -> None:
        super().reset()
        self.provenance: list[dict[str, Any]] = []
        self.last_bellman_action: np.ndarray | None = None
        if hasattr(self.actor, "reset"):
            self.actor.reset()

    def _observe(self, rl: Any, t: int, bt: "dict[str, Any]") -> np.ndarray:
        con = bt["con"]
        fn_l = float(con["left"]["fn"]) if con["left"] is not None else 0.0
        fn_r = float(con["right"]["fn"]) if con["right"] is not None else 0.0
        return np.array([bt["dtz"], bt["v_par"], _coin_speed(rl), _coin_spin(rl),
                         float(np.max(np.abs(rl.inner.data.qvel[:4]))), fn_l, fn_r, float(bt["both"]),
                         float(self._sqz), float(t) / max(self.horizon, 1.0), float(bt["in_release"])], np.float64)

    def dtau_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        bt = self._base_targets(rl, t)
        obs = self._observe(rl, t, bt)
        a = np.asarray(self.actor(obs), np.float64).ravel()[:3]      # BELLMAN ACTION (the actor emission only)
        a_clipped = np.clip(a, -1.0, 1.0)                            # safety: keep the tanh emission in range
        d = a_clipped * self.bounds.vec()                           # scaled residual added to the base targets
        qref = float(np.clip(bt["base_qref"] + d[0], 0.0, self.p.qdot_max))
        sqz = float(np.clip(bt["base_sqz"] + d[1], self.p.squeeze_min, self.p.squeeze_hold))
        kv = self.p.k_v * float(np.clip(1.0 + d[2], self.bounds.kv_lo, self.bounds.kv_hi))
        dtau = self._servo(rl, bt, qref, sqz, kv)
        self.last_bellman_action = a
        self.provenance.append({
            "t": int(t), "bellman_action": [round(float(x), 6) for x in a],
            "a_clipped": [round(float(x), 6) for x in a_clipped], "residual": [round(float(x), 6) for x in d],
            "base_qref": round(bt["base_qref"], 6), "corrected_qref": round(qref, 6),
            "base_sqz": round(bt["base_sqz"], 6), "corrected_sqz": round(sqz, 6), "kv": round(kv, 6),
            "clip_flags": {"a": bool(np.any(np.abs(a) > 1.0)), "qref": bool(qref != bt["base_qref"] + d[0]),
                           "sqz": bool(sqz != bt["base_sqz"] + d[1])},
            "in_release": bool(bt["in_release"]), "obs": [round(float(x), 6) for x in obs]})
        self.trace.append({"t": int(t), "phase": int(self.phase), "dtz_mm": round(bt["dtz"] * 1000, 2),
                           "v_par": round(bt["v_par"], 5), "qdot_max": round(float(np.max(np.abs(rl.inner.data.qvel[:4]))), 4),
                           "sqz": round(self._sqz, 4)})
        return dtau
