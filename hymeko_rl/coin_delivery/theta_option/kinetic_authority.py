"""Authority audit — learning-free reachability of the close-and-moving manifold under three residual-authority families.

R3-B localised the wall: the bounded residual (4-D per-joint, α = 0.15) over the frozen clone has a CLEAN coin-following ceiling
at ~50 mm. Before any more RL, this module asks — WITHOUT learning — whether a safe, expressible residual sequence from a HEALTHY
R2 frontier reaches ≤ 30 mm cleanly, for each of three authority families:

  A0  the current 4-D per-joint residual, α = 0.15                         (control)
  A1  the same 4-D per-joint residual, larger bound α ∈ {0.20, 0.25, 0.30}
  A2  a minimally-EXPANDED, STRUCTURED coin-following basis (NOT raw torque): left-tip forward-follow, right-tip forward-follow,
      common squeeze, left/right differential, common tangential pursuit — so the residual can track the two tips alongside the
      sliding coin, not merely nudge the clone's per-joint action.

Each family is searched by a bounded CEM over the residual sequence from the frontier (segment-local restart, restored clone
hidden). The verdict is the SMALLEST family that reaches `AUTHORITY_REACHABILITY_PASS` (min_dtz ≤ 30 mm, +v_par, light Fn, 0
stall/reversal/clamp, safe). No policy is trained here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, arm_jac_dir
from hymeko_rl.coin_delivery.mobile_conditioning import _fingertip_geoms, arm_inward_geom
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor, KineticCloneController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

_LEFT_DOF, _RIGHT_DOF = (0, 1), (2, 3)
A2_DIM = 5                                       # [L-fwd, R-fwd, squeeze, balance, tangential]
FAMILY_DIM = {"A0": ACT_DIM, "A1": ACT_DIM, "A2": A2_DIM}
CORRIDOR_MM = 30.0
FN_LIGHT = 2.0


def a2_structured_u(rl: Any, coeffs: np.ndarray, e_par: np.ndarray) -> np.ndarray:
    """Map the 5-D A2 coefficients to a 4-D joint correction via the LIVE tip Jacobians: left/right forward-follow (each tip
    toward the zone), common squeeze (inward grip), left/right differential, common tangential pursuit (both tips along the
    coin's sliding velocity). # Postconditions: length-4; each direction unit-norm so a coefficient is a bounded push."""
    gl, gr = _fingertip_geoms(rl.inner.model)
    coin = _coin_xy(rl)
    l_fwd, r_fwd = arm_jac_dir(rl, gl, _LEFT_DOF, e_par), arm_jac_dir(rl, gr, _RIGHT_DOF, e_par)
    inward = arm_inward_geom(rl, gl, _LEFT_DOF, coin) + arm_inward_geom(rl, gr, _RIGHT_DOF, coin)
    squeeze = inward / (np.linalg.norm(inward) + 1e-12)
    balance = l_fwd - r_fwd
    balance = balance / (np.linalg.norm(balance) + 1e-12)
    v = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    vdir = v / (np.linalg.norm(v) + 1e-9)
    tang = arm_jac_dir(rl, gl, _LEFT_DOF, vdir) + arm_jac_dir(rl, gr, _RIGHT_DOF, vdir)
    tang = tang / (np.linalg.norm(tang) + 1e-12)
    c = np.asarray(coeffs, np.float64).ravel()[:A2_DIM]
    return c[0] * l_fwd + c[1] * r_fwd + c[2] * squeeze + c[3] * balance + c[4] * tang


class SequenceResidual:
    """A precomputed per-step residual sequence (the CEM candidate); holds the last value past the end. # Invariants: stateful
    index; deterministic given the sequence."""

    def __init__(self, seq: np.ndarray) -> None:
        self.seq = np.asarray(seq, np.float64)
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def act(self, _obs: np.ndarray) -> np.ndarray:
        a = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return a


class KineticAuthorityController(KineticCloneController):
    """Frozen clone + a per-step authority residual (family A0/A1 per-joint, or A2 structured), optionally restarted segment-
    locally from a frontier. # Postconditions: |Δτ| ≤ slew; no teacher; release/coast/K6 downstream unchanged; at a zero residual
    reproduces the clone."""

    def __init__(self, snap: Any, clone: CloneActor, seq_actor: SequenceResidual, family: str, alpha: float,
                 *, start_kinetic: "dict | None" = None, **kw: Any) -> None:
        self.seq_actor = seq_actor
        self.family = family
        self.alpha = float(alpha)
        self._start_kinetic = start_kinetic
        super().__init__(snap, clone, **kw)

    def reset(self) -> None:
        super().reset()
        self.seq_actor.reset()
        if self._start_kinetic is not None:
            from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC as _KIN
            self.phase = _KIN
            self._kinetic_steps = 0
            self.actor.set_hidden(self._start_kinetic.get("clone_hidden"))

    def _transport_action(self, rl: Any, obs: np.ndarray) -> np.ndarray:
        u_clone = np.clip(np.asarray(self.actor.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        r = np.asarray(self.seq_actor.act(obs), np.float64)
        if self.family == "A2":
            e_par = np.asarray(rl.inner.direction_to_zone()[0], np.float64)
            corr = a2_structured_u(rl, r, e_par)
        else:
            corr = r[:ACT_DIM]
        return np.clip(u_clone + self.alpha * corr, -1.0, 1.0)


@dataclass(frozen=True)
class AuthorityCEMConfig:
    horizon: int = 14
    pop: int = 48
    iters: int = 6
    elite: int = 8
    init_std: float = 0.6
    seed: int = 20260728


def _cleanliness(kin: list[dict]) -> dict:
    vpar = [r["v_par"] for r in kin]
    fn = [min(r["fn_l"], r["fn_r"]) for r in kin]
    return {"stalls": int(sum(1 for v in vpar if v <= 0.0)), "clamps": int(sum(1 for f in fn if f > 4.0)),
            "reversals": int(sum(1 for i in range(1, len(vpar)) if vpar[i] * vpar[i - 1] < 0.0))}


def _rollout(frontier: Any, model: Any, norm: Any, seq: np.ndarray, family: str, alpha: float,
             bounds: ResidualBounds, cfg_env: Any) -> dict:
    ctrl = KineticAuthorityController(frontier, CloneActor(model, norm), SequenceResidual(seq), family, alpha,
                                      start_kinetic=frontier.start_state())
    m = velocity_rollout(frontier, ctrl, cfg_env)
    kin = [{"dtz_mm": r["dtz_mm"], "v_par": r["v_par"], "fn_l": r["fn_l"], "fn_r": r["fn_r"]}
           for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"]
    clean = _cleanliness(kin)
    min_dtz = _min_dtz_mm(frontier, m)
    safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
    exit_fn = min(kin[-1]["fn_l"], kin[-1]["fn_r"]) if kin else 0.0
    exit_v = kin[-1]["v_par"] if kin else 0.0
    return {"min_dtz": min_dtz, "safe": safe, "exit_fn": exit_fn, "exit_v": exit_v, **clean}


def _score(r: dict) -> float:
    if not r["safe"]:
        return -1e9
    pen = 40.0 * (r["stalls"] + r["reversals"] + r["clamps"])       # cleanliness dominates (no cheating by stalling)
    return -r["min_dtz"] - pen


def reachability_pass(m: dict) -> bool:
    """`AUTHORITY_REACHABILITY_PASS`: the residual sequence reached the corridor (≤ 30 mm) CLEANLY — still moving (+v_par),
    with light contact (Fn < 2 N), no stall/reversal/clamp, and safe. # Preconditions: `m` is a `_rollout`/best-metrics dict
    carrying `min_dtz, exit_v, exit_fn, stalls, reversals, clamps, safe`. # Postconditions: True iff every clause holds."""
    return bool(m.get("min_dtz", 9e9) <= CORRIDOR_MM and m.get("exit_v", 0.0) > 0.0 and m.get("exit_fn", 9.0) < FN_LIGHT
                and m.get("stalls", 1) == 0 and m.get("reversals", 1) == 0 and m.get("clamps", 1) == 0
                and bool(m.get("safe", False)))


def authority_cem(frontier: Any, model: Any, norm: Any, family: str, alpha: float, *, bounds: ResidualBounds,
                  cfg: AuthorityCEMConfig = AuthorityCEMConfig(), cfg_env: Any = DELIVERY_CFG) -> dict:
    """Bounded CEM over the residual sequence for one (family, α) from a healthy frontier; returns the best reachability
    (min_dtz + cleanliness) and the `AUTHORITY_REACHABILITY_PASS` verdict. Deterministic (fixed RNG). No learning."""
    dim = FAMILY_DIM[family]
    rng = np.random.default_rng(cfg.seed)
    mean = np.zeros((cfg.horizon, dim))
    std = np.full((cfg.horizon, dim), cfg.init_std)
    best = {"score": -1e18, "metrics": None, "seq": None}
    for _it in range(cfg.iters):
        pop = np.clip(mean[None] + std[None] * rng.standard_normal((cfg.pop, cfg.horizon, dim)), -1.0, 1.0)
        scored = []
        for s in pop:
            m = _rollout(frontier, model, norm, s, family, alpha, bounds, cfg_env)
            sc = _score(m)
            scored.append((sc, s, m))
            if sc > best["score"]:
                best = {"score": float(sc), "metrics": m, "seq": s}
        scored.sort(key=lambda z: z[0], reverse=True)
        elite = np.stack([s for _sc, s, _m in scored[:cfg.elite]])
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    m = best["metrics"] or {}
    reach = reachability_pass(m)
    return {"family": family, "alpha": round(alpha, 3), "min_dtz_mm": round(float(m.get("min_dtz", 9e9)), 2),
            "exit_v_par": round(float(m.get("exit_v", 0.0)), 4), "exit_fn": round(float(m.get("exit_fn", 0.0)), 4),
            "stalls": m.get("stalls"), "reversals": m.get("reversals"), "clamps": m.get("clamps"),
            "safe": m.get("safe"), "authority_reachability_pass": reach}
