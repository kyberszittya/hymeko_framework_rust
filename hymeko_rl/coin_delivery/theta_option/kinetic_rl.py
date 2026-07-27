"""R1/R2 — the bounded-residual OPTION environment + reward over the frozen KINETIC clone (matched TD3-vs-SAC).

Mirrors the R8 `ResidualOptionEnv` for the KINETIC clone: the Bellman action is the actor's bounded residual ``a ∈ [-1,1]^4``,
applied as a CONSTANT per-option residual ``α·a`` on top of the frozen clone over the whole deploy (one ``step`` = one full
frozen-chain rollout = one terminal semi-MDP option). The return is a task-tied reward computed from PHYSICAL signals only
(progress toward the zone, the close-and-moving release, strict K6, minus stall / clamp / early-contact-loss / sign-reversal /
safety) — crucially it does NOT reward raw v_par (the clone already accelerates to 0.37; rewarding speed would learn to overshoot
the coin). Everything except ``a`` is provenance; the frozen release/coast/K6 downstream and every torque/slew/joint limit are
unchanged; no teacher is in the loop. Plugs into the framework `train_semi_mdp` unchanged (it is an `OptionEnv`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor, KineticCloneController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import KineticResidualController, ResidualBounds
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

OBS_DIM = 41
REWARD_SCALE = 50.0                    # algorithmic critic reward-scale (identical across TD3/SAC; selection stays on min_dtz)
CORRIDOR_MM = 30.0                     # close+moving release corridor upper edge
EARLY_LOSS_MM = 55.0                   # losing bilateral contact above this dtz ⇒ "too early" (below the bank's coverage floor)
CLAMP_FN = 4.0                         # N — per-side normal force above this is a heavy clamp
STALL_VPAR = 0.0                       # v_par ≤ this on a transport step is a stall


@dataclass(frozen=True)
class RewardWeights:
    """Task-tied reward weights. Progress + close+moving + K6 are the goal; raw v_par is NOT rewarded (it is only kept positive
    via the stall penalty), so the RL cannot win by overshooting the coin."""

    dist: float = 1.0                  # per-mm improvement over the clone baseline (progress toward the zone)
    corridor: float = 30.0             # reached the 20–30 mm close+moving corridor
    k6: float = 100.0                  # strict K6 terminal
    stall: float = 8.0                 # per stalled transport step
    clamp: float = 5.0                 # per heavy-clamp step
    early_loss: float = 12.0           # lost bilateral contact too early (dtz > EARLY_LOSS_MM)
    reversal: float = 5.0              # per v_par sign reversal
    safety: float = 60.0               # motion-contract breach


def _kinetic_steps(clone_trace: list[dict]) -> list[dict]:
    return [r for r in clone_trace if r["kind"] == "KINETIC_CLONE"]


def kinetic_reward(m: dict, min_dtz: float, kin: list[dict], baseline_mm: float,
                   w: RewardWeights = RewardWeights()) -> "tuple[float, dict]":
    """Task-tied option return + its decomposition. # Preconditions: ``m`` a velocity_rollout metrics dict; ``kin`` the
    KINETIC-transport steps (v_par / fn per step); ``baseline_mm`` the frozen-clone min_dtz (46.2). # Postconditions: reward is
    a pure function of physical signals; no raw-v_par term."""
    safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
    vpar = [r["v_par"] for r in kin]
    fn = [min(r["fn_l"], r["fn_r"]) for r in kin]
    stalls = sum(1 for v in vpar if v <= STALL_VPAR)
    clamps = sum(1 for f in fn if f > CLAMP_FN)
    reversals = sum(1 for i in range(1, len(vpar)) if vpar[i] * vpar[i - 1] < 0)
    exit_dtz = kin[-1]["dtz_mm"] if kin else 999.0
    d = {
        "progress": w.dist * (baseline_mm - min_dtz),
        "corridor": w.corridor if min_dtz <= CORRIDOR_MM else 0.0,
        "k6": w.k6 if (m["k6_delivered"] and safe) else 0.0,
        "stall": -w.stall * stalls, "clamp": -w.clamp * clamps, "reversal": -w.reversal * reversals,
        "early_loss": -w.early_loss if (kin and exit_dtz > EARLY_LOSS_MM) else 0.0,
        "safety": -w.safety if not safe else 0.0}
    return float(sum(d.values())), {**{k: round(v, 3) for k, v in d.items()}, "safe": safe,
                                    "stalls": stalls, "clamps": clamps, "reversals": reversals}


class ConstResidual:
    """A constant per-option residual actor: returns the fixed Bellman action ``a`` for every step (the R8 constant-residual
    formulation). # Invariants: stateless; ``act`` ignores the observation."""

    def __init__(self, a: np.ndarray) -> None:
        self.a = np.asarray(a, np.float64).ravel()[:ACT_DIM]

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.a


def deploy_residual(snap: Any, clone: CloneActor, residual_vec: np.ndarray, bounds: ResidualBounds,
                    cfg: Any = DELIVERY_CFG) -> "tuple[dict, float, list]":
    """Deploy the frozen clone + a CONSTANT residual on the full frozen chain (teacher-free); returns (metrics, min_dtz,
    KINETIC steps). At ``residual_vec = 0`` this is exactly the K2 clone (the update-zero identity)."""
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    controller = KineticResidualController(snap, clone, ConstResidual(residual_vec), bounds)
    m = velocity_rollout(snap, controller, cfg)
    return m, _min_dtz_mm(snap, m), _kinetic_steps(controller.clone_trace)


def kinetic_init_obs(snap: Any, clone: CloneActor, cfg: Any = DELIVERY_CFG) -> np.ndarray:
    """The DETERMINISTIC initiation observation the actor conditions on — the first-KINETIC-step 41-D observation under the
    clone (zero residual), captured from a full-chain probe. # Postconditions: length-OBS_DIM float32."""
    from hymeko_rl.coin_delivery.theta_option.kinetic_contract import kinetic_observe
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    captured: dict = {}
    controller = KineticCloneController(snap, clone)

    def hook(rl: Any, t: int) -> None:
        from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC
        if controller.phase == KINETIC and "obs" not in captured:
            captured["obs"] = kinetic_observe(rl, [])[0]
    velocity_rollout(snap, controller, cfg, frame_hook=hook)
    return np.asarray(captured.get("obs", np.zeros(OBS_DIM)), np.float32)


@dataclass
class KineticResidualOptionEnv:
    """`OptionEnv` over the residual-parameterised KINETIC clone. One ``step`` executes ONE full frozen-chain rollout with a
    constant residual and is terminal. # Invariants: the Bellman action returned in ``info`` equals the clipped actor emission
    and drives the physics; no teacher; the reward uses physical signals only."""

    snap: Any
    clone: CloneActor
    reward_fn: Callable[[dict, float, list], "tuple[float, dict]"]
    init_obs: np.ndarray
    bounds: ResidualBounds = field(default_factory=ResidualBounds)
    cfg: Any = DELIVERY_CFG

    def reset(self, idx: "int | None" = None) -> np.ndarray:
        return np.asarray(self.init_obs, np.float32)

    def step(self, action: np.ndarray, *, search_seed: "int | None" = None) -> "tuple[np.ndarray, float, bool, dict]":
        a = np.clip(np.asarray(action, np.float64).ravel()[:ACT_DIM], -1.0, 1.0)   # the BELLMAN ACTION
        m, min_dtz, kin = deploy_residual(self.snap, self.clone, a, self.bounds, self.cfg)
        reward, decomp = self.reward_fn(m, min_dtz, kin)
        info = {"tau": float(max(1, len(kin))), "terminal": 1.0, "end": "handoff",
                "k6": float(bool(delivery_success(m, self.cfg))), "min_dtz": round(min_dtz, 2),
                "bellman_action": a.astype(np.float32), "reward_decomp": decomp,
                "peak_qdot": float(m["peak_qdot"]), "peak_coin_speed": float(m["peak_coin_speed"])}
        return np.asarray(self.init_obs, np.float32), reward, True, info


def make_dev_eval(snap: Any, clone: CloneActor, init_obs: np.ndarray, bounds: ResidualBounds, baseline_mm: float,
                  cfg: Any = DELIVERY_CFG) -> Callable[[Any], "tuple[float, dict]"]:
    """A `dev_eval_fn(actor) -> (score, aux)` that deploys the actor's MEAN residual on the full frozen chain; score = −min_dtz
    (closer is better). Selection stays on the physical metric, never the algorithmic reward scale."""
    x = torch.as_tensor(np.asarray(init_obs, np.float32)[None])

    def dev_eval(actor: Any) -> "tuple[float, dict]":
        with torch.no_grad():
            a = actor.mean_action(x)[0].numpy()
        m, min_dtz, _kin = deploy_residual(snap, clone, a, bounds, cfg)
        safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
        return -float(min_dtz), {"min_dtz": round(min_dtz, 2), "k6": bool(m["k6_delivered"] and safe),
                                 "improved": bool(min_dtz < baseline_mm - 1e-6), "residual": [round(float(v), 3) for v in a]}
    return dev_eval
