"""STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1 — expose the exact strict-dwell counter to the actor/critic/replay/targets,
and (Arm B) move the +30 graded delivery bonus from the K5 off-by-one step onto the K6 TERMINAL transition, paid once.

Arm A: exact-counter exposure; canonical reward + termination preserved bit-exact.
Arm B: exact-counter exposure; the graded terminal bonus fires ONCE on the K6 terminal transition (not the K5 pre-terminal
       step), so it cannot be farmed by reaching K5, breaking dwell, and rebuilding. No other reward term changes.

pi_0, physics, certifier, and all non-terminal reward terms are unchanged. The canonical v3 .hymeko is NOT overwritten —
Arm B composes a variant spec (v3 minus terminal_deliver_graded) plus a latched K6 bonus.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL, CoinRL4Dof

STRICT_K = HELD_DWELL           # 6
TERMINAL_WEIGHT = 30.0
CLEAN_BODY_PROGRESS = 0.005     # matches _term_terminal_deliver_graded


def strict_onehot(strict: int, k: int = STRICT_K) -> np.ndarray:
    """Exact one-hot of the dwell counter clamped to 0..k (k+1 dims). Distinct for every strict value 0..k."""
    v = np.zeros(k + 1, np.float32); v[int(min(max(strict, 0), k))] = 1.0
    return v


def augment_with_strict(state, rl, k: int = STRICT_K) -> np.ndarray:
    """Append the exact strict-counter one-hot to a base state (actor/critic input) — removes the strict aliasing."""
    return np.concatenate([np.asarray(state, np.float32), strict_onehot(int(rl._strict), k)]).astype(np.float32)


def terminal_grade(inner) -> float:
    """The graded delivery multiplier (fingertip-dominant +1.0 / body-assisted +0.2 / body-driven −0.5), replicating
    _term_terminal_deliver_graded's grade — evaluated at the K6 terminal for Arm B."""
    m = inner._planar_metrics
    if not bool(getattr(m, "in_zone", False)):
        return 0.0
    bp = float(getattr(inner, "_body_progress", 0.0)); fp = float(getattr(inner, "_fingertip_progress", 0.0))
    if bp <= CLEAN_BODY_PROGRESS:
        return 1.0
    return -0.5 if bp > fp else 0.2


def arm_b_terminal_bonus(strict: int, touched: bool, bonus_paid: bool, grade: float,
                         *, dwell_req: int = STRICT_K, weight: float = TERMINAL_WEIGHT):
    """Pure Arm-B bonus rule: pay ``weight*grade`` ONCE, only on the K6 terminal transition (strict≥dwell_req ∧ touched),
    latched by ``bonus_paid``. Returns (bonus, new_bonus_paid). No bonus at K5 ⇒ not farmable."""
    delivered = strict >= dwell_req and bool(touched)
    if delivered and not bonus_paid:
        return weight * grade, True
    return 0.0, bonus_paid


class CoinRL4DofAblation(CoinRL4Dof):
    """CoinRL4Dof with (arm) reward semantics and exact-counter exposure. ``arm='A'`` reproduces canonical reward +
    termination bit-exact; ``arm='B'`` moves the graded terminal bonus to the K6 terminal (latched, non-farmable)."""

    def __init__(self, arm: str = "A", horizon: int = 360):
        super().__init__(horizon=horizon)
        assert arm in ("A", "B")
        self.arm = arm
        if arm == "B":
            from hymeko_rl.env.reward import RewardSpec
            terms = self.inner.reward_spec.terms
            self._base_spec = RewardSpec(terms=tuple((k, w) for k, w in terms if k != "terminal_deliver_graded"))
            self._bonus_paid = False

    def reset(self, seed: int):
        o = super().reset(seed)
        if self.arm == "B":
            self._bonus_paid = False
        return o

    def step(self, a4):
        a = np.clip(np.asarray(a4, np.float32), -4.0, 4.0)
        self.inner.step(a); self._t += 1
        dtz = self._dtz(); m = self.inner._planar_metrics
        self._touched = self._touched or bool(m.left_contact or m.right_contact)
        strict_ok = (dtz <= CENTER_TOL) and (self._speed() < SETTLE_VEL)
        self._strict = self._strict + 1 if strict_ok else 0
        self.inner._success = int(self._strict); self.inner.success_steps = HELD_DWELL
        delivered = bool(self._strict >= HELD_DWELL and self._touched)
        if self.arm == "A":
            reward = float(self.inner.reward_spec.evaluate(self.inner, dtz, self.inner.data.ctrl))   # canonical (K5 bonus)
        else:
            reward = float(self._base_spec.evaluate(self.inner, dtz, self.inner.data.ctrl))           # v3 minus terminal
            bonus, self._bonus_paid = arm_b_terminal_bonus(self._strict, self._touched, self._bonus_paid, terminal_grade(self.inner))
            reward += bonus                                                                            # +30·grade at K6 only
        terminated = delivered
        truncated = (not terminated) and self._t >= self.horizon
        return self.obs(), reward, terminated, truncated, {"dtz": dtz, "strict": self._strict, "delivered": delivered}
