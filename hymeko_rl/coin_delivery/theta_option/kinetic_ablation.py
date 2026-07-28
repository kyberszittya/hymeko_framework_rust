"""R9 C1 component load-bearing ablation — is the learned K6 the clone/R2 heritage, or the authority-unlock TD3?

Two complementary probes, both over the FROZEN C1 stack `u = clip(u_clone + 0.15·a_R2 + 0.85·δ_expand, −1, 1)`:

  A. FROZEN-POLICY INTERVENTION (no training): evaluate a trained checkpoint with individual ACTION terms masked out — FULL /
     NO_EXPANSION / NO_R2 / NO_CLONE / EXPANSION_ONLY. Every term is still COMPUTED (so the clone GRU hidden that a_R2 and
     δ_expand condition on is unchanged); only its contribution to the summed action is switched off. This isolates each
     component's causal action contribution in the ALREADY-LEARNED policy. Truncated stacks are not expected to work — the point
     is the dependence, not the survival.

  B. RETRAINING (matched authority/reward/curriculum): F0 full C1 (clone + R2 + expansion, β = 0.85 over the frozen R2), F1
     clone + expansion (no R2; a zero R2 term, expansion β = 1.0 so the total residual authority ≈ 1.0 matches F0), F2
     expansion-only direct 4-D actor (no clone, no R2; a from-scratch policy on the 41-D KINETIC obs). F0 ≫ F1/F2 ⇒ the
     hierarchical skill stack is load-bearing; F0 ≈ F1 ≫ F2 ⇒ the clone matters, R2 only accelerates; F0 ≈ F1 ≈ F2 ⇒ the authority
     unlock is decisive and the old policies are mainly initialisation scaffolds.

All new code; the `8a0c1c7b` modules are imported unchanged.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import ALPHA0, AuthorityUnlockController
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor, KineticCloneController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import augmented_state

# Frozen-policy intervention masks: (include_clone, include_R2, include_expansion) over the summed action.
INTERVENTIONS: dict[str, tuple] = {
    "FULL": (True, True, True), "NO_EXPANSION": (True, True, False), "NO_R2": (True, False, True),
    "NO_CLONE": (False, True, True), "EXPANSION_ONLY": (False, False, True)}


def zero_residual_fn(_aug: np.ndarray) -> np.ndarray:
    """A residual_fn that always returns 0 — the R2 term for F1 (clone + expansion, no learned R2)."""
    return np.zeros(ACT_DIM, np.float64)


class AblationUnlockController(AuthorityUnlockController):
    """Frozen-policy intervention: the full three-term stack is computed every step (clone runs → GRU hidden; a_R2 and δ_expand
    condition on the SAME augmented state as in the learned policy), but a per-term ``include`` mask zeroes a component's
    CONTRIBUTION to the summed action. # Postconditions: FULL reproduces `AuthorityUnlockController`; every mask keeps the
    conditioning distribution intact so a single causal factor (one action term) is changed at a time; |Δτ| ≤ slew, safety guards
    unchanged."""

    def __init__(self, snap: Any, clone: CloneActor, expand_fn: Callable[[np.ndarray], np.ndarray],
                 bounds: ResidualBounds = ResidualBounds(), *, r2_fn: Callable[[np.ndarray], np.ndarray], beta: float,
                 include: tuple = (True, True, True), alpha0: float = ALPHA0, start_kinetic: "dict | None" = None,
                 **kw: Any) -> None:
        self.include = include
        super().__init__(snap, clone, expand_fn, bounds, r2_fn=r2_fn, beta=beta, alpha0=alpha0,
                         start_kinetic=start_kinetic, **kw)

    def _transport_action(self, rl: Any, obs: np.ndarray) -> np.ndarray:
        ic, ir, ie = self.include
        u_clone = np.clip(np.asarray(self.actor.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)   # always: sets clone hidden
        a_r2 = np.clip(np.asarray(self.r2_fn(augmented_state(self.actor, obs, self._prev_res_r2)),
                                  np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        aug_ex = augmented_state(self.actor, obs, self._prev_res)
        r_ex = np.clip(np.asarray(self.residual_fn(aug_ex), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        u = np.clip((u_clone if ic else 0.0) + (self.alpha0 * a_r2 if ir else 0.0)
                    + (self.beta * r_ex if ie else 0.0), -1.0, 1.0)
        self.aug_trace.append((aug_ex, r_ex.copy()))
        self.residual_trace.append({"a_r2": [round(float(x), 5) for x in a_r2],
                                    "r_ex": [round(float(x), 5) for x in r_ex], "u": [round(float(x), 5) for x in u]})
        self._prev_res_r2 = a_r2
        self._prev_res = r_ex
        return u


# ------------------------------------------------------------------------------------------------------------------------
# Retraining family F2 — expansion-only DIRECT 4-D actor (no clone, no R2): a from-scratch policy on the 41-D KINETIC obs.
# ------------------------------------------------------------------------------------------------------------------------
DIRECT_OBS_DIM = 41                              # the raw KINETIC observation (no clone hidden, no prev residual)


class DirectKineticController(KineticCloneController):
    """F2 scaffold-free baseline: frozen APPROACH / release / coast / K6, but the KINETIC transport action is a DIRECT 4-D actor
    on the raw 41-D observation (no clone action, no clone GRU memory, no R2). `policy_fn(obs_41) -> [−1,1]^4` IS the action.
    Records `aug_trace = (obs, u)` so the same `train_perstep` loop trains it (obs_dim = 41). # Postconditions: |Δτ| ≤ slew;
    safety unchanged; the clone/R2 heritage is entirely absent."""

    def __init__(self, snap: Any, clone: CloneActor, policy_fn: Callable[[np.ndarray], np.ndarray],
                 bounds: ResidualBounds = ResidualBounds(), *, start_kinetic: "dict | None" = None, **kw: Any) -> None:
        self.policy_fn = policy_fn
        self._start_kinetic = start_kinetic
        super().__init__(snap, clone, **kw)

    def reset(self) -> None:
        super().reset()
        self.aug_trace: list[tuple[np.ndarray, np.ndarray]] = []
        self.residual_trace: list[dict[str, Any]] = []
        if self._start_kinetic is not None:
            from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC as _KIN
            self.phase = _KIN
            self._kinetic_steps = 0
            self.actor.set_hidden(self._start_kinetic.get("clone_hidden"))

    def _transport_action(self, rl: Any, obs: np.ndarray) -> np.ndarray:
        x = self.actor.norm.apply(obs).astype(np.float64)                # normalise with the frozen stats (matched input scale)
        u = np.clip(np.asarray(self.policy_fn(x), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)   # the action IS the policy output
        self.aug_trace.append((x, u.copy()))
        self.residual_trace.append({"u": [round(float(v), 5) for v in u]})
        return u
