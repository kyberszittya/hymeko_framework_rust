"""R3-C — action-preserving authority unlock over the frozen R2 champion.

The torque-span diagnostic proved the ~36 mm ceiling was a residual-BOUND limit, not a scaffold/basis one: the teacher's
correction lies entirely in the current 4-D span but with magnitude ~0.72 (up to 1.07), which the α = 0.15 bound clipped away.
α ≈ 1.0 reaches the close-moving corridor cleanly; α ≈ 2.0 reaches the strict-K6 zone.

The unlock does NOT widen α on the R2 residual (that would apply 6–13× larger corrections in the first rollout) and does NOT build
a new basis or a direct 6-D actor. It KEEPS the frozen R2 residual at α = 0.15 and ADDS a zero-initialised state-dependent
expansion head β·tanh δ_expand on the SAME 4-D per-step basis and SAME augmented state:

    u = clip( u_clone + 0.15·a_R2(aug)  +  β·tanh δ_expand(aug) ,  −1, 1 )

At a zero expansion head this is BIT-IDENTICALLY the R2 champion (`AUTHORITY_EXPANSION_UPDATE_ZERO_IDENTITY`), so RL opens the
authority gradually from the known-good policy rather than starting as a suddenly-amplified one. Two families: C1 β = 0.85 (total
≈ 1.0), C2 β = 1.85 (total ≈ 2.0). The frozen clone, the R2 head, every torque/slew/joint limit and the release/coast/K6
downstream are unchanged; safety is still the final clip + slew + governor, independent of β.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from torch import nn

from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
    KineticTemporalResidualController, augmented_state, deterministic_residual)

ALPHA0 = 0.15                                    # the frozen R2 residual bound (unchanged)
C1_BETA, C2_BETA = 0.85, 1.85                    # expansion authority ⇒ total ≈ 1.0 / ≈ 2.0
R2_APPLIED_NOISE = ALPHA0 * 0.25                 # R2's applied exploration magnitude (α·expl_noise) — the unit to match


def zero_init_detactor(actor: nn.Module) -> nn.Module:
    """Zero the final linear layer of a `DetActor` (…, Linear(h, act), Tanh) so it outputs EXACTLY 0 (tanh(0)) everywhere — a
    bit-exact zero expansion head (the update-zero identity needs exact 0, not `distill_zero_residual`'s approximate 0).
    # Preconditions: ``actor.net`` ends in Linear→Tanh. # Postconditions: ``actor(x) == 0`` for all x."""
    final = actor.net[-2]                                              # the Linear before the terminal Tanh
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    return actor


def expl_noise_for(beta: float, applied: float = R2_APPLIED_NOISE) -> float:
    """Exploration σ (in the expansion actor's normalised output) whose APPLIED magnitude β·σ matches ``applied`` — so a larger β
    does not mean larger physical exploration (the user's applied-action-unit noise). Safety is unaffected either way (final clip
    + slew + governor); this is an exploration-efficiency scaling."""
    return float(applied / max(beta, 1e-6))


class AuthorityUnlockController(KineticTemporalResidualController):
    """Frozen clone + frozen R2 residual (α = 0.15) + a trainable zero-init expansion head (β). Two prev-residual trackers keep
    the R2 head's augmented state IDENTICAL to the R2 champion's (bit-exact update-zero identity) while the expansion head carries
    its own history. `aug_trace` records the EXPANSION (state, action) so `train_perstep` trains only δ_expand. # Postconditions:
    at a zero expansion head, bit-identical to the R2 champion; |Δτ| ≤ slew; no teacher; release/coast/K6 downstream unchanged."""

    def __init__(self, snap: Any, clone: CloneActor, expand_fn: Callable[[np.ndarray], np.ndarray],
                 bounds: ResidualBounds = ResidualBounds(), *, r2_fn: Callable[[np.ndarray], np.ndarray], beta: float,
                 alpha0: float = ALPHA0, start_kinetic: "dict | None" = None, **kw: Any) -> None:
        self.r2_fn = r2_fn
        self.beta = float(beta)
        self.alpha0 = float(alpha0)
        super().__init__(snap, clone, expand_fn, bounds, start_kinetic=start_kinetic, **kw)

    def reset(self) -> None:
        super().reset()                                               # sets self._prev_res (frontier: R2 context; entry: 0)
        self._prev_res_r2 = self._prev_res.copy()                     # the R2 head resumes the captured R2 residual context
        self._prev_res = np.zeros(ACT_DIM, np.float64)                # the expansion head always starts fresh

    def _transport_action(self, rl: Any, obs: np.ndarray) -> np.ndarray:
        u_clone = np.clip(np.asarray(self.actor.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)   # sets the clone hidden
        a_r2 = np.clip(np.asarray(self.r2_fn(augmented_state(self.actor, obs, self._prev_res_r2)),
                                  np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        aug_ex = augmented_state(self.actor, obs, self._prev_res)
        r_ex = np.clip(np.asarray(self.residual_fn(aug_ex), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        u = np.clip(u_clone + self.alpha0 * a_r2 + self.beta * r_ex, -1.0, 1.0)
        self.aug_trace.append((aug_ex, r_ex.copy()))                  # the RL trains the EXPANSION head on its own (state, action)
        self.residual_trace.append({"a_r2": [round(float(x), 5) for x in a_r2],
                                    "r_ex": [round(float(x), 5) for x in r_ex], "u": [round(float(x), 5) for x in u]})
        self._prev_res_r2 = a_r2
        self._prev_res = r_ex
        return u


def unlock_controller_factory(r2_fn: Callable[[np.ndarray], np.ndarray], beta: float,
                              alpha0: float = ALPHA0) -> Callable:
    """A `make_controller(start, clone, expand_fn, bounds, **kw)` binding the frozen R2 head + β for the shared collect/eval."""
    def make(start: Any, clone: CloneActor, expand_fn: Callable[[np.ndarray], np.ndarray],
             bounds: ResidualBounds, **kw: Any) -> AuthorityUnlockController:
        return AuthorityUnlockController(start, clone, expand_fn, bounds, r2_fn=r2_fn, beta=beta, alpha0=alpha0, **kw)
    return make


def champion_key_unlock(k6: bool, safe: bool, clean: bool, min_dtz: float, released: bool, k6_dwell: int = 0) -> tuple:
    """R3-C freeze order (higher is better): strict K6 ≻ safety ≻ cleanliness (0 clamp/stall/reversal) ≻ K6 dwell ≻ min_dtz ≻
    clean release. Strict K6 already implies safe + dwell ≥ 6, so pre-K6 this reduces to safety ≻ clean ≻ min_dtz ≻ released."""
    return (int(bool(k6 and safe)), int(safe), int(clean), int(k6_dwell), -round(min_dtz, 2), int(released))


def _clean(decomp: dict) -> bool:
    return bool(decomp["clamp"] == 0 and decomp["neg"] == 0 and decomp["reversal"] == 0)


def make_collect_unlock(entry_snap: Any, frontiers: list, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
                        w: krl2.Reward2Weights, r2_fn: Callable[[np.ndarray], np.ndarray], beta: float, *,
                        frac_frontier: float = 0.2, envelope_w: float = krl3.ENVELOPE_W, seed: int = 0,
                        alpha0: float = ALPHA0) -> Callable:
    """A `collect_override` drawing (1−frac)·KINETIC-entry / frac·frontier episodes through the authority-unlock controller
    (default 80 % entry / 20 % healthy frontier). Reuses the tested R3-B curriculum + reward3."""
    return krl3.make_collect3(entry_snap, frontiers, clone_factory, bounds, w, frac_frontier=frac_frontier,
                              envelope_w=envelope_w, seed=seed, make_controller=unlock_controller_factory(r2_fn, beta, alpha0))


def make_dev_eval_unlock(entry_snap: Any, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
                         w: krl2.Reward2Weights, r2_fn: Callable[[np.ndarray], np.ndarray], beta: float, *,
                         envelope_w: float = krl3.ENVELOPE_W, alpha0: float = ALPHA0) -> Callable:
    """A `champion_override(actor) -> (key, aux)`: evaluate the unlock policy on the full frozen chain and rank by the R3-C freeze
    order. ``aux['k6_strict']`` drives the immediate freeze-on-first-K6 stop."""
    factory = unlock_controller_factory(r2_fn, beta, alpha0)

    def ev(actor: Any) -> "tuple[tuple, dict]":
        ep = krl3.collect_episode3(entry_snap, clone_factory(), deterministic_residual(actor), bounds, w, envelope_w,
                                   frontier=False, make_controller=factory)
        d = ep.decomp
        clean = _clean(d)
        key = champion_key_unlock(ep.k6, ep.safe, clean, ep.min_dtz, bool(d["released"]))
        return key, {"min_dtz": round(ep.min_dtz, 2), "exit_dtz": round(ep.exit_dtz, 2), "k6": bool(ep.k6),
                     "k6_strict": bool(ep.k6 and ep.safe), "safe": bool(ep.safe), "clean": clean,
                     "released": bool(d["released"]), "stall_pen": d["neg"], "clamp_pen": d["clamp"],
                     "reversal_pen": d["reversal"]}
    return ev


def stop_on_strict_k6(_key: tuple, aux: dict) -> bool:
    """Early-stop predicate for `train_perstep`: the first teacher-free strict learned K6 (safe delivery)."""
    return bool(aux.get("k6_strict"))
