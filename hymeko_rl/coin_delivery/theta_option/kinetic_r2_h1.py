"""R2 training UNDER the explicit H1 HANDOFF_RESET contract — reproduce the first learned K6 as a LEARNING result, not a checkpoint.

The handoff-reset audit corrected the story: R2 (clone + per-step residual) is the first learned teacher-free s1 K6 transport
policy, but only under the explicit APPROACH→HANDOFF_RESET→KINETIC contract (H1); the plain direct handoff (H0) does not deliver.
The original R2 training rolled H0 episodes from the cradle. This module re-runs the SAME R2 training (same residual/reward/authority
α = 0.15) but with the KINETIC policy deployed through the H1 controller, 100 % from the canonical cradle — every episode is the
full uninterrupted chain `cradle → frozen APPROACH → explicit HANDOFF_RESET → KINETIC clone + learning-R2 → coast → K6`, no
offline frozen-entry artifact as episode-start, no snapshot injection.

Reuses the tested R3-B `collect_episode3`/`make_collect3` with `make_controller` = the H1 factory and `envelope_w = 0` (so the
reward is exactly the original R2 `reward2`, no R3-B envelope). All `8a0c1c7b` modules imported unchanged.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual

R2_ALPHA = 0.15                                  # the frozen R2 residual authority (unchanged)


def h1_temporal_factory() -> Callable:
    """A `make_controller(start, clone, residual_fn, bounds, **kw)` that deploys the KINETIC clone+R2 policy through the explicit
    H1 handoff-reset (one online transition-servo step before the first policy action)."""
    def make(start: Any, clone: CloneActor, residual_fn: Callable, bounds: ResidualBounds, **kw: Any) -> Any:
        return HandoffResetTemporalController(start, clone, residual_fn, bounds, **kw)
    return make


def make_collect_r2_h1(cradle: Any, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
                       w: krl2.Reward2Weights) -> Callable:
    """A `collect_override(residual_fn) -> transitions`: 100 % canonical-cradle H1 episodes, reward2 (envelope_w = 0)."""
    factory = h1_temporal_factory()

    def collect(residual_fn: Callable[[np.ndarray], np.ndarray]) -> list:
        return krl3.collect_episode3(cradle, clone_factory(), residual_fn, bounds, w, 0.0, frontier=False,
                                     make_controller=factory).transitions
    return collect


def make_eval_r2_h1(cradle: Any, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
                    w: krl2.Reward2Weights) -> Callable:
    """A `champion_override(actor) -> (key, aux)`: evaluate the R2 residual on the full H1 chain from the cradle, ranked
    strict-K6 ≻ safety ≻ clean ≻ min_dtz ≻ release; ``aux['k6_strict']`` drives freeze-on-first-K6."""
    factory = h1_temporal_factory()

    def ev(actor: Any) -> "tuple[tuple, dict]":
        ep = krl3.collect_episode3(cradle, clone_factory(), deterministic_residual(actor), bounds, w, 0.0,
                                   frontier=False, make_controller=factory)
        d = ep.decomp
        clean = bool(d["clamp"] == 0 and d["neg"] == 0 and d["reversal"] == 0)
        jerk = float(np.mean(np.abs(np.diff(ep.residuals, axis=0)))) if len(ep.residuals) > 1 else 0.0
        k6_strict = bool(ep.k6 and ep.safe)
        key = (int(k6_strict), int(ep.safe), int(clean), -round(ep.min_dtz, 2), int(bool(d["released"])), -round(jerk, 4))
        return key, {"min_dtz": round(ep.min_dtz, 2), "exit_dtz": round(ep.exit_dtz, 2), "k6": bool(ep.k6),
                     "k6_strict": k6_strict, "safe": bool(ep.safe), "clean": clean, "released": bool(d["released"]),
                     "stall_pen": d["neg"], "clamp_pen": d["clamp"], "reversal_pen": d["reversal"]}
    return ev
