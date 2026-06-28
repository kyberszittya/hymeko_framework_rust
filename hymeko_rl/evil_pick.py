"""Parametric "evil" environment generation for the FANUC pick-and-place.

The SAME HyMeKo arm+gripper structure, perturbed into ADVERSARIAL scenarios by a single ``difficulty`` knob
(0 = easy, 1 = evil): the object is pushed to the reach edge and wide angles, made heavy and small, and the
fingers are made slippery. Use it three ways:

* **domain randomization** — train on sampled difficulties for a robust policy;
* **curriculum** — raise difficulty as the policy improves;
* **adversarial accountability** — deliberately try to break the policy and report where it fails (the
  robustness curve), the honest counterpart to a single cherry-picked success.

    python -m hymeko_rl.evil_pick            # print the scripted expert's robustness curve
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from hymeko_rl.render_pick_place import fanuc_pick_env


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def evil_config(difficulty: float, rng: np.random.Generator) -> dict[str, Any]:
    """Adversarial env overrides scaled by ``difficulty`` ∈ [0, 1]. Each call jitters within the level so a
    suite at one difficulty is varied (not a single fixed scene).

    # Preconditions ``rng`` is a seeded Generator (reproducibility).
    # Postconditions a kwargs dict valid for :func:`fanuc_pick_env`; monotonically harder in ``difficulty``.
    """
    d = float(np.clip(difficulty, 0.0, 1.0))
    r0 = _lerp(0.30, 0.40, d)                                             # object pushed to the reach edge
    return dict(
        obj_radius=(r0, r0 + 0.04),
        obj_angle=(_lerp(-0.2, -0.6, d), _lerp(0.2, 0.6, d)),            # wider, off-axis spawns
        box_mass=_lerp(0.10, 0.50, d) * float(rng.uniform(0.85, 1.15)),  # heavier
        box_half=_lerp(0.020, 0.014, d),                                # smaller (harder to grasp)
        finger_friction=_lerp(4.0, 0.6, d) * float(rng.uniform(0.85, 1.15)),  # slippery
    )


def make_evil_env(difficulty: float, seed: int, **overrides: Any) -> Any:
    """Build a FANUC pick env adversarially configured at ``difficulty`` (seeded). ``overrides`` win."""
    cfg = evil_config(difficulty, np.random.default_rng(seed))
    cfg.update(overrides)
    return fanuc_pick_env(**cfg)


def _expert_place_rate(difficulty: float, n_seeds: int) -> float:
    """Fraction of episodes the scripted expert places the object, over ``n_seeds`` evil scenes at this level."""
    placed = 0
    for s in range(n_seeds):
        env = make_evil_env(difficulty, seed=1000 + s)
        env.reset(seed=1000 + s)
        done = success = False
        while not done:
            _o, _r, term, trunc, info = env.step(env.expert_action)
            success = success or bool(info["reached"])
            done = term or trunc
        placed += int(success)
    return placed / n_seeds


def robustness_sweep(difficulties: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
                     n_seeds: int = 8) -> list[tuple[float, float]]:
    """The scripted expert's place rate vs difficulty — the robustness curve. A learned policy would be swept
    the same way (swap the rollout for the policy's ``action_mean``)."""
    return [(d, _expert_place_rate(d, n_seeds)) for d in difficulties]


def main(argv: list[str] | None = None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=8)
    a = ap.parse_args(argv)
    curve = robustness_sweep(n_seeds=a.n_seeds)
    print(json.dumps({"robustness_curve": [{"difficulty": d, "place_rate": r} for d, r in curve]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
