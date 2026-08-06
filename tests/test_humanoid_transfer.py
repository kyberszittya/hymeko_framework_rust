"""Reduced-model → embodied humanoid transfer: the arm reaction-wheel benefit does not carry over (honest).

The certified PD-hold scaffold is robust to moderate pitch perturbations; where it fails, a bounded arm-swing
residual adds no recovery — the reduced model's arm/torso inertia ratio over-predicted the arms' authority.
Skipped where MuJoCo / the built CLI are absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from scenarios.humanoid.humanoid_transfer import (  # noqa: E402
    arm_reaction_wheel_controller,
    baseline_controller,
    survival_rate,
)


def test_certified_baseline_is_robust_to_moderate_perturbation() -> None:
    assert survival_rate(baseline_controller, 2.0, n_seeds=4) == 1.0     # a=0 PD-hold recovers ≤2 rad/s


def test_arm_residual_does_not_transfer_the_benefit() -> None:
    """In the baseline's failure regime, the arm reaction-wheel residual adds no recovery (negative transfer)."""
    base = survival_rate(baseline_controller, 4.0, n_seeds=4)
    rw = survival_rate(arm_reaction_wheel_controller(k=4.0, sign=1), 4.0, n_seeds=4)
    assert base < 1.0                                                    # 4 rad/s is in the failure regime
    assert rw <= base + 0.2                                              # arms add nothing beyond the scaffold
