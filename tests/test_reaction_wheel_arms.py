"""Arms as a reaction wheel — the flight-phase balance port. Contract tests.

The arms genuinely extend the recoverable balance basin (they stabilise the torso in flight, where the foot
cannot), and they do so within their mechanical range — a hard-limited swing, not an unphysical windmill.
"""

from __future__ import annotations

from scenarios.humanoid.reaction_wheel_arms import ArmBalanceConfig, recoverable_basin


def test_arms_extend_the_recoverable_basin() -> None:
    """Reaction-wheel arms recover markedly more of the perturbation grid than stance-only foot torque."""
    cfg = ArmBalanceConfig()
    foot = recoverable_basin(cfg, use_arm=False)
    arm = recoverable_basin(cfg, use_arm=True)
    assert arm["recovered_fraction"] > foot["recovered_fraction"] + 0.1   # measured ~+0.30 (flight-phase authority)


def test_arm_swing_respects_the_mechanical_limit() -> None:
    """The arms use their range but never exceed it (honest — a real arm cannot windmill past its stop)."""
    cfg = ArmBalanceConfig()
    arm = recoverable_basin(cfg, use_arm=True)
    assert 0.0 < arm["max_arm_swing"] <= cfg.arm_range + 1e-9


def test_balance_is_deterministic() -> None:
    cfg = ArmBalanceConfig()
    assert recoverable_basin(cfg, use_arm=True) == recoverable_basin(cfg, use_arm=True)
