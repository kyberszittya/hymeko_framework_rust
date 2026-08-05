"""Certified balance on the embodied humanoid — the verification arc (viability + HSTL monitor) transfers.

Unlike the control-gain transfer, the VERIFICATION carries over: the scaffold's recoverable region is a viability
boundary, and the HSTL monitor over the combined balance margin certifies recovery (robustness > 0) / flags falls
(robustness < 0) with an early warning. Skipped where MuJoCo / the CLI are absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from scenarios.humanoid.humanoid_certified_balance import (  # noqa: E402
    monitor_balance,
    scaffold_recovers,
)


def test_scaffold_has_a_recoverable_region() -> None:
    assert scaffold_recovers(2.0, range(4)) == 1.0                   # recovers moderate perturbations
    assert scaffold_recovers(5.0, range(4)) < 1.0                    # fails past its viability boundary


def test_monitor_certifies_recovering_balance() -> None:
    report = monitor_balance(2.0, seed=0)
    assert not report["fell"] and report["satisfied"] and report["robustness"] > 0.05


def test_monitor_flags_a_fall_with_early_warning() -> None:
    """A falling balance: robustness goes negative, the spec is unsatisfied, and the monitor warns before the fall."""
    report = monitor_balance(4.5, seed=0)
    assert report["fell"] and not report["satisfied"] and report["robustness"] <= 0.05
    assert 0 <= report["warn_step"] <= report["fall_step"] and report["lead_steps"] > 0


def test_lateral_push_viability_bound() -> None:
    """The scaffold's lateral-push viability boundary is a sharp cliff (~2.3 m/s); above it, in-place fails."""
    from scenarios.humanoid.humanoid_certified_balance import lateral_push_bound, lateral_push_recovers
    assert 1.5 < lateral_push_bound(seeds=range(6)) < 3.0
    assert lateral_push_recovers(3.0, range(4)) == 0.0        # a hard cliff — no gradual stepping-recoverable band
