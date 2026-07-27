"""Humanoid COM-Lyapunov + generic certificate (fast, no env)."""

from __future__ import annotations

from scenarios.humanoid.lyapunov import (
    HumanoidCOMLyapunov,
    evaluate_lyapunov,
    lyapunov_certificate,
)


def test_com_lyapunov_zero_at_equilibrium() -> None:
    V = HumanoidCOMLyapunov(h_ref=0.645)             # measured standing COM height
    eq = {"com_z": 0.645, "com_xy_off": 0.0, "com_speed": 0.0, "uprightness": 1.0}
    assert V(eq) == 0.0
    fallen = {"com_z": 0.4, "com_xy_off": 0.2, "com_speed": 2.0, "uprightness": 0.3}
    assert V(fallen) > 1.0


def test_start_at_equilibrium_passes() -> None:
    # V ~ 0 throughout (bounded + converged, not strictly decreasing) -> stable
    vs = [1e-6] * 50
    assert evaluate_lyapunov(vs)["passes"]


def test_collapse_diverges_fails() -> None:
    vs = [0.0] + [0.5 * k for k in range(1, 40)]  # grows unboundedly
    r = evaluate_lyapunov(vs)
    assert not r["passes"] and not r["bounded"]


def test_converging_from_perturbation_passes() -> None:
    vs = [0.4 * (0.9 ** k) for k in range(40)]  # descends to ~0
    assert evaluate_lyapunov(vs)["passes"]


def test_certificate_discriminates() -> None:
    cert = lyapunov_certificate("balance", HumanoidCOMLyapunov())

    class _Stable:
        signals = ({"com_z": 0.645, "com_xy_off": 0.0, "com_speed": 0.0, "uprightness": 1.0},) * 20

    class _Fall:
        signals = tuple({"com_z": 0.645 - 0.02 * k, "com_xy_off": 0.01 * k,
                         "com_speed": 0.1 * k, "uprightness": 1.0} for k in range(20))

    assert cert.evaluate(None, _Stable()) is True
    assert cert.evaluate(None, _Fall()) is False
