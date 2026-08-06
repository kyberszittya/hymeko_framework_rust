"""Lyapunov certificate + AIBO Lyapunov function (fast, no env)."""

from __future__ import annotations

from scenarios.aibo.lyapunov import AIBOLyapunov, evaluate_lyapunov, lyapunov_certificate


def test_evaluate_passes_on_converging_descent() -> None:
    vs = [1.0 * (0.9 ** k) for k in range(30)]  # monotone -> 0
    r = evaluate_lyapunov(vs)
    assert r["passes"] and r["converged"] and r["net_decrease"]


def test_evaluate_rejects_divergence() -> None:
    vs = [0.1 * (1.1 ** k) for k in range(30)]  # grows
    assert not evaluate_lyapunov(vs)["passes"]


def test_evaluate_rejects_non_convergence() -> None:
    vs = [1.0] * 30  # flat, never reaches ~0
    r = evaluate_lyapunov(vs)
    assert not r["passes"] and not r["converged"]


def test_aibo_lyapunov_zero_at_equilibrium() -> None:
    V = AIBOLyapunov(reach_target=0.42)
    eq = {"dist_to_goal": 0.42, "heading_error": 0.0, "speed": 0.0}
    assert V(eq) == 0.0
    far = {"dist_to_goal": 1.0, "heading_error": 0.5, "speed": 0.2}
    assert V(far) > 0.0


def test_lyapunov_certificate_reward_independent_signature() -> None:
    import inspect

    cert = lyapunov_certificate("stab", AIBOLyapunov())
    params = list(inspect.signature(cert.evaluate).parameters)
    assert params == ["state", "trace"]  # no reward

    class _T:
        signals = ({"dist_to_goal": 1.0, "heading_error": 0.5, "speed": 0.1},) * 5

    assert cert.evaluate(None, _T()) is False  # flat/high V -> not stable
