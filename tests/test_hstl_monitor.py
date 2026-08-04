"""M2⁺ — HSTL runtime monitor contract tests.

The Python backend reproduces robust-STL (ρ = min margin); the Rust backend is a documented slot; and the
certificate-tied monitor gives a graded online safety margin whose verdict matches the certificate — satisfied
and safe inside {V ≤ c}, flagged (no later than the fall) outside. Most tests use a fast numpy stand-in for the
certificate so the monitor logic is exercised without a torch fit; one light test wires the real certificate.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from scenarios.humanoid.centroidal import CentroidalConfig, centroidal_rollout
from scenarios.humanoid.hstl_monitor import make_monitor, monitor_trajectory
from scenarios.humanoid.neural_certificate import NeuralLyapunovCertificate


class _QuadCert:
    """A fast numpy stand-in certificate: V = L² + pitch² (the fall-relevant coordinates)."""

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return x[:, 2] ** 2 + x[:, 3] ** 2


def test_python_htl_backend_is_robust_stl() -> None:
    """G(m ≥ 0) robustness is the min margin over the trace; satisfaction flips when a margin goes negative."""
    mon = make_monitor("G(m >= 0)", "python")
    for t, m in enumerate([0.4, 0.2, 0.05]):
        mon.observe(t, {"m": m})
    assert mon.satisfied() and abs(mon.robustness() - 0.05) < 1e-9
    mon.observe(3, {"m": -0.1})
    assert not mon.satisfied() and abs(mon.robustness() + 0.1) < 1e-9


def test_rust_backend_parity_with_python() -> None:
    """The Rust (hymeko.HtlMonitor) backend is bit-identical to the Python engine on the same stream."""
    pytest.importorskip("hymeko")                                   # the built pyo3 extension; skip in a headless build
    rust = make_monitor("G(m >= 0)", "rust")
    py = make_monitor("G(m >= 0)", "python")
    for t, m in enumerate([0.5, 0.4, 0.42, 0.1, -0.05, 0.2, -0.2]):
        assert abs(rust.observe(t, {"m": m}) - py.observe(t, {"m": m})) < 1e-9
    assert rust.satisfied() == py.satisfied()
    with pytest.raises(ValueError):
        make_monitor("G(m >= 0)", "nonsense")


def test_monitor_certifies_a_state_inside_the_certified_set() -> None:
    """A state inside {V ≤ c} that recovers keeps both specs satisfied with a positive margin."""
    cfg = CentroidalConfig()
    res = monitor_trajectory(np.array([cfg.z0, 0.0, 0.0, 0.05]), _QuadCert(), level=0.5, cfg=cfg)
    assert res.satisfied_certified and res.satisfied_upright
    assert not res.fell and res.min_cert_margin > 0.0


def test_monitor_flags_a_falling_state_no_later_than_the_fall() -> None:
    """Outside/falling: the certificate spec is violated, and the warning is raised no later than the fall."""
    cfg = CentroidalConfig()
    res = monitor_trajectory(np.array([cfg.z0, 0.0, 0.0, 1.35]), _QuadCert(), level=0.5, cfg=cfg)
    assert res.fell and not res.satisfied_certified
    assert res.warn_step >= 0 and res.warn_step <= res.fall_step   # never warns after the fall


def test_monitor_margin_is_graded_by_depth() -> None:
    """The robustness is a graded safety margin: deeper inside the set ⇒ larger margin."""
    cfg = CentroidalConfig()
    deep = monitor_trajectory(np.array([cfg.z0, 0.0, 0.0, 0.02]), _QuadCert(), level=0.5, cfg=cfg)
    near = monitor_trajectory(np.array([cfg.z0, 0.0, 0.0, 0.65]), _QuadCert(), level=0.5, cfg=cfg)
    assert deep.min_cert_margin > near.min_cert_margin > 0.0


def test_monitor_is_deterministic() -> None:
    cfg = CentroidalConfig()
    x = np.array([cfg.z0, 0.0, 1.0, 0.3])
    a = monitor_trajectory(x, _QuadCert(), level=0.5, cfg=cfg)
    b = monitor_trajectory(x, _QuadCert(), level=0.5, cfg=cfg)
    assert a == b


def test_monitor_rejects_a_batched_state() -> None:
    with pytest.raises(ValueError):
        monitor_trajectory(np.zeros((3, 4)), _QuadCert(), level=0.5, cfg=CentroidalConfig())


def test_monitor_integrates_with_the_neural_certificate() -> None:
    """Light integration: the real torch certificate drives the monitor — a fall is flagged, the nominal does not fall."""
    cfg = dataclasses.replace(CentroidalConfig(grid_n=7), horizon=0.6)
    x0, _ = centroidal_rollout(cfg)
    cert = NeuralLyapunovCertificate(cfg, seed=0).fit(x0, iters=200)
    level = cert.certified_level(x0)
    falling = monitor_trajectory(np.array([cfg.z0, 0.0, 0.0, 1.35]), cert, level, cfg)
    nominal = monitor_trajectory(np.array([cfg.z0, 0.0, 0.0, 0.0]), cert, level, cfg)
    assert falling.fell and not falling.satisfied_certified     # a real fall is flagged (high V ⇒ negative margin)
    assert not nominal.fell                                      # the nominal gait does not fall


def test_monitor_within_wall_budget() -> None:
    """One monitored run (numpy stand-in) is cheap (median of 5)."""
    cfg = CentroidalConfig()
    x = np.array([cfg.z0, 0.0, 0.5, 0.2])
    monitor_trajectory(x, _QuadCert(), level=0.5, cfg=cfg)      # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        monitor_trajectory(x, _QuadCert(), level=0.5, cfg=cfg)
        times.append(time.perf_counter() - t0)
    assert sorted(times)[2] < 3.0
