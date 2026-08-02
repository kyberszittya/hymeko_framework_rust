"""M1 — verified Lyapunov certificate contract tests (self-validating on the pendulum).

The certificate is a genuine Lyapunov function (PSD, zero at the target), its certified sublevel set is verified
(near-zero decrease/no-cross violation) and CONSERVATIVE (never exceeds the analytic barrier c*), and it recovers
the analytic ROA when seeded by H_d — with the seed shown to be load-bearing. The shared closed-loop step is
pinned so the certificate is about the same dynamics that produced the M0 labels.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from scenarios.humanoid.viability import (
    LyapunovCertificate,
    ViabilityConfig,
    closed_loop_step,
    control_torque,
    sample_viability,
    separatrix_level,
)


def _fit(cfg: ViabilityConfig, seed_from_hd: bool = True) -> LyapunovCertificate:
    x, _ = sample_viability(cfg)
    return LyapunovCertificate(cfg, seed_from_hd=seed_from_hd).fit(x)


def test_certificate_is_psd_and_zero_at_the_target() -> None:
    """V is a genuine Lyapunov candidate: V(x*)=0, P ≻ 0, V>0 away from the target."""
    cfg = ViabilityConfig(grid_n=21)
    cert = LyapunovCertificate(cfg)
    assert float(cert.value(np.array([[cfg.target, 0.0]]))[0]) == 0.0
    assert np.all(np.linalg.eigvalsh(cert.matrix()) > 0)
    assert float(cert.value(np.array([[cfg.target + 0.5, 0.0]]))[0]) > 0


def test_seeded_certificate_recovers_the_analytic_roa() -> None:
    """Seeded by H_d, the verified certified set matches the analytic ROA with a near-zero violation rate."""
    cfg = ViabilityConfig(grid_n=41)
    report = _fit(cfg, seed_from_hd=True).verify()
    assert report["iou_vs_analytic"] >= 0.9                              # certified set ≈ analytic ROA
    assert report["violation_rate"] <= 0.02                             # the decrease/no-cross condition holds


def test_certified_level_is_conservative() -> None:
    """Safety: the certified level never exceeds the true barrier c* (no over-certification of unsafe states)."""
    cfg = ViabilityConfig(grid_n=41)
    report = _fit(cfg, seed_from_hd=True).verify()
    cstar = separatrix_level(cfg)
    assert 0.8 * cstar <= report["certified_level"] <= cstar             # tight but strictly inside the barrier


def test_hd_seed_is_load_bearing() -> None:
    """Honest scoping: the H_d seed carries the certificate; an unseeded fit recovers a much smaller set."""
    cfg = ViabilityConfig(grid_n=41)
    seeded = _fit(cfg, seed_from_hd=True).verify()["iou_vs_analytic"]
    unseeded = _fit(cfg, seed_from_hd=False).verify()["iou_vs_analytic"]
    assert seeded >= unseeded + 0.2                                      # the seed does the heavy lifting


def test_shared_closed_loop_step_matches_the_manual_integrator() -> None:
    """Refactor / one-shared-step contract: closed_loop_step equals the inline control + semi-implicit update."""
    cfg = ViabilityConfig()
    theta, thetadot = np.array([2.0]), np.array([1.0])
    tau = control_torque(theta, thetadot, cfg)
    tdot = thetadot + (-cfg.mgl * np.sin(theta) - cfg.b * thetadot + tau) / cfg.inertia * cfg.dt
    th_manual = theta + tdot * cfg.dt
    th, td = closed_loop_step(theta, thetadot, cfg)
    assert np.allclose(th, th_manual) and np.allclose(td, tdot)


def test_certificate_fit_and_verify_within_wall_budget() -> None:
    """Fit + verify must stay under budget (median of 5 after warm-up)."""
    cfg = ViabilityConfig(grid_n=41)
    x, _ = sample_viability(cfg)
    LyapunovCertificate(cfg).fit(x).verify()                            # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        LyapunovCertificate(cfg).fit(x).verify()
        times.append(time.perf_counter() - t0)
    median = statistics.median(times)
    assert median < 10.0, f"median {median:.2f}s exceeds 10s budget"
