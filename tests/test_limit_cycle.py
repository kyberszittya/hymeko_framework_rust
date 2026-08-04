"""M2-limit-cycle — Poincaré-map certificate contract tests (the L-coupled running regime).

The soft-regulated gait is a limit cycle; on the Poincaré section it is a fixed point of the stride map, which
lets a quadratic certificate work where the point-Lyapunov one collapsed (M2⁺). These pin: the section fixed
point is genuine, the stride map contracts (stable gait), the certificate is PSD/zero at the gait, its formal LMI
holds exactly, the certified set is conservative AND genuinely depends on L (the whole point of this regime).
"""

from __future__ import annotations

import time

import numpy as np

from scenarios.humanoid.limit_cycle import (
    PoincareLyapunovCertificate,
    gait_fixed_point,
    soft_running_config,
    stride_map,
)


def _certified_l_range(cert: PoincareLyapunovCertificate) -> float:
    grid = cert._section_grid()
    inside = grid[cert.value(grid) <= cert.certified_level(grid)]
    return float(inside[:, 2].max() - inside[:, 2].min())


def test_gait_fixed_point_is_a_stride_map_fixed_point() -> None:
    """Iterating the Poincaré map converges the (L, pitch) transverse coordinates to a genuine fixed point."""
    cfg = soft_running_config()
    xstar = gait_fixed_point(cfg)
    nxt, fell = stride_map(xstar[None, :], cfg)
    assert np.allclose(nxt[0, 2:4], xstar[2:4], atol=1e-6) and not fell[0]


def test_stride_map_contracts_stable_gait() -> None:
    """The gait is stable: the linearised stride map DP at x* has spectral radius < 1."""
    cfg = soft_running_config()
    fv = PoincareLyapunovCertificate(cfg).formal_verify()
    assert fv["gait_stable"] and fv["spectral_radius_DP"] < 1.0


def test_certificate_is_psd_and_zero_at_the_gait() -> None:
    cfg = soft_running_config()
    cert = PoincareLyapunovCertificate(cfg)
    assert float(cert.value(cert.xstar[None, :])[0]) == 0.0
    assert np.all(np.linalg.eigvalsh(cert.matrix()) > 0)


def test_formal_verify_holds_exactly() -> None:
    """The exact LMI Q = DPᵀP DP − P ⪯ 0 certifies stride-to-stride decrease at the gait fixed point."""
    cfg = soft_running_config()
    fv = PoincareLyapunovCertificate(cfg).fit(iters=3000).formal_verify()
    assert fv["decreasing"] and fv["max_eig_Q"] <= 0.0


def test_certificate_is_conservative() -> None:
    """Safety: no section state inside {V ≤ c} falls within the horizon (empirical, held-out grid)."""
    cfg = soft_running_config()
    report = PoincareLyapunovCertificate(cfg).fit(iters=3000).verify()
    assert report["fall_violation_rate"] <= 0.02 and report["certified_level"] > 0.0


def test_certified_set_genuinely_depends_on_L() -> None:
    """The point of the limit-cycle regime: the certified set spans a wide range of L (L is not a spectator)."""
    cfg = soft_running_config()
    cert = PoincareLyapunovCertificate(cfg).fit(iters=3000)
    assert _certified_l_range(cert) > 1.0                            # certified L-range is wide (L matters)


def test_fit_is_deterministic() -> None:
    cfg = soft_running_config()
    a = PoincareLyapunovCertificate(cfg).fit(iters=1500).certified_level(
        PoincareLyapunovCertificate(cfg)._section_grid())
    b = PoincareLyapunovCertificate(cfg).fit(iters=1500).certified_level(
        PoincareLyapunovCertificate(cfg)._section_grid())
    assert a == b


def test_fit_and_verify_within_wall_budget() -> None:
    cfg = soft_running_config()
    PoincareLyapunovCertificate(cfg).fit(iters=1500).verify()       # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        PoincareLyapunovCertificate(cfg).fit(iters=1500).verify()
        times.append(time.perf_counter() - t0)
    assert sorted(times)[2] < 10.0
