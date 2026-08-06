"""M2 — centroidal capturability + neural Lyapunov certificate contract tests.

The reduced closed loop has a genuine basin (both recover and fall present); the torch certificate is a genuine
Lyapunov function (V ⪰ 0, V(x*) = 0), its certified sublevel set is CONSERVATIVE (a state inside {V ≤ c} almost
never falls — the safety property) and covers the recoverable set non-trivially, and the fit is deterministic.
Verification is empirical (sampling), stated as such — not a formal proof.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest
import torch

from scenarios.humanoid.centroidal import CentroidalConfig, centroidal_rollout, centroidal_step
from scenarios.humanoid.neural_certificate import NeuralLyapunovCertificate

torch.set_num_threads(4)


@pytest.fixture(scope="module")
def fitted() -> tuple:
    """Fit one certificate on the real-dimensional basin (shared across the property tests)."""
    cfg = CentroidalConfig(grid_n=9)
    x0, _ = centroidal_rollout(cfg)
    cert = NeuralLyapunovCertificate(cfg, seed=0).fit(x0)   # default iters/sep
    return cert, cfg, cert.verify()


def test_centroidal_step_is_deterministic_and_bounded_near_nominal() -> None:
    """The shared step is deterministic; a near-nominal state does not fall."""
    cfg = CentroidalConfig()
    s = np.array([[cfg.z0, 0.0, 0.2, 0.05]])
    a = centroidal_step(s, 0.0, cfg)
    b = centroidal_step(s, 0.0, cfg)
    assert np.array_equal(a, b) and a.shape == s.shape
    _, y = centroidal_rollout(cfg, np.array([[cfg.z0, 0.0, 0.0, 0.05]]))
    assert bool(y[0])                                                    # near-nominal recovers


def test_centroidal_basin_has_both_recover_and_fall() -> None:
    """A genuine capturability basin: the sampled box contains both recovering and falling states."""
    _, y = centroidal_rollout(CentroidalConfig(grid_n=9))
    assert 0.0 < y.mean() < 1.0


def test_neural_certificate_is_nonnegative_and_zero_at_target(fitted) -> None:
    cert, cfg, _ = fitted
    assert float(cert.value(np.array([[cfg.z0, 0.0, 0.0, 0.0]]))[0]) == 0.0
    assert float(cert.value(np.array([[cfg.z0, 0.0, 3.0, 0.6]]))[0]) > 0.0


def test_neural_certificate_is_conservative(fitted) -> None:
    """The safety property: a state inside the certified set {V ≤ c} almost never falls (empirical)."""
    _, _, report = fitted
    assert report["fall_violation_rate"] <= 0.04                        # certified ⇒ rarely falls (conservative)
    assert report["certified_level"] > 0.0


def test_neural_certificate_covers_the_basin_nontrivially(fitted) -> None:
    """The certified inner-approximation covers a non-trivial fraction of the recoverable set (held-out)."""
    _, _, report = fitted
    assert report["iou_vs_recoverable"] >= 0.40


def test_lipschitz_formal_verify_is_sound_and_refines(fitted) -> None:
    """SOUND (Lipschitz) per-cell decrease+no-fall guarantee: coverage refines and the core shrinks as r→0."""
    cert, cfg, _ = fitted
    coarse = cert.lipschitz_formal_verify(cfg, grid_n=81)
    fine = cert.lipschitz_formal_verify(cfg, grid_n=161)
    assert 0.0 <= coarse["sound_fraction"] <= 1.0
    assert fine["sound_fraction"] > coarse["sound_fraction"]              # sound coverage improves with resolution
    assert fine["uncertifiable_core_V"] < coarse["uncertifiable_core_V"]  # the uncertifiable core shrinks (∝ r)
    assert cert.spectral_lipschitz() > 0.0                               # finite positive Lipschitz bound


def test_neural_certificate_fit_is_deterministic() -> None:
    """Same seed ⇒ identical certificate (torch CPU + numpy, no RNG in fit beyond seeded init)."""
    cfg = CentroidalConfig(grid_n=7)
    x0, _ = centroidal_rollout(cfg)
    a = NeuralLyapunovCertificate(cfg, seed=0).fit(x0, iters=200).certified_level(x0)
    b = NeuralLyapunovCertificate(cfg, seed=0).fit(x0, iters=200).certified_level(x0)
    assert a == b


def test_neural_certificate_fit_within_wall_budget() -> None:
    """Fit + verify under budget (median of 5 after warm-up), torch CPU."""
    cfg = CentroidalConfig(grid_n=7)
    x0, _ = centroidal_rollout(cfg)
    NeuralLyapunovCertificate(cfg, seed=0).fit(x0, iters=200).verify()   # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        NeuralLyapunovCertificate(cfg, seed=0).fit(x0, iters=200).verify()
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 30.0
