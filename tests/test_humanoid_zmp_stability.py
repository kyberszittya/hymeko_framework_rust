"""Vukobratović-ZMP balance certificate for the humanoid — genuine (not vacuous) support-polygon stability.

The same proper ZMP (with Ḣ) that certifies the AIBO turn certifies humanoid balance: PASS for a small
pitch perturbation (ZMP in support), FAIL for a large one (ZMP leaves support — before the fall). Locks
the multi-embodiment stability core + that the certificate distinguishes stability from survival.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv
from scenarios.humanoid.zmp_stability import (
    certified_zmp_margin,
    foot_bodies,
    vukobratovic_zmp,
    zmp_balance_certificate,
)


def _env(perturb: float) -> HumanoidBalanceEnv:
    return HumanoidBalanceEnv(BalanceConfig(perturb_lo=perturb, perturb_hi=perturb, max_steps=300))


def test_small_perturbation_keeps_zmp_in_support() -> None:
    ok, mn, _ = certified_zmp_margin(_env(0.0), perturb=0.0, steps=250)
    assert ok and mn > 0.0                                   # standing → ZMP in support (certifies)


def test_large_perturbation_zmp_leaves_support() -> None:
    ok, mn, _ = certified_zmp_margin(_env(3.0), perturb=3.0, steps=250)
    assert not ok and mn < 0.0                               # big push → ZMP leaves support (fails), before the fall


def test_margin_is_monotone_in_perturbation() -> None:
    _ok0, m0, _ = certified_zmp_margin(_env(0.0), perturb=0.0, steps=200)
    _ok3, m3, _ = certified_zmp_margin(_env(3.0), perturb=3.0, steps=200)
    assert m0 > m3                                           # larger perturbation → smaller (worse) margin


def test_zmp_is_finite() -> None:
    e = _env(0.0)
    e.reset(seed=0)
    dt = float(e.model.opt.timestep) * 10
    pv = np.asarray(e.data.subtree_linvel[0]).copy()
    ph = np.asarray(e.data.subtree_angmom[0]).copy()
    e.step(np.zeros(e.model.nu, np.float32))
    zmp, _v, _h = vukobratovic_zmp(e.model, e.data, pv, ph, dt)
    assert zmp.shape == (2,) and np.all(np.isfinite(zmp))
    assert len(foot_bodies(e.model)) == 2


def test_certificate_fn_passes_positive_fails_negative() -> None:
    cert = zmp_balance_certificate()
    ok = SimpleNamespace(signals=[{"zmp_margin": 0.08}, {"zmp_margin": 0.04}])
    bad = SimpleNamespace(signals=[{"zmp_margin": 0.08}, {"zmp_margin": -0.06}])
    assert cert.evaluate(None, ok) is True
    assert cert.evaluate(None, bad) is False
