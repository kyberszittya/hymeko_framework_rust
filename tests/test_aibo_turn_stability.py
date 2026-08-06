"""Vukobratović-ZMP turning-stability certificate — the capturability boundary for fast turning.

Locks: the ZMP (with the angular-momentum-rate term) stays inside support for a STABLE turn and leaves it
for a FAST turn (the certificate FAILS before the actual fall — predictive); and the certificate ``_fn``
passes a positive-margin trace, fails a negative one.
"""

from __future__ import annotations

from types import SimpleNamespace

from scenarios.aibo.locomotion_gait import RotationalTurnGait
from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv
from scenarios.aibo.turn_stability import turn_zmp_margins, vukobratovic_zmp, zmp_stability_certificate

_turn = RotationalTurnGait()


def _margins(g: float, steps: int = 300):
    rt = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat"), seed=0)
    gov = rt._gov
    return turn_zmp_margins(rt._env, lambda e: gov.govern(e, _turn.action(e, turn=g)), steps=steps, seed=0)


def test_stable_turn_keeps_zmp_in_support() -> None:
    assert min(_margins(1.0)) > 0.0                          # g=1.0 stable → ZMP never leaves support


def test_fast_turn_zmp_leaves_support() -> None:
    assert min(_margins(1.6)) < 0.0                          # g=1.6 tips → ZMP leaves support (certificate fails)


def test_zmp_is_finite() -> None:
    import numpy as np
    rt = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat"), seed=0)
    rt.reset(seed=0)
    dt = float(rt._env.model.opt.timestep) * int(rt._env.frame_skip)
    pv = np.asarray(rt._env.data.subtree_linvel[0]).copy()
    ph = np.asarray(rt._env.data.subtree_angmom[0]).copy()
    rt._env.step(rt._gov.govern(rt._env, _turn.action(rt._env, turn=1.0)))
    zmp, _v, _h = vukobratovic_zmp(rt._env.model, rt._env.data, pv, ph, dt)
    assert zmp.shape == (2,) and np.all(np.isfinite(zmp))


def test_certificate_fn_passes_positive_fails_negative() -> None:
    cert = zmp_stability_certificate()
    ok = SimpleNamespace(signals=[{"zmp_margin": 0.3}, {"zmp_margin": 0.2}, {"zmp_margin": 0.4}])
    bad = SimpleNamespace(signals=[{"zmp_margin": 0.3}, {"zmp_margin": -0.1}, {"zmp_margin": 0.2}])
    assert cert.evaluate(None, ok) is True
    assert cert.evaluate(None, bad) is False
