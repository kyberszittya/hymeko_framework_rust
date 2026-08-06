"""Regression tests for the centroidal-plan whole-body tracking — the humanoid running the momentum plan.

Each pins a bug/contract from the build (CLAUDE.md §3):
- the plan is solved for the REAL humanoid mass (a hardcoded 18 kg feed-forward under-pushed by ~2× and the
  CoM sank — the mass mismatch bug);
- tracking the plan drives the humanoid FORWARD with a genuine flight phase (both feet off), not backward.
These are integration tests (a few strides), kept short.
"""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.centroidal_run import CentroidalRunConfig
from scenarios.humanoid.centroidal_track import CentroidalRunner, TrackConfig


def test_plan_is_solved_for_the_real_humanoid_mass() -> None:
    """Regression: the feed-forward used a hardcoded 18 kg while the humanoid is ~35 kg → under-push → sink.

    The runner must read the model's total mass and solve/feed-forward with IT."""
    r = CentroidalRunner(CentroidalRunConfig(target_speed=1.0))
    assert r._mass > 25.0                                    # the sagittal humanoid is ~35 kg, not the 18 kg default
    assert abs(r.plan.speed - 1.0) < 0.2                     # the plan (re-solved for the real mass) still hits the speed


def test_tracking_drives_the_humanoid_forward_with_flight() -> None:
    """The centroidal-plan tracking makes the humanoid RUN forward (net +x) with a genuine flight phase."""
    r = CentroidalRunner(CentroidalRunConfig(target_speed=0.9),
                         TrackConfig(pel_w=120, post_w=4, post_kp=20, push_boost=1.1, com_w=200, fall_z=0.40))
    fwd, flight, upright, _fell = r.run(n_strides=6)
    assert fwd > 0.25                                        # decisively forward (the CEM hop netted ~0.1 m; this runs)
    assert flight > 0.03                                     # a real both-feet-off flight phase emerges
    assert upright > 0.7                                     # stays broadly upright through the run


def test_flight_phase_uses_no_ground_contact() -> None:
    """During the (contact-less) flight phase the airborne detector must fire — the plan's flight is real."""
    r = CentroidalRunner(CentroidalRunConfig(target_speed=1.0), TrackConfig(fall_z=0.40))
    _fwd, flight, _up, _fell = r.run(n_strides=4)
    assert flight > 0.02                                     # some fraction of the stride has both feet off the floor
    assert np.allclose(r.plan.force[~r.plan.contact], 0.0)   # and the plan those flight knots track carries zero force
