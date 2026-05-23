"""Unit tests for the coherence + estimator + simulator triad."""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRIAD_PATH = REPO_ROOT / "data" / "coalitions" / "triad_hri.hymeko"


# ─── coherence: σ-cycle product ─────────────────────────────────────


def test_sigma_three_positive_is_one():
    from signedkan_wip.src.rapport.coherence import sigma_cycle
    from signedkan_wip.src.rapport.coalition import SigmaCycle
    cyc = SigmaCycle(name="c", members=["e1", "e2", "e3"])
    s = sigma_cycle({"e1": 0.8, "e2": 0.9, "e3": 0.7}, cyc)
    # all positive → sign +1, min |w| = 0.7
    assert abs(s - 0.7) < 1e-9


def test_sigma_one_negative_is_minus():
    from signedkan_wip.src.rapport.coherence import sigma_cycle
    from signedkan_wip.src.rapport.coalition import SigmaCycle
    cyc = SigmaCycle(name="c", members=["e1", "e2", "e3"])
    s = sigma_cycle({"e1": -0.8, "e2": 0.9, "e3": 0.7}, cyc)
    # one negative → sign -1, min |w| = 0.7
    assert abs(s + 0.7) < 1e-9


def test_sigma_two_negative_is_positive():
    """Cartwright-Harary: two negatives in a 3-cycle = balanced."""
    from signedkan_wip.src.rapport.coherence import sigma_cycle
    from signedkan_wip.src.rapport.coalition import SigmaCycle
    cyc = SigmaCycle(name="c", members=["e1", "e2", "e3"])
    s = sigma_cycle({"e1": -0.8, "e2": -0.9, "e3": 0.7}, cyc)
    # two negatives → sign +1
    assert s > 0
    assert abs(s - 0.7) < 1e-9


def test_sigma_zero_weight_collapses():
    from signedkan_wip.src.rapport.coherence import sigma_cycle
    from signedkan_wip.src.rapport.coalition import SigmaCycle
    cyc = SigmaCycle(name="c", members=["e1", "e2"])
    assert sigma_cycle({"e1": 0.5, "e2": 0.0}, cyc) == 0.0


# ─── estimator: EMA over observed signed nudges ─────────────────────


def test_estimator_positive_observations_drive_weight_toward_plus_one():
    from signedkan_wip.src.rapport.coalition import load_coalition
    from signedkan_wip.src.rapport.estimator import CoalitionEstimator, Observation
    coalition = load_coalition(TRIAD_PATH)
    est = CoalitionEstimator(coalition, alpha=0.3, decay_to_prior=0.0)
    # Repeated positive observations on alice-bob.
    for t in range(50):
        est.step([Observation(t=t, kind="gaze_at", src="alice", dst="bob")])
    w_ab = est.weights["r_ab"]
    # Should be close to the schema's positive target (0.30) — settled toward it
    # given alpha=0.3 and 50 steps.
    assert w_ab > 0.2
    assert w_ab <= 1.0


def test_estimator_negative_observations_flip_sign():
    from signedkan_wip.src.rapport.coalition import load_coalition
    from signedkan_wip.src.rapport.estimator import CoalitionEstimator, Observation
    coalition = load_coalition(TRIAD_PATH)
    est = CoalitionEstimator(coalition, alpha=0.3, decay_to_prior=0.0)
    # Start positive (initial weight is +1.0 from the .hymeko file).
    assert est.weights["r_ab"] > 0
    # Apply sustained tone_negative observations.
    for t in range(50):
        est.step([Observation(t=t, kind="tone_negative", src="alice", dst="bob")])
    # Weight should now be negative.
    assert est.weights["r_ab"] < 0


def test_estimator_no_observations_decays_toward_prior():
    from signedkan_wip.src.rapport.coalition import load_coalition
    from signedkan_wip.src.rapport.estimator import CoalitionEstimator, Observation
    coalition = load_coalition(TRIAD_PATH)
    est = CoalitionEstimator(coalition, alpha=0.0, decay_to_prior=0.2)
    # Set a relation to a non-prior value.
    est.weights["r_ab"] = -0.5
    prior = est.prior["r_ab"]
    # No observations → decay toward prior.
    for _ in range(20):
        est.step([])
    # After 20 steps with 0.2 decay rate, should have moved meaningfully
    # toward the prior.
    assert est.weights["r_ab"] > -0.5
    assert abs(est.weights["r_ab"] - prior) < 0.5


# ─── simulator: scripted conflict scenario ──────────────────────────


def test_simulator_emits_observations_at_baseline():
    from signedkan_wip.src.rapport.coalition import load_coalition
    from signedkan_wip.src.rapport.simulator import (
        Simulator, SimulatorConfig,
    )
    coalition = load_coalition(TRIAD_PATH)
    cfg = SimulatorConfig(n_frames=100, baseline_rate=3.0, seed=0)
    sim = Simulator(coalition, cfg)
    total = 0
    for t in range(100):
        obs = sim.step(t)
        total += len(obs)
    # Average rate ~3 obs/frame × 100 frames → expect ~300, allow 0.5x-2x
    assert 100 < total < 600


def test_simulator_conflict_scenario_drives_sigma_negative():
    """End-to-end falsifier test (Claim b): an injected conflict
    drives σ(triad) < -0.2 within a few frames."""
    from signedkan_wip.src.rapport.coalition import load_coalition
    from signedkan_wip.src.rapport.estimator import CoalitionEstimator
    from signedkan_wip.src.rapport.coherence import sigma_cycle
    from signedkan_wip.src.rapport.simulator import (
        ConflictScenario, Simulator, SimulatorConfig,
    )
    coalition = load_coalition(TRIAD_PATH)
    cfg = SimulatorConfig(
        n_frames=80,
        conflict_scenarios=(
            ConflictScenario(start=20, end=60, agent_a="alice", agent_b="bob"),
        ),
        seed=0,
    )
    sim = Simulator(coalition, cfg)
    est = CoalitionEstimator(coalition, alpha=0.2)
    sigma_trace = []
    for t in range(80):
        obs = sim.step(t)
        w = est.step(obs)
        s = sigma_cycle(w, coalition.cycles["triad"])
        sigma_trace.append(s)
    # σ should be positive (balanced) before conflict
    pre_conflict = sigma_trace[:20]
    assert max(pre_conflict) > 0, f"pre-conflict max σ = {max(pre_conflict)}"
    # σ should drop below -0.2 during conflict
    conflict_min = min(sigma_trace[20:60])
    assert conflict_min < -0.2, f"in-conflict min σ = {conflict_min}, expected < -0.2"
