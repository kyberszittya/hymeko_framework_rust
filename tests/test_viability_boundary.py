"""Learned viability boundary + differential-geometry curvature — the M0 contract tests.

Self-validation: on the pendulum the ROA boundary is analytic (c* = ½kπ²), so a learned boundary is checked
against ground truth. Curvature tests pin the geometry module against the hand-verified scratch values and
the internal identity scalar = 2·Gauss (2-D). Performance asserts a numeric wall-time budget.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import sympy as sp

from scenarios.humanoid.geometry import RiemannianMetric, scalar_curvature_numeric
from scenarios.humanoid.symbolic_ph import (
    ida_pbc_potential_shaping,
    pendulum_ph,
    two_link_leg_ph,
)
from scenarios.humanoid.viability import (
    LearnedBoundary,
    ViabilityConfig,
    analytic_labels,
    control_torque,
    hamiltonian_d,
    in_roa,
    sample_viability,
    separatrix_level,
    validate_boundary,
)

# ---- viability boundary ---------------------------------------------------------------------------------

def test_separatrix_level_is_half_k_pi_squared() -> None:
    cfg = ViabilityConfig(k=24.0)
    assert abs(separatrix_level(cfg) - 0.5 * 24.0 * np.pi ** 2) < 1e-9


def test_in_roa_matches_the_hamiltonian_sublevel_set() -> None:
    """in_roa is exactly {H_d < c*}: a low-energy state is in, a state just past the saddle is out."""
    cfg = ViabilityConfig()
    assert bool(in_roa(cfg.target + 0.2, 0.0, cfg))                       # near the target → inside
    # at θ* enough kinetic energy to clear the saddle (½Iθ̇² > c* ⇔ θ̇ > π√(k/I) ≈ 15.4) → outside
    assert not bool(in_roa(cfg.target, 16.0, cfg))
    assert bool(hamiltonian_d(cfg.target, 0.0, cfg) < separatrix_level(cfg))


def test_control_torque_matches_symbolic_ida_pbc() -> None:
    """Controller-drift guard: the numeric potential-shaping control equals the symbolic IDA-PBC law."""
    cfg = ViabilityConfig(kd=0.0)                                         # isolate the potential-shaping part
    ph = pendulum_ph(m=cfg.m, ell=cfg.ell, g=cfg.grav, b=cfg.b)
    th = ph["q"][0]
    tau_sym = ida_pbc_potential_shaping(ph, sp.Rational(1, 2) * cfg.k * (th - cfg.target) ** 2)[0]
    f = sp.lambdify(th, tau_sym, "numpy")
    for theta in [cfg.target - 1.0, cfg.target - 0.2, cfg.target + 0.5]:   # |θ−θ*|<π so wrap is identity
        assert abs(float(control_torque(theta, 3.3, cfg)) - float(f(theta))) < 1e-9


def test_sample_viability_is_deterministic() -> None:
    cfg = ViabilityConfig(grid_n=21)
    _, y1 = sample_viability(cfg)
    _, y2 = sample_viability(cfg)
    assert np.array_equal(y1, y2)


def test_rollout_labels_agree_with_the_analytic_roa() -> None:
    """The closed-loop simulator respects H_d: rollout recover/fall labels match the analytic ROA (thin shell)."""
    cfg = ViabilityConfig(grid_n=41)
    x, y_roll = sample_viability(cfg)
    y_true = analytic_labels(x, cfg)
    agreement = float(np.mean(y_roll == y_true))
    assert agreement >= 0.92                                              # gap = the damping-enlarged boundary shell


def test_learned_boundary_recovers_the_analytic_separatrix() -> None:
    """M0 self-validation: the (u²,v²) model class recovers the exact separatrix from ground-truth labels."""
    cfg = ViabilityConfig(grid_n=41)
    x, _ = sample_viability(cfg)
    model = LearnedBoundary(cfg).fit(x, analytic_labels(x, cfg))          # train on the analytic ROA
    report = validate_boundary(model, cfg)
    assert report["iou"] >= 0.97                                          # near-exact recovery of the ellipse
    assert report["err_recover"] <= 0.05 and report["err_fall"] <= 0.05


def test_boundary_learned_from_rollout_data_is_close_to_analytic() -> None:
    """From DATA (rollout labels) the learned boundary still tracks the analytic ROA; the gap is the damping shell."""
    cfg = ViabilityConfig(grid_n=41)
    x, y_roll = sample_viability(cfg)
    report = validate_boundary(LearnedBoundary(cfg).fit(x, y_roll), cfg)
    assert report["iou"] >= 0.9                                           # data-driven boundary ≈ analytic ROA
    assert report["err_recover"] <= 0.1                                   # does not wrongly condemn safe states
    # honest physics: damping enlarges the true ROA, so rollout labels are a mild super-set of {H_d<c*}
    assert y_roll.mean() >= analytic_labels(x, cfg).mean()


def test_learned_boundary_requires_fit_before_predict() -> None:
    cfg = ViabilityConfig(grid_n=11)
    try:
        LearnedBoundary(cfg).predict(np.zeros((3, 2)))
        raise AssertionError("expected RuntimeError before fit()")
    except RuntimeError:
        pass


# ---- differential geometry (curvature) ------------------------------------------------------------------

def test_gauss_curvature_of_two_link_leg_matches_scratch_and_changes_sign() -> None:
    """The leg's kinetic metric has a computable Gauss curvature that flips sign with the knee angle."""
    leg = two_link_leg_ph(m1=4.0, m2=3.0, l1=0.34, l2=0.34)
    metric = RiemannianMetric(leg["M"], leg["q"])
    k_expr = metric.gauss_curvature()
    q2 = leg["q"][1]
    assert float(k_expr.subs({q2: 0.05})) > 5.5                          # knee near-straight: geodesics focus (K≈+6.2)
    assert float(k_expr.subs({q2: 3.0})) < -5.0                          # knee folded: geodesics diverge (K≈−5.9)


def test_scalar_curvature_is_twice_gauss_in_two_d() -> None:
    """Internal cross-check of the curvature machinery via the 2-D identity R = 2K."""
    leg = two_link_leg_ph()
    metric = RiemannianMetric(leg["M"], leg["q"])
    diff = sp.simplify(metric.scalar_curvature() - 2 * metric.gauss_curvature())
    assert float(diff.subs({leg["q"][1]: 0.7})) == 0.0 or abs(float(diff.subs({leg["q"][1]: 0.7}))) < 1e-9


def test_flat_metric_has_zero_curvature() -> None:
    q = list(sp.symbols("q0 q1", real=True))
    metric = RiemannianMetric(sp.eye(2), q)
    assert sp.simplify(metric.scalar_curvature()) == 0


def test_bakry_emery_equals_the_shaping_gain() -> None:
    """The closed loop's Bakry–Émery Ricci is the shaping gain k — the entropy-relaxation rate bound."""
    th = sp.Symbol("theta", real=True)
    k = sp.Symbol("k", positive=True)
    metric = RiemannianMetric(sp.Matrix([[sp.Integer(1)]]), [th])         # flat kinetic line
    be = metric.bakry_emery(sp.Rational(1, 2) * k * (th - sp.pi) ** 2)
    assert sp.simplify(be[0, 0] - k) == 0


def test_numeric_scalar_curvature_matches_symbolic() -> None:
    """The finite-difference curvature (the path used for MuJoCo mj_fullM) matches the symbolic value."""
    leg = two_link_leg_ph(m1=4.0, m2=3.0, l1=0.34, l2=0.34)
    metric = RiemannianMetric(leg["M"], leg["q"])
    q0, q1 = leg["q"]
    sym = float(metric.scalar_curvature().subs({q0: 0.3, q1: 0.6}))
    m_fn = sp.lambdify((q0, q1), leg["M"], "numpy")
    num = scalar_curvature_numeric(lambda q: np.asarray(m_fn(q[0], q[1]), float), [0.3, 0.6], eps=1e-4)
    assert abs(num - sym) < 1e-2 * (1 + abs(sym))


# ---- performance ----------------------------------------------------------------------------------------

def test_sample_viability_within_wall_budget() -> None:
    """Vectorised rollouts over the default grid must stay under budget (median of 5 after warm-up)."""
    cfg = ViabilityConfig(grid_n=41)
    sample_viability(cfg)                                                 # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sample_viability(cfg)
        times.append(time.perf_counter() - t0)
    median, worst = statistics.median(times), max(times)
    assert median < 5.0, f"median {median:.2f}s exceeds 5s budget (worst {worst:.2f}s)"
