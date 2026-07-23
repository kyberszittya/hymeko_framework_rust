"""Mixed-teacher-averaging audit unit tests (pure, deterministic; no env). Synthetic teachers construct the two regimes:
a clearly-averaging actor (learned = midpoint of two far-apart teachers, in locally-mixed neighborhoods) must be
CONFIRMED; an on-target actor (learned == its label, agreeing teachers) must be NOT_CONFIRMED."""
import numpy as np
import pytest

from hymeko_rl.coin_delivery.coin_mixed_teacher_audit import _knn_indices, _segment_position, mixed_teacher_metrics

ACT = 4


def _flags(n):
    return [{"safe": True, "improving": True} for _ in range(n)]


def test_segment_position_midpoint_and_endpoints():
    a = np.zeros((3, ACT), np.float32); b = np.ones((3, ACT), np.float32)
    learned = np.stack([a[0], b[0], 0.5 * (a[0] + b[0])])
    t, perp, d = _segment_position(learned, a, b)
    assert np.allclose(t, [0.0, 1.0, 0.5], atol=1e-5) and np.allclose(perp, 0.0, atol=1e-5)
    assert np.allclose(d, np.sqrt(ACT))


def test_knn_excludes_self_and_finds_cluster():
    X = np.array([[0, 0], [0.1, 0.0], [0.0, 0.1], [10, 10]], np.float32)
    nn = _knn_indices(X, 1)
    assert nn[0, 0] in (1, 2) and 0 not in nn[0]        # self excluded, nearest is a cluster member


def test_averaging_actor_confirmed():
    rng = np.random.default_rng(0); n = 120
    X = rng.normal(size=(n, 6)).astype(np.float32)      # states overlap => mixed neighborhoods
    pi0 = rng.normal(size=(n, ACT)).astype(np.float32)
    planner = pi0 + 2.0                                 # teachers 2.0*sqrt(ACT) apart (large gap)
    prov = ["planner" if i % 2 else "pi0_fallback" for i in range(n)]   # interleaved modes => high NN disagreement
    label = np.where(np.array(prov)[:, None] == "planner", planner, pi0).astype(np.float32)
    learned = 0.5 * (pi0 + planner)                     # the actor averages: midpoint, off BOTH labels
    m = mixed_teacher_metrics(X, pi0, planner, label, prov, _flags(n), learned, k=8)
    assert m["between_teachers_frac"] > 0.9 and m["nn_mode_disagreement_mean"] >= 0.25
    assert m["planner_states_mean_segment_pos"] < 0.6
    assert m["verdict"] == "MIXED_TEACHER_AVERAGING_CONFIRMED"


def test_on_target_actor_not_confirmed():
    rng = np.random.default_rng(1); n = 120
    X = rng.normal(size=(n, 6)).astype(np.float32)
    pi0 = rng.normal(size=(n, ACT)).astype(np.float32)
    planner = pi0.copy()                                # teachers agree => no averaging possible
    prov = ["pi0_fallback"] * n
    label = pi0.copy(); learned = pi0.copy()            # actor hits its label exactly
    m = mixed_teacher_metrics(X, pi0, planner, label, prov, _flags(n), learned, k=8)
    assert m["teacher_gap"]["median"] < 0.5 and m["first_action_error"]["mean"] < 1e-5
    assert m["verdict"] == "MIXED_TEACHER_AVERAGING_NOT_CONFIRMED"


def test_preconditions_enforced():
    n = 20; X = np.zeros((n, 3), np.float32); a = np.zeros((n, ACT), np.float32)
    with pytest.raises(AssertionError):
        mixed_teacher_metrics(X, a, a, a, ["bogus"] * n, _flags(n), a, k=4)     # bad provenance token
    with pytest.raises(AssertionError):
        mixed_teacher_metrics(X, a, a, a, ["planner"] * n, _flags(n), a, k=n + 5)  # k >= N
