"""Unit tests for multi-seed aggregation + the 2-proportion tie-test."""
from __future__ import annotations


from hymeko_rl.eval.multiseed import aggregate, compare_ftdom, two_proportion_ztest


def test_aggregate_mean_std_and_violation_dist() -> None:
    per = [{"ft_dom": 0.75, "monitor_score": 0.30, "violation_reason": "a"},
           {"ft_dom": 0.65, "monitor_score": 0.40, "violation_reason": "b"},
           {"ft_dom": 0.70, "monitor_score": 0.35, "violation_reason": "a"}]
    st = aggregate(per, n_eval=24)
    assert st.n_seeds == 3 and st.n_eval == 24
    assert abs(st.mean["ft_dom"] - 0.70) < 1e-9
    assert st.std["ft_dom"] > 0.0
    assert st.violation_dist == {"a": 2, "b": 1}


def test_ftdom_count_recovery() -> None:
    # two seeds at 0.5 and 0.75 over n=24 → 12 + 18 = 30 delivered of 48
    st = aggregate([{"ft_dom": 0.5}, {"ft_dom": 0.75}], n_eval=24)
    c, n = st.ftdom_count()
    assert (c, n) == (30, 48)


def test_two_proportion_ztest_identical_is_tied() -> None:
    z, p = two_proportion_ztest(30, 48, 30, 48)
    assert abs(z) < 1e-9 and p == 1.0


def test_two_proportion_ztest_clear_difference_significant() -> None:
    z, p = two_proportion_ztest(5, 100, 60, 100)   # 5% vs 60%
    assert p < 0.001


def test_compare_ftdom_small_delta_is_tied() -> None:
    # baseline 18/24, candidate 16/24 replicated over 4 seeds → the 2-episode/seed delta must read as 'tied'
    base = aggregate([{"ft_dom": 18 / 24}] * 4, n_eval=24)
    cand = aggregate([{"ft_dom": 16 / 24}] * 4, n_eval=24)
    d = compare_ftdom(base, cand)
    assert d.decision == "tied"
    assert d.p >= 0.05


def test_compare_ftdom_large_drop_is_worse() -> None:
    base = aggregate([{"ft_dom": 0.75}] * 4, n_eval=48)
    cand = aggregate([{"ft_dom": 0.30}] * 4, n_eval=48)
    d = compare_ftdom(base, cand)
    assert d.decision == "worse" and d.p < 0.05
    assert d.delta < 0.0
