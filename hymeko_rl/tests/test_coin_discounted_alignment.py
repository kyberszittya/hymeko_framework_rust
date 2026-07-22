"""COIN_DISCOUNTED_REWARD_ALIGNMENT gate (Option B §9): under the OUTER canonical K=6 env's v3 reward, the DISCOUNTED
return of strict K=6 delivery dominates every non-success behavior class (both SAC and TD3 γ), and no repeatable
non-success loop's infinite-horizon upper bound reaches strict delivery. The internal ordering among failures is
diagnostic, not asserted.
"""
from __future__ import annotations

import pytest

from hymeko_rl.coin_delivery.discounted_alignment import (
    discounted_return,
    resolve_gammas,
    strict_delivery_reference,
)
from hymeko_rl.coin_delivery.run_discounted_alignment import run
from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig, make_delivery_rl_env


@pytest.fixture(scope="module")
def result():
    return run()


def test_verdict_is_pass(result):
    assert result["verdict"] == "COIN_DISCOUNTED_REWARD_ALIGNMENT_PASS"


def test_strict_reference_actually_delivers_at_k6(result):
    # the strict demonstration latches robot-attribution via the acquisition grasp, which is coin-position dependent;
    # the MAJORITY of seeds reach the K=6 terminal (run() aggregates by median, so the reference return is a delivery).
    delivered = [row for row in result["strict_reference_detail"] if row["terminated"] and row["final_dwell"] == 6]
    assert len(delivered) >= (len(result["strict_reference_detail"]) + 1) // 2, \
        f"strict reference reached K=6 on too few seeds: {result['strict_reference_detail']}"


@pytest.mark.parametrize("label", ["sac", "td3"])
def test_strict_dominates_every_failure(result, label):
    g = result["per_gamma"][label]
    assert not g["failures_dominating_strict"], f"{label}: failures ≥ strict: {g['failures_dominating_strict']}"
    assert g["strict_delivery_return"] > max(g["failure_returns"].values())


@pytest.mark.parametrize("label", ["sac", "td3"])
def test_no_repeatable_loop_farms_above_strict(result, label):
    g = result["per_gamma"][label]
    assert not g["farmers_dominating_strict"], f"{label}: farming loops dominate strict: {g['farmers_dominating_strict']}"
    for name, f in g["no_farming"].items():
        assert not f["dominates_strict"], f"{label}: loop {name} infinite bound {f['cycle_upper_bound']} ≥ strict"


def test_gammas_are_read_from_the_actual_configs():
    g = resolve_gammas()
    assert g["sac"] == 0.99 and g["td3"] == 0.99      # if a future config changes γ, this documents the re-measure


def test_discounted_return_is_monotone_in_a_terminal_bonus():
    # a unit sanity check on the discounting: a late +30 terminal raises the return vs the same trajectory without it.
    base = [-1.0] * 7
    assert discounted_return(base[:-1] + [-1.0 + 30.0], 0.99) > discounted_return(base, 0.99)


def test_strict_reference_consumes_the_outer_env_reward():
    # the reference reward stream comes from CoinDeliveryTrainEnv.step (not a re-derivation): a delivering run ends
    # with a large positive terminal step embedded in the OUTER reward.
    env = make_delivery_rl_env(DeliveryRLConfig())
    row = strict_delivery_reference(env, seed=0)
    assert row["terminated"] and row["final_dwell"] == 6
    assert max(row["rewards"]) > 10.0, "the OUTER env's v3 terminal (graded +30) must appear in the reward stream"
