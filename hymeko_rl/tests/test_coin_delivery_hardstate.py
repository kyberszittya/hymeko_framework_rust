"""Tests for COIN-DELIVERY-RL-2 hard-state machinery (train.coin_delivery_hardstate + the RL-1 env additions).

Covers: failure-class precedence (each class reachable), CoinDeliveryProblem immutability, the residual-oracle
plumbing (tiny CEM budget) + recoverability labelling keys, the problem generator's oversampling / per-class δ /
recoverable-seed sampling, and the env's delta_override + start_geometry additions.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.train.coin_delivery_hardstate import (
    CoinDeliveryProblem,
    CoinDeliveryProblemGenerator,
    FailureClass,
    OracleConfig,
    Recoverability,
    build_problems,
    classify_state,
    oracle_recoverability,
    residual_oracle_cem,
)
from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig, make_delivery_rl_env

_ZH = 0.04


def _row(**kw) -> dict:
    base = {"center_reach": False, "zone_entry": False, "handoff_event": False, "start_dtz": 0.12, "min_dtz": 0.12,
            "final_dtz": 0.12, "contact_lost": False}
    base.update(kw)
    return base


# ── classification precedence ────────────────────────────────────────────────────────────────────────────────────────
def test_classify_center_success() -> None:
    assert classify_state(_row(center_reach=True), zone_half=_ZH) == FailureClass.CENTER_SUCCESS


def test_classify_zone_only() -> None:
    assert classify_state(_row(zone_entry=True), zone_half=_ZH) == FailureClass.ZONE_ONLY


def test_classify_near_miss() -> None:
    # not in zone, but min_dtz within ~1.5*zone_half
    assert classify_state(_row(min_dtz=0.05), zone_half=_ZH) == FailureClass.NEAR_MISS


def test_classify_contact_loss() -> None:
    assert classify_state(_row(min_dtz=0.10, contact_lost=True, handoff_event=True),
                          zone_half=_ZH) == FailureClass.CONTACT_LOSS


def test_classify_transport_stall() -> None:
    # grasped (handoff) but no transport progress, far from zone, no contact loss
    r = _row(min_dtz=0.118, start_dtz=0.12, handoff_event=True)
    assert classify_state(r, zone_half=_ZH) == FailureClass.TRANSPORT_STALL


def test_classify_geometric_hard() -> None:
    # never grasped, made some progress but far, no contact loss
    r = _row(min_dtz=0.09, start_dtz=0.13, handoff_event=False)
    assert classify_state(r, zone_half=_ZH) == FailureClass.GEOMETRIC_HARD


# ── CoinDeliveryProblem ──────────────────────────────────────────────────────────────────────────────────────────────
def test_problem_is_frozen() -> None:
    p = CoinDeliveryProblem("64000", FailureClass.ZONE_ONLY.value, Recoverability.CURRENT.value, 0.05, 0.3, ("center",))
    with pytest.raises(Exception):
        p.state_id = "x"  # type: ignore[misc]


def test_build_problems_fuses_class_and_oracle() -> None:
    classified = [{"seed": 1, "failure_class": FailureClass.CENTER_SUCCESS.value, "row": _row(center_reach=True), "geom": {}},
                  {"seed": 2, "failure_class": FailureClass.ZONE_ONLY.value, "row": _row(zone_entry=True, start_dtz=0.05), "geom": {}}]
    oracle = {2: {"recoverability": Recoverability.WIDER.value, "recovered_at_delta": 0.5}}
    problems = build_problems(classified, oracle)
    by_id = {p.state_id: p for p in problems}
    assert by_id["1"].recoverability == Recoverability.CURRENT.value  # success is trivially "current"
    assert by_id["2"].recoverability == Recoverability.WIDER.value
    assert by_id["2"].required_residual_scale == 0.5


# ── generator ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _problems() -> list[CoinDeliveryProblem]:
    return [
        CoinDeliveryProblem("1", FailureClass.CENTER_SUCCESS.value, Recoverability.CURRENT.value, 0.03, 0.0, ("center",)),
        CoinDeliveryProblem("2", FailureClass.ZONE_ONLY.value, Recoverability.CURRENT.value, 0.05, 0.3, ("center",)),
        CoinDeliveryProblem("3", FailureClass.NEAR_MISS.value, Recoverability.WIDER.value, 0.11, 0.75, ("center", "zone")),
        CoinDeliveryProblem("4", FailureClass.GEOMETRIC_HARD.value, Recoverability.UNREACHABLE.value, 0.13, float("nan"), ("acquisition",)),
    ]


def test_generator_recoverable_seeds() -> None:
    gen = CoinDeliveryProblemGenerator(_problems())
    assert set(gen.recoverable_seeds) == {2, 3}


def test_generator_weights_zero_for_unreachable() -> None:
    gen = CoinDeliveryProblemGenerator(_problems())
    # the UNREACHABLE state (id 4) must get zero sampling weight
    assert gen._weights[3] == 0.0
    assert gen._weights.sum() == pytest.approx(1.0)


def test_generator_delta_for_uses_required_scale() -> None:
    gen = CoinDeliveryProblemGenerator(_problems())
    assert gen.delta_for(3, default=0.3) == 0.75
    assert gen.delta_for(4, default=0.3) == 0.3  # NaN required scale → default


def test_generator_samples_recoverable_dominantly() -> None:
    gen = CoinDeliveryProblemGenerator(_problems(), easy_weight=0.2)
    rng = np.random.default_rng(0)
    picks = [gen.sample_seed(rng) for _ in range(200)]
    assert 4 not in picks                        # unreachable never sampled
    assert sum(p in (2, 3) for p in picks) > 150  # recoverable dominates


# ── env additions: delta_override + start_geometry ───────────────────────────────────────────────────────────────────
def test_env_delta_override_scales_residual() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    raw = np.full(6, 2.0, np.float32)            # a fixed non-trivial raw action
    env._delta_override = 0.3
    env.reset(seed=64_000)
    env.step(raw)
    small = env.residual_norm()
    env._delta_override = 1.0
    env.reset(seed=64_000)
    env.step(raw)
    big = env.residual_norm()
    assert big > small > 0


def test_env_start_geometry_keys() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    env.reset(seed=64_000)
    g = env.start_geometry()
    for k in ("coin_x", "coin_y", "zone_x", "zone_y", "coin_to_zone_x", "coin_to_zone_y", "start_dist"):
        assert k in g
    assert g["start_dist"] >= 0


# ── oracle plumbing (tiny CEM budget) ────────────────────────────────────────────────────────────────────────────────
def test_oracle_recoverability_returns_valid_label() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    ocfg = OracleConfig(segments=2, pop=4, elite=2, iters=2, horizon=40)
    rng = np.random.default_rng(0)
    res = oracle_recoverability(env, 64_000, (0.3, 1.0), ocfg, rng, zone_half=cfg.zone_half)
    assert res["recoverability"] in {r.value for r in Recoverability}
    assert "per_delta" in res and set(res["per_delta"]) <= {0.3, 1.0}


def test_residual_oracle_cem_returns_metrics() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    ocfg = OracleConfig(segments=2, pop=4, elite=2, iters=2, horizon=40)
    best = residual_oracle_cem(env, 64_000, 0.5, ocfg, np.random.default_rng(0))
    for k in ("center", "zone", "min_dtz", "start_dtz", "contact_frac", "score"):
        assert k in best
