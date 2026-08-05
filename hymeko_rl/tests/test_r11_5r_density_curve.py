"""Tests for the R11.5R density-ablation curve: wide-train pool, deterministic subsample, aggregation + verdict."""
import json
from pathlib import Path

from hymeko_rl.experiments.r11_5r_density_curve import (
    _curve_verdict,
    _summ,
    aggregate,
    subsample,
    wide_train_ids,
)


def test_wide_train_ids_filters_split_and_status(tmp_path: Path) -> None:
    rows = [{"scenario_id": "a", "split": "train", "status": "WIDE_RECERTIFIED"},
            {"scenario_id": "b", "split": "train", "status": "NARROW_ONLY"},
            {"scenario_id": "c", "split": "dev", "status": "WIDE_RECERTIFIED"},
            {"scenario_id": "d", "split": "train", "status": "NO_NOMINAL_K6"}]
    p = tmp_path / "merged.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert wide_train_ids(p) == {"a"}                       # only wide AND train


def test_subsample_deterministic_sized_and_seed_varies() -> None:
    pool = list(range(20))
    a = subsample(pool, 8, 0)
    assert len(a) == 8 and a == sorted(a) and a == subsample(pool, 8, 0)   # deterministic, order-preserving
    assert subsample(pool, 8, 1) != a                                       # a different seed picks a different subset
    assert subsample(pool, 50, 0) == pool                                   # k > len clamps to the whole pool


def test_summ_k6_rate_and_mean_dtz() -> None:
    rows = [{"ridge": {"k6": True, "dtz_mm": 5.0}}, {"ridge": {"k6": False, "dtz_mm": 15.0}}]
    s = _summ(rows, "ridge")
    assert s["k6_rate"] == 0.5 and s["dtz_mm"] == 10.0


def _pt(k: int, seed: int, ridge_k6: float, ridge_dtz: float) -> dict:
    base = {"k6_rate": 0.0, "dtz_mm": 20.0}
    return {"k": k, "seed": seed, "held": {p: dict(base) for p in ("mean_theta", "nearest_schedule", "mlp_bc")}
            | {"ridge": {"k6_rate": ridge_k6, "dtz_mm": ridge_dtz}}}


def test_aggregate_and_density_verdict_rising() -> None:
    # ridge held-out K6 climbs 0.0 (k=10) -> 0.4 (k=38): density-limited
    pts = [_pt(10, s, 0.0, 30.0) for s in (0, 1)] + [_pt(38, s, 0.4, 12.0) for s in (0, 1)]
    agg = aggregate(pts)
    assert agg["verdict"] == "R11_5R_DENSITY_LIMITED_DENSIFY_INDICATED"
    assert agg["curve"]["38"]["ridge"]["k6_rate"] == 0.4 and agg["n_seeds"] == 2


def test_density_verdict_flat_is_descriptor_limited() -> None:
    # ridge/mlp held-out unchanged across k -> descriptor-limited
    agg = {10: {p: {"k6_rate": 0.17, "dtz_mm": 20.0} for p in ("ridge", "mlp_bc")},
           38: {p: {"k6_rate": 0.17, "dtz_mm": 19.5} for p in ("ridge", "mlp_bc")}}
    assert _curve_verdict(agg) == "R11_5R_DESCRIPTOR_LIMITED_DENSIFY_UNLIKELY_TO_HELP"
