"""Tests for the R11.5R retrieval characterization experiment: survival join, aggregation, deployment verdict."""
import json
from pathlib import Path

from hymeko_rl.experiments.r11_5r_retrieval_policy import CELLS, aggregate, survival_map


def test_survival_map_wide_uses_t1_else_t0(tmp_path: Path) -> None:
    rows = [{"scenario_id": "w", "status": "WIDE_RECERTIFIED", "t0": {"survival1": 0.3}, "t1": {"survival1": 0.9}},
            {"scenario_id": "n", "status": "NARROW_ONLY", "t0": {"survival1": 0.4}, "t1": {"survival1": 0.4}}]
    p = tmp_path / "merged.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    m = survival_map(p)
    assert m["w"] == 0.9 and m["n"] == 0.4                       # WIDE takes robust t1; fallback takes nominal t0


def _row(sid: str, split: str, **cells: bool) -> dict:
    r = {"scenario_id": sid, "split": split}
    for name, _ in CELLS:
        hit = cells.get(name, False)
        r[name] = {"k6": hit, "safe": True, "dtz_mm": 5.0 if hit else 40.0}
    return r


def test_aggregate_rates_best_cell_and_certificate() -> None:
    rows = [_row("d0", "dev", std_nearest=True, std_widest3=True),
            _row("d1", "dev", std_widest3=True),
            _row("t0", "test", std_nearest=False),
            _row("tr0", "train", std_nearest=True)]
    a = aggregate(rows)
    assert a["cells"]["std_nearest"]["held"] == round(1 / 3, 3)          # 1 of 3 held-out
    assert a["cells"]["std_widest3"]["held"] == round(2 / 3, 3)          # widest3 beats nearest here
    assert a["best_cell"] == "std_widest3" and a["deploy_beaten_by"] == "std_widest3"
    assert a["is_deployable"] and a["certificate"]["oracle_free"]
    assert a["verdict"] == "R11_5R_RETRIEVAL_DEPLOYABLE_MEETS_050"       # best held 0.667 >= 0.50


def test_verdict_below_050_when_best_is_low() -> None:
    rows = [_row(f"d{i}", "dev", std_nearest=(i == 0)) for i in range(7)] \
        + [_row(f"t{i}", "test") for i in range(5)]                       # best held = 1/12 < 0.5
    a = aggregate(rows)
    assert a["verdict"] == "R11_5R_RETRIEVAL_DEPLOYABLE_BELOW_050" and a["best_cell"] == "std_nearest"
