"""Tests for the F11/F21 campaign's ACTOR classification + analysis (subprocess launch exercised by the real run)."""
from __future__ import annotations

import json
from pathlib import Path

from hymeko_rl.experiments.coin_f11_f21_campaign import _classify_actor, analyze


def _agg(**overrides) -> dict:
    base = {k: dict(median=0.0, boot95=[0.0, 0.0]) for k in
            ("s2_cov", "s2_loose", "s1_ret", "strong_ret", "s1_bilat", "s1_clean")}
    for k, (med, ci) in overrides.items():
        base[k] = dict(median=med, boot95=ci)
    return base


def test_classify_actor_positive() -> None:
    per = [dict(t_s2cov=2), dict(t_s2cov=1)]
    assert _classify_actor(_agg(s2_cov=(1.5, [0.5, 2.5])), per) == "ACTOR_POSITIVE"


def test_classify_actor_negative_on_transport_loss() -> None:
    per = [dict(t_s2cov=0)]
    assert _classify_actor(_agg(s2_loose=(-2.0, [-3.0, -0.5])), per) == "ACTOR_NEGATIVE"


def test_classify_actor_contact_positive() -> None:
    per = [dict(t_s2cov=0)]
    assert _classify_actor(_agg(s1_bilat=(0.2, [0.05, 0.35])), per) == "ACTOR_CONTACT_POSITIVE"


def test_classify_actor_no_effect() -> None:
    assert _classify_actor(_agg(), [dict(t_s2cov=0)]) == "NO_EFFECT"


def test_classify_negative_precedes_contact_positive() -> None:
    per = [dict(t_s2cov=0)]
    agg = _agg(s1_bilat=(0.2, [0.05, 0.35]), s1_ret=(-2.0, [-3.0, -0.4]))
    assert _classify_actor(agg, per) == "ACTOR_NEGATIVE"           # gain that destroys retention loses


def _bm(s2_cov: int, s1_cov: int, bilat: float = 0.5) -> dict:
    return dict(stage1=dict(coverage=s1_cov, loose=s1_cov, max_certified_clearance=-9.9,
                            P_bilat=bilat, P_clean=0.5, P_attr=0.5),
                stage2=dict(coverage=s2_cov, loose=max(s2_cov, 0), max_certified_clearance=-9.9,
                            P_bilat=bilat, P_clean=0.5, P_attr=0.5),
                strong_strict=False, s64102_strict=False)


def test_analyze_roundtrip(tmp_path: Path) -> None:
    for s in range(4):
        for r in range(2):
            for cell in ("F11", "F21"):
                d = tmp_path / f"{cell}_s{s}r{r}"
                d.mkdir(parents=True)
                hist = [{"bank_diag": {"mode_occupancy_transport": 0.1}}] if cell == "F21" else []
                (d / "run.json").write_text(json.dumps({"best_metrics": _bm(0, 4), "eval_history": hist}))
    out = analyze(tmp_path)
    assert out["n_pairs"] == 8 and out["classification"] == "NO_EFFECT"
    assert out["mean_transport_occupancy"] == 0.1
    assert (tmp_path / "f11_f21_comparison.json").exists()


def test_analyze_blocked_when_no_pairs(tmp_path: Path) -> None:
    assert analyze(tmp_path)["classification"] == "BLOCKED"
