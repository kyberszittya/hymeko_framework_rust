"""Tests for the F11/F12 campaign's pure analysis + classification (subprocess launch is exercised by the real run)."""
from __future__ import annotations

import json
from pathlib import Path

from hymeko_rl.experiments.coin_f11_f12_campaign import _classify, _pair_deltas, analyze


def _bm(s2_cov: int, s1_cov: int, *, s2_clr: float = -9.9, bilat: float = 0.5, clean: float = 0.5,
        attr: float = 0.5, strong: bool = False, r64102: bool = False) -> dict:
    """A best_metrics dict shaped like the eval_fn output (STAGE1/STAGE2 stage metrics + strict flags)."""
    return dict(stage1=dict(coverage=s1_cov, loose=s1_cov, max_certified_clearance=-9.9,
                            P_bilat=bilat, P_clean=clean, P_attr=attr),
                stage2=dict(coverage=s2_cov, loose=max(s2_cov, 0), max_certified_clearance=s2_clr,
                            P_bilat=bilat, P_clean=clean, P_attr=attr),
                strong_strict=strong, s64102_strict=r64102)


def test_pair_deltas_is_f12_minus_f11() -> None:
    c = _bm(s2_cov=0, s1_cov=4, clean=0.30)
    t = _bm(s2_cov=2, s1_cov=5, s2_clr=0.045, clean=0.55, strong=True)
    d = _pair_deltas(c, t)
    assert d["s2_cov"] == 2 and d["s1_ret"] == 1 and d["strong_ret"] == 1
    assert d["s1_clean"] == 0.25 and d["s2_maxclr"] == round(0.045 - 0.0, 4)   # F11 clr -9.9 → clamped to 0
    assert d["t_s2cov"] == 2 and d["c_s2cov"] == 0


def _agg(**overrides) -> dict:
    """A neutral aggregate (all deltas zero); override specific endpoints with (median, [lo, hi])."""
    base = {k: dict(median=0.0, boot95=[0.0, 0.0]) for k in
            ("s2_cov", "s1_ret", "strong_ret", "s1_bilat", "s1_clean")}
    for k, (med, ci) in overrides.items():
        base[k] = dict(median=med, boot95=ci)
    return base


def test_classify_critic_positive() -> None:
    per = [dict(t_s2cov=2), dict(t_s2cov=1)]                                     # F12 certifies STAGE2
    agg = _agg(s2_cov=(1.5, [0.5, 2.5]))                                         # improvement CI above zero
    assert _classify(agg, per) == "CRITIC_POSITIVE"


def test_classify_critic_negative_on_retention_loss() -> None:
    per = [dict(t_s2cov=0), dict(t_s2cov=0)]
    agg = _agg(s1_ret=(-2.0, [-4.0, -0.5]))                                      # retention CI below zero, no S2 gain
    assert _classify(agg, per) == "CRITIC_NEGATIVE"


def test_classify_mechanism_positive() -> None:
    per = [dict(t_s2cov=0), dict(t_s2cov=0)]
    agg = _agg(s1_clean=(0.2, [0.05, 0.35]))                                     # clean-mechanism CI above zero
    assert _classify(agg, per) == "CRITIC_MECHANISM_POSITIVE"


def test_classify_no_effect() -> None:
    assert _classify(_agg(), [dict(t_s2cov=0)]) == "NO_EFFECT"                   # everything spans zero


def test_classify_negative_precedes_mechanism() -> None:
    """A mechanism gain that destroys retention is NEGATIVE, not MECHANISM_POSITIVE (the guard wins)."""
    per = [dict(t_s2cov=0)]
    agg = _agg(s1_clean=(0.2, [0.05, 0.35]), strong_ret=(-1.0, [-1.0, -0.2]))
    assert _classify(agg, per) == "CRITIC_NEGATIVE"


def test_analyze_roundtrip_writes_verdict(tmp_path: Path) -> None:
    for s in range(4):
        for r in range(2):
            (tmp_path / f"F11_s{s}r{r}").mkdir(parents=True)
            (tmp_path / f"F12_s{s}r{r}").mkdir(parents=True)
            (tmp_path / f"F11_s{s}r{r}" / "run.json").write_text(json.dumps({"best_metrics": _bm(0, 4)}))
            (tmp_path / f"F12_s{s}r{r}" / "run.json").write_text(json.dumps({"best_metrics": _bm(0, 4)}))
    out = analyze(tmp_path)
    assert out["n_pairs"] == 8 and out["classification"] == "NO_EFFECT"          # identical → no effect
    assert (tmp_path / "f11_f12_comparison.json").exists()


def test_analyze_blocked_when_no_pairs(tmp_path: Path) -> None:
    out = analyze(tmp_path)
    assert out["classification"] == "BLOCKED" and out["n_pairs"] == 0
