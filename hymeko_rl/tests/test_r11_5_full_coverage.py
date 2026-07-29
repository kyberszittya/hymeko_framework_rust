"""Tests for the R11.5 full-coverage gate logic (pure) + bank enumeration."""
from pathlib import Path

from hymeko_rl.experiments.r11_5_full_coverage import (
    R11_4A_BANK,
    _cclass,
    _count_by_class,
    _coverage_passed,
    bank_facts,
)


def test_cclass_and_count() -> None:
    assert _cclass("bank_c2_+0.015_+0.015") == "c2"
    assert _cclass("bank_c0_0") == "c0"
    assert _cclass("bank_c3_r9_a+45") == "c3"
    assert _count_by_class(["bank_c0_0", "bank_c1_x", "bank_c1_y", "bank_c3_z"]) == {"c0": 1, "c1": 2, "c3": 1}


def test_coverage_passed_predicate() -> None:
    cov_ok = {"c0": 1.0, "c1": 0.6, "c2": 0.5, "c3": 0.7}
    sp = {"train": 30, "dev": 2, "test": 1}
    assert _coverage_passed(46, cov_ok, sp, 0, True, True) is True
    assert _coverage_passed(44, cov_ok, sp, 0, True, True) is False              # <45 overall
    assert _coverage_passed(46, {**cov_ok, "c2": 0.49}, sp, 0, True, True) is False   # a class <50%
    assert _coverage_passed(46, cov_ok, {"train": 30, "dev": 0, "test": 1}, 0, True, True) is False  # no dev
    assert _coverage_passed(46, cov_ok, {"train": 30, "dev": 2, "test": 0}, 0, True, True) is False  # no test
    assert _coverage_passed(46, cov_ok, sp, 1, True, True) is False              # a nudge-only K6
    assert _coverage_passed(46, cov_ok, sp, 0, False, True) is False             # safety regression
    assert _coverage_passed(46, cov_ok, sp, 0, True, False) is False             # incomplete energy


def test_bank_facts_enumerates_the_51_failures_and_7_r2() -> None:
    if not Path(R11_4A_BANK).exists():
        return  # bank present only in the coin worktree
    fails, r2_by_class, r2_total = bank_facts(R11_4A_BANK)
    assert len(fails) == 51 and r2_total == 7
    assert all(sp in ("train", "dev", "test") for _sid, sp in fails)
