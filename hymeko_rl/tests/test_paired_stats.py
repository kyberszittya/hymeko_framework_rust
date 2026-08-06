"""Tests for the canonical paired-bootstrap primitive (:mod:`hymeko_rl.eval.paired_stats`)."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.paired_stats import boot_ci, paired_stats


def test_boot_ci_deterministic_in_seed() -> None:
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert boot_ci(x, seed=7) == boot_ci(x, seed=7)                  # same seed → byte-identical CI (resume/provenance)
    lo, hi = boot_ci(x, seed=7)
    assert lo < np.mean(x) < hi                                      # the mean lies strictly inside its own CI


def test_boot_ci_constant_sample_is_degenerate() -> None:
    lo, hi = boot_ci([2.5, 2.5, 2.5, 2.5], seed=0)
    assert lo == hi == pytest.approx(2.5)                            # zero-variance sample → point CI


def test_boot_ci_positive_sample_excludes_zero() -> None:
    lo, hi = boot_ci([3.0, 4.0, 5.0, 3.5, 4.5], seed=1)             # clearly-positive → CI above 0
    assert lo > 0.0 and hi > lo


def test_boot_ci_rejects_empty_and_bad_percentiles() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        boot_ci([], seed=0)
    with pytest.raises(ValueError, match="lo < hi"):
        boot_ci([1.0, 2.0], seed=0, lo=97.5, hi=2.5)


def test_paired_stats_sign_split_and_shape() -> None:
    st = paired_stats([+2.0, 0.0, -1.0, +3.0, 0.0], seed=42)
    assert st["n"] == 5 and st["pos"] == 2 and st["zero"] == 2 and st["neg"] == 1
    assert st["pos"] + st["zero"] + st["neg"] == st["n"]            # sign counts partition n
    assert st["median"] == pytest.approx(0.0)
    assert st["mean"] == pytest.approx(0.8)                          # (2+0-1+3+0)/5
    assert isinstance(st["boot95"], list) and len(st["boot95"]) == 2
    assert st["boot95"][0] <= st["boot95"][1]                        # well-ordered CI


def test_paired_stats_all_zero_is_flat_ci() -> None:
    st = paired_stats([0.0] * 8, seed=3)
    assert st["mean"] == 0.0 and st["boot95"] == [0.0, 0.0]         # matched no-effect → exactly [0,0]
    assert st["pos"] == 0 and st["neg"] == 0 and st["zero"] == 8
