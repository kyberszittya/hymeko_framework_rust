"""Pytest entry: HSiKAN pipeline integrity tests.

Run:
    pytest signedkan_wip/tests/test_pipeline_integrity.py -v

The env-var audit catches partial-wiring bugs *without* running the
full training pipeline, so it's fast (~5 seconds total). The
invariant smoke test runs a 1-epoch training cell on bitcoin_alpha
(~30 seconds) to catch numerical / NaN / shape regressions before
they burn through a long sweep.
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from signedkan_wip.src.test_harness.env_var_audit import cases as audit_cases
from signedkan_wip.src.test_harness.env_var_audit import _env_block
from signedkan_wip.src.test_harness.invariant_check import smoke_test


@pytest.mark.parametrize("case", audit_cases(), ids=lambda c: c.name)
def test_env_var_takes_effect(case):
    """Each HSIKAN_* env var must actually change model state."""
    with _env_block(case.env):
        model = case.setup()
        ok, detail = case.assertion(model)
    assert ok, f"{case.name}: {detail}"


@pytest.mark.slow
def test_bitcoin_alpha_invariants():
    """One-epoch smoke test on bitcoin_alpha to catch numerical
    regressions (NaN AUC, etc.). Takes ~30 seconds."""
    assert smoke_test("bitcoin_alpha", epochs=1)
