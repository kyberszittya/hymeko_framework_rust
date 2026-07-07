"""Tests for the committed pick-place clearance diagnostics (`hymeko_rl.eval.pick_clearance`).

Pure aggregation/gate logic is unit-tested on synthetic episodes; one bounded integration test rolls the v1
scripted expert (1 episode) and asserts it reproduces the known DIRTY signature (early table strike + negative
clearance) — the same signature the local smoke and the 2026-07-06 forensics recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hymeko_rl.eval.pick_clearance import (
    EpisodeClearance,
    aggregate,
    gate_verdict,
    run_clearance,
    write_outputs,
)


def _ep(seed: int, *, lift: int = 1, place: int = 1, forbidden: bool = True, clr: float = -0.02,
        transit: float = 0.45, over: "int | None" = 200, fingtab: "int | None" = 50) -> EpisodeClearance:
    return EpisodeClearance(
        seed=seed, lift=lift, place=place, obj_to_target=0.05, length=500,
        first_finger_table_step=fingtab, first_gripper_table_step=None, first_over_object_step=over,
        min_transit_clearance=clr, forbidden_pre_object=forbidden, transit_finger_contact_frac=transit,
        phase=None, diverged=False)


def test_aggregate_dirty_episodes() -> None:
    agg = aggregate([_ep(i) for i in range(4)])
    assert agg["episodes"] == 4
    assert agg["lift_rate"] == 1.0
    assert agg["forbidden_pre_object_rate"] == 1.0
    assert agg["min_clearance_min"] is not None and agg["min_clearance_min"] < 0.0
    assert agg["transit_finger_contact_rate"] > 0.1


def test_gate_fails_on_dirty() -> None:
    verdict = gate_verdict(aggregate([_ep(i) for i in range(4)]))
    assert verdict["pass"] is False
    assert verdict["crit1_no_early_strike"] is False        # forbidden pre-object contact
    assert verdict["crit3_positive_min_clearance"] is False  # negative clearance


def test_gate_passes_on_clean() -> None:
    clean = [_ep(i, forbidden=False, clr=0.05, transit=0.0, fingtab=None) for i in range(4)]
    verdict = gate_verdict(aggregate(clean))
    assert verdict["pass"] is True
    assert verdict["crit1_no_early_strike"] and verdict["crit2_transit_contact_near_zero"]
    assert verdict["crit3_positive_min_clearance"]
    assert verdict["pref4_lift_ge_0_90"] and verdict["pref5_place_ge_0_80"]


def test_gate_crit2_transit_contact_tolerance() -> None:
    # a tiny transit-contact rate (<= near-zero tol) still passes crit2; above it fails.
    near = gate_verdict(aggregate([_ep(i, forbidden=False, clr=0.05, transit=0.01, fingtab=None) for i in range(2)]))
    over = gate_verdict(aggregate([_ep(i, forbidden=False, clr=0.05, transit=0.10, fingtab=None) for i in range(2)]))
    assert near["crit2_transit_contact_near_zero"] is True
    assert over["crit2_transit_contact_near_zero"] is False


def test_forbidden_pre_object_requires_strike_before_over() -> None:
    # a clean episode whose only table contact is AFTER it is over the object is NOT forbidden.
    late = _ep(0, forbidden=False, clr=0.05, transit=0.0, over=100, fingtab=None)
    agg = aggregate([late])
    assert agg["forbidden_pre_object_rate"] == 0.0


def test_run_clearance_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        run_clearance(3, 1, 0)
    with pytest.raises(ValueError):
        run_clearance(1, 0, 0)


def test_write_outputs_creates_json_and_csv(tmp_path: Path) -> None:
    out = tmp_path / "clr"
    payload = write_outputs(out, 2, 4, 0, [_ep(i, forbidden=False, clr=0.05, transit=0.0, fingtab=None)
                                           for i in range(4)], plot=False)
    assert out.with_suffix(".json").exists()
    assert out.with_suffix(".csv").exists()
    parsed = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert parsed["version"] == 2 and len(parsed["per_episode"]) == 4
    assert payload["gate"]["pass"] is True


@pytest.mark.slow
def test_run_clearance_v1_reproduces_dirty_signature() -> None:
    """Integration: the v1 scripted expert strikes the table BEFORE it is over the object, with negative
    clearance — the frozen `v1_dirty` signature. Bounded to 1 episode."""
    eps = run_clearance(version=1, episodes=1, seed0=50000)
    assert len(eps) == 1
    e = eps[0]
    assert e.lift == 1                                          # v1 succeeds despite being dirty
    assert e.forbidden_pre_object is True                      # strike before over-object
    assert e.min_transit_clearance < 0.0                       # penetrates the tabletop plane
    assert e.first_finger_table_step is not None
    assert e.first_over_object_step is not None
    assert e.first_finger_table_step < e.first_over_object_step
