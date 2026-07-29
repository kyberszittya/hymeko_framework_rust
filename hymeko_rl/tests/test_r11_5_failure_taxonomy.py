"""Tests for the R11.5 complete failure taxonomy — classifier paths + full-24 coverage + bounded-pilot invariants."""
from pathlib import Path

from hymeko_rl.experiments.r11_5_failure_taxonomy import (
    FailureCategory,
    build_taxonomy,
    classify,
    select_pilot,
)
from hymeko_rl.experiments.r11_5_full_coverage import R11_4A_BANK

COVERAGE = Path("reports/2026-07-30-r11-5-coverage/coverage.jsonl")


def test_classify_paths() -> None:
    # no certified grasp -> capture-support, regardless of (absent) delivery metrics
    assert classify(False, 0.0, 0.0, 0.0, 0.0) is FailureCategory.CAPTURE_SUPPORT_FAILURE
    # certified but never moved -> handoff-to-kinetic failure
    assert classify(True, 0.0, 90.0, 0.0, 1.0) is FailureCategory.HANDOFF_TO_KINETIC_FAILURE
    # driven net AWAY from target -> directional bias (the negative-x tail: forward progress but gap widened)
    assert classify(True, -0.71, 126.8, 0.0, 21.4) is FailureCategory.DIRECTIONAL_BIAS
    # moved toward target, closed part of the gap, stalled short -> insufficient progress
    assert classify(True, 0.70, 22.4, 0.0, 61.4) is FailureCategory.INSUFFICIENT_PROGRESS
    # reached the zone but too fast to settle
    assert classify(True, 0.9, 15.0, 0.09, 80.0) is FailureCategory.ZONE_NEAR_SPEED_FAILURE
    # entered the zone, slow, but no held dwell
    assert classify(True, 0.9, 15.0, 0.0, 80.0) is FailureCategory.ZONE_ENTRY_WITHOUT_DWELL


def test_classify_gap_sign_is_the_directional_discriminator() -> None:
    """Forward coin_progress with a WIDENING gap is directional bias, not progress (would have passed pre-taxonomy)."""
    toward = classify(True, 0.30, 60.0, 0.0, 40.0)
    away = classify(True, -0.30, 160.0, 0.0, 40.0)          # same forward progress, gap widened
    assert toward is FailureCategory.INSUFFICIENT_PROGRESS
    assert away is FailureCategory.DIRECTIONAL_BIAS


def test_build_taxonomy_is_complete_and_partitions_24() -> None:
    if not (Path(R11_4A_BANK).exists() and COVERAGE.exists()):
        return  # artifacts present only in the coin worktree
    recs = build_taxonomy(COVERAGE, Path(R11_4A_BANK))
    ids = [r.scenario_id for r in recs]
    assert len(recs) == 24 and len(set(ids)) == 24                       # every uncovered scenario exactly once
    counts = {c: sum(r.category is c for r in recs) for c in FailureCategory}
    assert counts[FailureCategory.CAPTURE_SUPPORT_FAILURE] == 10
    assert counts[FailureCategory.DIRECTIONAL_BIAS] == 4
    assert counts[FailureCategory.INSUFFICIENT_PROGRESS] == 10
    assert sum(counts.values()) == 24                                    # partition — no scenario double-counted


def test_select_pilot_covers_groups_and_heldout_splits() -> None:
    if not (Path(R11_4A_BANK).exists() and COVERAGE.exists()):
        return
    pilot = select_pilot(build_taxonomy(COVERAGE, Path(R11_4A_BANK)))
    assert len(pilot) == 12
    cats = [r.category for r in pilot]
    assert cats.count(FailureCategory.CAPTURE_SUPPORT_FAILURE) == 4
    assert cats.count(FailureCategory.DIRECTIONAL_BIAS) == 4
    assert cats.count(FailureCategory.INSUFFICIENT_PROGRESS) == 4
    splits = {r.split for r in pilot}
    assert "dev" in splits and "test" in splits                         # gate needs held-out representation
    # capture pick tests both the systematic +/+ gap and the stochastic-regen cases
    caps = [r for r in pilot if r.category is FailureCategory.CAPTURE_SUPPORT_FAILURE]
    assert {r.subtype for r in caps} == {"systematic_pp", "stochastic_regen"}
    # the transport fix (progress group) must be validated off-train, not train-only
    prog = [r for r in pilot if r.category is FailureCategory.INSUFFICIENT_PROGRESS]
    assert len({r.split for r in prog}) >= 2 and any(r.split in ("dev", "test") for r in prog)
