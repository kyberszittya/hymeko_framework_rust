"""Tests for the retracted-seed kinematic feasibility probe (`hymeko_rl.eval.pick_retract_probe`)."""
from __future__ import annotations

from hymeko_rl.eval.pick_retract_probe import probe_episode


def test_probe_finds_clean_hover_branch_and_is_feasible() -> None:
    """The multi-start probe finds a table-clear, collision-free config AT the object hover and judges
    HOME_RETRACT_OR_PRESHAPE feasible on a known seed — the object hover IS reachable clean (the v2 stall was the
    PATH from the over-extended home, not the pose). Deterministic (fixed reset seed + fixed multi-start RNG)."""
    ep = probe_episode(50000)
    assert ep["home_retract_feasible"] is True
    assert ep["best_seed"] is not None
    cands = {c["name"]: c for c in ep["candidates"]}
    assert set(cands) == {"arm_home", "cf_hover", "cf_mid_retract", "manual_elbow_up"}
    cf = cands["cf_hover"]
    assert cf["seed_collision_free"] is True
    assert cf["seed_clearance"] > 0.0              # table-clear at the object hover
    assert cf["seed_to_hover_reached"] is True
    assert cf["feasible"] is True


def test_probe_rejects_table_penetrating_candidate() -> None:
    """A candidate whose finger geometry penetrates the table (the manual guess) is judged infeasible."""
    manual = {c["name"]: c for c in probe_episode(50000)["candidates"]}["manual_elbow_up"]
    assert manual["seed_clearance"] < 0.0
    assert manual["feasible"] is False
