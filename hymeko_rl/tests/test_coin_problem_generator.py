"""Validity + provenance tests for the HyMeKo Coin-Delivery structured problem generator (§5). Physical validity only
(never policy-success); one named relation per variant; frozen TRAIN/HELD disjoint + balanced; canonical mirror."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar
from hymeko_rl.experiments.coin_problem_generator import (
    ATTRIBUTION_BOUNDARY,
    CERTIFIED_NEIGHBORHOOD,
    LEFT_RIGHT_SYMMETRY,
    _mirror,
    _perturb,
    _parent_snapshot,
    validate,
)

_GEN = Path("experiments/2026_07_20_coin_problem_generator")


@pytest.fixture(scope="module")
def env():
    from hymeko_rl.experiments.coin_two_arm_sac import direct_env
    return direct_env()


def _load(name):
    import pickle
    with open(_GEN / f"{name}_configs.pkl", "rb") as f:
        return pickle.load(f)


def test_frozen_corpora_balanced_and_disjoint() -> None:
    man = json.loads((_GEN / "generator_manifest.json").read_text())
    assert man["train_by_family"] == {CERTIFIED_NEIGHBORHOOD: 32, ATTRIBUTION_BOUNDARY: 32, LEFT_RIGHT_SYMMETRY: 32}
    assert man["held_by_family"] == {CERTIFIED_NEIGHBORHOOD: 16, ATTRIBUTION_BOUNDARY: 16, LEFT_RIGHT_SYMMETRY: 16}
    assert man["n_train"] == 96 and man["n_held"] == 48
    assert set(man["train_hashes"]).isdisjoint(man["held_hashes"])              # generated eval never seen in training


def test_no_generated_config_starts_in_zone(env) -> None:
    for c in _load("train") + _load("held"):
        assert c.not_initially_successful                                       # §5: rejected if starts in zone
        restore_planar(env.inner, c.snapshot)
        assert not env.inner.planar_metrics.in_zone


def test_restore_is_deterministic(env) -> None:
    for c in (_load("train")[:10] + _load("held")[:10]):
        restore_planar(env.inner, c.snapshot)
        h1 = snapshot_hash(snapshot_planar(env.inner))
        restore_planar(env.inner, c.snapshot)
        h2 = snapshot_hash(snapshot_planar(env.inner))
        assert h1 == h2 == c.state_hash


def test_families_present_and_labeled(env) -> None:
    fams = {c.family for c in _load("train")}
    assert fams == {CERTIFIED_NEIGHBORHOOD, ATTRIBUTION_BOUNDARY, LEFT_RIGHT_SYMMETRY}
    for c in _load("train"):
        assert c.parent_seed in (64102, 64201, 64111) and c.changed_relation and c.left_reachable and c.right_reachable


def test_single_relation_changes_only_its_qpos(env) -> None:
    base = _parent_snapshot(env, 64102)
    lat = _perturb(base, "coin_lateral_offset", 0.03)
    assert not np.allclose(lat.qpos[4], base.qpos[4]) and np.allclose(lat.qpos[:4], base.qpos[:4])  # only coin_x
    gap = _perturb(base, "lr_contact_gap", 0.05)
    assert np.allclose(gap.qpos[4:], base.qpos[4:]) and not np.allclose(gap.qpos[1], base.qpos[1])  # only arm joints


def test_mirror_is_canonical_lr_reflection(env) -> None:
    base = _parent_snapshot(env, 64201)
    mir = _mirror(base)
    zx = base.zone[0]
    assert np.isclose(mir.qpos[4], 2 * zx - base.qpos[4])                       # coin x reflected about the zone
    assert np.isclose(mir.qpos[0], -base.qpos[2]) and np.isclose(mir.qpos[2], -base.qpos[0])  # L/R shoulder swapped


def test_validate_rejects_policy_independent_only(env) -> None:
    # a config whose coin sits on the zone must be rejected on PHYSICAL grounds (starts_in_zone), never policy outcome
    base = _parent_snapshot(env, 64102)
    on_zone = base.__class__(qpos=base.qpos.copy(), qvel=base.qvel, qacc_warmstart=base.qacc_warmstart,
                             time=base.time, zone=base.zone, latches=base.latches, disk_to_zone=0.0)
    on_zone.qpos[4], on_zone.qpos[5] = base.zone
    ok, reason, _ = validate(env, on_zone)
    assert not ok and reason == "starts_in_zone"
