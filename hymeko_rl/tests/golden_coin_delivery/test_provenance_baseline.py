"""RECOVERY-BASELINE-0 §8 — baseline provenance gates. These enforce scientific provenance (hashes, explicit StateId,
restore determinism, K1 model equality, tracked source) WITHOUT redesigning behaviour."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.coin_delivery.provenance.state_identity import CorpusId, StateId, legacy_seed_to_index, snapshot_hash

_ART = Path("artifacts/coin_recovery_baseline")


def _bank():
    from hymeko_rl.experiments.pedc_selection import _load_pkl_bank
    return _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False)


def test_corpus_hash_matches_manifest() -> None:
    man = json.loads((_ART / "corpus_manifest.json").read_text())
    live = hashlib.sha256(open(man["path"], "rb").read()).hexdigest()
    assert live == man["corpus_id"]["sha256"]                    # §8.1 corpus hash == manifest


def test_evaluation_reports_duplicate_states() -> None:
    mp = json.loads((_ART / "state_mapping.json").read_text())   # §8.2 duplicates surfaced, not hidden
    assert mp["episode_count"] == 90 and mp["unique_state_count"] == 82 and mp["duplicate_state_count"] == 8
    assert mp["unique_state_count"] < mp["episode_count"]        # "n=90 independent" is false


def test_stateid_requires_hash_not_bare_seed() -> None:
    # §8.3 a seed integer cannot masquerade as a StateId — StateId needs a corpus + snapshot hash
    with pytest.raises(TypeError):
        StateId(64000)  # type: ignore[call-arg]
    sid = StateId(CorpusId("c1_heldseed_bank.pkl", "deadbeef"), 0, "abc123")
    assert isinstance(sid.snapshot_sha256, str)


def test_golden_stateids_restore() -> None:
    from hymeko_rl.experiments.pedc_selection import _env
    from hymeko_rl.train.coin_transport import restore_planar
    bank = _bank()
    env = _env()
    for idx in json.loads((_ART / "golden_results.json").read_text())["state_indices"]:
        restore_planar(env._env, bank[idx]["snap"])              # §8.4 every golden StateId restores
        assert env._env._planar_metrics.disk_to_zone >= 0.0


def test_restore_is_history_independent() -> None:
    from hymeko_rl.experiments.pedc_selection import _env
    from hymeko_rl.train.coin_transport import restore_planar
    bank = _bank()
    env = _env()
    snap = bank[0]["snap"]
    restore_planar(env._env, snap)
    q1 = env._env.data.qpos.copy()
    for _ in range(7):
        env._env.step(np.zeros(env._env.model.nu, np.float32))   # perturb
    restore_planar(env._env, snap)
    q2 = env._env.data.qpos.copy()
    assert np.array_equal(q1, q2)                                # §8.5 restore history-independent


def test_k1_variant_models_identical() -> None:
    # §8.6 K1 neutral/aware/scramble share ONE compiled model
    fp = json.loads((_ART / "model_fingerprints.json").read_text())
    assert fp["K1_neutral_aware_scramble_identical"] is True
    assert fp["P0_neutral"]["hash"] == fp["P1_geometry_aligned"]["hash"] == fp["P3_orientation_scramble"]["hash"]


def test_coin_production_source_tracked() -> None:
    # §8.7 the coin production source is committed (not untracked) on the recovery branch
    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n"))
    for f in ("hymeko_rl/train/coin_delivery_actor.py", "hymeko_rl/train/pad_aware_control.py",
              "hymeko_rl/env/contact_formation_env.py"):
        assert f in tracked, f"{f} is untracked"


def test_legacy_seed_is_index_selector_not_identity() -> None:
    bank = _bank()
    # the same seed deterministically selects the same index; different seeds may collide on one state
    assert legacy_seed_to_index(64000, len(bank)) == legacy_seed_to_index(64000, len(bank))
    h = snapshot_hash(bank[0]["snap"])
    assert len(h) == 64 and h == snapshot_hash(bank[0]["snap"])  # content hash is stable
