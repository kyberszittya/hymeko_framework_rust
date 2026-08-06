"""R11.7A U4 — O0 parity regression guard (frozen provenance hash).

The full production smoke lives in ``hymeko_rl.experiments.r11_7a_o0_parity`` (model + collision + rollout
bit-exactness). This test locks the single strongest invariant into the automated suite: the R11.6C
acquisition, now reading the manipuland from the HyMeKo scene (``galambos_env.hymeko`` → ``EnvSpec.object``)
through the unpinned ``_make_env``, reproduces the frozen s1-cradle physical-state hash bit-for-bit. That
hash is the committed reference recorded across the H2 bv-identification / teacher-bank / cradle-scout
artifacts — a change here means the unification perturbed the certified pipeline.
"""
from __future__ import annotations

import pytest

# The frozen s1-cradle (seed 14250) physical-state hash — committed reference:
# reports/2026-07-26-h2-bv-identification/bv_identification.json, teacher_bank.json, cradle_scout.json,
# reports/2026-07-26-h2-control-to-contact-velocity-identification.md ("s1 16778d7df544b9e8").
_FROZEN_S1_HASH = "16778d7df544b9e8"
_S1_SEED = 14250


@pytest.mark.slow
def test_unpinned_make_env_reproduces_frozen_s1_cradle_hash() -> None:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import FROZEN_SEEDS, acquire_snapshot, load_harness

    assert _S1_SEED in FROZEN_SEEDS
    snap, meta = acquire_snapshot(load_harness(), _S1_SEED)
    assert snap is not None, f"frozen cradle no longer certifies (invalid={meta.get('invalid_snapshot')})"
    assert bool(meta.get("certified")) is True
    assert snap.post_release_hash == _FROZEN_S1_HASH, (
        "O0 parity regression: the HyMeKo-scene-sourced path changed the frozen s1-cradle physical-state hash "
        f"({snap.post_release_hash} != {_FROZEN_S1_HASH}) — the unification perturbed the certified pipeline")
