"""COIN-DELIVERY-RECOVERY-BASELINE-0 §3 — explicit, hash-anchored state identity for the delivery corpus.

ISOLATED provenance types only — this module does NOT change production rollout behaviour (the legacy path still
selects a bank item via ``np.random.default_rng(seed).integers(len(bank))``). It makes explicit that an evaluation
*seed* is NOT a semantic state identifier: it is an RNG that selects a bank index. A ``StateId`` is the real identity
(corpus + snapshot index + snapshot hash). Future rollout code should consume ``StateId``; this stage only defines it
and audits the historical seed→state mapping.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorpusId:
    """A content-addressed corpus reference. # Invariants: sha256 identifies the exact bank bytes."""
    name: str
    sha256: str


@dataclass(frozen=True)
class StateId:
    """The explicit identity of one evaluation state. A random seed may drive stochasticity but must NEVER be used as
    a StateId. # Invariants: snapshot_sha256 pins the exact restored qpos/qvel/zone."""
    corpus: CorpusId
    snapshot_index: int
    snapshot_sha256: str


def snapshot_hash(snap) -> str:
    """Content hash of a bank snapshot (qpos+qvel+zone) — the anchor for a StateId. # Preconditions: snap has qpos/qvel."""
    h = hashlib.sha256()
    for arr in (np.asarray(snap.qpos), np.asarray(snap.qvel), np.asarray(getattr(snap, "zone", np.zeros(0)))):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def legacy_seed_to_index(seed: int, n_items: int) -> int:
    """Reproduce ContactFormationEnv.reset's selection: seed → RNG → bank index (deterministic). Documents the legacy
    behaviour under audit; it is NOT an endorsement of seed-as-state-id."""
    return int(np.random.default_rng(int(seed)).integers(int(n_items)))
