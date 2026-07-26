"""DEV-CRADLE EXPANSION — deduplicate the certified-cradle inventory into UNIQUE development cradles.

The certification scout sweeps ``seed = 14000 + 250·si``, but `_reconstruct` scans a 1600-wide sub-seed window per
seed_lo, so adjacent seeds overlap ~84% and the raw certification count is NOT the unique-cradle count. This module
turns the scout inventory into a clean pool:

    certified  →  EXACT dedup (post-release state hash)  →  NEAR dedup (geometry fingerprint L2)
               →  exclude the frozen held-out cradles (s4,s7) AND their near-duplicates  →  unique DEV-eligible cradles

The geometry fingerprint is the frozen 42-D causal feature vector (`dataset.structured_features`): two cradles with
near-identical fingerprints are the same geometry. Held snapshots are reused by the delivery pass (no re-acquisition).
The near-dedup threshold is a parameter and the full pairwise distances are reported, so the choice is auditable
(measure before deciding).
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.theta_option.dataset import flatten_features, structured_features
from hymeko_rl.coin_delivery.theta_option.teacher_bank import FROZEN_SEEDS, acquire_snapshot

HELDOUT_SEEDS = (15000, 15750)                       # s4, s7 — the frozen held-out panel, excluded from the dev pool
# FROZEN near-duplicate threshold (fingerprint L2), frozen BEFORE any delivery outcome and justified from the
# distinct-cradle distance scale: the four genuinely-distinct frozen cradles s1/s3/s4/s7 are pairwise 1.53–3.22 apart
# (closest distinct pair s4–s7 = 1.534), so 0.30 is 5× below the distinct scale — it merges only near-identical
# duplicates, never distinct cradles. The full pairwise matrix is reported so the choice is auditable, not post-hoc.
NEAR_TOL = 0.30


def geometry_fingerprint(snap: CradleSnapshot) -> np.ndarray:
    """The cradle's geometry fingerprint = the frozen 42-D normalised causal feature vector at the handoff. Deterministic;
    near-identical fingerprints ⇒ the same cradle geometry."""
    return flatten_features(structured_features(snap))


def replay_state_hash(snap: CradleSnapshot) -> str:
    """FULL replay-relevant state hash (not just qpos+qvel): qpos, qvel, ctrl, qacc_warmstart, act, plus the exported
    controller state (prev_tau, q_hold) and the frozen contact frames/identities. Two cradles with the same replay hash
    are the SAME operating point. (`snap.post_release_hash` is only a physical_state_hash of qpos+qvel.)"""
    d = snap._rl.inner.data
    h = hashlib.md5()
    for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "act"):
        arr = getattr(d, name, None)
        if arr is not None:
            h.update(np.ascontiguousarray(np.asarray(arr, np.float64)).tobytes())
    for arr in (snap.prev_tau, snap.q_hold, np.asarray(snap.fn0, np.float64)):
        h.update(np.ascontiguousarray(np.asarray(arr, np.float64)).tobytes())
    for side in ("left", "right"):
        fr = snap.frames[side]
        h.update(np.ascontiguousarray(np.asarray(fr["x_c"], np.float64)).tobytes())
        h.update(np.ascontiguousarray(np.asarray(fr["n"], np.float64)).tobytes())
        h.update(f"{fr['pair']}|{fr['geom_tip']}".encode())
    return h.hexdigest()[:16]


class CradleEntry:
    """A re-acquired certified cradle held in memory for dedup + the delivery pass (snapshot reused, never re-acquired).
    ``hash`` is the FULL replay-state hash (the dedup key); ``physical_state_hash`` is the coarse qpos+qvel hash."""

    def __init__(self, seed: int, snap: CradleSnapshot):
        self.seed = int(seed)
        self.snap = snap
        self.hash = replay_state_hash(snap)
        self.physical_state_hash = snap.post_release_hash
        self.fingerprint = geometry_fingerprint(snap)


def acquire_certified_pool(harness: tuple[Any, ...], certified_seeds: list[int],
                           progress: "Any | None" = None) -> list[CradleEntry]:
    """Re-acquire each certified seed and compute its hash + geometry fingerprint. ``progress(entry, dt)`` is called live
    per cradle. Skips any seed that fails to re-acquire (should not happen for a scout-certified seed)."""
    import time as _time
    pool: list[CradleEntry] = []
    for seed in certified_seeds:
        t0 = _time.time()
        snap, _meta = acquire_snapshot(harness, seed)
        if snap is not None:
            e = CradleEntry(seed, snap)
            pool.append(e)
            if progress is not None:
                progress(e, _time.time() - t0)
    return pool


def dedup_and_split(pool: list[CradleEntry], *, near_tol: float = NEAR_TOL,
                    heldout_seeds: tuple[int, ...] = HELDOUT_SEEDS) -> dict[str, Any]:
    """Exact (hash) + near (fingerprint L2) dedup, then exclude the held-out cradles and their near-duplicates. Returns
    the per-cradle report (duplicate_of / near_dup_of / held_out / held_out_near_dup / dev_eligible), the dev-eligible
    seed list, and the full pairwise fingerprint-distance summary. # Postconditions: dev_eligible cradles are pairwise
    hash-distinct, pairwise fingerprint-distance ≥ near_tol, and ≥ near_tol from every held-out cradle."""
    n = len(pool)
    seeds = [e.seed for e in pool]
    # EXACT dedup by post-release state hash (first seed per hash is the representative)
    rep_by_hash: dict[str, int] = {}
    duplicate_of: list[int | None] = [None] * n
    for i, e in enumerate(pool):
        if e.hash in rep_by_hash:
            duplicate_of[i] = seeds[rep_by_hash[e.hash]]
        else:
            rep_by_hash[e.hash] = i
    hash_unique = [i for i in range(n) if duplicate_of[i] is None]
    # pairwise fingerprint distances among hash-unique cradles
    dist = {}
    for a in hash_unique:
        for b in hash_unique:
            if b > a:
                dist[(a, b)] = float(np.linalg.norm(pool[a].fingerprint - pool[b].fingerprint))
    # NEAR dedup: the later hash-unique cradle within near_tol of an earlier one is a near-duplicate
    near_dup_of: list[int | None] = [None] * n
    kept = []
    for i in hash_unique:
        merged = None
        for j in kept:
            d = dist.get((j, i), dist.get((i, j)))
            if d is not None and d < near_tol:
                merged = seeds[j]
                break
        if merged is None:
            kept.append(i)
        else:
            near_dup_of[i] = merged
    # held-out exclusion: by seed AND fingerprint proximity to any held-out cradle
    held_idx = [i for i in kept if seeds[i] in heldout_seeds]
    held_out = [seeds[i] in heldout_seeds for i in range(n)]
    held_out_near = [False] * n
    for i in kept:
        if held_out[i]:
            continue
        for h in held_idx:
            d = float(np.linalg.norm(pool[i].fingerprint - pool[h].fingerprint))
            if d < near_tol:
                held_out_near[i] = True
                break
    dev_eligible_idx = [i for i in kept if not held_out[i] and not held_out_near[i]]
    rows = [{"seed": seeds[i], "replay_hash": pool[i].hash,
             "physical_state_hash": getattr(pool[i], "physical_state_hash", None),
             "duplicate_of": duplicate_of[i], "near_dup_of": near_dup_of[i],
             "held_out": bool(held_out[i]), "held_out_near_dup": bool(held_out_near[i]),
             "dev_eligible": bool(i in dev_eligible_idx), "is_frozen_seed": bool(seeds[i] in FROZEN_SEEDS)}
            for i in range(n)]
    dvals = list(dist.values())
    return {"near_tol": near_tol, "heldout_seeds": list(heldout_seeds), "n_certified": n,
            "n_hash_unique": len(hash_unique), "n_after_near_dedup": len(kept),
            "n_dev_eligible": len(dev_eligible_idx), "dev_eligible_seeds": [seeds[i] for i in dev_eligible_idx],
            "held_out_present": [seeds[i] for i in held_idx],
            "pairwise_fingerprint_dist": {"min": (round(min(dvals), 4) if dvals else None),
                                          "median": (round(float(np.median(dvals)), 4) if dvals else None),
                                          "max": (round(max(dvals), 4) if dvals else None),
                                          "pairs_below_tol": int(sum(1 for d in dvals if d < near_tol))},
            "rows": rows}
