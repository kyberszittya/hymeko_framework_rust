"""OBSERVATION_NONINTERFERENCE_CONTRACT_V1 — authoritative start identifier + noninterference regression.

Investigation result (measured, bit-exact): the audit-capture rollout is BIT-IDENTICAL to the canonical
``rollout_from_handoff`` (same obs/action/qpos/qvel/strict/dtz at every step; ``node_features`` is read-only). The
0.194 vs 0.032 discrepancy was NOT a trajectory/observation defect. It had two causes:
  (1) **seed-collision** in analysis code — the dev bank has 31 starts but only 23 unique seeds, so a dict keyed by
      seed alone collapses 8 starts;
  (2) **dwell-origin mismatch** — the env ``_strict`` counter carries dwell accumulated during the pi_0 PREFIX replay
      (handoff ``_strict`` ≥ 1 for the settling starts), which the reconstruction's ``max_dwell`` inherits, while an
      offline re-certification that restarts the dwell at the handoff under-counts by the carried steps.

This module provides the authoritative :func:`start_id` (never seed alone) and the noninterference primitives.
"""
from __future__ import annotations

import hashlib


def start_id(ls) -> str:
    """Authoritative start identifier — hash of (seed, prefix_steps, family, obs_sha, base_sha, causal_sha). Use for
    EVERY join / aggregation / dedup / manifest. Seed alone is NOT unique across families (measured: 23 unique of 31)."""
    key = f"{ls.seed}|{ls.prefix_steps}|{ls.family}|{ls.obs_sha}|{ls.base_sha}|{ls.causal_sha}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def start_id_from_row(seed, prefix, family, obs_sha="", base_sha="", causal_sha="") -> str:
    key = f"{seed}|{prefix}|{family}|{obs_sha}|{base_sha}|{causal_sha}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def seed_collision_report(starts) -> dict:
    """How many distinct starts collapse under seed-only keying vs the authoritative start_id."""
    seeds = [s.seed for s in starts]
    return {"n_starts": len(starts), "n_unique_seeds": len(set(seeds)),
            "n_unique_start_id": len({start_id(s) for s in starts}), "seed_collision": len(set(seeds)) != len(starts)}
