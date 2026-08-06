---
campaign: COIN-DELIVERY-RECOVERY-BASELINE-0
title: Corpus provenance + explicit state identity
date: 2026-07-20
parent: 2026-07-20-coin-delivery-recovery-baseline.md
---

# Corpus provenance + state identity (§3)

**Created-at:** 2026-07-20 17:40 JST.

## The corpus is a stale artifact of a prior campaign

| property | value |
|---|---|
| name | `c1_heldseed_bank.pkl` |
| sha256 | `9262f6dc842b…` |
| path | `experiments/2026_07_18_arcrl/c1_heldseed_bank.pkl` |
| **originating campaign** | **`2026_07_18_arcrl` — a PRIOR campaign; reused unchanged by the coin-delivery arc** |
| tracked | no (untracked binary; referenced by hash) |
| n_items | 376 |
| snapshot schema | `snap` + `{mode, drift, dist, high_coin_y, source, side, second_dist, difficulty}` |
| qpos / qvel dim | 7 / 7 (K1 pads to 9 at hinge indices 2, 5) |
| source seed universe | 62000–62312 (`pedc_selection._build_plan_corpus`) |

`corpus_manifest.json`. The delivery arc never regenerated a corpus of its own; it consumes this July-18 bank.

## The evaluation "seed" is an RNG index-selector, not a state identity

`ContactFormationEnv.reset(seed)` does `np.random.default_rng(seed).integers(len(bank))` — the seed selects a **random
bank index**. It is deterministic and *paired* (the same seed selects the same index for K0 and K1), but it is **not a
semantic state identifier**. `state_mapping.json` (the 90 eval seeds 64000–64089):

| episode count | unique state count | duplicate state count | max duplication |
|---|---|---|---|
| 90 | **82** | 8 | 4 |

So a report claiming "n = 90 independent states" is false — there are 82 unique states, and one snapshot is evaluated 4×.
Duplicate weighting shifts a continuous metric (see the golden-reproduction report) though not a qualitative verdict.

## Explicit state identity (new, isolated — no rollout change)

`hymeko_rl/coin_delivery/provenance/state_identity.py`:

```python
@dataclass(frozen=True)
class CorpusId:      name: str; sha256: str
@dataclass(frozen=True)
class StateId:       corpus: CorpusId; snapshot_index: int; snapshot_sha256: str
```

Plus `snapshot_hash(snap)` (content hash of qpos+qvel+zone) and `legacy_seed_to_index(seed, n)` (documents the legacy
selector under audit — not an endorsement). This stage **defines** the identity; wiring `rollout` to consume `StateId`
(replacing seed-as-selection) belongs to the refactor campaign.

## Provenance gates (§8) — 8 tests, all pass

Corpus-hash-matches-manifest · duplicate-states-surfaced (82/90) · `StateId` requires a hash (a bare seed raises) ·
every golden StateId restores · restore history-independent · K1 variant models identical · coin production source
tracked · seed-is-index-selector-not-identity.
