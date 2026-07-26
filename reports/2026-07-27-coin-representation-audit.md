# Decision-time representation audit (Step 1) — why the current 42-D space fails, and what R1 must fix

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-decision-representation` · **Base:** tag
`coin-multimodality-insufficient` (`1ca1edcb`). No training; development-only; held-out is frozen diagnosis.

## Context

Coverage (`COVERAGE_ALONE_INSUFFICIENT`) and proposal modality (`MULTIMODALITY_PRESENT_BUT_UPDATE_ZERO_STILL_FAILS`) are
both empirically excluded. This audit asks, without training: **in what coordinate system is the cradle→working-θ map
even learnable, and what specifically is broken about the current 42-D features?**

## Findings

Artifact `reports/2026-07-27-coin-decision-representation/representation_audit.json`. Wall 185 s, RSS 0.25 GB.

### 1. The features have NO canonical left-right frame — the dominant defect

The 42-D features concatenate per-side quantities (fingertip forces, contact normals, tip-coin offsets, per-arm joints)
in **fixed left-then-right order**. Swapping L↔R (the permutation a mirror-symmetric cradle induces) moves the feature
vector by:

| | L/R full-swap deficit | (relative to ‖φ‖) |
|---|:-:|:-:|
| s1 | 4.29 | 1.00 |
| s3 | 3.27 | 0.86 |
| 16500 | 3.53 | 0.84 |
| 17750 | 3.78 | 0.93 |
| 19500 | 2.97 | 0.81 |
| 24000 | 3.13 | 0.81 |
| **mean** | **3.49** | **~0.87** |

**The mean L/R-swap deficit (3.49) is larger than the median distance between two *different* cradles (2.58).** A cradle
and its own mirror are farther apart in this feature space than two genuinely distinct cradles. The representation is more
sensitive to arbitrary left/right *labeling* than to actual geometry — it effectively doubles the cradle population
(every cradle + its mirror as a distinct point) and prevents any invariant sharing of the mirrored relation. `canonical =
False`.

### 2. The map is locally smooth — the failure is NOT a jagged/unlearnable coordinate system

Over the dev canonical θ (15 pairs): **corr(‖Δφ‖, ‖Δθ‖) = 0.71** (positive), **Lipschitz ratio max = 0.88** (min 0.29,
median 0.53) — no coordinate-dependent break (a small feature change never maps to a large θ change). So the current
space is not intrinsically jagged; a model *can* fit the dev cradles (the B0 did, dev 2/2). The problem is transfer, not
local fitting.

### 3. Feature-proximity does not predict θ-transferability — training-free retrieval fails

Propose the nearest-feature *other* cradle's canonical θ and run the same budget-8 search (R0 baseline, training-free):

| target | nearest-φ source | dφ | K6 | dtz_end |
|---|---|:-:|:-:|---|
| s1 | 24000 | 2.55 | ✗ | 75 mm |
| s3 | 24000 | 1.15 | ✗ | 48 mm |
| 16500 | 24000 | 1.50 | ✗ | 70 mm |
| 17750 | s1 | 2.58 | ✗ | 94 mm |
| 19500 | s3 | 1.18 | **✓** | 18 mm |
| 24000 | s3 | 1.15 | ✗ | 61 mm |
| **dev** | | | **1/6** | |
| s4 (held) | 17750 | 1.45 | ✗ | 80 mm |
| s7 (held) | 17750 | 1.29 | ✗ | 103 mm |
| **held-out** | | | **0/2** | |

Nearest neighbours are 1.15–2.58 apart in features (θ 0.69–1.21), well beyond the budget-8 (std 0.15) search reach. The
one success (19500←s3) is the case where the neighbour happens to be close enough. The raw 42-D space does **not** support
retrieval-based transfer between cradles.

## Verdict

`CURRENT_42D_DOES_NOT_ADMIT_A_LEARNABLE_MAP_AS_IS` — defects `NO_CANONICAL_LEFT_RIGHT_ORDERING` +
`FEATURE_PROXIMITY_DOES_NOT_PREDICT_THETA_TRANSFER_ON_DEV`.

**Precise reading (measured vs inferred):** *measured* — the space is locally smooth (corr 0.71, no breaks) yet
retrieval-transfers only 1/6 on dev and 0/2 held-out, and its L/R-swap sensitivity (3.49) exceeds the inter-cradle
distance (2.58). *Inferred* — the generalisation failure is consistent with (a) the mirror-doubling inflating all
distances and blocking invariant sharing, and (b) the current (world/joint-ish) coordinates placing physically-similar
cradles too far apart for a budget-8 search to bridge. The axis is confirmed: the blocker is the input *coordinate
system*, and it is a fixable one, not an intrinsic wall.

### What this hands R1 (concrete, measurable targets)

1. **Canonical left-right ordering** — collapse the 3.49 swap deficit toward 0 (a deterministic side-canonicalisation), so
   a cradle and its mirror map to the *same* vector.
2. **Target/contact-frame coordinates** — express coin state, tip offsets, contacts, and authority in the target-aligned
   frame so physically-similar cradles become *close*, bringing nearest-neighbour θ within search reach.

**R0 reference the ladder must beat:** NN-retrieval dev **1/6**, held-out **0/2**; and the learned single-θ/K-head R0
update-0 = 2/4 (held 0/2).

## Tests

`hymeko_rl/tests/test_coin_representation_audit.py` — 7 fast (smoothness hi/lo correlation; NN never-self + held-out-style
query; swap swaps contact pairs; deficit zero-symmetric/nonzero-asymmetric; verdict both branches). ruff clean.

## Files touched

- `hymeko_rl/coin_delivery/theta_option/representation_audit.py` (new).
- `hymeko_rl/experiments/coin_theta_rl_benchmark.py` (`--rep-audit` mode).
- `hymeko_rl/tests/test_coin_representation_audit.py` (new).

**CORE.YAML items touched:** none. **Performance:** RSS 0.25 GB; wall 185 s. Single-threaded.

## Next

Build the equal-budget ladder: **R0** (current 42-D, this baseline) → **R1** (canonical-ordered target/contact-frame
features) → **R2** (HyMeKo relational structure). Same teacher bank, acceptable sets, model budget, budget-8 search,
dev-only CV, one frozen-panel deploy, no held-out fitting. The audit predicts R1 should move the needle if canonical
ordering + target-frame coordinates are the missing invariances; R2 then tests whether explicit HyMeKo relations add
load-bearing value over hand-engineered R1.
