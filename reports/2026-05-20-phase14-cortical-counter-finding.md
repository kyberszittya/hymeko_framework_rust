# Phase 14 cortical counter-finding — 2026-05-20 overnight queue

## Summary

Built a cortical sister of Phase 14's combined HSIKAN fixture:
multi-cost dimensions + by-product injection on `binning_shallow`
(which the cortical exhaustive sweep showed gets beaten by
`binning_deep` at d≥8). Ran the same 6-corner lever-interaction
matrix.

**Headline counter-finding: the by-product filter is *adversarial*
under cost-only weight on this cortical fixture.** It forces ABB
to pick d4+deep (mean r² 0.235), which the exhaustive sweep
already established is the **worst** of all 12 architectures —
worse by −0.050 than the relaxed d4+shallow it was supposed to
"upgrade".

So Phase 14's HSIKAN sub-additivity finding does NOT generalize
uniformly. By-product filtering is only useful when the injection
targets an architectural choice that is **actually dominated at
every relevant width**, not just at large widths. Guessing wrong
makes things worse.

## Files touched

| File | Status |
| --- | --- |
| `data/hsikan/sweep_msg_cortical_combo.hymeko` | **new** (75 LOC) — multi-cost + `binning_shallow` by-product |

## CORE.YAML items touched

None.

## The 6-corner matrix (5 seeds, synthetic Cichy-92, ResNet backbone)

| weights | mode | ABB picks | mean r² (V1/V2/V4) | gain vs A |
| --- | --- | --- | --- | --- |
| `1,0,0` (gpu only) | relaxed (no filter) | `d4+shallow+pls25` (A) | **0.2844** | baseline |
| `1,0,0` (gpu only) | strict (filter on) | `d4+deep+pls25` (B) | **0.2349** | **−0.0495 (regression)** |
| `0,1,0` (quality only) | relaxed | `d16+deep+pls25` (C) | **0.3966** | **+0.1122** |
| `0,1,0` (quality only) | strict | `d16+deep+pls25` (same C) | 0.3966 | +0.1122 |
| `1,5,1` (balanced) | relaxed | `d16+deep+pls25` (same C) | 0.3966 | +0.1122 |
| `1,5,1` (balanced) | strict | `d16+deep+pls25` (same C) | 0.3966 | +0.1122 |

## Three substantive findings

### 1. By-product filter HURTS at cost-only weight (counter to HSIKAN Phase 14)

On HSIKAN, the strict-mode filter at cost-only weight moves the
pick from `m4+h8+short` (AUC 0.430) → `m4+h8+long` (AUC 0.491,
**+0.061**). The filter targeted `train_short`, which is dominated
at every cycle width.

On cortical, the filter at cost-only weight moves the pick from
`d4+shallow` (r² 0.284) → `d4+deep` (r² 0.235, **−0.050**). The
filter targeted `binning_shallow`, which is dominated **only at
d≥8**, not at d=4.

The mechanism is correct in both cases. The *injection choice* was
wrong on cortical: `binning_shallow` is not dominated at d=4. The
exhaustive sweep had already shown this (d4+shallow 0.284,
d4+deep 0.235; d8+shallow 0.348, d8+deep 0.357 — crossover at
d=8) but the Phase-14 cortical fixture treated shallow uniformly
as the dominated choice.

### 2. The MO quality lever still does what it's supposed to

Both relaxed and strict modes under quality-only or balanced
weights pick the same `d16+deep+pls25` (r² 0.397) — a +0.112 gain
over the relaxed cost-only baseline. The MO mechanism is robust
to the by-product mis-injection because it's operating on
ground-truth quality_drop values.

### 3. Lever-interaction rule (refined post-cortical)

Rewriting Phase 14's "sub-additive" claim more honestly:

> **By-product injection works only when the targeted unit is
> dominated at every relevant point along the orthogonal axes
> the search considers.** When the user mis-injects (treats a
> non-dominated unit as dominated), the strict-mode filter
> regresses the cost-only result.
>
> **Multi-objective quality weighting is robust to mis-specified
> by-product injection** because it operates on numerical
> quality_drop values, not on schema-level assertions.
>
> When both levers are available and quality weights are
> available, **prefer MO weights over by-product injection** —
> the failure mode is graceful (smaller gain) rather than
> adversarial (regression).

## Why this is the more honest Phase 14 result

The HSIKAN Phase 14 report ended with "engineering implication:
pick the right tool per use case." The cortical counter-finding
sharpens that into a real caution: **by-product injection is a
sharp knife.** Used correctly it removes dominated architectures
automatically; used incorrectly it removes the right one.

A real NAS pipeline would either:
- (a) Inject by-products only on units whose dominance is
  empirically established at every relevant orthogonal-axis
  point;
- (b) Use MO weights with quality_drop derived from measurements,
  which is failure-tolerant.

Phase 11 and Phase 12.5 satisfied (a) — the dominated units were
empirically measured on the actual workload. The cortical
combined fixture I built here violated (a) by treating shallow
as uniformly dominated. **The mechanism is fine; the schema author
made a wrong call.**

This is a useful negative result for the audit.

## Test results

No new tests in Phase 14 cortical — the schema is data, the
mechanism is tested in Phase 11. The cortical exhaustive sweep
already pinned the d4+shallow vs d4+deep behaviour numerically.

`cargo test -p hymeko_pgraph` full sweep: 96/96. Slice 1 cortical
tests: 21/21.

## Open follow-ups

1. **Refined cortical combined fixture.** Inject by-product
   only on units that are dominated at d=4 *and* d=8 *and*
   d=16 — which on this synthetic data is none of them (the
   exhaustive sweep showed each unit is at least once on the
   Pareto frontier). The honest schema would have no by-product
   injection and only use MO weights.
2. **Lever-selection guidance in the book.** Phase 13's
   `jq` recipe should grow a sibling: when to use MO vs
   by-product injection, with this cortical counter-finding
   as the cautionary example.
3. **Bidirectional check on HSIKAN.** Re-verify the Phase 11
   dominance claim against the actual 5-seed AUC distribution
   across all 8 architectures — confirm `train_short` is
   dominated at every cycle/hidden combination, not just at the
   ones Phase 8 measured.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted).
- **Wall time:** ~12 s for the 5-seed × 3-architecture training A/B.
- **No regressions on prior tests.**

## Acceptance check

- [x] Cortical combined fixture parses; canonical PASS;
      extension FAIL on `wasted_retinotopy`.
- [x] 6-corner matrix run; 3 distinct architectures emerge.
- [x] 5-seed training A/B confirms the **counter-finding**:
      by-product filter regresses the cost-only result by
      −0.050 r².
- [x] MO quality weight still delivers +0.112 r² over relaxed
      cost-only.
- [x] Lever-selection guidance refined and documented.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
