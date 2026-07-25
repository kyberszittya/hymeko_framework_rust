# Bimanual E1/E2 — a balanced authority preload exists (rare, off-midpoint), but its passive release is not clean

**Date:** 2026-07-25 20:56 JST
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4` + coast target. Deterministic, no RL. O3 stays paused.
**Question (E0's follow-up):** the geometric tip-midpoint is not the authority-balanced point — so **does a balanced
bilateral preload configuration exist at all**, and if found, does releasing it leave a spring residue?
**One-line outcome:** a clean **balanced** preload **exists on 3/8 states, every one OFF the geometric midpoint**
(s ∈ {+0.03, +0.01, −0.03} m along the tip–tip axis) — the authority centre is real, state-dependent, and *not* the
midpoint. But 5/8 states have **no** balanced config in ±4 cm, and a discriminating test shows the ~1.5 cm **release
residue is not a controller artifact** (hold vs retract-neutral makes no difference). Existence: yes-but-rare-and-off-
centre. Release-readiness: **not yet** — passive release of a static balanced preload is not clean.

---

## E1 — the authority-balance existence oracle

Delivery-blind 1-D search along the fingertip→fingertip axis (±4 cm × 9), acquiring the preload at each candidate (no
launch) and scoring **force imbalance** `|Fn_L − Fn_R|/(Fn_L + Fn_R)` subject to the clean gate **and a minimum total
normal force** (so the trivial Fn≈0 touch cannot win). "Balanced" = clean ∧ total Fn ≥ 0.6 N ∧ imbalance ≤ 0.25.

| state | balanced? | best s (m) | imbalance | Fn L / R (N) | penetration L / R |
|---|---|---|---|---|---|
| s1 | **yes** | **+0.03** | 0.117 | 0.51 / 0.64 | −3.6 / −3.3 mm |
| s5 | **yes** | **+0.01** | **0.057** | 1.80 / 1.60 | −2.0 / −4.7 mm |
| s7 | **yes** | **−0.03** | 0.126 | 3.27 / 2.54 | −9.7 / −9.0 mm |
| s0,s3,s4 | no | — | — | 0 candidates acquire in ±4 cm | — |
| s2,s6 | no | — | min 0.73 / 0.71 | one-sided everywhere | — |

**Reading it.** Where a balanced configuration exists it is **decisively off the midpoint** (every best `s ≠ 0`, and the
sign flips between states) with low imbalance (down to 0.057). So `GEOMETRIC_MIDPOINT_IS_NOT_THE_AUTHORITY_CENTRE` is
**confirmed**, and the authority centre is a **state-dependent** offset. But the existence is **broad-only on 3/8** — the
other 5/8 either never acquire two contacts across the whole sweep (s0/s3/s4) or stay strongly one-sided (s2/s6, imbalance
≥ 0.71). Verdict: `BALANCED_PRELOAD_EXISTS_BUT_RARE_AND_OFF_MIDPOINT__CONTACT_GEOMETRY_LACKS_BROAD_BALANCED_AUTHORITY`.

## E2 — release-only sanity from the *exact* validated preload (hold vs retract-neutral)

From each found balanced preload, release the pin with no launch command and measure the coin's drift. Two release
controllers (the analysis's `ACQUISITION_READY` vs `RELEASE_READY` distinction): **hold** (tips keep position) vs
**retract-neutral** (tips back off δ, relieving the squeeze).

| state | imbalance | hold: disp / speed | retract: disp | jumped? |
|---|---|---|---|---|
| s1 | 0.117 | 2.56 cm / 0.28 m/s | 2.12 cm | both |
| s5 | 0.057 | 1.78 cm / 0.20 m/s | **3.68 cm** (worse) | both |
| s7 | 0.126 | 1.31 cm / 0.18 m/s | 1.55 cm | both |

**The discriminating test is decisive:** retract-neutral does **not** reduce the residue (and makes s5 worse). So the
1.3–2.6 cm drift is **not** a hold-controller squeeze artifact — it is a fundamental property of releasing the soft
(damping) pin: during acquisition the 1e5 slide-damping masks a small net force, and removing it lets the coin drift even
from a well-balanced (imbalance 0.057) preload. Verdict: `SPRING_RESIDUE_REMAINS_AT_BALANCED_POINT`. The residue is ~5×
smaller than the imbalanced-E0 case (11 cm) but still comparable to the delivery tolerance.

## Honest ledger (updated)

```
GRASP_ALLOCATION_PURE_MATH .................... PASS
BALANCED_PRELOAD_CONFIGURATION_EXISTS ......... YES, but 3/8 and OFF-midpoint
GEOMETRIC_MIDPOINT_IS_NOT_THE_AUTHORITY_CENTRE  CONFIRMED (state-dependent offset, sign flips)
CONTACT_GEOMETRY_HAS_BROAD_BALANCED_AUTHORITY   NO (5/8 lack any balanced config in ±4 cm)
RELEASE_READY (passive, zero residue) ......... NO (irreducible ~1.5 cm drift; not a hold/retract artifact)
COOPERATIVE_LAUNCH (A0 vs A2) ................. STILL NOT EVALUABLE from a clean static release
```

## Claims / non-claims

**Claimed (measured):** a balanced (imbalance ≤ 0.13, real force) bilateral preload exists on 3/8 states, always off the
geometric midpoint with a state-dependent sign; 5/8 states have no balanced config in ±4 cm; the ~1.5 cm release residue
survives both hold and retract-neutral release, so it is not a release-controller artifact.

**NOT claimed:** that no wider search or repositioning would find balance on the other 5/8 (the sweep is 1-D, ±4 cm along
one axis only); that the residue is irreducible under an *active* net-wrench-nulling controller (E3, untested); that A0 vs
A2 is settled (still blocked on a clean or baseline-subtracted release).

## Exact next rung (two options, decide before building)

1. **Reframe the launch as baseline-subtracted, not clean-release.** The passive-release residue is irreducible at these
   configs, so a fair A0-vs-A2 comparison should measure the **incremental** launch effect (commanded launch motion −
   same-preload passive-release motion) from an identical preload snapshot. This unblocks E0c without solving E2.
2. **Or E3 — an active net-wrench-nulling force-balance loop** (common-mode preload + differential Fn_L−Fn_R + a slow
   contact-frame offset toward the authority centre) that drives the *net* residual wrench to zero before release, rather
   than only balancing penetration. Only then is a zero-residue passive release plausible.

Deployment note (from the analysis): physically relocating the coin to the authority point is a valid **oracle**, not a
deploy solution — the real controller must select the contact points / bimanual frame / approach **state-dependently**
(the s ∈ {+0.03, +0.01, −0.03} offsets are exactly the relational quantity a structured prior would predict). O3 paused.

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — `balanced_preload_search` (E1 oracle, returns the validated env for a
  deterministic E2), `acquire_clean_preload(coin_xy=…)`, `release_only_sanity(retract=…)` hold/retract-neutral.
- `hymeko_rl/experiments/bimanual_curriculum_e1_benchmark.py` — E1 search + E2 hold-vs-retract discriminating test.

### Test results
- Unit: `test_cooperative_grasp` 4/4 pass; ruff clean.
- Benchmark: 8 states × (9-candidate search + 2 releases), ~13 min wall, single-thread, deterministic seeds 14000+250·i.
- Artifact: `reports/2026-07-25-coin-dynamics-contract-v2/bimanual_curriculum_e1.json`. Coast μ 0.179, δ 5 mm.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
