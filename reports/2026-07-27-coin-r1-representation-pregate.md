# R1 representation pre-gate — v2 (canonical target-frame + control authority)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-decision-representation` · No training; dev-only; held-out is
frozen diagnosis.

## Ledger

| stage | status |
|---|---|
| R1 equivariance core (state + θ, 6 mandatory tests) | PASS (`1ab9e62f`) |
| R1 full feature extraction (34-D target/authority) | PASS (`37e5c18b`) |
| **first unnormalised pre-gate** | **INVALID — scale artefact** (B_τ condition ≈189, fn ≈8.7 dominated; NN dist measured units, not physics) |
| fixed physical group normalisation (`R1_NORM_SCALES`) | PASS (`4cf70ec2`; max\|v\| 189→2.0) |
| **valid R1 v2 pre-gate** | **MIXED-POSITIVE** (this report) |
| signed-directional-authority fallback | PRE-REGISTERED (next) |
| R2 HyMeKo relational encoder | NOT YET AUTHORISED |
| SAC / TD3 | BLOCKED |

## The scaling catch (why the first pre-gate was void)

The unnormalised 34-D vector was dominated by the B_τ **condition number** (~189) and **fn** (~8.7); every other component
was O(0.1–2). So the nearest-neighbour distance essentially ranked cradles by their condition number — `retrieval 0/6` was
a **units artefact**, not an R1 verdict. Fix: drop the raw condition number (O(100+), and redundant with the singular
values), and apply fixed per-group physical scales (`R1_NORM_SCALES`, causal — no held-out statistics, no dataset-size
dependence, identical on both sides ⇒ mirror-equivariant, deterministic at deploy). Verified `max|v| 189 → 2.0`.

## Valid pre-gate result (post-fix)

| metric | R0 (42-D) | R1 v2 (34-D canonical) | |
|---|:-:|:-:|---|
| mirror invariance (real cradles) | — | **all 8 True** (deficit 3.49→0 by construction) | ✅ |
| feature rank (centred, 6 cradles ⇒ ≤5) | — | **5/6** = no sample-level collapse (relative sanity only) | ✅ |
| corr(dφ,dθ) *(secondary; noisy at 6 cradles, multimodal θ-sets)* | 0.71 | **0.71** (recovered from v1's 0.42) | ✅ |
| **mean nearest-acceptable-set distance (dev)** *(PRIMARY)* | **0.874** | **0.697** | **R1 IMPROVED −20%** |
| within budget-8 reach (<0.45) | 0/6 | 0/6 | — |
| NN-retrieval K6 (dev) | 1/6 | 0/6 | — |

Per-cradle acc-set distance (R0 → R1): s1 0.94→0.94, s3 0.57→0.57 (same NN), 16500 1.11→**0.60**, 17750 1.18→**0.77**,
19500 0.80→**0.66**, 24000 0.64→0.64 (same NN). R1 is closer on 4/6 and never worse. Artifact
`r0_vs_r1_acceptable_set_distance.txt`.

## Classification — MIXED-POSITIVE

Per the decision tree: acceptable-set distance **improves meaningfully** (0.87→0.70) but retrieval K6 does not flip and
neither representation reaches within budget-8 (0/6 both). This is the **middle branch**: the invariant coordinatisation
**does** pull the proposal toward a working basin (a genuine mechanistic win for the target/contact-frame hypothesis) —
but the residual (~0.70) still exceeds the budget-8 (std 0.15) search reach, so a raw 1-NN proposal cannot deliver.

**Interpretation (measured vs inferred).** *Measured:* R1 reduces the nearest-acceptable-set distance by 20% on the same
sets while holding corr and preserving mirror invariance. *Inferred:* the residual gap is consistent with the **missing
signed directional authority** — the magnitude-only B_τ features say *how much* authority exists but not *which* torque
sign pushes forward vs brakes, and that sign is plausibly what a proposal needs to land inside a delivering basin.

## Next (pre-registered, not R2)

**SIGNED_DIRECTIONAL_AUTHORITY** enrichment: replace/augment the magnitude B_τ features with signed canonical directional
projections — forward push / brake authority, lateral ±, squeeze ±, spin ±, and the ± slew-admissible margins — stored
equivariantly in the canonical frame (no raw joint-angle convention). If this pushes the acc-set distance within reach and
retrieval > R0, run the R1 update-0 gate. Only a clean negative *after* the signed ablation yields
`ENGINEERED_R1_REPRESENTATION_INSUFFICIENT → R2`.

Also adding to the pre-gate artifact (per request): raw feature / normalised feature / normalisation scale / per-group
distance contribution — to immediately expose any remaining post-normalisation dominance (e.g. contact forces or slew
headroom).

## Addendum — signed authority + the second scale bug (clean result)

**Signed directional authority (v3, `ef6bf8f4`+integration):** reachable-set object authority (forward/lateral/brake over
the asymmetric slew box), contact-side squeeze/balance (B_τ null-space), deploy-faithful FD (caught joint-0 at 81 %
governor-attenuation), Jacobian equivariance test (`B_coin(Mx)=S_yB_coin(x)P_τ`) — all pass. Integrated equivariantly
(43-D, 6/6 mandatory tests, s3 mirror invariance holds).

**Second scale bug, caught by the per-group distance-contribution diagnostic (`fd4c76ad`):** `friction_util = |v_t|/(|v_n|+ε)`
is unbounded (explodes as v_n→0), variance 37.5, **95.7 % of the pairwise-distance sum** — same class as the B_τ condition
number. Fixed to a bounded slip fraction ∈[0,1]. (Note: it dominated the distance *sum* via outlier pairs but not the
argmin NN — the fix left the NN selections unchanged, so the retrieval conclusions were stable, not artefacts.)

**First clean pre-gate (both scale bugs fixed):** mean acc-set **0.79**, **0/6 within reach**, retrieval **0/6**, mirror
invariance all, rank 5/6. The signed authority does **not** help (only flips s1's NN to a worse match, 0.94→1.50; 0.8 % of
the distance — the reachable-set sign collapses for these non-saturated, symmetric-box cradles). **Best R1 = magnitude
authority (no signed):** per-state acc-set improved over R0 on 3/6 (16500 1.11→0.60, 17750 1.18→0.77, 19500 0.80→0.66),
mean 0.87→0.70; 0/6 within reach; retrieval 0/6.

**Classification (ChatGPT A/B/C, per-state not mean):** the best R1 is **case B** — the canonical representation is
load-bearing (primary metric improves over R0) but no cradle is within budget-8 reach and 1-NN retrieval (a coarse proxy)
does not deliver. The pre-registered signed-authority ablation is **exhausted** (no add; sign collapses for these
cradles). Two scale-dominance bugs found and fixed, so case-C's "no scale error" precondition is now satisfiable — but the
primary metric *did* improve, so it is not a clean case C either.

**The remaining open axis = learned amortisation.** 1-NN retrieval is the coarse proxy the tree itself warns against; the
learned R1 update-0 (a model that can interpolate/weight, not a single nearest neighbour) is untested and is what the
ledger flags OPEN. Decision fork: (1) **R1 update-0 gate** on the best clean R1 — justified because the primary metric
improved over R0 (the user's stated update-0 trigger) and the learned model is the actual deploy; or (2) **R2** (HyMeKo
relational encoder over the *same* canonical + signed quantities as node/edge attributes) if the flat representation is
judged at its limit for search-init.

## Files

- `hymeko_rl/coin_delivery/theta_option/canonical_frame.py` (R1 core unchanged; feature extractor + `R1_NORM_SCALES`).
- `hymeko_rl/experiments/coin_theta_rl_benchmark.py` (`--r1-check` with the acceptable-set-distance primary metric).
- Artifacts: `r1_representation_check.json`, `r0_vs_r1_acceptable_set_distance.txt`.

**CORE.YAML:** none. **Performance:** ~4 min/pre-gate, RSS < 0.3 GB.
