# R1 learned update-0 gate — FLAT_R1_LEARNED_AMORTISATION_FAILS (flat ceiling established → R2)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-decision-representation` · **Base:** `4bdc329d`.

## What this tested

The decisive learned-amortisation test on the flat canonical R1 representation. **Only** the representation (42-D →
R1 v3 canonical 43-D) and the labels (raw θ → canonical θ) changed vs the frozen multimodal baseline. Everything else
frozen: K-head architecture, **dev-only** LODO K-selection, dev acceptable sets, optimiser/epochs/seed, budget-8 +
centre-inclusion, physical option, K6 monitor, frozen 4-state panel. Deploy: canonicalise → R1 features → predict
canonical θ heads → **decode (inverse T_θ)** → fair budget-8 search → frozen K6.

## Result

- **K_main = 4** (dev-only LODO: K=1 0/6, K=2 0/6, K=4 1/6 — capacity-capable; K=1 is a side-diagnostic).
- **MAIN (K=4): dev 2/2 · held-out 0/2 · total 2/4.** K=1 side-diagnostic: 2/4.

| state | split | K6 | head | dtz_end | search displ. | failure |
|---|---|:-:|:-:|---|---|---|
| s1 | dev | ✅ | 1/4 | 15.7 mm | 0.00 (head centre delivers) | — |
| s3 | dev | ✅ | 1/4 | 11.6 mm | 0.43 (search bridged) | — |
| s4 | held-out | ❌ | 2/4 | 64.7 mm | 0.22 | NEVER_REACHED_ZONE |
| s7 | held-out | ❌ | 0/4 | 67.2 mm | 0.00 | NEVER_REACHED_ZONE |

## Verdict — FLAT_R1_LEARNED_AMORTISATION_FAILS

Per the pre-registered result tree: total **2/4, held-out 0/2 → `FLAT_R1_LEARNED_AMORTISATION_FAILS` → the flat
representation ceiling is established → R2 authorised. SAC/TD3 remains BLOCKED.**

**This is interpretable representation science, not a pipeline artefact:**

- **No dev regression** — dev **2/2** delivers cleanly *through the full decode pipeline* (s1's decoded canonical head
  centre delivers directly; s3's is bridged by the budget-8 search). So the canonical-θ **encode/decode and head/θ₀/θ_exec
  provenance are sound** (the tree's "dev regression → audit encode/decode first" branch does not apply).
- **Held-out is a WRONG-BASIN failure, not near-but-unbridged** — s4/s7 end ~65 mm from the zone (`NEVER_REACHED_ZONE`),
  far beyond the budget-8 search reach; the learned model proposes the wrong region for held-out geometry.

**Decisive comparison:** this equals the **raw-42-D multimodal ceiling** (also dev 2/2, held 0/2, total 2/4 at K=4). So the
flat canonical representation *improved the geometry* (pre-gate acc-set 0.87→0.70) but did **not** improve *learned
held-out delivery* — the flat-representation ceiling is 2/4 for **both** the raw and the canonical feature organisation.
The remaining untested axis is therefore **relational organisation** (R2), not more flat features.

## Ledger

```
coverage                          CLOSED NEGATIVE
raw-θ multimodality               CLOSED NEGATIVE (2/4)
canonical representation          LOAD-BEARING (acc-set 0.87→0.70) but learned update-0 = 2/4 (= raw ceiling)
signed authority                  EXHAUSTED
two scale-dominance bugs          FOUND + FIXED (B_τ condition, friction_util)
physical delivery                 SOLVED 4/4 (oracle/search)
flat learned amortisation         FAILS (2/4, held 0/2 — wrong basin)  ← ceiling established
R2 HyMeKo relational encoder      AUTHORISED (next isolated axis)
SAC / TD3                         BLOCKED
```

## Next — R2 (isolated axis)

R2 keeps R1's *same* canonical + signed physical quantities but reorganises them as **HyMeKo node/edge attributes**
(coin, target, two canonical fingertips/contacts; contact point/normal/tangent; push/brake/lateral authority;
squeeze/balance authority; slew headroom; phase) → a relational encoder → the **same** K-head acceptable-set output →
the **same** budget-8 search. So R2's test is purely whether relational *organisation* extracts held-out generalisation
from the *same physically-correct information* the flat representation could not. A 4/4 (incl. held-out 2/2) would be
`HYMEKO_STRUCTURAL_REPRESENTATION_LOAD_BEARING` and authorise SAC/TD3; any honest failure keeps RL blocked.

## Files

- `hymeko_rl/coin_delivery/theta_option/multimodal_proposal.py` (K-head net feature-dim parameterised).
- `hymeko_rl/experiments/coin_theta_rl_benchmark.py` (`--r1-update0` mode + `_r1_deploy_one` decode deploy).
- Artifact: `reports/2026-07-27-coin-r1-update0/r1_update_zero.json` + `r1_khead_K{1,4}.pt`.

**CORE.YAML:** none. **Performance:** wall 358 s, RSS < 0.35 GB.
