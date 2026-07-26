# Acceptable-set multimodality test (M0) — is the coin held-out failure MODALITY or REPRESENTATION?

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-acceptable-set-proposal` · **Base:** tag
`coin-coverage-alone-insufficient` (`5fbf3c16`).

## Why this test comes first

The coverage curve closed the coverage axis: single-θ B0 deploys 2/4 (held-out 0/2 at **every** N=2,4,6) while the oracle
(teacher θ + the same budget-8 search) is 4/4. A delivering θ exists; the actor's single θ₀ misses it. Two hypotheses
remain, needing **different** fixes:

- **(A) modality** — the actor averages several valid, distant modes into a physically-bad mean → a **K-head multimodal
  proposal** is the fix.
- **(B) representation** — the features cannot read out the right mode at all → **representation** is the blocker; a
  K-head cannot help.

Per the directive, we do **not** build the multimodal model until we have proven distinct basins exist. This is the
discriminating gate.

## Method

- **Acceptable set (DEV cradles only):** per dev cradle, the local delivering basin (`sample_dev_basin`, frozen) **plus**
  a global uniform enrichment over the option box, keeping every θ that is **frozen-K6 delivered AND motion-contract
  compatible** (bounded joint/coin speed, braked to rest). Global sampling (not local jitter) is what can reveal a
  *second* basin. Measured global delivery rate ≈ 1.3 %, so the local basin carries the density and the global pass
  probes for separated modes.
- **Cluster** the pooled successful θ in normalised option-space by **single-linkage connected components** (a basin =
  a connected region; hop < `link_tol`). Report the basin count across a `link_tol` sweep so the choice is auditable.
- **Averaging test:** roll each state's acceptable-set **centroid** (what an MSE regressor targets). A non-delivering
  centroid is direct evidence that averaging modes is harmful (mechanism A).
- **Held-out overlay (eval-only, no fit):** the already-recorded delivering teacher θ for s4/s7 is assigned to the
  nearest pooled dev basin, or flagged an **orphan** (outside every dev basin → the delivering region is OOD for the dev
  manifold → representational, not recoverable by a dev-trained K-head). The failing N=6 actor θ₀ is overlaid to show
  where the single-θ regressor pointed.
- Harness: `coin_theta_rl_benchmark --acceptable-set`; logic in `hymeko_rl/coin_delivery/theta_option/acceptable_set.py`
  (reuses the frozen rollout / `sample_dev_basin`; the downstream K-head will bind to the existing
  `option_rl.MultimodalBudgetSearch`, which already splits a fixed total budget across K modes).

## Results

Artifact `reports/2026-07-27-coin-acceptable-set/acceptable_set_multimodality.json`; figure `acceptable_set_basins.png`.
121 pooled dev acceptable θ; wall 316 s, peak RSS 0.24 GB.

**1. Averaging demonstrably fails (robust, threshold-independent) — the primary signal.** The acceptable-set **centroid**
(what an MSE regressor targets) is **NON-delivering for 4 of 6 dev states**:

| state | n accepted | centroid delivers? | centroid dtz_end |
|---|:-:|:-:|---|
| s1 | 24 | ✅ | 9.4 mm |
| s3 | 12 | ❌ | 56.8 mm |
| 16500 | 4 | ❌ | 96.9 mm |
| 17750 | 68 | ✅ | 14.9 mm |
| 19500 | 10 | ❌ | 23.0 mm |
| 24000 | 3 | ❌ | 20.5 mm |

For those four states, averaging valid solutions yields a θ that **misses** — a single deterministic MSE centre cannot
represent their acceptable set.

**2. The single-θ regressor collapses the held-out proposals (mechanism A, visualised).** The N=6 actor emits **nearly
identical θ₀ for the two different held-out states** — ‖θ₀(s4) − θ₀(s7)‖ = **0.215** — and that collapsed point sits
**1.45 (s4) / 1.59 (s7)** from where each state's teacher θ actually delivers. On the PCA (`acceptable_set_basins.png`)
the two grey ✗ (actor θ₀) pile up in the upper-left while the red ★ (delivering teacher θ) sit in the populated lower-
centre cloud. The regressor averages the distant per-state modes into one wrong region.

**3. The held-out delivering θ are INSIDE the dev manifold (not OOD).** Point-based (nearest dev *canonical*, not
centroid): s4's teacher θ is **0.65** from s1's delivering θ; s7's is **0.67** from 24000's — ~the intra-basin hop scale
(0.87). Each held-out delivering solution lies near a *specific* dev cradle's strategy, so a proposal that offers the dev
cradles' distinct strategies (instead of their average) can reach it.

**Basin structure (reported honestly, not over-read):** at `link_tol` 0.9 the pool is one dominant connected basin
(n = 94) plus 14 scattered satellites (mostly singletons); the count is threshold-sensitive (sweep 0.4→80, 0.8→28,
0.9→15, 1.1→3, 1.5→1), so the *robust* multimodality evidence is the centroid-averaging failure (1) and the collapse (2),
**not** a clean basin count. inter/intra = 1.03/0.87 (marginal separation).

**Metric caveat (self-corrected):** the code's `assign_to_basins` "orphan" flag (distance to basin *centroid* > link_tol)
mislabeled s4's teacher θ as OOD. It is an artifact of the sprawling basin-0 centroid — the **same** flag fires on the N=6
actor θ₀ for the *delivering* dev state s1, which plainly is not OOD. The point-based check (3) refutes the OOD reading;
the `held_out_ood_warning` in the JSON is **downgraded** and not carried forward.

## Verdict

**MULTIMODAL_BASINS_PRESENT** — `justifies_k_head = True`. The blocker is **modality, not representation**: averaging the
valid modes fails (4/6 non-delivering centroids) and the single-θ regressor provably collapses the two held-out proposals
into one wrong region, while the delivering θ sit inside the dev manifold near distinct dev-cradle strategies.

→ **Proceed to M1** (K-head set predictor implementing `option_rl.MultimodalProposalPolicy`, permutation-invariant
acceptable-set loss; deploy through `MultimodalBudgetSearch` at total budget 8). Expectation to test at M2: emitting modes
that cover the dev cradles' distinct delivering strategies (rather than their average) recovers held-out. Open risk
carried forward: the delivering θ are ~0.65 from the nearest dev mode — a K-head must place a mode close enough that the
budget-8 (std 0.15) search can bridge it; if not, the honest outcome could be `MULTIMODALITY_PRESENT_BUT_UPDATE_ZERO_STILL_FAILS`.

## Tests

`hymeko_rl/tests/test_coin_acceptable_set.py` — 10 fast pure-logic tests (single-linkage clustering incl. the
link_tol merge threshold and empty set; orphan-vs-assigned overlay; the 3 verdict branches incl. centroid-fail-on-one-
cluster and not-separated-when-inter<intra; held-out OOD flag). All pass; ruff clean.

## Files touched

- `hymeko_rl/coin_delivery/theta_option/acceptable_set.py` (new).
- `hymeko_rl/experiments/coin_theta_rl_benchmark.py` (`--acceptable-set` mode + PCA viz).
- `hymeko_rl/tests/test_coin_acceptable_set.py` (new).

**CORE.YAML items touched:** none. **Performance:** peak RSS 0.238 GB (« 16 GB cap); wall 316.2 s. Single-threaded.

## Follow-up

- **M1 (next):** K-head set predictor (`option_rl.MultimodalProposalPolicy`) over the frozen B0 encoder → shared trunk →
  K bounded 6-D heads; permutation-invariant acceptable-set loss (bidirectional: every acceptable mode covered by a head
  = recall; every head near a real acceptable θ = precision) + mild head-collapse penalty gated on the set being multimodal.
  Model selection on **dev-only** CV; freeze arch/loss/seed rule; deploy via `MultimodalBudgetSearch` (total budget 8,
  K×(8/K)); one frozen-panel evaluation; all seeds reported.
- **Downgraded, not carried:** the centroid-distance OOD flag (metric artifact).
