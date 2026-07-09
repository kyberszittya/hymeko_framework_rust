# LiNGAM-SH Step 4A — per-tail mechanism weights

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration`
**Status:** done. A mechanism can now carry **real-valued per-tail loadings**, so one hyperedge represents
asymmetric reward terms. Binary incidence remains the default. No training; no reward ablation.

---

## Summary

The binary factorization gives every mechanism one shared strength (uniform tail weights), so a single mechanism
`{progress_score, near_fraction} → {total_reward}` could not represent the asymmetric coffee-push reward
(`near→reward ≈ 0.97` ≫ `progress→reward`). Step 4A adds `fit_loadings_least_squares`: **`A_in` becomes
real-valued** (per-tail loadings) with `Σ = I`, so `B_hat = A_out · A_inᵀ`. For a single-output mechanism the
fitted loading of tail `x` equals `B[y, x]` exactly. On the coffee-push evidence the HyMeKo reward mechanism's
explained energy **doubles, 0.275 → 0.550**. The mechanism *structure* is unchanged — loadings are metadata, not
new DAG edges — so cross-view still passes.

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/causal/mechanism_factorization.py` | + `fit_loadings_least_squares` (weighted), `_tail_columns`, `_declared_loading`, `_count_loading_sign_mismatch` |
| `hymeko_rl/eval/causal/mechanism_proposal.py` | explicit proposals store per-tail declared weights in `evidence['loadings']` (additive) |
| `hymeko_rl/eval/causal/__init__.py` | export `fit_loadings_least_squares` |
| `hymeko_rl/eval/cip/reward_mechanism_integration.py` | comparison adds `hymeko_reward_weighted` |
| `hymeko_rl/tests/test_mechanism_factorization.py` | +7 weighted tests (spec A–G) |
| `hymeko_rl/tests/{test_reward_mechanism_integration,test_metaworld_cip}.py` | updated expectations / robust dep-test |

`CORE.YAML` / `pyproject.toml` / FANUC v2 / coin-collab v2b untouched; no dependency added.

## API decision

- **Separate fitted structure, not a proposal-API break.** The weighted fit is a new function
  `fit_loadings_least_squares` returning a `MechanismFactorization` whose `a_in` holds the real loadings — the
  existing dataclass already types `a_in` as `np.ndarray`, so **no schema change**. `factorize_from_proposals`
  (binary indicator) and `fit_sigma_least_squares` (shared strength) are untouched and remain the default.
- **Declared loadings ride on `evidence`.** Explicit reward/monitor proposals now stash their per-tail declared
  weights in `evidence['loadings']` (additive — existing tests unaffected). `use_declared=True` uses them as the
  loadings instead of fitting; `metrics['n_sign_mismatch']` counts tails whose *fitted* loading sign contradicts
  the *declared* one.
- **Multi-output choice (documented):** one loading per tail, **shared across the mechanism's heads**
  (`B_hat[j,i] = [j∈head]·loading_i`). This is the direct real-valued generalization of the single shared strength
  (`|tail|` parameters, not `|tail|×|head|`). For single-output mechanisms it is exact.

## Binary vs weighted (coffee-push multi-seed B)

| Candidate | mechanism | explained energy | params |
|---|---|---|---|
| none | — | 0.000 | 0 |
| common_child | `{near, action_noise} → reward` | 0.215 | 1 |
| **hymeko_reward (binary)** | `{progress, near} → reward` (1 strength) | **0.275** | 1 |
| **hymeko_reward_weighted** | `{progress, near} → reward` (per-tail loadings) | **0.550** | 2 |
| raw_pairwise | 4 degenerate edges | 1.000 | 4 |

**Explained energy before/after: 0.275 → 0.550** — the weighted HyMeKo reward mechanism reconstructs the strong
`near→reward` edge *exactly* (loading = 0.973) instead of the shared-strength compromise (0.487 across both tails),
and no longer posits a spurious uniform `progress→reward` contribution. It uses 2 parameters (one loading per tail)
vs the binary 1; the residual now is the reward's *other* parent (`action_noise→reward`, not in the HyMeKo tail)
and the non-reward edges.

## MetaWorld reward SoT example

The HyMeKo reward declaration provides per-term weights (`mw_in_place 8, mw_near 1, …`) that map to CIP variables
(`progress_score 8, near_fraction 1`). Those are carried as `evidence['loadings']` on the reward proposal, so the
weighted fit can **initialize / sign-check against the declared reward weights** (`use_declared=True`), or fit them
from `B` and flag disagreement — the SoT weights and the observed loadings are directly comparable.

## Cross-view verification

**Passes.** The weighted mechanism is the *same* structural hyperedge; `factorization.to_causal_hypergraph(name)`
→ `check_acyclicity` = True → star-expand → engine → **`cross_view.agree = True`** (declared ≡ tensor, Blake3 hash).
Loadings do not appear as extra edges (spec requirement 6). Tested (`test_weighted_converts_and_cross_view`).

## Tests

`pytest -p no:randomly`. **154 passed** across the CIP suite; ruff / radon (no block ≥ C, after splitting
`_count_loading_sign_mismatch`) / mypy `--strict` clean. Spec §7 A–H:
A weighted-beats-binary (asymmetric) · B weighted==raw-pairwise (single-output) · C binary unchanged ·
D declared loadings usable · E sign mismatch reported · F → `CausalHypergraph` · G cross-view passes ·
H existing Step 1/2/3A/3B/reward-SoT tests unchanged.

## Did existing behavior change? — No

Binary incidence remains the default (`factorize_from_proposals`, `fit_sigma_least_squares` untouched); the
`evidence['loadings']` addition is additive; the two updated test expectations reflect *new* outputs
(`hymeko_reward_weighted` key) and a *more robust* metaworld-dependency check, not changed behavior.

## What remains before the reward ablation

1. **Run the ablation** — with per-tail loadings the reward mechanism is now expressive; `ablate_reward(drop=[…])`
   recomputes the HyMeKo reward and we re-fit the CIP DAG on **pick-place** (where `mw_grasp` is in-frame) to test
   whether the `grasp → total_reward` loading collapses — the coin Stage-A intervention, on MetaWorld.
2. **Generic-frame integration** — comparison on the generic-sweep B (`near, grasp, obj_to_target_delta` all
   present) so all four reward terms map and the full `{near, grasp, progress, dist} → reward` mechanism is fit.
3. **Selection with weighted mechanisms** — extend `select_mechanism_subset` to score weighted candidates (it
   currently uses the binary `fit_sigma`).

## Constraints honored

No training · no reward ablation · no discovery / stochastic search / DirectLiNGAM change · FANUC v2 / coin-collab
v2b / `CORE.YAML` / `pyproject.toml` untouched · no existing report/artifact overwritten.
