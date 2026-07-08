# LiNGAM-SH step 3B — deterministic mechanism selection + closed-form Σ

**Date:** 2026-07-08 · Aiko · branch `hymeko-neuro-migration`
**Status:** implemented (deterministic selection + least-squares Σ). **No stochastic search, no discovery, no
DirectLiNGAM change, no training.** Spec §5/§7. Builds on step 3A.

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/causal/mechanism_factorization.py` | + `fit_sigma_least_squares` (closed-form Σ via `lstsq`; reports fitted-vs-proposal sign mismatch) |
| `hymeko_rl/eval/causal/mechanism_selection.py` | **new** — `MechanismSelectionResult`, `select_mechanism_subset` (exhaustive/greedy) |
| `hymeko_rl/eval/causal/__init__.py` | exports |
| `hymeko_rl/tests/test_mechanism_selection.py` | **new** — 11 tests (spec §8) |

No existing module's behaviour changed. `CORE.YAML` / `pyproject.toml` untouched; no dependency added.

## Selection API

```python
fit_sigma_least_squares(variables, pairwise_edges, proposals) -> MechanismFactorization
select_mechanism_subset(variables, pairwise_edges, proposals, *,
                        method="greedy"|"exhaustive", max_exhaustive=12, complexity_penalty=1.0
                        ) -> MechanismSelectionResult

@dataclass(frozen=True)
class MechanismSelectionResult:
    variables; selected; rejected; factorization; candidate_scores; selection_score; method
```

## Least-squares Σ convention

One rank-1 basis `A_out[:,k] · A_in[:,k]ᵀ` per proposal (0/1 incidence, `B[effect,cause]`); `Σ` = `np.linalg.lstsq`
solution of `Σ_k σ_k · basis_k ≈ B` (deterministic, min-norm on rank-deficient / overlapping bases). The proposal
**signs are metadata**; the *fitted* σ may disagree — `metrics["n_sign_mismatch"]` counts mechanisms whose fitted
sign contradicts the proposed sign, and `candidate_scores[name]["sign_match"]` flags it per candidate.

## Selection score

```
selection_score = fro_error + complexity_penalty · n_parameters · ln(n_observed + 1) / max(n_observed, 1)
```
`n_observed` = number of non-zero entries of `B`. Exhaustive tests every subset (empty included) when
`len(proposals) ≤ max_exhaustive`; else deterministic greedy forward selection (add the improving candidate with
the lowest resulting score; stop when none improves). Fewer mechanisms are preferred when reconstruction is
comparable (penalty rises with `n_parameters`); the empty subset is a valid baseline.

## Example — grouped mechanism beats pairwise DAG edges

`edges: near→reward (0.9), progress→reward (0.9)`; candidates: grouped `{near, progress}→{reward}` and the two
pairwise `{near}→{reward}`, `{progress}→{reward}`. Exhaustive selection:
- **selected = `[grouped]`**, rejected = `[p1, p2]`; `fro_error` **1.273 → 0.0** (exact), `selection_score` 0.549.
- The grouped mechanism (1 parameter) and the two pairwise (2 parameters) both reconstruct exactly, so the
  complexity penalty picks the single grouped mechanism.

## Example — a wrong mechanism is rejected

`edges: x→y (0.8)`; candidates: `correct {x}→{y}`, `wrong {y}→{x}` (reversed). Selection:
- **selected = `[correct]`**, rejected = `[wrong]`. The wrong mechanism's basis lands on `B[x,y]=0`, so its
  least-squares σ = 0 (`candidate_scores["wrong"].fro_error_alone = 0.8`, no improvement) → dropped.

## Sign mismatch

`edges: x→y (−0.8)`, proposal `{x}→{y}` declared `sign=+1`: fitted σ = **−0.8**, `n_sign_mismatch = 1`,
`candidate_scores["m"].sign_match = 0.0` — the fit reveals the declared sign is wrong.

## Test results

`pytest -p no:randomly`. **134 passed** across the CIP suite (11 new + 123 prior); ruff / radon (no block ≥ C) /
mypy `--strict` clean on the new module. Covers spec §8 tests 1–13: lstsq exact/improvement, exhaustive
correct-over-wrong and two-when-needed, greedy determinism, empty baseline, complexity-penalty grouped preference,
overlap determinism, sign-mismatch reporting, → `CausalHypergraph`, `check_acyclicity`, `cross_view_verify`, and
the existing suite unchanged.

## Cross-view verification

**Passes** — `result.factorization.to_causal_hypergraph(name)` star-expands and cross-view-verifies against the
native engine (`test_selected_converts_to_hypergraph_and_cross_view`); the selected graph also passes
`check_acyclicity`.

## What remains for Step 4

- **Discovery** — propose mechanisms from shared non-Gaussian residual and multi-seed edge co-presence (spec §6)
  rather than only step-2's common-child / contract heuristics; feed those into this selector.
- **Identifiability** — the theorem: when non-Gaussianity identifies the *grouping* `(A_in, A_out)`, not just edges.
- **Intervention scoring** — rank selected mechanisms by ablation support (the coin Stage-A method), the arbiter of
  causal truth per doctrine.

## Constraints honored

No stochastic search · no discovery beyond step 2 · no non-Gaussian grouping · no multi-seed aggregation · no
intervention · no training · no MetaWorld rerun · FANUC v2 / coin-collab v2b / `CORE.YAML` untouched ·
`pyproject.toml` not edited · no existing report/artifact overwritten.
