# LiNGAM-SH step 3A — deterministic mechanism factorization `B ≈ A_out · Σ · A_inᵀ`

**Date:** 2026-07-08 · Aiko · branch `hymeko-neuro-migration`
**Status:** implemented (deterministic evaluation over a **fixed** candidate set). **No search, no optimization,
no discovery, no DirectLiNGAM change, no training.** Spec: `docs/plans/2026-07-08-lingam-sh-causal-hypergraph/spec.md`
§5. Builds on step 1 (representation) + step 2 (proposals).

---

## Summary

Given the pairwise LiNGAM coefficient matrix `B` and a fixed set of proposed mechanisms (step 2), evaluate the
signed-incidence factorization `B ≈ A_out · Σ · A_inᵀ`: each mechanism is a **rank-1 block** that contributes its
`sign · strength` to every tail×head entry of `B_hat`. The structure (`A_in`, `A_out`) is the proposal support and
`Σ` comes straight from the proposal scores — this is the **deterministic evaluation** Step 3B will optimize/select
over. It changes **no** existing pairwise behaviour.

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/causal/mechanism_factorization.py` | **new** — `MechanismFactorization`, `build_pairwise_b`, `factorize_from_proposals`, `score_mechanism_set` |
| `hymeko_rl/eval/causal/__init__.py` | export the factorization API |
| `hymeko_rl/tests/test_mechanism_factorization.py` | **new** — 11 tests (spec §8) |

No existing module modified. `CORE.YAML` / `pyproject.toml` untouched; no dependency added.

## Factorization API

```python
@dataclass(frozen=True, eq=False)          # eq disabled: fields are np.ndarray
class MechanismFactorization:
    variables: tuple[str, ...]
    mechanisms: tuple[MechanismProposal, ...]
    a_in: np.ndarray                         # (d, m)
    a_out: np.ndarray                        # (d, m)
    sigma: np.ndarray                        # (m, m) diagonal
    b_hat: np.ndarray                        # (d, d)
    residual: np.ndarray                     # (d, d) = B - B_hat
    metrics: Mapping[str, float]
    def to_causal_hypergraph(name) -> CausalHypergraph   # → step-1 mechanism graph (star-expand + verify)

build_pairwise_b(variables, pairwise_edges) -> np.ndarray            # B[effect, cause] = weight (duplicates sum)
factorize_from_proposals(variables, pairwise_edges, proposals, *, normalize=True) -> MechanismFactorization
score_mechanism_set(b, b_hat, *, n_mechanisms, n_parameters) -> dict[str, float]
```

## Exact matrix convention

- **`B[effect, cause] = weight`** — rows are effects, columns are causes (matches DirectLiNGAM `adjacency`).
- **`A_in[i, k] ≠ 0` iff variable `i` is a TAIL of mechanism `k`**; **`A_out[j, k] ≠ 0` iff `j` is a HEAD of `k`**.
- **`Σ[k, k] = sign_k · strength_k`** (a diagonal matrix). `B_hat = A_out · Σ · A_inᵀ`, so
  `B_hat[j, i] = Σ_k A_out[j,k] · Σ[k,k] · A_in[i,k]` = `sign_k·strength_k` for every `(i∈tail_k, j∈head_k)`.
- **`normalize=True`** unit-scales the incidence columns (`A_in[:,k] = 1/√|tail|`, `A_out[:,k] = 1/√|head|`, and
  `Σ` absorbs `√(|tail||head|)`) — a pure representation choice; **`B_hat` is identical either way** (tested).
- **Overlapping mechanisms sum** into shared `(effect, cause)` entries (spec §8, tested).
- `B_hat`'s nonzero support **is** the pairwise projection of the mechanism graph (spec §7).

## Worked example (coffee-push multi-seed evidence)

`edges = near_fraction→total_reward (0.96), progress_score→total_reward (0.80)`;
one proposed mechanism `{near_fraction, progress_score} → {total_reward}`, `strength=0.88, sign=+1`
(variables ordered `near_fraction, progress_score, total_reward`; `normalize=False` shown):

```
B =            A_in =     A_out =    Σ =        B_hat =
[[0    0    0 ] [[1]      [[0]       [[0.88]]   [[0    0    0 ]
 [0    0    0 ]  [1]       [0]                   [0    0    0 ]
 [0.96 0.80 0 ]] [0]]      [1]]                  [0.88 0.88 0 ]]
```

**Reconstruction metrics:**

| metric | value |
| --- | --- |
| `fro_error` | 0.1131 |
| `fro_error_baseline` (‖B‖, B_hat=0) | 1.2496 |
| `relative_error` | 0.0905 |
| `explained_energy` | **0.9918** |
| `n_mechanisms` / `n_parameters` | 1 / 1 |
| `bic_like_score` | −56.80 |

A single 1-parameter mechanism explains **99.2%** of `B`'s energy — the multi-input reward mechanism is a strong
rank-1 summary of the two pairwise edges (residual is the ±0.08 spread between 0.96/0.80 and the 0.88 mean).

## Test results

`pytest -p no:randomly`. **123 passed** across the CIP suite (11 new + the unchanged 112); ruff / radon (no block
≥ C) / mypy `--strict` clean on the new module.

| Test (spec §8) | Result |
| --- | --- |
| 1. `build_pairwise_b` respects `B[effect,cause]` | ✅ |
| 2. degenerate `{x}→{y}` reconstructs a single edge | ✅ |
| 3. multi-input single-output block | ✅ |
| 4. multi-input multi-output block | ✅ |
| 5. empty proposals → `B_hat=0`, baseline residual | ✅ |
| 6. matching proposal improves reconstruction | ✅ |
| 7. wrong proposal worse than matching | ✅ |
| 8. overlapping mechanisms sum deterministically | ✅ |
| 9. → mechanism-form `CausalHypergraph` | ✅ |
| 10. mechanism graph passes `cross_view_verify` | ✅ |
| 11. existing Step 1/2/CIP tests still pass | ✅ (112 unchanged) |
| (extra) `normalize` leaves `B_hat` invariant | ✅ |

## Cross-view verification

**Passes.** `factorization.to_causal_hypergraph(name)` yields the step-1 mechanism graph, which star-expands, is
compiled by the native `hymeko` engine, and cross-view-verifies (declared ≡ tensor, Blake3 hash) —
`test_factorization_converts_to_hypergraph_and_cross_view`. It also passes `check_acyclicity`.

## Did any existing pairwise behaviour change? — No

This is a **new, additive** module; `build_pairwise_b` reads a matrix and nothing writes back into DirectLiNGAM,
`CausalHypergraph`, or the pairwise emit. The 112 prior CIP tests pass unchanged.

## What remains for Step 3B / Step 4

- **Step 3B — selection / group-sparse fit.** This step *evaluates* a fixed proposal set; 3B should **select** the
  subset that best trades reconstruction against complexity (the `bic_like_score` / MDL criterion is the deterministic
  seed for that), and optionally solve for `Σ` by (still-deterministic, closed-form) least squares given fixed
  `A_in/A_out` instead of taking `sign·strength` verbatim.
- **Step 4 — discovery + identifiability.** Propose mechanisms from shared non-Gaussian residual and multi-seed
  edge co-presence (spec §6), and the identifiability theorem (when non-Gaussianity identifies the *grouping*).
- **Intervention scoring** — rank confirmed mechanisms by ablation support (the coin Stage-A method), the ultimate
  arbiter per doctrine §9.

## Constraints honored

No factorization search / stochastic optimization / group-sparse search · no intervention logic · no training ·
no MetaWorld rerun (static synthetic fixtures only) · FANUC v2 / coin-collab v2b untouched · `CORE.YAML` untouched
· `pyproject.toml` not edited · no existing report/artifact overwritten.
