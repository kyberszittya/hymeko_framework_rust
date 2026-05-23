# Fuzzy signature view of HSIKAN — 2026-05-20

## Summary

HSIKAN factors each per-edge prediction into a sum of per-cycle
contributions; each cycle has an inherent edge-sign product
(its "vote": balanced = +1, unbalanced = −1) and a membership
weight in the query's neighbourhood (its "firing strength").
The fuzzy signature view makes these directly observable.

For a query edge $e_q = (u, v)$, the signature is the set
$\{(c, \sigma_c, \alpha_c, h_c, k_c) : M_e[e_q, c] \neq 0\}$
where $\sigma_c \in \{+1, -1\}$ is the Cartwright-Harary
balance vote ($\prod$ edge signs), $\alpha_c$ is the
attention or uniform-pool membership, $h_c$ is the per-cycle
embedding, and $k_c$ is the arity slot. Plotted as a
per-arity stacked bar over $\sigma$ plus a scatter of
individual cycles, you get a paper-grade interpretation:
"this prediction was driven by 60% balanced cycle2 cycles
voting +, 40% unbalanced walk4 walks voting −".

**Headline:** On Bitcoin Alpha with a quick 30-epoch HSIKAN
at the Optuna SOTA mix (c2,c5,w2,w3,w4), the three picked
queries each show interpretable structure that matches the
model's confidence:

| query | true | $p(+)$ | net $\sigma \cdot \alpha$ |
| --- | --- | --- | --- |
| high\_positive | +1 | 1.000 | **+4.200** |
| high\_negative | −1 | 0.002 | **−2.658** |
| decision\_boundary | −1 | 0.507 | **−1.667** (conflicted) |

The decision-boundary case is particularly informative:
**cycle5 votes balanced (+)** while **walk2/3/4 vote
unbalanced (−)** — the model is conflicted, and that
conflict is now visible.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/interpret/__init__.py` | new | 23 |
| `signedkan_wip/src/interpret/fuzzy_signature.py` | new | 332 (dataclasses + extractor + plot) |
| `signedkan_wip/src/mixed_arity_signedkan/encoding_full.py` | extended | +22 (side-channel capture hook) |
| `signedkan_wip/tests/test_fuzzy_signature.py` | new | 219 (10 unit tests) |
| `signedkan_wip/experiments/runs/demo_fuzzy_signature.py` | new | 207 (Bitcoin Alpha demo) |
| `docs/plans/2026-05-20-fuzzy-signature-view/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}` | new | 4 plan formats |
| `reports/2026-05-20-fuzzy-signature-view.md` | new | this file |
| `reports/figures/fuzzy_signature_bitcoin_alpha/{*.png,summary.json}` | new | 3 demo figures + JSON |

## CORE.YAML items touched

None.

## Interface

```python
@dataclass
class CycleContribution:
    arity: int
    arity_kind: str               # 'cycle' | 'walk'
    cycle_idx: int
    vertices: tuple[int, ...]
    sigma_assignment: tuple[int, ...]  # per-vertex σ_i (structural)
    edge_signs: tuple[int, ...]   # per-edge signs (interpretive)
    sigma_prod: int               # ±1 = Π edge_signs (the VOTE)
    balanced: bool
    membership: float             # α_c — fuzzy firing strength
    embedding: np.ndarray         # per-cycle h_c

@dataclass
class FuzzySignature:
    query_edge: tuple[int, int]
    query_idx: int
    contributions: list[CycleContribution]
    arity_alpha: np.ndarray
    arity_kinds: list[str]
    logit: float | None
    prob_positive: float | None
    # methods: vote_by_arity(), net_vote(), total_membership()

def extract_signature(model, per_arity_inputs, query_edges, query_idx,
                       arity_kinds=None, arity_edge_signs=None) -> FuzzySignature
def plot_signature(sig, axes=None)
```

## Semantic fix: σ-product is the EDGE-sign product, not the per-vertex σ product

This was caught in the first demo run. HSIKAN's per-vertex
σ_i is computed so that **Π σ_i ≡ +1 always**: each negative
edge in a cycle flips parity at exactly two vertices, so the
sum of parities is always even and the product of σ_i =
$(-1)^{\sum c_i} = (-1)^{2 n_{neg}} = +1$.

The interpretive vote — "is this cycle balanced?" — is the
Cartwright-Harary balance flag, equivalently the product of
**edge signs** (not per-vertex σ). The extractor now takes
an optional `arity_edge_signs` list and uses it to compute
`sigma_prod`. Without it, the fallback per-vertex product is
returned with a warning that the result is structurally
uninformative.

The new tests pin this: `test_sigma_prod_is_product_of_edge_signs_when_provided`
and `test_sigma_prod_fallback_warns_and_is_always_plus_one`.

## Demo results

Bitcoin Alpha, hidden=8, 30 epochs (quick — interpretability is
independent of absolute AUC), mixed arities
`(c2, c5, w2, w3, w4)`. Three picked queries from the test set.

### High-confidence positive

Edge (3730, 563), true=+, $p(+)=1.000$, net $\sigma \cdot \alpha = +4.200$:

| arity | bal vote | unbal vote |
| --- | --- | --- |
| cycle2 | 1.000 | 0.000 |
| cycle5 | 1.000 | 0.000 |
| walk2 | 1.000 | 0.000 |
| walk3 | 1.000 | 0.000 |
| walk4 | 0.600 | 0.400 |

Every arity slot is **fully balanced** except for a modest
walk4 minority (0.4 unbalanced). Model confidence is unanimous
because the rule firings are unanimous.

### High-confidence negative

Edge (4, 3699), true=−, $p(+)=0.002$, net $\sigma \cdot \alpha = -2.658$:

| arity | bal vote | unbal vote |
| --- | --- | --- |
| cycle2 | 0.000 | 1.000 |
| cycle5 | 0.684 | 0.316 |
| walk2 | 0.000 | 1.000 |
| walk3 | 0.344 | 0.656 |
| walk4 | 0.143 | 0.857 |

cycle2 and walk2 (the direct adjacency views) are **fully
unbalanced** — the model sees the direct edge as negative and
the 1-hop walks confirm. cycle5 actually leans balanced but
its αₖ weight is only 0.045 (vs 0.567 for cycle2), so it
gets out-voted.

### Decision boundary (the interesting case)

Edge (543, 3725), true=−, $p(+)=0.507$, net $\sigma \cdot \alpha = -1.667$:

| arity | bal vote | unbal vote |
| --- | --- | --- |
| cycle2 | 0.500 | 0.500 |
| cycle5 | **1.000** | 0.000 |
| walk2 | 0.000 | **1.000** |
| walk3 | 0.000 | 1.000 |
| walk4 | 0.167 | 0.833 |

The model is **internally conflicted**: cycle5 says "balanced
→ predict +", walks all say "unbalanced → predict −". The
αₖ weighting (cycle2 0.567, walk2 0.312, cycle5 0.045) tips
the net vote slightly negative but the BCE-via-logit
non-linearity produces a confidence near 0.5. The model is
honestly uncertain here, and the signature now exposes WHY.

This is the kind of failure mode that's invisible to a
black-box AUC report but legible in the fuzzy signature.

## Learned αₖ on this demo run

```
cycle2: 0.567   ← dominant
cycle5: 0.045
walk2:  0.312   ← secondary
walk3:  0.044
walk4:  0.032
```

The model relies almost entirely on direct adjacency (cycle2)
and 1-hop walks (walk2). Higher-arity cycles and walks
contribute < 5% each at this configuration. This matches the
"Bitcoin Alpha is locally signed-balance-y" intuition.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_fuzzy_signature.py` | **10 / 10 pass** |
| All prior side / mixed-arity / Phase 22 suites | no regression (13 + 12 + 22 + ...) |
| `cargo test -p hymeko_pgraph` | 96 / 96 + 1 ignored doctest |

## §6.5 anti-pattern audit

- `extract_signature` is a single function with a config-
  style kwarg surface (no Cartesian wrappers).
- The `_signature_capture` side channel mirrors the existing
  `_attn_entropy_terms` pattern — additive, opt-in via setting
  the attribute, defaults to None so it's a no-op when
  unused.
- `interpret/` is a new package with one file (332 LOC, well
  under §6.2 ceilings).
- No env-var feature flags at deep call sites; capture is
  attribute-driven from a clear caller.

Clean.

## Open follow-ups

1. **Per-vertex σ\_i view.** The current view aggregates over
   cycles. A complementary "frustration map" would show
   which vertices in the cycle are frustrated (odd negative-
   edge count) — this is the per-vertex σ_i that drives
   HSIKAN's KAN spline. A second plot mode.
2. **Embedding heatmap.** `embedding` is currently per-cycle
   but unused in the plot. Adding a PCA-2D projection or a
   t-SNE of the cycle embeddings, coloured by σ\_c, would
   show whether the model has learned to spatially separate
   balanced and unbalanced cycles. Single-figure addition.
3. **Batched-encode support.** The current extractor only
   works through `encoding_full.py`; the batched path
   (`cycle_batch_size > 0`) needs a parallel capture hook.
   Same pattern, ~15 LOC.
4. **Demo on Slashdot.** Slashdot is walk-dominant
   ([[project-edge-cr-5seed-2026-05-09]]); the αₖ ratio
   here would tell a different story (walks > cycles).
5. **Per-vertex contribution heatmap for cycles vs walks.**
   Walks vote by Π edge_signs along the walk; cycles by
   the cyclic Π. Same primitive, different semantics — worth
   a panel.

## Experiment provenance

- **Git SHA:** uncommitted (post-Phase-22 branch
  `refactor/extract-hymeko-hre`).
- **Dataset:** Bitcoin Alpha (n\_nodes=3783, n\_edges=24186).
- **Demo training:** 30 epochs at hidden=8, lr=5e-2,
  Adam, full mixed mix. Light training — interpretation
  works regardless.
- **Demo wall:** 7.3 s training + ~5 s for 3 signature
  extractions and plots. Total < 1 min including cycle
  enum (cached).
- **Figures:** `reports/figures/fuzzy_signature_bitcoin_alpha/{signature_high_positive,signature_high_negative,signature_decision_boundary}.png`
- **Seeds:** seed=0 only (interpretation, not statistics).

## Acceptance check

- [x] Plan in 4 formats on disk (tex/pdf/tikz/mmd).
- [x] CORE.YAML items touched = 0.
- [x] 10 / 10 unit tests pass.
- [x] Demo produces three PNGs + summary.json.
- [x] **Semantic correctness:** `sigma_prod` is the edge-
      sign product (Cartwright-Harary balance vote), not the
      structurally-always-+1 per-vertex product.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
