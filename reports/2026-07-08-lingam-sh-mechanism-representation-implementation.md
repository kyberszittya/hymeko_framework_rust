# LiNGAM-SH step 1 — mechanism-form causal hypergraph (representation + projection + cross-view)

**Date:** 2026-07-08 · Aiko · branch `hymeko-neuro-migration`
**Status:** implemented (representation only). **No discovery, no factorization, no training.** Spec:
`docs/plans/2026-07-08-lingam-sh-causal-hypergraph/spec.md` (§"representation first, discovery second").

---

## Summary

Extended the existing pairwise `CausalHypergraph` into a **mechanism-form causal hypergraph** — causal mechanisms
as directed hyperedges (tail set → head set) — **without changing DirectLiNGAM behaviour or the pairwise
`.hymeko` emit**. Implemented only the representation + projections + `.hymeko` emit + cross-view verification path.
Hyperedge discovery and the `B ≈ A_out Σ A_inᵀ` factorization are **not** implemented (next step).

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/causal/hymeko_emit.py` | + `Mechanism`, `StarProjection`, `AcyclicityResult`, `_topo_rank`/`_indegrees`; extended `CausalHypergraph` (mechanism field + methods); mechanism branch in `to_hymeko_source`; generalized `cross_view_verify` counts |
| `hymeko_rl/eval/causal/__init__.py` | export `Mechanism`, `StarProjection`, `AcyclicityResult` |
| `hymeko_rl/tests/test_causal_hymeko_emit.py` | +11 mechanism tests (A–H + validation) |

No other module touched. `CORE.YAML` / `pyproject.toml` untouched; no dependency added.

## New dataclasses / API

```python
@dataclass
class Mechanism:                 # a directed causal hyperedge
    name: str                    # becomes the star-expansion hub vertex
    tail: tuple[str, ...]        # input variables (non-empty, unique)
    head: tuple[str, ...]        # output variables (non-empty, unique, disjoint from tail)
    strength: float = 1.0
    sign: int = 1                # -1 | +1
    evidence: Mapping[str, Any] | None = None
    @classmethod
    def pairwise(cls, cause, effect, weight, *, name=None) -> Mechanism   # degenerate {cause}→{effect}
    @property
    def weight(self) -> float    # sign · |strength|

CausalHypergraph:                # extended, not duplicated
    mechanisms: list[Mechanism]  # NEW field; empty ⇒ pairwise form (unchanged), non-empty ⇒ mechanism form
    from_mechanisms(name, mechanisms) -> CausalHypergraph
    star_projection() -> StarProjection          # variables + hubs + directed incidence
    pairwise_projection() -> list[(cause, effect, weight)]   # tail × head per mechanism (a PROJECTION)
    to_pairwise_hypergraph() -> CausalHypergraph
    check_acyclicity() -> AcyclicityResult       # bipartite variable/mechanism DAG check
    n_binary_edges() -> int

@dataclass
class AcyclicityResult:  acyclic: bool; rank: dict|None; cycle_nodes: list|None   # __bool__ = acyclic
@dataclass
class StarProjection:   variables: list; hubs: list; incidence: list[(src, dst, sign)]
```

## How pairwise compatibility is preserved

- **`CausalHypergraph.from_lingam` is unchanged** — it produces the pairwise form (`mechanisms == []`). The
  DirectLiNGAM → `.hymeko` path is byte-identical (the pairwise branch of `to_hymeko_source` is the original code;
  `cross_view_verify` uses `n_binary_edges()`, which equals `len(edges)` for the pairwise form).
- **Degenerate equivalence is tested.** A pairwise edge `x → y` is representable as `Mechanism.pairwise(x, y, w)`;
  a degenerate mechanism graph's `pairwise_projection()` recovers the original edges, and
  `to_hymeko_source(mech.to_pairwise_hypergraph())` is **byte-identical** to the direct pairwise emit
  (`test_degenerate_mechanism_equivalent_to_pairwise`).
- **All existing CIP/coin/MetaWorld callers are unaffected** — they construct pairwise graphs via `from_lingam`
  and call `cross_view_verify`; 102 CIP tests pass unchanged.

## Emit / projection semantics (mechanism form)

- **`.hymeko` emit = star expansion.** Each mechanism becomes a hub vertex; each tail arc `tail → hub` (sign +1)
  and each head arc `hub → head` (sign = mechanism sign) is a 2-member signed hyperedge — reusing the existing
  binary-edge grammar, so the engine parses it and the cross-view machinery (binary-edge reparse, star invariant,
  Blake3 hash) applies unchanged.
- **Star projection** (`{a,b}→{c,d}`): `a→hub, b→hub, hub→c, hub→d` (contains the hub node).
- **Pairwise projection** (`{a,b}→{c,d}`): `a→c, a→d, b→c, b→d` — explicitly a *projection*, not the primary model.
- **Acyclicity** is checked on the bipartite variable/mechanism digraph (tail→mech, mech→head) via Kahn topo-sort;
  a cyclic graph returns `acyclic=False` with the trapped nodes (dynamic feedback / time-lag deferred, per spec §4).

### Example emitted mechanism `.hymeko`

`{contact_score, near_fraction} → {total_reward}` (`Mechanism(name="reward_mech", …, sign=+1)`):

```
CoinReward{}
coinreward
{
    contact_score {}
    near_fraction {}
    total_reward {}
    reward_mech {}

    @c0{ (+contact_score, +reward_mech); }
    @c1{ (+near_fraction, +reward_mech); }
    @c2{ (+reward_mech, +total_reward); }
}
```

## Test results

`pytest -p no:randomly`. **102 passed** across the CIP suite; ruff / radon (no block ≥ C, after splitting
`_topo_rank`/`_indegrees`) / mypy `--strict` clean on the changed module.

| Test (per spec §8) | Result |
| --- | --- |
| A. pairwise DirectLiNGAM path unchanged | ✅ `test_pairwise_path_unchanged_from_lingam` |
| B. pairwise edge as degenerate mechanism (byte-identical collapse) | ✅ `test_degenerate_mechanism_equivalent_to_pairwise` |
| C. multi-input single-output mechanism emits valid `.hymeko` | ✅ `test_multi_input_single_output_mechanism_emits_valid_hymeko` |
| D. multi-input multi-output → expected pairwise edges | ✅ `test_multi_output_mechanism_pairwise_projection` |
| E. star projection contains hub nodes | ✅ `test_star_projection_contains_hub_nodes` |
| F. acyclic mechanism graph passes | ✅ `test_acyclic_mechanism_graph_passes` |
| G. cyclic mechanism graph fails | ✅ `test_cyclic_mechanism_graph_fails` |
| H. cross-view passes for a small mechanism graph | ✅ `test_multi_input_…`, `test_delivery_mechanism_cross_view` |
| I. existing CIP/coin/MetaWorld tests still pass | ✅ 102 passed |
| (extra) negative sign survives cross-view; self-cycle / hub-collision rejected | ✅ |

## Cross-view verification

**Passes for the mechanism form.** A mechanism graph emits to `.hymeko`, star-expands, is compiled by the native
`hymeko` engine, and the declared star-incidence signed-edge set equals the engine-reparsed set with the star
invariant satisfied and a canonical Blake3 hash present (`test_multi_input_…`, `test_delivery_mechanism_cross_view`,
`test_negative_mechanism_sign_survives_cross_view`). The existing **pairwise** cross-view is not weakened (same
code path, 102 tests green).

## What remains for the next step (not this task)

- **Hyperedge discovery** — propose candidate mechanisms from common parent/child sets, reward-term decomposition,
  monitor-contract structure, shared non-Gaussian residual, multi-seed edge co-presence (spec §6).
- **Factorization** — `B ≈ A_out Σ A_inᵀ`, the group-sparse rank-1-per-mechanism fit (spec §5).
- **Scoring** — presence/IQR/reconstruction/stability/interpretability/intervention (spec §7).
- Wiring the CIP consumers (`contact_reward_ablation`, `metaworld_cip`) to *optionally* emit mechanism-form graphs
  once discovery exists — the representation is ready; nothing consumes it as primary yet.

## Constraints honored

No training · FANUC v2 untouched · coin-collab v2b untouched · MetaWorld monitors not modified · `CORE.YAML`
untouched · `pyproject.toml` not edited · no existing report/artifact overwritten.
