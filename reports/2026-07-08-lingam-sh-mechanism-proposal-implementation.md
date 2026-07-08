# LiNGAM-SH step 2 — deterministic mechanism proposal

**Date:** 2026-07-08 · Aiko · branch `hymeko-neuro-migration`
**Status:** implemented (deterministic proposal only). **No factorization, no search, no intervention, no
training.** Spec: `docs/plans/2026-07-08-lingam-sh-causal-hypergraph/spec.md` §6–§7. Builds on step 1
(`reports/2026-07-08-lingam-sh-mechanism-representation-implementation.md`).

---

## Summary

Added a **deterministic** mechanism-proposal layer that turns existing pairwise causal evidence + explicit
task/reward/monitor structure into candidate `Mechanism` hyperedges, and assembles them into a mechanism-form
`CausalHypergraph` (which star-expands and cross-view-verifies, from step 1). No stochastic search, no
`B ≈ A_out Σ A_inᵀ` factorization — those remain the next step.

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/causal/mechanism_proposal.py` | **new** — `MechanismProposal` + `propose_common_child` / `propose_reward_terms` / `propose_monitor_contract` / `proposals_to_causal_hypergraph` |
| `hymeko_rl/eval/causal/__init__.py` | export the proposal API |
| `hymeko_rl/tests/test_mechanism_proposal.py` | **new** — 10 tests (spec §8) |

No existing module modified. `CORE.YAML` / `pyproject.toml` untouched; no dependency added.

## Proposal API

```python
@dataclass(frozen=True)
class MechanismProposal:
    name: str; tail: tuple[str, ...]; head: tuple[str, ...]
    strength: float; sign: int; confidence: float
    source: str                       # "common_child" | "reward_terms" | "monitor_contract"
    evidence: Mapping[str, Any]
    def to_mechanism(self) -> Mechanism

propose_common_child(edges, *, min_parents=2, name_prefix="mech_to") -> list[MechanismProposal]
propose_reward_terms(terms, output="total_reward", *, weights=None, name="reward_mechanism") -> MechanismProposal
propose_monitor_contract(monitor_terms, output="monitor_pass", *, weights=None, name="monitor_mechanism") -> MechanismProposal
proposals_to_causal_hypergraph(variables, proposals, *, name) -> CausalHypergraph
```

**Deterministic scoring** (spec §7, simple form):
- `strength` = **mean absolute** supporting edge weight (`1.0` for explicit reward/monitor proposals without weights);
- `sign` = **sign of the summed** supporting weights (ties / all-zero → `+1`);
- `confidence` = **coverage ratio** `len(tail) / (n_variables − 1)` for common-child, clamped to `[0,1]`; **`1.0`**
  for explicit reward/monitor-contract proposals;
- `source` records the proposal type; `evidence` records the supporting edges / declared term set.
- Ordering is deterministic (children and parents sorted by name), so the same input yields an identical proposal.

## Example proposed mechanisms

**Common-child** (grouping the multi-seed coffee-push evidence):
```
edges: near_fraction→total_reward (0.96), progress_score→total_reward (0.80)
proposal: name=mech_to_total_reward  tail=(near_fraction, progress_score)  head=(total_reward,)
          strength=0.88  sign=+1  confidence=1.0  source=common_child
          evidence={child: total_reward, supporting_edges: [[near_fraction,total_reward,0.96],
                                                             [progress_score,total_reward,0.8]]}
```
**Reward-terms:** `{near_fraction, contact_score} → {total_reward}` (source `reward_terms`, confidence 1.0).
**Monitor-contract:** `{delivery_score, progress_score, stagnation_duration} → {monitor_pass}` (source
`monitor_contract`, confidence 1.0).

## How proposals map to a mechanism-form `CausalHypergraph`

`proposals_to_causal_hypergraph(variables, proposals, name=...)` materializes each proposal via `to_mechanism()`
and builds a mechanism-form `CausalHypergraph` (variables = the caller's set **unioned** with every referenced
tail/head, so a referenced variable can't be accidentally omitted). It then reuses the step-1 machinery unchanged:
star expansion → `.hymeko` emit → cross-view. The `{near_fraction, progress_score} → {total_reward}` proposal
emits:

```
CoinReward{}
coinreward
{
    near_fraction {}
    progress_score {}
    total_reward {}
    mech_to_total_reward {}

    @c0{ (+near_fraction, +mech_to_total_reward); }
    @c1{ (+progress_score, +mech_to_total_reward); }
    @c2{ (+mech_to_total_reward, +total_reward); }
}
```

## Test results

`pytest -p no:randomly`. **112 passed** across the CIP suite (10 new proposal tests + the unchanged 102);
ruff / radon (no block ≥ C) / mypy `--strict` clean on the new module.

| Test (spec §8) | Result |
| --- | --- |
| 1. common-child groups two parent edges into one mechanism | ✅ `test_common_child_groups_two_parents` |
| 2. single parent not multi-input unless allowed (`min_parents`) | ✅ `test_single_parent_not_grouped_by_default` |
| 3. signs/strengths deterministic | ✅ `test_signs_and_strengths_deterministic` |
| 4. reward-term → `{terms} → {total_reward}` | ✅ `test_reward_term_proposal` |
| 5. monitor-contract → `{terms} → {monitor_pass}` | ✅ `test_monitor_contract_proposal` |
| 6. proposals → mechanism-form `CausalHypergraph` | ✅ `test_proposals_convert_to_mechanism_hypergraph` |
| 7. result passes `check_acyclicity` | ✅ `test_proposed_graph_is_acyclic` |
| 8. result passes `cross_view_verify` | ✅ `test_proposed_graph_cross_view_passes` |
| 9. pairwise projection recovers the grouped edges | ✅ `test_pairwise_projection_recovers_grouped_edges` |
| 10. existing pairwise/CIP tests still pass | ✅ 102 unchanged |

## Cross-view verification

**Passes for proposed mechanism graphs.** A proposal-derived mechanism graph star-expands, is compiled by the
native `hymeko` engine, and its declared star-incidence set matches the engine reparse with the star invariant +
a Blake3 hash (`test_proposed_graph_cross_view_passes`). The pairwise path is unchanged.

## What remains before full LiNGAM-SH factorization

- **`B ≈ A_out Σ A_inᵀ`** — the group-sparse rank-1-per-mechanism factorization of the pairwise coefficient matrix
  (spec §5); the proposals here are the *initialization / candidate set* for that fit, not the fit itself.
- **Richer discovery** — shared non-Gaussian residual grouping, and multi-seed edge co-presence as a proposal
  source (spec §6); currently common-child uses a single edge set.
- **Scoring §7 (full)** — reconstruction-improvement of `B` and cross-seed stability as scores (beyond the simple
  coverage/median form here); intervention support when available.
- **Wiring** — let the CIP consumers optionally emit proposal-derived mechanism graphs; nothing consumes proposals
  as primary yet (representation + proposal are ready, discovery/fit are not).

## Constraints honored

No factorization · no stochastic search · no intervention logic · no training · FANUC v2 untouched · coin-collab
v2b untouched · `CORE.YAML` untouched · `pyproject.toml` not edited · no existing report/artifact overwritten.
