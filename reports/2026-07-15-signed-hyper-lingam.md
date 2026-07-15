---
title: "SignedHyperLiNGAM — native signed-hyperedge causal discovery (vs DirectLiNGAM)"
date: 2026-07-15
author: Aiko (Opus 4.8)
branch: feat/spec-reward-close-loop
stage: thread-B-signed-hyper-lingam
status: COMPLETE — ties on additive SEMs, WINS on joint-interaction mechanisms pairwise LiNGAM misses
tags: [causal, lingam, signed-hypergraph, hymeko_pgraph, hsikan, discovery]
---

# SignedHyperLiNGAM — causal discovery of joint (hyperedge) mechanisms pairwise LiNGAM misses

**[2026-07-15 19:28 JST]**

## Summary

DirectLiNGAM recovers a *pairwise* linear SEM `x = Bx + e`; the CIP pipeline then post-hoc splits `B = A⁺ − A⁻`
and groups pairwise edges into mechanism hyperedges. **SignedHyperLiNGAM** makes the causal primitive **native**:
for each effect (in DirectLiNGAM's non-Gaussian causal order — *reused*, not re-implemented), it selects the
effect's **signed-hyperedge tail** — the subset of prior variables that *jointly* produce it — by reducing
tail-selection to the **same `hymeko_pgraph` SSG solution-structure search the spec arbiter uses**, scored by an
**interaction-aware** fit, then signs each member (`A⁺`/`A⁻`, or *joint* when the linear part is ≈0). The result is
emitted directly as a `CausalHypergraph` (HyMeKo IR) — no matrix to translate. It runs **alongside** DirectLiNGAM
so the two are tested head-to-head (the user's "both alive, so we can test them").

**The measured result (§3 — measured, not asserted; 12 seeds, 6-var DAGs, median [IQR] support-recall):**

| SEM regime | DirectLiNGAM | SignedHyperLiNGAM | verdict |
|---|--:|--:|---|
| linear | 1.00 [1.0, 1.0] | 1.00 [1.0, 1.0] | **TIE** |
| additive-nonlinear (per-edge monotone) | 1.00 [1.0, 1.0] | 0.94 [0.8, 1.0] | slight DirectLiNGAM edge |
| **joint-interaction** (`x_i = x_a·x_b + …`) | **0.73** [0.6, 1.0] | **1.00** [0.71, 1.0] | **SignedHyperLiNGAM WINS** |

The win is the contribution: on a **joint-interaction** mechanism a parent's *marginal* effect vanishes
(`corr(x_a, x_a·x_b) ≈ 0` for independent `x_a,x_b`), so pairwise DirectLiNGAM **structurally** cannot see it
(recall 0.73), while SignedHyperLiNGAM scores the *subset jointly* and recovers the hyperedge `{x_a,x_b}→x_i`
(recall 1.00). This is exactly **"signed hypergraphs capture joint causal structure pairwise misses"** — the
HyMeKo premise, demonstrated on causal discovery. Figure:
`reports/figures/2026_07_15_19_27_signed_hyper_lingam/signed_hyper_lingam.png`.

**Honest scope (do not overclaim):**
- On **additive** SEMs the two **tie** (linear) or DirectLiNGAM is *slightly* ahead (additive-nonlinear, 0.94 vs
  1.0). DirectLiNGAM already recovers additive support perfectly — **no headroom**; the 0.94 is a minor cost of
  SignedHyperLiNGAM's interaction-aware parsimony occasionally dropping a weak monotone edge. The win is **only**
  where mechanisms are genuinely **non-additive**.
- The ordering is **reused from DirectLiNGAM**, so SignedHyperLiNGAM inherits its non-Gaussian ordering
  assumptions; a mechanism whose non-additivity breaks the ordering caps recall regardless of tail selection (the
  joint SEM keeps mechanisms mild enough that the order holds — reported, not hidden).
- **Doctrine unchanged:** it PROPOSES structure; ablations decide. Observational discovery, not proof.

## Files (new, non-core; worktree `feat/spec-reward-close-loop`)

| file | LOC | note |
|---|---:|---|
| `hymeko_rl/eval/causal/signed_hyper_lingam.py` | 176 | `SignedHyperLiNGAM`, `SignedHyperResult` (→ signed `B`-form + `CausalHypergraph`), `sample_interaction_sem` |
| `hymeko_rl/experiments/exp_signed_hyper_lingam.py` | 133 | 3-regime head-to-head vs DirectLiNGAM (`recovery_metrics`); JSON + recall plot |
| `hymeko_rl/tests/test_signed_hyper_lingam.py` | 118 | 10 tests |

Reused (no re-implementation, §6.1): `DirectLiNGAM._search_causal_order`/`.fit`, `solve_ssg` +
`predicates_to_pgraph_hymeko` (the arbiter's SSG reduction), `recovery_metrics`, `generate_signed_dag`/`sample_sem`
(harness), `proposals_to_causal_hypergraph`/`CausalHypergraph`. No §6.5 anti-patterns.

## CORE.YAML items touched

**None.** No new dependency (`pgraph` binary already built; numpy/scipy present).

## Test results

`test_signed_hyper_lingam.py`: **10 passed** (3.5 s) — ordering-reuse, shape/finite validation, additive-tie
(no false win), the **joint-win regression** (`median(SHL) > median(DirectLiNGAM) + 0.1`), interaction-SEM
non-additivity (marginal `<0.1`, product `>0.5`), additive-edge sign correctness, `CausalHypergraph` acyclicity,
frozen-config, and the head-to-head runner smoke (joint = `SignedHyperLiNGAM_WINS`). `ruff`: clean.
`mypy --strict`: clean (both source files). `radon cc -a -nc`: no function at rank C or worse (all A/B) — §6.2 gate
passes.

## Performance

- `SignedHyperLiNGAM.fit` at d=6, n=3500: well under 1 s (SSG subset search over ≤5 predecessors +
  interaction-R²). Full 12-seed × 3-regime head-to-head < 30 s CPU. Peak RSS < 0.5 GB.

## Experiment provenance

- Worktree `hymeko_spec_reward_wt` on `feat/spec-reward-close-loop`; Apple-Silicon Mac, `.venv` cpython-3.11,
  numpy 2.x, the built `target/debug/pgraph` (symlinked into the worktree). Seeds 0–11; 6-var DAGs, density 0.4,
  n_samples 3500.

## Open issues / follow-ups

1. **Additive-nonlinear tie:** add the per-column monotone basis (`tanh`/signed-square/softsign) to the
   interaction-R² so SignedHyperLiNGAM also captures purely-monotone edges and closes the 0.94→1.0 gap — a
   principled improvement, deferred to avoid over-tuning against the joint result.
2. **HSiKAN over the discovered hyperedge:** the natural downstream — fit `build_hsikan_operator` over
   SignedHyperLiNGAM's `(A⁺,A⁻)` to *model* the joint mechanism (the `hsikan_mechanism` doctrine); wire the sink-R²
   comparison vs the linear operator.
3. **Thread A (compositional-spec benchmark)** remains staged in the same worktree (WIP re-applied) — the second
   live thread.
