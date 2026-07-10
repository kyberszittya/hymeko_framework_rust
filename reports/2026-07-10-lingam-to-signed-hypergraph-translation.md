# LiNGAM → signed-HSiKAN operator bridge (operator-compatibility result)

**Date:** 2026-07-10 · Aiko · branch `hymeko-neuro-migration` · CPU. An **operator-compatibility** result and its
implementation: the DirectLiNGAM linear-SEM operator `B` splits by sign into HyMeKo's signed causal adjacency, and
the one-step linear operator embeds as a restricted HSiKAN/SignedKAN layer. Code:
`hymeko_rl/eval/causal/hsikan_mechanism.py`; tests: `hymeko_rl/tests/test_hsikan_mechanism.py`.

> **Scope.** This increment does **not** complete the structural-leverage experiment. It establishes an *operator
> compatibility* result — a principled bridge from a discovered linear causal operator to the HSiKAN input — and
> nothing about empirical advantage. The falsification (harness + directed scramble) is a separate, later step.

## The model

DirectLiNGAM (`hymeko_rl/eval/causal/lingam.py`) estimates a linear structural equation model
`x = B x + e` with the convention `B[effect, cause]` and non-Gaussian `e` — a proposed signed, weighted DAG
(doctrine: LiNGAM proposes, controlled ablations decide).

## The translation (weighted / faithful)

Split `B` by sign:

    A⁺ = max(B, 0)   (excitatory causes)
    A⁻ = max(−B, 0)  (inhibitory causes)
    B  = A⁺ − A⁻

`A⁺`, `A⁻` are nonnegative with disjoint support; `A⁺ − A⁻` reconstructs `B` exactly (tested). This matches the
signed causal adjacency convention used by the HSiKAN/SignedKAN operator when `dst = effect` and `src = cause` —
the same `B[effect, cause]` convention DirectLiNGAM's `adjacency` and `build_pairwise_b` already use, so no
re-indexing. Sign **and** magnitude are preserved. `signed_adjacency_split(B, min_abs, row_normalize)` implements it
(`min_abs` prunes tiny coefficients; `row_normalize` optionally matches HSiKAN's normalised convention, off by
default so the reconstruction is exact); `lingam_to_signed_adjacency(LingamResult)` is the wrapper.

## Allowed statement (operator compatibility)

> The one-step linear LiNGAM operator can be embedded as a restricted HSiKAN/SignedKAN layer when using the raw
> signed split, identity activation, zero self term, and matching linear weights. This gives a principled bridge:
> LiNGAM can propose a signed causal operator, and HSiKAN can model nonlinear/multi-hop functions over that operator.

Supporting fact (tested): with `φ = id`, `W_self = 0`, `W₊ = W₋ = I`, and the raw split, a SignedKAN layer evaluates
to `h' = A⁺x − A⁻x = B x`; the operator identity `(A⁺ − A⁻)x = B x` is a passing test.

## Explicit non-claims

This increment does **not** claim, and the code/report must not be read to claim:

- HSiKAN empirically beats LiNGAM.
- HSiKAN generally contains the full LiNGAM SEM solution.
- Structural leverage is proven.
- Causal discovery is proven.
- Any RL advantage is proven.
- This replaces the directed-scramble ablation.
- This replaces the Stage 0 / Stage 1 falsification pilot.

## Two paths — and they are not equivalent

| path | function | preserves | use |
|---|---|---|---|
| **weighted signed operator** | `lingam_to_signed_adjacency` → `(A⁺, A⁻)` | sign **+ magnitude**; `A⁺−A⁻ = B` | feed `SignedKANBackbone` directly — the faithful operator bridge |
| **topological / hypergraph star** | `causal_hg_to_structure` → `HypergraphState` | sign + mechanism-hub structure | HyMeKo representation + cross-view verification |

The **weighted** path is the faithful operator bridge. The **topological** path encodes mechanism structure
(`{tail}→{head}` hubs, star expansion) and is useful for HyMeKo representation and cross-view verification, but it
routes through `dense_signed_adj` (row-normalised arc counts) and therefore does **not** preserve the full weighted
LiNGAM operator unless weights/signs are explicitly carried through.

## Primitives in this increment

- `signed_adjacency_split` / `lingam_to_signed_adjacency` — the weighted signed split (this file's subject).
- `causal_hg_to_structure` / `node_features_from_frame` — the topological bridge + sink-masked feature layout.
- `scramble_directed_signed_incidence` (`hymeko_rl/experiments/incidence_scramble.py`) — degree/sign-preserving
  **directed** scramble of `(A⁺, A⁻)`, the H2 decider for the (later) harness. Directed because a causal graph is
  directed, unlike the Stage-0 symmetric scramble.

## Status

Bridge + translation + directed scramble implemented and tested (mechanism + directed-scramble suites green; 105
existing CIP causal tests still green; ruff + mypy clean). **No harness, no experiment run.**

## Next (separate step, not this commit)

Build a synthetic-nonlinear-SEM harness (`reports/2026-07-10-hsikan-lingam-operator-harness.md`) that asks whether
HSiKAN over the LiNGAM-derived signed operator improves modelling over baselines **only when** the data-generating
process is nonlinear over a meaningful signed structure — comparing a linear-SEM/LiNGAM-loadings predictor, a
params-matched MLP, a DeepSets/bag baseline, HSiKAN over raw `(A⁺, A⁻)`, and HSiKAN over the **scrambled** operator.
Expected: linear baseline matches HSiKAN on linear-SEM data (no fake win); HSiKAN's advantage appears only on
nonlinear-over-structure data and **collapses under the directed scramble**; no advantage on flat/irrelevant
structure. That returns to the actual H1 (advantage ≈0 on flat/linear, grows with structure-rich nonlinear
mechanisms) and H2 (scramble collapses it). On the Mac first (synthetic); real coffee-push frames are a kato15 job
(`metaworld` is not installed on this Mac).
