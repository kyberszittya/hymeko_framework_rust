# HymeKo-Gömb — overview

**Gömb** is HyMeKo's three-shell cascade architecture for signed-graph
prediction. It composes three architecturally-orthogonal layers — each
adding an independent **inductive bias** — over a shared signed
hypergraph representation. The shells are:

1. **OuterFIRShell** — a Clifford-algebra-based **Finite Impulse Response
   (FIR)** kernel bank that filters signed neighborhoods through `M`
   geometric kernels. Operates in a graded multivector space; the
   "outer" name refers to the *outer product* / wedge structure of
   geometric algebra.
2. **MiddleHSiKAN** — the **Hypergraph Signed Kolmogorov–Arnold
   Network** middle layer. Catmull–Rom (or B-spline / Kochanek–Bartels)
   splines per sign branch over per-arity cycle σ-products. This is
   the *learnable activation* layer.
3. **InnerCPMLCore** — the **Capsule Path-routed Multi-Layer** core. A
   tier-stratified capsule router with optional Highway-residual,
   Capsule-routed, or KAN-routed paths. Produces the final per-edge
   embedding consumed by the classification head.

The three shells are deliberately **orthogonal in design space**:
geometric-algebra (FIR) × spline-activation (HSiKAN) × capsule-routing
(CPML). See the comprehensive treatment in
[Orthogonal neural dimensions](../research/orthogonal-neural-dimensions.md)
for how each axis is independently variable.

## Why "Gömb"

`Gömb` is Hungarian for *sphere* — chosen because the three shells form
a concentric architecture: outermost geometric filtering, middle
spline-activation, innermost capsule routing. The outer-to-inner
information flow mirrors a spherical wave-front collapsing onto a focal
classifier.

## Architectural anchor: cycle σ-products on signed hypergraphs

All three shells consume the same fundamental object: **per-cycle
σ-products** over a signed hypergraph `H = (V, E, σ)`. For a k-cycle
`c = (v_1, …, v_k)` with edge signs `σ_i ∈ {±1}`:

$$ \pi(c) = \prod_{i=1}^{k} \sigma_i $$

`π(c) = +1` iff the cycle is **structurally balanced**
(Cartwright-Harary 1956); the σ-product features encode this balance
directly.

The three shells transform the same per-cycle σ-products differently:

| Shell | Transform | Inductive bias |
| --- | --- | --- |
| Outer (FIR) | Clifford-algebra graded-product filtering | geometric / scale-equivariant |
| Middle (HSiKAN) | Per-sign-branch spline activation | learnable non-linearity per arity |
| Inner (CPML) | Tier-stratified capsule routing | hierarchical evidence aggregation |

## Strict-by-construction protocol

Gömb's cycle pool enumerates over **training edges only**
(`enumerate_top_k_cycles_rs(e_tr, s_tr, ...)` in
[run_gomb_smoke.py](../../../../signedkan_wip/src/run_gomb_smoke.py)). Test
edges never participate in σ-products. This means:

- No transductive σ-leakage (which the canonical Bitcoin/Slashdot
  signed-link convention silently permits).
- Label-shuffle on training edges drops AUC to chance (`test_AUROC =
  0.5402` on Bitcoin Alpha — see
  [strict-protocol-benchmark](./strict-protocol-benchmark.md)).
- The reported AUC numbers reflect **honest architectural learning**,
  not protocol-permitted feature leakage.

This makes Gömb the architecturally-clean reference for signed-link
prediction — distinct from HSiKAN's transductive setup, which inherits
the canonical-but-leaky convention.

## Variants

The [`signedkan_wip/src/hymeko_gomb/`](../../../../signedkan_wip/src/hymeko_gomb/)
package exposes four ablation classes:

- `HymeKoGomb` — full three-shell cascade.
- `GombNoOuter` — Middle-HSiKAN + InnerCPML only.
- `GombNoMiddle` — OuterFIR + InnerCPML.
- `GombNoInner` — OuterFIR + Middle-HSiKAN.

Plus two mix-fusion variants:

- `MixedArityGomb` — multiple cycle arities (k=3..6) without
  cross-arity α-fusion.
- `JointMixGomb` — c3 + c4 + w2 + w3 stacks fused via learned α
  (the standard 5-seed configuration).

## Cross-references

- [Three-shell strict-protocol 5-seed benchmark](./strict-protocol-benchmark.md)
  — the 2026-05-14 paper-table numbers.
- [HSiKAN architecture](../research/hsikan.md) — Middle shell details.
- [CPML routes: Highway · Capsule · KAN](../research/cpml-routing-highway-capsule-kan.md)
  — Inner shell routing options.
- [Orthogonal neural dimensions](../research/orthogonal-neural-dimensions.md)
  — comprehensive treatment of why the three shells compose.
- [Gömb-orthogonal: meanings](../research/gomb-orthogonal.md) —
  earlier exploration of the "orthogonal" framing across compute,
  loss, and architecture axes.
