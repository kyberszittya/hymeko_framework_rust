---
name: project-fuzzy-defuzzification-heads
description: "Revisit-soon idea (Dr. Hajdu, 2026-06-17): the rotor/KAN readout heads are structurally defuzzification — unify the head design under a fuzzy-signature lens. softplus/σ belong on heads (positivity/membership on an emitted crisp scalar), NOT inside manifold propagation (scale-free → exp/gate)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 1e9da94f-88c9-4ee6-b9d5-3692a874c26a
---

Dr. Hajdu's observation while reviewing the rotor-propagation self-weight
parameterisation (softplus→exp switch, 2026-06-17): there are **many structural
relationships between our readout heads and fuzzy / fuzzy-signature theory**, and
this is worth a dedicated revisit soon (not now — we're pushing the rotor numbers
first).

**The parallel, sharpened.** A readout has two distinct moments:
1. **Defuzzifying collapse** = the attention/pool step (σ-weighted triad mean,
   centroid-style) that collapses a distribution-over-members to one crisp value.
   This is defuzzification *proper* (centroid / mean-of-maxima), NOT softplus.
2. **Constraint activation** = softplus / σ shaping the *emitted* scalar: a
   positive magnitude (rate, concentration, temperature, variance) via softplus,
   or a [0,1] membership via σ. This is where softplus legitimately lives.

Inside `SignedRotorPropagation` there is **no scalar emitted and no distribution
collapsed** — only normals interpolated and renormalised on S³ — so the self-weight
is a scale-free ratio → log-space `exp(θ)` (≡ a sigmoid residual gate), and neither
defuzzification nor softplus belongs there. The placement rule: defuzzification +
positivity-activation at the **head**; manifold interpolation + gate **inside**.

**Why revisit.** The project already has a fuzzy line that is essentially a set of
defuzzification heads: `fuzzy-signature-layer`, `balance-as-activation`,
`fuzzy-pose-detection` (BACKLOG "Architectures / fuzzy / sequence"). Framing the
rotor/KAN heads and these fuzzy heads under **one defuzzification-head abstraction**
could (a) unify the readout design (a `DefuzzHead` Strategy: pool/collapse → membership/
magnitude activation), (b) connect the signed-balance triad pool to Davis weak
balance as a fuzzy membership, (c) reuse one positivity/membership convention across
the rotor line and the fuzzy-signature layer.

**How to apply.** When the fuzzy-signature / balance-as-activation backlog items are
picked up, design them and the rotor-line heads (`bilinear`, geom-attn pool,
`RotorRelativeHead`) against a shared defuzzification-head contract rather than
ad-hoc per-head. Until then this is a *lens*, not a task. The empirical rotor work
([[project-hsikan-geometric-attention-berge]]) showed the readout algebra is NOT the
AUROC bottleneck (real bilinear ≥ complex/quat/geodesic) — so a fuzzy-defuzzification
reframe is about *design unification / interpretability*, not expected AUROC gain.
See also [[project-cayley-rotor-idea]].

**NOW A CONCRETE PLAN (2026-06-25):** `docs/plans/2026-06-25-fuzzy-cr-outer-layer/` (4 artifacts, built+validated).
An **explicit CR-membership fuzzy outer layer** for HSiKAN & Gömb: fuzzify via learnable Catmull-Rom membership
functions (μ_fj = clip[0,1] CR(c_fj, x_f), reusing `signed_kan.catmull_rom`) → T-norm rule firing + σ-vote →
defuzzify (Sugeno centroid) readout. Key grounding: `interpret/fuzzy_signature.py` (FuzzySignature) already reads
HSiKAN AS a fuzzy rule base (per-cycle σ-vote × membership α → net_vote=Σσ_c·α_c); the plan makes that EXPLICIT +
LEARNABLE in the forward pass, so the signature reads exact memberships (not reconstructed). New module
`signed_kan/fuzzy.py` + a FuzzyHead (changeable-head slot); signedkan_wip stays pristine. Open decisions for Kato:
rule source (per-cycle rec., avoid K^F explosion), T-norm (product rec.), defuzz (Sugeno centroid rec.), plug
point (head first, Gömb outer-shell later), membership bounding. Realizes this lens as the readout layer.
