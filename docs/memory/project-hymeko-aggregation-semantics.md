---
name: project-hymeko-aggregation-semantics
description: "Kato/Hajdu 2026-06-25 — lift the COMBINATION ALGEBRA (sum/product/T-norm) into HyMeKo as a declared `aggregate` tag on a relation, so one signed incidence becomes linear-conv / balance-parity / fuzzy-rule. 4-artifact plan on disk; grammar = CORE (gated)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

**Origin (2026-06-25, deep modeling thread with Dr. Hajdu, "Kato is right again").** Established that the
declarative reward IS a convolution `R = wᵀ f(M x)` over a SEPARATE signed term-incidence `M` (terms × vertices),
NOT entries of the kinematic matrix — a two-layer hub conv: vertices →(M)→ term-nodes →(w bundle weights)→ scalar.
Reward terms should be HYPERNODES (data points / 0-cochains carrying a scalar), only the bundle is a hyperedge;
this is also what the arc-weight design presupposes (a node is reusable across bundles with different arc weights;
an edge isn't). Observation CHANNELS stay hyperedges (they fan OUT 1→N onto vertices); TERMS are nodes (fan IN
N→1 to a scalar). See [[project-fuzzy-defuzzification-heads]], the structural critic in
[[project-actor-critic-shared-reasoning]] (same `M⊛x` algebra — reward & critic are dual objects in one chain
complex).

**The plan (`docs/plans/2026-06-25-hymeko-aggregation-semantics/`, 4 artifacts compile/validate).** Kato's point:
the MULTIPLICATION (combination operator) is currently HARD-CODED ADDITIVE in the engine, but the project needs
three over the SAME signed incidence: **additive Σ** (Laplacian / HSiKAN A± / reward), **multiplicative Π** (signed
parity = Cartwright balance = `hymeko_graph::BalanceScorer`), **T-norm** (fuzzy-signature firing). Fix = a declared
`aggregate ∈ {sum, signed_sum, product, signed_product, tnorm_min, tnorm_prod}` tag on a hyperedge/bundle; default
`sum` ⇒ everything unchanged. Differential/spectral payoff: the tag IS the choice of operator/cohomology —
additive→∂+graph-Laplacian (Hodge; Ng potential-based shaping = the exact 1-forms, e.g. goal_progress telescopes),
multiplicative→F₂ sign-parity, T-norm→residuated lattice. "Declare the multiplication early" = "declare which
(co)homology this relation computes in."

**THEORY FOUNDATION (2026-06-25, user-requested 'so the extension can be done'):**
`docs/theory/hypergraph_convolution_semantics.tex/.pdf` — the algebra: signed incidence B = boundary ∂;
convolution `(B⊛x)_e = ⊕_{v∈e} σ_ev w_ev ⊗ x_v` = semiring matvec. WELL-DEFINED iff the declared algebra is a
COMMUTATIVE MONOID (⊕ assoc+comm+identity for unordered members/padding); signed-weighted needs a SEMIRING (⊗
distributes over ⊕); nesting (hyperedge-on-hyperedge) composes iff distributive. Catalog↔tags: additive (ℝ,+,×)=
sum; parity ({±1},×)=signed_product=Cartwright balance/F₂ cohomology; tropical (min,+); fuzzy ([0,1],T-conorm,
T-norm) — CAVEAT product/Łukasiewicz T-norms are NOT distributive (residuated lattice not semiring → mark
single-layer/non-composable, Gödel min/max IS distributive). Each algebra → its own cohomology (additive=Hodge/
Laplacian + Ng-shaping=exact-forms; parity=F₂; tropical=max-plus eigenvalues; fuzzy=residuum not transpose). The
extension recipe = grammar tag → validate (reject non-monoid, mark non-distributive) → one pick_semiring(tag)
reader → parity guard vs BalanceScorer.

**WALK-SPIKES = the tropical instantiation (2026-06-25, Kato/Hajdu).** Spikes are NOT a new architecture — they
are the min-plus (tropical) semiring read as EVENTS: arc weights = delays, ⊗=+ accumulates, ⊕=min keeps earliest
arrival; a signed walk emits a spike at time Στ carrying its parity → signed spike train ("walk-spikes"), a depth-k
stack = LIF forward with structural delays. Two SEPARABLE testable claims (in the theory note §event-based): (1)
REPRESENTATIONAL bet — additive readout is path-length-BLIND; tropical weights the EARLIEST/shortest signed walk =
"trust the shortest balanced path", principled for imperfectly-balanced graphs → test tropical-vs-additive on OTC
AUROC (adds nothing on a perfectly balanced graph since parity is order-independent — honest caveat). (2)
EFFICIENCY near-certain — tropical idempotent (a⊕a=a) → sparse event-driven → neuromorphic/low-power = the fast
REFLEX of the rate-asymmetric controller (event-based inference, embedded). Framework now spans rate(additive) ·
spike(tropical) · balance(parity) · fuzzy(residuated) over ONE signed incidence = a unifying theory (stronger
paper than "another KAN variant"). Opens neuromorphic venues.

**HARD LINE (the real risk = scope creep):** declare the OPERATOR (semiring), NOT the learnable program. No
HSiKAN W, no Catmull-Rom splines, no shapes/weights in the grammar — those stay runtime. Structure + algebra
declarative; parameters in the engine. If a proposal needs a shape, it's engine, not grammar.

**CORE GATE:** the parser crate + `**/*.lalrpop` are `lockdown: full` → the grammar production for the tag is a
core edit needing a verbatim `APPROVED-CORE-EDIT: <slug>` token BEFORE any grammar change. NOT given yet. MVP
minimises the delta to ONE optional annotation reusing existing tokens; the non-core readers
(`pick_aggregator(tag)` Strategy in `hymeko_rl/env/{reward,_profile}.py` + HypergraphState) + vocab tags can stage
behind a flag meanwhile. Parity test: `signed_product` over a cycle ≡ `BalanceScorer`. **How to apply:** when
picked up, get the APPROVED-CORE-EDIT token first; build readers+MVP vocab behind the flag while waiting. Phase-2
arc = the convolution semiring on ANY hyperedge (HSiKAN A± aggregation declared not assumed) — ties
[[project-unify-hsikan-core]].
