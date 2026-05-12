# Predictive coding on signed graphs (entropy replaces backprop) — 2026-05-09

This plan is the **Phase B** of `docs/plans_entropy_learning_2026_05_08.md`, scoped now as a standalone post-SMC research direction.  Tonight's Phase A1 result gives the empirical green light: α-mixer entropy carries paired-Δ signal at +5σ on Slashdot, so entropy *does* contain usable training-time information on signed graphs.  The radical question — *can entropy-driven local updates **replace** backpropagation entirely?* — is now justified to attempt.

## Goal

Train HSiKAN end-to-end with **only local, entropy-derived updates** instead of global backpropagation, on the validated 2026-05-08 SOTA-beating recipe (joint-mix + Highway-quat).  If successful, it's an architectural-level claim about graph-native learning: the same signed-incidence message-passing structure used in the forward pass also carries the learning signal, with no separate backward graph.

## Why this is its own paper (separate from SMC)

The SMC paper claims architectural primitives (cycles, walks, attention) and their compounding ladder.  Predictive coding (PC) replaces *the optimiser*, not the architecture.  These are orthogonal claims — combining them would dilute both.  The right venue for PC is NeurIPS / ICML, where backprop-replacement methods (predictive coding, equilibrium propagation, forward-forward) have an active research community.

The headline framing: *"On signed graphs, local entropy-driven prediction-error updates train HSiKAN to within 1σ of backprop, with no global gradient and no autograd memory cost."*

## Method choice

Among entropy-driven backprop alternatives, **Predictive Coding (PC)** is the best fit, for reasons documented in the entropy-learning plan (`docs/plans_entropy_learning_2026_05_08.md` §B0).  Briefly:

- PC has the most-developed PyTorch implementations (Bogacz tutorials, NGC framework)
- Maps cleanly to signed-incidence message passing: the generative path uses $M_{vt}^\top$ (already sparse and known), no new graph structure needed
- HSiKAN's $L=2$ shared-layer depth is tractable for PC (deeper PC is research-grade)
- Equilibrium Propagation needs a real energy function, hard to derive for σ-masked aggregation
- Forward-Forward's per-layer "goodness" is poorly defined for cycle aggregation

## PC formulation for HSiKAN

For each layer $\ell$:
- Forward produces cycle embeddings $\mathbf{H}^{({\rm e}, \ell)}$ from vertex embeddings $\mathbf{H}_v^{(\ell-1)}$ via the shared SignedKAN layer.
- A *generative* path predicts $\mathbf{H}_v^{(\ell-1)}$ from $\mathbf{H}^{({\rm e}, \ell)}$ via $M_{vt}^\top$ (vertex-tuple incidence transposed — already in the codebase).
- Per-layer prediction error: $\varepsilon^{(\ell)} = \mathbf{H}_v^{(\ell-1)} - M_{vt}^\top \mathbf{H}^{({\rm e}, \ell)}$
- Layer parameters update via local rule: $\Delta \theta^{(\ell)} \propto \mathrm{outer}(\varepsilon^{(\ell)}, \mathbf{H}_v^{(\ell-1)})$

The final layer's prediction is the edge-classification output; its prediction error = (predicted_logit − true_label) is the *only* label-aware signal.  All deeper-layer updates depend on the final error and the local prediction errors.  No global backprop chain.

**Convergence semantics**: PC iterates forward + generative paths to a fixed point (10–100 inner iterations per training step), then applies local parameter updates from the converged errors.  Activations and parameters update on different schedules.

## Experiments (3-tier acceptance)

### B-1. Tier 1 — PC works at all on Bitcoin Alpha

- Implement generative path + relaxation loop on $L=1$ (single-layer)
- Train PC + HSiKAN on Bitcoin Alpha
- Compare AUC against backprop baseline on the same recipe

**Acceptance**: PC AUC within 5pp of backprop's 0.9845 (i.e., $\geq 0.93$).  Shows entropy-driven learning works at all on signed graphs.  If PC fails to converge or lands at random, terminate.

### B-2. Tier 2 — PC matches backprop on the SOTA-beating recipe

- Extend to $L=2$ (production setting, joint-mix + Highway-quat)
- Train on Bitcoin Alpha → Bitcoin OTC → Slashdot
- 5-seed paired comparison with backprop

**Acceptance**: PC mean within 1σ of backprop on at least 2 of 3 datasets.  Real claim: entropy-driven learning matches the global-gradient SOTA recipe.

### B-3. Tier 3 — PC scales deeper / more memory-efficient than backprop

- Try $L=4$, $L=8$ HSiKAN layers (where backprop's vanishing-gradient pathology bites)
- Measure peak activation memory: PC has no need to store activations (relaxation recomputes), so it should win
- Compare AUC at deeper $L$

**Acceptance**: at $L=4$, PC AUC > backprop AUC by > 0.5σ paired.  Or memory at $L=4$ is < 50% of backprop's.  Strong claim: entropy-driven learning *outperforms* backprop on this architecture in some regime.

## Implementation cost

| component | scope | LOC |
|---|---|---|
| Generative path: predict vertex embeddings from cycle embeddings via $M_{vt}^\top$ | new module per arity | ~80 |
| Layer-local error tensor management | new state in MixedAritySignedKAN | ~50 |
| Inner relaxation loop: iterate forward + generative until errors converge | new training-step structure | ~100 |
| Local update rule (Hebbian-like, per-arity per-layer) | replace optimiser step | ~80 |
| Local update for Catmull-Rom splines (4-point support; derivable) | non-trivial math + impl | ~150 |
| Eval harness against backprop-trained baseline (paired AUC) | new measurement script | ~50 |
| **Total** | | **~500 LOC** |

Plus integration with the existing α-mixer + Highway gate (PC update rule for those needs to be derived).

## Risk register

| risk | probability | mitigation |
|---|---|---|
| PC inner relaxation never converges on σ-masked aggregation | high | inherit from PC literature: damping, fewer cycles per inner step, slower lr; if still divergent, try a smoother aggregation (drop σ-mask) |
| Local rule on Catmull-Rom splines is incorrect — autograd does the right thing implicitly | medium | derive update by hand; validate against autograd-computed gradient on a single-layer toy problem |
| The α-mixer's softmax doesn't have a natural local PC update | medium | treat α as a separately-trained variational parameter (gradient-free EM) |
| 3 weeks of engineering doesn't reach Tier 1 | high | hard cutoff at 3 weeks; if Tier 1 not reached, write the negative result |
| PC achieves Tier 1 but plateaus far below backprop on harder datasets (Slashdot) | medium | acceptable as a workshop / methods note, less so as a NeurIPS paper |

## Milestones

- **Week 1**: implement generative path + relaxation on Bitcoin Alpha at $L=1$.  Verify activations converge.
- **Week 2**: extend to $L=2$ joint-mix.  Tier-1 gate (within 5pp of backprop on BA).
- **Week 3**: extend to Slashdot recipe.  Tier-2 gate (within 1σ of backprop on Slashdot).
- **Weeks 4-6**: deeper-stack experiments; Tier-3 gate.
- **Week 7-8**: paper draft.

If Tier 1 fails at end of week 2, terminate the plan and write the negative result as a note.

## Acceptance for the plan as a whole

- **Tier 1** (workshop): cheap claim — PC works on signed graphs.  Submit to NeurIPS structured-prediction workshop.
- **Tier 2** (NeurIPS / ICML): real claim — PC matches backprop on the SOTA-beating recipe.
- **Tier 3** (NeurIPS top-tier): scale claim — PC outperforms or scales deeper than backprop.

## Files this plan will touch

- `signedkan_wip/src/predictive_coding.py` — new, ~300 LOC
- `signedkan_wip/src/run_pc_train.py` — new, ~100 LOC
- `signedkan_wip/src/mixed_arity_signedkan.py` — extend with PC hooks (generative path, error tensors)
- `paper/predictive_coding_signedgraph/main.tex` — new venue submission
- `docs/plans_predictive_coding_signedgraph_2026_05_09.md` — this file

## What this plan does NOT do

- Doesn't replace HSiKAN's architecture.  Same model, different optimiser.
- Doesn't claim biological plausibility (despite the PC framing being inspired by neuroscience).
- Doesn't extend to deep stacks beyond $L=8$ — those are pretrained-LLM territory and out of scope.
- Doesn't promise convergence.  PC has many failure modes.

## Connection to other plans

- **Phase A entropy aux losses** (`docs/plans_entropy_learning_2026_05_08.md`) — Phase A1 result is this plan's prerequisite.  Already validated tonight.
- **Mesh matching** (`docs/plans_mesh_matching_2026_05_09.md`) — orthogonal application; PC could be tested there too.
- **Structural-KA theorem** (`docs/plans_structural_ka_theorem_2026_05_09.md`) — PC convergence to the same fixed-point as backprop would be empirical evidence that the structural-KA representation is *learnable* without explicit gradient signal, which is itself a constructivity result.
- **Tabular benchmarks** (`docs/plans_hsikan_tabular_benchmarks_2026_05_09.md`) — PC on Iris is a clean tractable test, smaller than Bitcoin Alpha.

## Deeper theoretical implication if Tier 2 lands

If PC matches backprop on signed-cycle HSiKAN, the implication is that **the signed-incidence graph itself carries enough information for learning** — the gradient signal of backprop is in some sense redundant with the graph structure.  This would be a non-trivial claim about the role of structure in learning, and aligns with the "structural KA" framing: the graph encodes the function, and the function encodes the gradient.
