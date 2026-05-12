# Entropy-based learning on signed graphs — Phase A and Phase B (2026-05-08)

User's question (paraphrased): *replace backpropagation with entropy-based learning on graphs, since everything in graph-based propagation is applicable.*

This plan splits the question into two concrete experimental phases. Phase A is additive (aux loss alongside backprop) and tests the hypothesis "entropy carries signal on signed graphs." Phase B is the radical replacement and tests "entropy can REPLACE the gradient signal." Both are scoped against the validated baselines from the 2026-05-08 SOTA-beating campaign (joint-mix HSiKAN on BA/OTC/Slashdot).

## Why two phases

| | Phase A | Phase B |
|---|---|---|
| approach | aux entropy loss in addition to BCE | replace BCE backward with local entropy-driven layer updates |
| scope | regularizer | optimizer replacement |
| risk | low — at worst α=0 reduces to baseline | high — could fail to converge |
| effort | half-day to ~3 days | weeks |
| paper-bullet | regularizer that helps walk-rich datasets | architectural-level claim about graph-native learning |
| go/no-go decision feeds into | Phase B (only worth attempting if A confirms entropy carries signal) | a separate paper / journal extension |

Phase A is a **prerequisite**: if entropy doesn't even add signal as an auxiliary, there's no reason to attempt the replacement.

---

# Phase A — auxiliary entropy losses

Goal: measure whether entropy-derived auxiliary losses lift HSiKAN's validated AUCs on the existing benchmarks. The validated baselines are the 2026-05-08 5-seed numbers:

| dataset | baseline (5-seed) | recipe |
|---|---|---|
| Bitcoin Alpha | 0.9845 ± .0028 | joint c3,c4,w2,w3, h=16 |
| Bitcoin OTC | 0.9801 ± .0057 | same |
| Slashdot | 0.9035 ± .0044 | c2,c3,c4,c5,w2,w3 + Highway-quat, h=4 |
| Epinions | TBD | same Slashdot recipe (run in flight) |

Each Phase-A variant is added as a SEPARATE knob, so they can be combined or ablated.

### Variant A1 — α-mixer entropy

The α-mixer is `softmax(ℓ)` with per-slot logits. Auxiliary loss:

$$\mathcal{L}_{\alpha} = -\lambda_\alpha \cdot H(\boldsymbol{\alpha}) = \lambda_\alpha \sum_\kappa \alpha_\kappa \log \alpha_\kappa$$

Sign: minimising the negative-entropy term *maximises* H(α), i.e., keeps the routing spread across slots. Small λ_α (1e-3 to 1e-1) is the natural range; larger values force uniform mixing and would erase the "walks attend, cycles uniform" pattern we're trying to *detect* in the routing signal.

**Implementation**: 5 lines in `run_final_cell.py` training loop. Env-var `HSIKAN_ALPHA_ENTROPY_LAMBDA`. Already covered by `model.alpha()` — straightforward.

**Hypothesis**: Mild α-entropy will not change BA/OTC much (slots are already spread there) but may help Slashdot/Epinions where one slot can dominate (w3 at 0.36 in c5full).

**Acceptance**: paired Δ AUC > 1σ on at least one of the four datasets at λ_α ∈ {0.001, 0.01, 0.1}.

### Variant A2 — per-edge attention entropy

Already exists in `signedkan_wip/src/attention.py::attention_entropy_loss` for the old `SignedTriadAttention` path. NOT wired into the current `_QuaternionAttentionM_e` / `_AttentionM_e` Highway path.

For each edge $e$ and its incident attention distribution $a^{(\kappa)}_{e, \cdot}$ (already softmax-normalized inside `_QuaternionAttentionM_e.forward`), the per-edge entropy is

$$H_{\rm attn}(e) = -\sum_{t \in \mathcal{N}_\kappa(e)} a^{(\kappa)}_{e,t} \log a^{(\kappa)}_{e,t}$$

Auxiliary loss: $\mathcal{L}_{\rm attn} = -\lambda_{\rm attn} \cdot \mathrm{mean}_{e,\kappa} H_{\rm attn}(e)$.

**Implementation**: extend the attention head's forward to optionally return raw scores; compute per-edge entropy via scatter-mean. ~30 lines. Env-var `HSIKAN_ATTN_ENTROPY_LAMBDA`.

**Hypothesis**: Slashdot's gate values [c5=0.52, w2=0.66, w3=0.63] suggest attention is concentrating; entropy reg might prevent the concentration from going too sharp. Could compound with Highway gate cap.

**Acceptance**: paired Δ on Slashdot > 1σ at λ_attn ∈ {0.001, 0.01, 0.1}. If null on Slashdot, this variant is dropped.

### Variant A3 — per-arity cycle-embedding spectral entropy

The existing `EntropyRegulariser` (`entropy_reg.py`) computes spectral entropy of an embedding matrix and is currently applied to `model.node_embed.weight`. Extending it to per-arity cycle embeddings $\mathbf{H}^{(\kappa)} \in \mathbb{R}^{T_\kappa \times d_{\rm jk}}$ tests whether spreading cycle-representation rank helps.

For each arity $\kappa$:

$$\mathcal{L}_{\rm spec, \kappa} = \lambda_{\rm spec} \cdot R(\mathbf{H}^{(\kappa)})$$

where $R(\cdot)$ is the existing `EntropyRegulariser` term (spectral KL + target).

**Implementation**: thread `EntropyRegulariser` through `MixedAritySignedKAN` so it can call back with each $\mathbf{H}^{(\kappa)}$ during the forward pass. ~50 lines, more invasive because of the encoder modification. Env-var `HSIKAN_CYCLE_SPEC_ENTROPY_LAMBDA`.

**Hypothesis**: cycle embeddings on Slashdot may collapse to a low-rank subspace because of the dense softmax attention (the gate values suggest strong concentration). Spectral spread could re-diversify.

**Acceptance**: paired Δ on Slashdot/Epinions > 1σ. Compatible with A1 and A2.

### Variant A4 — graph-native: balance-distribution KL (the "graph-theoretic" entropy)

The most theoretically interesting variant — the one that genuinely uses **signed-graph** structure rather than generic representation-spread.

For each edge $e$ and arity $\kappa$, define the **balance distribution**:

$$p^{\rm bal}_\kappa(e) = \frac{|\{t \in \mathcal{N}_\kappa(e) : \mathrm{balanced}(t)\}|}{|\mathcal{N}_\kappa(e)|}$$

where $\mathrm{balanced}(t)$ is the Cartwright-Harary flag (product of edge signs is +1) — already computed in the `SignedNTuple` dataclass.

The model's attention weights $a^{(\kappa)}_{e, \cdot}$ also induce a distribution over balanced/unbalanced cycles:

$$q^{\rm bal}_\kappa(e) = \sum_{t \in \mathcal{N}_\kappa(e)} a^{(\kappa)}_{e,t} \cdot \mathbb{1}[\mathrm{balanced}(t)]$$

Auxiliary loss: KL divergence between attention-induced balance distribution and ground-truth balance distribution (per edge per arity, averaged):

$$\mathcal{L}_{\rm bal} = \lambda_{\rm bal} \cdot \mathrm{mean}_{e, \kappa} \mathrm{KL}\!\left(q^{\rm bal}_\kappa(e)\;\|\;p^{\rm bal}_\kappa(e)\right)$$

This aligns the attention to the structural-balance prior. With λ=0 the attention is free; with large λ, attention is forced to match balance.

**Implementation**: needs balance flag passed alongside the M_e indices. ~80 lines. Env-var `HSIKAN_BAL_KL_LAMBDA`.

**Hypothesis**: Heider's structural balance is the only known principle for *what makes good cycles* on signed graphs. If the model's attention learns to attend to balance-informative cycles, AUC should lift. If it doesn't, attention is finding something other than balance.

**Acceptance**: paired Δ on Slashdot/Epinions > 1σ at λ_bal ∈ {0.001, 0.01, 0.1}. Strong success criterion: Slashdot mean shifts above 0.91.

### Phase A measurement protocol

1. **5-seed paired (within-config)**: same seed list as the 2026-05-08 5-seed grid. Each Phase-A variant adds one knob; baseline = same recipe with λ=0. Paired Δ over 5 seeds.

2. **3 datasets**: Slashdot (where attention helps most, Phase A is most likely to land), Bitcoin Alpha (where joint-mix is already winning, tests whether Phase A regresses), Epinions (whose 5-seed is a separate result anyway).

3. **λ sweep**: {0.001, 0.01, 0.1} per variant. Stop at the first λ that lands a positive Δ; report best.

4. **Compounding**: once individual variants are tested, combine A1 + A2 (cheap) and A1 + A4 (theoretical compound).

5. **Acceptance threshold**: paired Δ > 1σ on the harder datasets (Slashdot/Epinions). On BA/OTC, no-regression is the bar.

### Phase A risk register

| risk | probability | mitigation |
|---|---|---|
| All variants null at small λ | medium | the validated baselines are already strong; small-λ entropy regs add noise, not signal |
| Variant A4 too expensive to compute per epoch | medium | precompute balance flag once per arity at setup |
| Phase A confounds the Schmidhuber routing pattern | low | the per-slot α and gate values stay observable in the JSON output; routing pattern can be verified post hoc |
| Variant A3 backward through eigenvalue decomposition is unstable | medium | the existing `EntropyRegulariser` already handles this on node_embed; use the same numerical safeguards |

### Phase A go / no-go for Phase B

If at least one Phase-A variant lands paired Δ > 1σ on Slashdot or Epinions, entropy carries signal on signed graphs. Phase B becomes a meaningful research direction. If ALL Phase-A variants are null after the λ sweep, Phase B is not worth attempting — entropy is not a learnable signal in this architecture.

---

# Phase B — entropy-driven learning, replacing backprop

Goal: train HSiKAN end-to-end with **only local, entropy-derived updates** instead of global backpropagation. If successful, this is an architectural-level claim about graph-native learning: the same signed-incidence message-passing structure used in the forward pass also carries the learning signal.

### B0 — fixing the family

Several flavours of entropy-driven learning exist; we pick ONE and commit.

| family | local rule? | published work | maps to HSiKAN how |
|---|---|---|---|
| Predictive Coding (PC) | yes — per-layer prediction error | Whittington & Bogacz 2017 | each SignedKAN layer predicts the next; minimise prediction error per cycle |
| Equilibrium Propagation (EP) | yes — two equilibrium states | Scellier & Bengio 2017 | energy-based formulation of the BCE objective; train via state-perturbation gradient |
| Forward-Forward (FF) | yes — per-layer goodness | Hinton 2022 | each layer maximises a local "goodness" (e.g., activation energy / entropy) |
| Free Energy Principle (FEP) | yes — variational free energy | Friston et al. | layer outputs as posterior beliefs; loss = -log p(label) + KL(q || prior) |
| InfoMax / IB | layer-pair only | Tishby; InfoGraph | maximise I(layer; label) − β I(layer; input) |

**Pick**: **Predictive Coding**. Reasons:
1. Most-developed PyTorch implementations (e.g. Bogacz's tutorials, NGC framework).
2. Directly maps to layer-by-layer signed-incidence propagation: each layer's prediction error is local, and the error itself is propagated backward via the same incidence structure used in forward — no separate backward graph needed.
3. The HSiKAN architecture has L=2 shared layers; PC at depth 2 is a tractable setting (deeper PC is research-grade).
4. Equilibrium Propagation requires a true energy function; deriving one for the signed-cycle aggregation is non-trivial.
5. Forward-Forward's per-layer "goodness" is poorly defined for cycle aggregation (what's "good" for a per-cycle embedding?).

If PC fails, fallback is FF with goodness = α-entropy per slot.

### B1 — concrete PC formulation for HSiKAN

Replace the global BCE loss + autograd with **per-layer prediction error minimisation**:

For each layer $\ell$:
- Forward produces cycle embeddings $\mathbf{H}^{({\rm e}, \ell)}$ from vertex embeddings $\mathbf{H}_v^{(\ell-1)}$.
- A *generative* path predicts $\mathbf{H}_v^{(\ell-1)}$ from $\mathbf{H}^{({\rm e}, \ell)}$ via $M_{vt}^\top$ (the transpose of the vertex-tuple incidence — already sparse and known).
- Per-layer prediction error: $\varepsilon^{(\ell)} = \mathbf{H}_v^{(\ell-1)} - M_{vt}^\top \mathbf{H}^{({\rm e}, \ell)}$.
- Layer parameters update via local rule: $\Delta \theta^{(\ell)} \propto \mathrm{outer}(\varepsilon^{(\ell)}, \mathbf{H}_v^{(\ell-1)})$.

The final layer's prediction is the edge-classification output. Its prediction error = (predicted_logit − true_label) is the only label-aware signal. All deeper-layer updates depend ONLY on this final error and the local prediction errors — *no backprop chain*.

**Convergence semantics**: PC is run iteratively to a fixed point (typically 10-100 inner iterations per training step). Once converged, the parameter updates use the converged errors. Unlike backprop, the activations and the parameters are updated separately.

### B2 — what new code is required

| component | scope | LOC estimate |
|---|---|---|
| Generative path: predict vertex embeddings from cycle embeddings via $M_{vt}^\top$ | new module per arity | ~80 |
| Layer-local error tensor management (one per layer per arity) | new state in MixedAritySignedKAN | ~50 |
| Inner relaxation loop: iterate forward + generative until errors converge | new training-step structure | ~100 |
| Local update rule (Hebbian-like, per-arity per-layer) | replace optimizer step | ~80 |
| Differentiation of CR splines for the local rule (currently autograd handles this; PC needs explicit local update) | non-trivial: CR has 4-point support, local update derivable | ~150 |
| Eval harness against backprop-trained baseline (paired AUC) | new measurement script | ~50 |
| **Total** | | **~500 LOC** |

### B3 — Phase B success criteria

PC has historically had two failure modes:
1. **Convergence failure**: relaxation doesn't settle; activations oscillate.
2. **AUC gap to backprop**: PC converges but lands ~10-20pp below backprop on standard tasks.

Acceptance for Phase B as a meaningful contribution:

- **Tier 1** *(cheap claim)*: PC trains HSiKAN to within 5pp of the backprop baseline on Bitcoin Alpha. This shows entropy-driven learning *can* work on signed graphs at all.
- **Tier 2** *(real claim)*: PC trains HSiKAN to within 1σ of backprop on Slashdot. This shows entropy-driven learning matches backprop on the SOTA-beating recipe.
- **Tier 3** *(research-paper claim)*: PC outperforms backprop on at least one dataset, OR PC scales to deeper HSiKAN (L=4+) where backprop's vanishing-gradient pathology starts to bite. This justifies a separate paper.

If PC fails Tier 1, the experiment is a negative result and Phase B closes.

### B4 — Phase B risk register

| risk | probability | mitigation |
|---|---|---|
| PC inner relaxation never converges on the signed-cycle aggregation | high | inherit from existing PC literature: damping, fewer cycles per inner step, slower learning rate |
| Local rule on CR splines is incorrect — backprop's autograd does the right thing implicitly that local rule misses | medium | derive the local update for CR by hand, validate against autograd-computed gradient on a single layer toy problem |
| The α-mixer's softmax doesn't have a natural local PC update | medium | treat α as a separately-trained variational parameter (gradient-free EM-like update) |
| Engineering cost overruns 3 weeks | high | hard cutoff at 3 weeks; if Tier 1 not reached, write the negative result and stop |

### B5 — Phase B milestones

1. **Week 1**: implement generative path + relaxation loop on Bitcoin Alpha at L=1 (no-stack). Verify activations converge in inner loop.
2. **Week 2**: extend to L=2 (the production setting). Measure AUC vs backprop. Tier 1 gate.
3. **Week 3**: extend to Slashdot recipe (joint mix + attention). Measure AUC vs backprop. Tier 2 gate.

If Tier 1 fails at end of week 2, terminate. Write the negative result.

---

# Order of operations

1. **Phase A1 (α-entropy)** — easiest, ~30 min code + 5-seed Slashdot. **Start tomorrow.**
2. **Phase A2 (per-edge attention entropy)** — extend existing `attention_entropy_loss` to new path. ~1 day.
3. **Phase A3 (cycle-embedding spectral entropy)** — modify encoder to expose per-arity intermediates. ~1 day.
4. **Phase A4 (balance-distribution KL)** — most theoretically motivated, also most code. ~2 days.
5. **Phase A compound experiments** — combine top variants. ~half day.
6. **Decision gate**: if any Phase A variant lands > 1σ on Slashdot/Epinions, proceed to Phase B. Else close.
7. **Phase B**: 3 weeks if green-lit.

Phase A in total: ~5 days. Phase B in total: ~3 weeks. Combined: ~4 weeks.

# Files this plan will touch when executed

```
signedkan_wip/src/run_final_cell.py             — wire Phase-A env-vars into training loop
signedkan_wip/src/mixed_arity_signedkan.py      — expose per-arity intermediates for A2/A3
signedkan_wip/src/attention.py                  — extend attention_entropy_loss to new path
signedkan_wip/src/entropy_reg.py                — generalise EntropyRegulariser to per-arity cycle embeddings
signedkan_wip/src/balance_kl.py                 — NEW, A4 only
signedkan_wip/src/predictive_coding.py          — NEW, B only
signedkan_wip/src/run_pc_train.py               — NEW, B only
signedkan_wip/experiments/run_phaseA_5seed_*.sh — measurement scripts per variant
docs/plans_entropy_learning_2026_05_08.md       — this file (close out with results)
```

# What this plan deliberately does NOT do

- Does not commit to the radical replacement upfront. Phase A's gate is non-negotiable; without it, Phase B is unsupported speculation.
- Does not chase deep-net entropy methods (e.g. NeuralODE-style equilibrium learning). HSiKAN is shallow; the depth motivation isn't there.
- Does not couple Phase B to GPU-native cycle enumeration or sparse-attention scaling — those are independent threads.
- Does not propose entropy-based pretraining (e.g. masked-cycle prediction). The current architecture is supervised end-to-end; that's the comparison.
- Does not use Phase B as a paper claim before it works. If it fails, document as a negative result; don't soften the SMC HSiKAN paper to depend on it.

# Connection to the structural Kolmogorov-Arnold framing

Phase A operationalises the structural-KA claim that primitives compose under a learnable mix. Adding entropy aux losses tests whether the *mix itself* (the α distribution, the attention weights, the cycle-embedding spectrum) carries usable signal independent of the BCE label gradient. Phase B asks the deeper question: can the structure of the signed-incidence graph carry the learning signal *itself*, with no global gradient at all? Both phases are tests of *how much information the graph structure encodes for learning* — Phase A measures it in addition to gradient; Phase B asks whether it's enough alone.
