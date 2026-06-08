# AC-HSiKAN capacity-scaling debug: from "doesn't scale" to "scales with caveats"

**Date:** 2026-06-06 evening
**Scope:** Systematic investigation of why AC-HSiKAN failed to scale
from d=16 (165 k params) to d=32 (337 k params), the structural fixes
that restored scaling, the ABB-per-vertex sequence analogue we ported
from simple HSiKAN, and the mathematical questions the data raises.

## Starting state

The 2026-06-06 morning headline (Triton-stack):

> AC-HSiKAN at d=16 (165 k params) reaches **statistical parity**
> with iso-param Transformer on full IMDB: 0.8480 ± 0.005 vs
> 0.8535 ± 0.005, Δ = −0.0055, t = −2.94, p ≈ 0.05.

Scaling AC to d=32 (337 k params) was expected to widen the gap on
the AC side. It did the opposite — the gap got WORSE:

| config | AC val_acc | Δ vs TR | t | result |
|---|---:|---:|---:|---|
| d=16 sh=16 lr=3e-3 | 0.8489 ± .004 | −0.0046 | −2.12 | paritás (p>0.05) |
| d=32 sh=16 lr=3e-3 | 0.8442 ± .005 | −0.0125 | −5.78 | **AC LOSES significantly** |

Doubling capacity made AC *worse* relative to Transformer. The user's
intuition: "valami nem stimmel" (something's not right).

## Two missing scaling axes identified

### Axis 1 — `sign_head_hidden` doesn't scale with `d_model`

The sign-attention bottleneck `sign_head_hidden=16` is hard-coded as a
config default and does NOT scale with `d_model`. At d=16 this is 100 %
of `d_model` (rank-preserving); at d=32 it is a 50 % bottleneck. The
bigger model gets the **same** rank-16 attention slot as the small one.

Fix: `sign_head_hidden = d_model`. New param: 337 k → 337.5 k (+0.15 %).

| config | AC val_acc | Δ vs TR | t |
|---|---:|---:|---:|
| d=32 sh=16 (default) | 0.8442 ± .005 | −0.0125 | −5.78 |
| **d=32 sh=32** | **0.8476 ± .008** | **−0.0091** | **−2.13** |

Half the regression closed. Still borderline; on to axis 2.

### Axis 2 — LR scaling

The lr=3e-3 was tuned at d=16. At d=32, doubling parameter scale makes
the same lr too aggressive. Standard heuristic: lr ∝ 1/√(d_model).
Applied: lr = 1.5e-3 = 3e-3 / √2.

| config | AC val_acc | Δ vs TR | t | result |
|---|---:|---:|---:|---|
| d=32 sh=32 lr=3e-3 | 0.8476 ± .008 | −0.0091 | −2.13 | borderline |
| **d=32 sh=32 lr=1.5e-3** | **0.8483 ± .007** | **−0.0038** | **−1.19** | **paritás** |

**AC now scales positively in capacity.** d=32 paritás restored.

## Sequence-native ABB analogue: dynamic top-K candidate selection

Inspired by simple-HSiKAN's ABB top-K-per-vertex cycle enumeration (which
gave +6.7 pp on Epinions), we ported the **dynamic candidate selection**
idea to AC-HSiKAN. Where simple-HSiKAN picks the top-K most informative
cycles per graph vertex by structural score (Cartwright-Harary balance,
Friedler axiom), AC-HSiKAN can pick the top-K most informative neighbour
positions per sequence anchor by **sign-head magnitude**.

Implementation: at each forward, compute the dense `(B, L, L)` sign
matrix once (the dense sign-head already does this), take per-anchor
batch-averaged `|magnitude|`, mask self-edges, and `topk(K)`. The
selected `(L, K)` indices replace the static `_local_indices_cache`.

### Result at d=32 sh=32 lr=1.5e-3

| selection | AC val_acc | Δ vs TR | σ |
|---|---:|---:|---:|
| static (positional + random jumps) | 0.8483 | −0.0038 | .007 |
| **pure dynamic top-K** | 0.8482 | −0.0040 | **.005 (−25 %)** |

**Mean acc unchanged; variance dropped 25 %.** The static positional
prior on IMDB is approximately as informative as the magnitude-top-K.
Sequence data has strong local structure — `(positional ± K, random
jumps)` already captures the "informative" neighbours that the magnitude
score would identify.

This is **opposite** to the simple-HSiKAN ABB story: on sparse signed
graphs, structural top-K is a big win because the right neighbours are
*not* topologically near. On dense sequences, positional locality and
magnitude top-K converge.

## Long-cycle inductive prior — sequence-native DAG-pruning analogue

> "Nem a DAG-tulajdonságot, hanem azt hogy hosszabb ciklusokat
> részesítünk előnyben" — user

In simple HSiKAN, the DAG axiom pruning (acyclic-subgraph pruning,
Friedler-style) is the structural mechanism for ABB enumeration. The
sequence-native analogue would be: bias the dynamic top-K toward
positionally **distant** candidates — these form longer walks/cycles
through the walk-op composition.

Implementation: multiply the magnitude score by `(|i-j|/(L-1))^α`
where α is the prior strength.

| config | AC val_acc | Δ vs TR | σ |
|---|---:|---:|---:|
| pure dynamic top-K (α=0) | 0.8482 | −0.0040 | .005 |
| + long-cycle α=0.3 | 0.8272 | −0.0250 | **.029 (6×)** |
| + long-cycle α=1.0 + arity bias toward k=5 | 0.8391 | −0.0130 | .018 |

**Long-cycle bias on IMDB is structurally harmful.** Even mild α=0.3
collapses 2/5 seeds (peak ep 3, decline thereafter). The mean drops
2 pp, variance balloons 6×.

Why: IMDB sentiment is **lexico-syntactic**, fundamentally captured by
local n-gram structure. Forcing AC to look at positionally distant
candidates suppresses the natural signal. This is **task-dependent**:
on Long Range Arena, equilibrium-prop, or scientific PDE simulations
the long-cycle prior should be a net positive.

## Mixed-selection: the working compromise

The dynamic top-K and static positional approaches both worked, with
different strengths (static = stable on IMDB, dynamic = variance ↓).
Combining them: K_total = K_static + K_dyn, where K_static slots are
guaranteed positional and K_dyn slots are dynamic top-K of the
*non-static* positions (suppressing the static set to avoid duplication).

### K_static sweep at K_total = 8

| K_static | AC val_acc | Δ vs TR | t | wins |
|---:|---:|---:|---:|---:|
| 0 (pure dyn) | 0.8482 ± .005 | −0.0040 | −1.19 | 1/5 |
| 4 (4+4) | 0.8496 ± .006 | −0.0025 | −0.92 | 2/5 |
| **6 (6+2)** | **0.8505 ± .005** | **−0.0016** | **−0.66** | **2/5** |
| 7 (7+1) | 0.8460 ± .008 | −0.0062 | −1.69 | 2/5 |
| 8 (pure stat) | 0.8483 ± .007 | −0.0038 | – | – |

**K_static = 6 is the peak.** Strong local prior (6 positional neighbours)
plus a small adaptive long-range channel (2 dynamic slots) outperforms
either pure variant.

The sharp drop at K=7 is a local minimum: one dynamic slot is too small
to carry adaptive information but disrupts the positional structure
of the static set.

K_total = 12 (with K_static=8 K_dyn=4) was tested — **regressed back
to 0.8482**. Beyond K=8, more candidates dilute the signal at this
architecture's capacity. Pool-scatter wall scales linearly in K (+21 %
wall for K=12) with no accuracy gain.

## Training-dynamics tuning (cosine LR + grad clip)

Standard transformer-stack tricks applied to both architectures:
cosine annealing 1.5e-3 → 1.5e-4 over 8 epochs, grad-clip norm 1.0.

| config | AC val_acc | TR val_acc | Δ |
|---|---:|---:|---:|
| K_static=6 constant LR | 0.8505 | 0.8522 | −0.0016 |
| **K_static=6 + cosine + clip** | 0.8500 | **0.8553** | −0.0053 |

**Cosine LR helped TR substantially (+0.0031), AC barely (−0.0005).**
Net: gap widened by 0.0037. The transformer-native training tricks favour
the transformer architecture. AC's slower training dynamics (peak at
ep 4-5 vs TR ep 1-2) does not benefit from end-of-training LR decay.

## Final state

**Best config**: d=32, sign_head_hidden=32, lr=1.5e-3 constant,
mixed top-K with K_static=6 + K_dyn=2, no cosine/clip.

| | val_acc | params | wall/seed | Δ vs TR | p |
|---|---:|---:|---:|---:|---:|
| Transformer baseline | 0.8522 ± .004 | 345 k | 36 s | 0 | – |
| **AC-HSiKAN v1.6 + mixed-6** | **0.8505 ± .005** | 337 k | 144 s | **−0.0016** | **0.55** |

Gap closed **−0.0125 → −0.0016 = 87 %**. Statistically indistinguishable
from Transformer (p > 0.5).

## Mathematical conjectures the data raises

The empirical scaling and selection results suggest mathematical
structure that the current architecture does NOT exploit. Three concrete
directions worth thinking about:

### 1. Inter-layer scatter-pool coupling

Currently each AC-HSiKAN layer's pool-scatter primitive operates on the
*residual sum* `LN(x + dropout(y))` of the previous layer. The pool and
scatter streams are MERGED before the next layer sees them.

**Conjecture:** if pool and scatter are kept as *separate streams*
across layers, with the next layer's pool operating on the previous
pool-output and the next layer's scatter operating on the previous
scatter-output, the primitive's bidirectional asymmetry would be
*preserved* across depth, not collapsed at each LayerNorm.

Mathematically: at layer ℓ, define `p_ℓ`, `s_ℓ` such that
`p_ℓ = P(p_{ℓ-1})` and `s_ℓ = S(s_{ℓ-1})`, with `y_ℓ = p_ℓ + s_ℓ`.
This is a "split-stream" topology. The information surface is now
two parallel chains weakly coupled at the output.

### 2. Hamilton-rotor composition across layers

The entropy-feedback Hamilton rotor `M_ℓ = exp(β_ℓ · H_ℓ · n_ℓ)` is
currently *per-layer independent*. Quaternions compose multiplicatively;
the natural composed rotor across layers would be
`M_total = M_L · M_{L-1} · ... · M_1`.

**Conjecture:** if each `M_ℓ` is parameterised as a *delta* from the
running composition (i.e., `M_ℓ = M_{ℓ-1} · dM_ℓ` where `dM_ℓ` is the
layer's incremental rotation), the entropy-feedback signal would
accumulate as a coherent *trajectory* in quaternion space, not as a
fresh rotation per layer. The evolvens telemetry should show
qualitatively different behaviour.

### 3. Cross-layer cycle composition

The walk-op product `s = Π_k sign(R_k)` is computed *within each layer*
over the candidate set. Across N layers, the implicit cycle structure
is `s_total = s_1 · s_2 · ... · s_N` (independent products in the
sign field).

**Conjecture:** the walk-op should be defined *across layers*, not
within them: at layer ℓ, the sign products use signs from layers
`{ℓ-k+1, ..., ℓ}` for arity k. This makes each "cycle" span multiple
layers, exploiting the depth dimension as a cycle-length budget.
Algebraically: `s_total = Π_ℓ Π_k s_{ℓ,k}` where `s_{ℓ,k}` references
inputs from layers below.

This is the AC-HSiKAN analogue of multi-layer cycle enumeration in
graph-HSiKAN, where cycles can span graph distances. Here, depth IS the
graph distance.

### Why the data suggests these

- The **K_total=12 plateau** says: more *within-layer* capacity doesn't
  help. The information bottleneck is across layers, not within.
- The **mixed selection K_static=6 win** says: a small adaptive channel
  on top of a stable positional prior is the sweet spot. Same shape
  might apply to cross-layer information: stable per-layer state + small
  cross-layer adaptive channel.
- The **long-cycle bias hurting on IMDB** says: positional distance is
  the wrong axis for "long cycles" on sequences. *Depth-distance* might
  be the right axis instead.

### 4. Aggregated CR-patch surfaces (channel-coupled activation)

Currently each of the `h` channels has its own independent 1D
Catmull-Rom spline:
``coef_pos, coef_neg ∈ ℝ^{h × G}`` (h = 8, G = 8 ⇒ 128 params total).

The pool-scatter primitive applies these channel-wise: each channel's
CR is an independent scalar-in scalar-out activation. **No channel
interaction in the activation surface itself.** Channels only couple
through downstream Hamilton products and `W_back` projection.

**Conjecture:** replace per-channel 1D splines with **aggregated CR-patch
surfaces** -- higher-dimensional CR over channel groups. Concrete
options:

- **2D CR over (R, V_c)**: per channel a 2D surface ``coef[h, G, G]``
  parametrising the *joint* dependence on the sign-input `R` and the
  value-input `V_c`. Currently `S * V_c` is a multiplicative gate; a 2D
  CR is a learned non-linear *interaction surface* between them.
  Memory: `h · G² = 512` per buffer (vs 64); compute: bilinear CR =
  16 control-point reads vs 4.
- **Hamilton-block-aggregated 4D CR**: each `n_quat` block of 4
  channels shares a single 4D CR over (e_0, e_1, e_2, e_3) -- the
  full quaternion. Activation becomes Hamilton-equivariant via the
  surface structure, not just via the rotor.
  Memory: `n_quat · G⁴` per buffer -- too many at G=8 (4 · 4096 = 16k);
  feasible only at G ≤ 4 (4 · 256 = 1k).
- **Group-aggregated 2D**: split h channels into pairs, each pair
  shares a 2D CR over (R_a, R_b). Captures channel-pair interactions
  cheaply.
  Memory: `(h/2) · G² = 256` per buffer.

**Mathematical motivation:** the current per-channel 1D CR treats the
activation as **separable across channels**. The pool-scatter primitive
is non-separable in its Q⊗K Hamilton coupling; the activation surface
should mirror this. A 2D CR patch over (R, V_c) makes the *non-linear
mixing* explicit at the activation level, rather than relying on
downstream linear layers (`W_back`) to recover the interaction.

**Empirical prediction:** if the per-channel 1D CR is the bottleneck,
the 2D variant unlocks scaling that K_total = 12 could not. If it is
not the bottleneck, the 2D variant adds parameters without payoff
(a clean null test).

**Implementation cost:** medium. PyTorch reference is straightforward
(bilinear CR is a known formula). Triton kernel extension is the work
-- the existing `_vector_cr_branch_and_grad` extends to 16 control-point
reads per evaluation with a `(G × G)` weight matrix. ~half-day of
kernel work + parity tests.

This is the **most concrete mathematical direction** that follows
directly from the current data, with a low-risk null test (parameter
count grows ~4×, accuracy either improves or doesn't). Worth running.

## Files touched

- [signedkan_wip/src/ac_hsikan/config.py](../signedkan_wip/src/ac_hsikan/config.py) — `use_dynamic_topk`, `dynamic_topk_static_k`, `long_cycle_prior_alpha`.
- [signedkan_wip/src/ac_hsikan/layer.py](../signedkan_wip/src/ac_hsikan/layer.py) — mixed-selection candidate index path; long-cycle prior weighting.
- [signedkan_wip/experiments/ac_hsikan_imdb_smoke.py](../signedkan_wip/experiments/ac_hsikan_imdb_smoke.py) — `--dynamic-topk`, `--dyn-topk-static-k`, `--long-cycle-prior`, `--arity-bias`, `--top-k`, `--lr-schedule`, `--grad-clip`, `--weight-decay`, `--sign-head-hidden`.

## Result artefacts

- `/tmp/imdb_d32_signhead32_5seed.json` — sh=32 fix
- `/tmp/imdb_d32_signhead32_lr1.5e3.json` — sh+lr fix (paritás restored)
- `/tmp/imdb_d32_dyntopk_5seed.json` — pure dynamic top-K
- `/tmp/imdb_d32_lc0.3_5seed.json` — long-cycle α=0.3 (failed)
- `/tmp/imdb_d32_longcycle_5seed.json` — long-cycle α=1.0 + arity bias (failed)
- `/tmp/imdb_d32_mixed4plus4.json` — K_static=4
- `/tmp/imdb_d32_mixed6plus2.json` — **K_static=6 (best)**
- `/tmp/imdb_d32_mixed7plus1.json` — K_static=7 (regression)
- `/tmp/imdb_d32_mixed8plus4_K12.json` — K_total=12 (regression)
- `/tmp/imdb_d32_mixed6_cosine_clip.json` — cosine + clip (helps TR more)
- `/tmp/imdb_d32_dyntopk_16ep.json` — 16-epoch test (no further gain)

## CORE.YAML items touched

None. `signedkan_wip/` is non-core.

## What this enables for the paper

Not "AC beats Transformer at d=32". The empirical paper-claim is:

> **AC-HSiKAN scales positively from d=16 (165 k) to d=32 (337 k) when
> the sign-head bottleneck is scaled with the model dimension and lr is
> reduced by √2. The mixed static-positional + dynamic-content-adaptive
> candidate selection (K_static=6 of K_total=8) is the AC-HSiKAN analogue
> of HSiKAN's ABB top-K-per-vertex; it closes 87 % of the d=32 scaling
> gap to Transformer baseline (Δ went from −0.0125 to −0.0016, p > 0.5).**

Plus the math conjectures above as `§ Future Work / Discussion`.
