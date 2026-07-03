# HSiKAN — overview & capabilities

**HSiKAN = Highway Signed KAN.** Signed-hypergraph message passing whose edge activations are **K**olmogorov–
**A**rnold **N**etwork splines (Catmull-Rom), with **Highway** gating (Srivastava, Greff & Schmidhuber, 2015) so
deep signed message passing does not degrade. One body, two conventions (RL control + signed-graph link-sign),
unified in the pure-torch `hymeko_neuro/core/` package.

---

## 1. The architecture

A signed-KAN layer over a (hyper)graph:

```
h' = skip( σ_CR( W_self·h + W₊·(A⁺h) + W₋·(A⁻h) ), h )
```

- **Signed message passing** — separate positive (`A⁺`) and negative (`A⁻`) incidence; a vertex aggregates
  friend- and foe-messages independently (the *signed* graph premise).
- **CR activation** (`σ_CR`) — a learnable per-channel **Catmull-Rom** spline (the KAN nonlinearity), not a fixed
  ReLU/tanh. This is the **K** in HSiKAN.
- **Highway skip** — `T·H + (1−T)·h`, `T = σ(W_T·h)`, bias −2 (carry-dominant init). This is the **H**.
- **Pool / readout** — a head turns per-vertex reps into the task output (below).

## 2. Four orthogonal capability axes (all config, no forked variants)

| axis | values | where |
|---|---|---|
| **aggregation backend** | `DenseBatchedBackend` · `SparseSignedBackend` · (Triton, legacy) | `hymeko_neuro/core/backends.py` |
| **edge spline** | Catmull-Rom (`cr`) · ReLU/tanh (ablation) · B-spline/KB (legacy) | `hymeko_neuro/core/splines.py` |
| **skip / highway** | `none` · `residual` · `highway` | `hymeko_neuro/core/layer.py` |
| **incidence** | `fixed` · `learned` · `weighted` | `hymeko_neuro/core/backbone.py` |

**Incidence modes** (how `A±` carries its arc weights):
- `fixed` — the binary kinematic/graph structure (sign + row-normalised magnitude).
- `learned` — the full `A±` is trainable (any real value on any pair; sparsity not preserved).
- `weighted` — the structural **mask** is fixed, a **per-arc weight is learned** (init 1.0 → parity): free real
  weights on the *real* arcs only. This is the signed-hypergraph premise (real arc weights) without losing the
  sparsity prior. See [`hymeko_arc_weights.md`](hymeko_arc_weights.md).

## 3. One body, changeable head (the unification)

The body — `hymeko_neuro.core.SignedKANBackbone` (CR + highway + weighted incidence) — is shared; only the **input
adapter** and the **head** change per convention:

| convention | input adapter | head | output |
|---|---|---|---|
| **RL control** | dynamic per-vertex features `(B, N, feat)` | pooled actor/critic (`hymeko_rl.ActorCritic`) | action mean + value |
| **signed graph** | transductive node embeddings (`SignedGraphHSiKAN`) | `hymeko_neuro.core.EdgeSignHead` | per-edge sign logit |

So a signed-graph link-sign task runs on the **same CR HSiKAN** as the robot controller — `SignedGraphHSiKAN`
= `Embedding → SignedKANBackbone(cr) → EdgeSignHead`, scalable via `SparseSignedBackend`.

## 4. What's where

| file | role |
|---|---|
| `hymeko_neuro/core/splines.py` | the canonical Catmull-Rom evaluator (shared by all lines) + `CatmullRomActivation` |
| `hymeko_neuro/core/backends.py` | aggregation Strategy: dense (RL), sparse (large graphs) |
| `hymeko_neuro/core/layer.py` | `SignedKANLayer` + `HighwaySkip` |
| `hymeko_neuro/core/backbone.py` | `SignedKANBackbone` (incidence + stack + pool) |
| `hymeko_neuro/core/heads.py` | `EdgeSignHead` (signed-graph head) |
| `hymeko_neuro/core/graph_model.py` | `SignedGraphHSiKAN` (signed-graph convention assembly) |
| `hymeko_rl/policy.py` | `HSiKANBackbone` = thin `hg_state` adapter over the core; `ActorCritic` (RL head) |
| `hymeko_neuro/hyperedge/signedkan.py` | **legacy** k-uniform triad + dual-spline layer (owns the current OTC numbers; imports the shared CR) |

## 5. Capabilities summary

- Signed (friend/foe) message passing over arbitrary signed incidence.
- **Learnable KAN edge functions** (Catmull-Rom; B-spline / Kochanek-Bartels available in the legacy line).
- **Highway / residual** skips for depth (Schmidhuber).
- **Real-valued arc weights** on the existing structure (`incidence="weighted"`) — not binary {0,±1}.
- **Dense** (batched, inductive — RL) and **sparse** (transductive, large-graph — signed link-sign) backends.
- **Pluggable head**: actor/critic (control) or edge-sign (link prediction) over one body.
- Trained-policy round-trip to/from `.hymeko` (`hymeko_rl.policy_store`); state-dict-stable across the
  core refactor (old checkpoints still load).

## 6. Design history (why each capability exists)

- **Binary {0,±1} incidence defeats *signed* hypergraphs** — the whole point is real-valued arc weights →
  `incidence="weighted"`.
- **The RL backbone was a *truncated* HSiKAN** (no highway, no residual, binary incidence) → highway + weighted
  incidence restored; the "HSiKAN ≈ MLP" Galambos tie was measured on the truncated backbone (a confound).
- **CR is HSiKAN's identity** — re-homed as the single canonical spline; the legacy OTC config used B-spline, so
  "OTC on CR" is the open validation experiment.
- **The two lines are different algorithms, not one with a backend swap** (legacy = k-uniform triad + dual spline;
  RL = pairwise single spline) → unified as *one CR body + changeable head*, not a forced layer merge.

## 7. Open / next

- **OTC-on-CR A/B**: train `SignedGraphHSiKAN` (CR) on Bitcoin-OTC link-sign, compare AUC to the legacy
  triad-B-spline (the user's "OTC should use CR").
- **Nonlinear reward / action head / strategy**: push the CR primitive to the readout sites — see
  [`nonlinear_rl_handout.md`](nonlinear_rl_handout.md).
