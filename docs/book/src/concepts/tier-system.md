# Concept: The Tier system

Layers are organised into tiers reflecting their compositional level.

## Tier 1: feed-forward primitives

Plain `nn.Linear` / activation building blocks. No internal hypergraph topology — just a parameter tensor + an activation.

| layer kind | emits |
|---|---|
| `linear_layer` | `nn.Linear(d_in, d_out)` |
| `relu_layer` | `nn.ReLU()` |
| `sigmoid_layer` | `nn.Sigmoid()` |
| `tanh_layer` | `nn.Tanh()` |

## Tier 2: composite blocks

Named composite patterns shipped with `ehk_torch_stub`. Carry internal structure but expose a flat field surface; the IR captures the shape contract, not the internal topology (multi-input dataflow not yet expressible at this tier).

| layer kind | structure |
|---|---|
| `residual_block` | `y = x + Linear(ReLU(Linear(x)))` |
| `highway_block` | `y = T(x)·F(x) + (1−T(x))·x`, `T = sigmoid(Linear(x))` |
| `hypergraph_conv` | `X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ` (Feng-Yu-Zhang-Ji 2019) |

## Tier 3: signed-cycle KAN primitives

The HSiKAN family. Multi-input dataflow + per-sign branch aggregation + spline activations.

| layer kind | role |
|---|---|
| `signedkan_layer` | One Option-C signed-incidence aggregation for arity k |
| `walk_layer` | Open-walk sibling of signedkan_layer (length-L walks) |
| `arity_mixer` | αₖ-weighted softmax over per-arity outputs + sparse `M_e^{(k)}` apply |
| `signed_classifier` | Linear head over edge embeddings |

## Why tiers matter

The codegen and analysis paths can specialize per tier:
- **Tier 1**: spectral-entropy regulariser walks `.weight` directly
- **Tier 2**: walks composite block's internal Linears (skip / gate connections omitted)
- **Tier 3**: walks `inner.weight` + `outer.weight` of the spline pair, plus mixer / classifier weights

`transforms/torch_dataflow/template.py`'s `spectral_weights()` method has one isinstance branch per tier-2/tier-3 type — see the file for the conventions.

## Adding a new tier

If you need a layer family that's compositional in a new way (e.g. Tier 4 for transformer blocks with internal multi-head attention), add the layer kinds in `meta_nn.hymeko`, register collections in `transforms/torch_dataflow/queries.hymeko`, and add the per-kind isinstance branch in `spectral_weights()`. See [Add a new layer kind](../recipes/add-a-layer-kind.md).

## See also

- [Generate a PyTorch nn.Module](../quickstart/06-emit-torch.md) — emit-side perspective
- [Build an HSiKAN architecture](../quickstart/08-hsikan-architecture.md) — Tier-3 in action
