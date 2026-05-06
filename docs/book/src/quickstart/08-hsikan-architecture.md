# Quickstart: Build an HSiKAN architecture

HSiKAN — Hypergraph Signed Kolmogorov–Arnold Network — is a signed-cycle graph neural network. Its architecture is expressible as a `.hymeko` description. This quickstart walks through writing one from scratch.

## What HSiKAN is

For each arity `k ∈ {2, 3, 4, 5}`, a `signedkan_layer` realises Option-C signed-incidence aggregation:

$$h_c = \sum_{s \in \{+, -\}} \phi_e^s\!\left(\sum_{i: \sigma_i = s} \phi_v^s(h_{v_i})\right)$$

with batched Catmull–Rom (or B-spline / Kochanek–Bartels) splines on `φ_v^s` and a diagonal-fused `φ_e^s`. Per-arity outputs are combined by an `arity_mixer` with learnable softmax weights `αₖ` plus per-arity sparse signed-incidence mm `M_e^{(k)}`.

## Write the description

Create `/tmp/my_hsikan.hymeko`:

```hymeko
my_hsikan_description {
    @"meta_nn.hymeko";
    using nn.layers as lyr;
    using nn.tensors as ten;
}

my_hsikan: lyr, ten {
    // ── Input + cycle structure tensors ────────────────────────
    x: ten.t_input { shape [16]; }

    // For each arity k: vertex IDs + signs + sparse incidence M_e
    triad_v_k3:     ten.t_input { shape [3]; }
    triad_sigma_k3: ten.t_input { shape [3]; }
    triad_v_k4:     ten.t_input { shape [4]; }
    triad_sigma_k4: ten.t_input { shape [4]; }
    M_e_k3: ten.t_input { shape [1]; }
    M_e_k4: ten.t_input { shape [1]; }

    // ── Activations + outputs ──────────────────────────────────
    cyc3_emb: ten.t_activation { shape [16]; }
    cyc4_emb: ten.t_activation { shape [16]; }
    edge_emb: ten.t_activation { shape [16]; }
    y: ten.t_output { shape [1]; }

    // ── Layers ─────────────────────────────────────────────────
    sk3: lyr.signedkan_layer {
        hidden 16; arity 3; spline_kind "catmull_rom"; grid 5;
    }
    sk4: lyr.signedkan_layer {
        hidden 16; arity 4; spline_kind "catmull_rom"; grid 5;
    }
    mixer: lyr.arity_mixer { hidden 16; mix_K 2; }
    head:  lyr.signed_classifier { d_in 16; d_out 1; }

    // ── Dataflow ───────────────────────────────────────────────
    @flow_sk3: lyr.dataflow { (+ x, + triad_v_k3, + triad_sigma_k3, ~ sk3, - cyc3_emb); }
    @flow_sk4: lyr.dataflow { (+ x, + triad_v_k4, + triad_sigma_k4, ~ sk4, - cyc4_emb); }
    @flow_mix: lyr.dataflow {
        (+ cyc3_emb, + cyc4_emb, + M_e_k3, + M_e_k4, ~ mixer, - edge_emb);
    }
    @flow_head: lyr.dataflow { (+ edge_emb, ~ head, - y); }
}
```

Three things to notice:

1. **Input tensors include cycle structure**: `triad_v_k3` (vertex IDs of each cycle), `triad_sigma_k3` (signs of each cycle's edges). These flow into the per-arity `signedkan_layer` alongside the vertex features `x`.
2. **`M_e` is also an input**: the per-arity sparse incidence matrix that maps cycle embeddings to edge embeddings.
3. **Multi-input dataflow**: the `@flow_mix` arc has 4 `+` operands (cyc embeddings + M_e per arity), which compile to `mixer(cyc3_emb, cyc4_emb, M_e_k3, M_e_k4)`.

## Emit a runnable Module

```bash
target/release/hymeko emit /tmp/my_hsikan.hymeko \
    --format torch_dataflow --name MyHSiKAN -o /tmp/my_hsikan.py
```

Inspect the emitted `forward`:

```python
def forward(self, x, triad_v_k3, triad_sigma_k3, triad_v_k4, triad_sigma_k4, M_e_k3, M_e_k4):
    cyc3_emb = self.sk3(x, triad_v_k3, triad_sigma_k3)
    cyc4_emb = self.sk4(x, triad_v_k4, triad_sigma_k4)
    edge_emb = self.mixer(cyc3_emb, cyc4_emb, M_e_k3, M_e_k4)
    return self.head(edge_emb)
```

## Train it (next quickstart)

This emits the architecture. To run a full training cell driven by HyMeKo, see [HyMeKo-controlled training](./09-hsikan-training.md).

## Architectural variants

To experiment with different setups, just edit the `.hymeko`:

| change | what it explores |
|---|---|
| `arity 5` on a new `sk5` layer | k=5 cycles (the αₖ regime-compass story) |
| `spline_kind "kochanek_bartels"` | TCB-parameterised splines (sharp/smooth control) |
| `hidden 32` | wider features |
| add a `walk_layer { hidden 16; walk_len 3; }` | open-walk variant (Walk-HSiKAN) |
| add a second `signedkan_layer` per arity | n_layers=2 stack (with shared weights) |

The codegen path picks each up automatically — no Python edits needed.

## Next

- [HyMeKo-controlled training](./09-hsikan-training.md) — run the architecture on a real dataset
- [Generate a PyTorch nn.Module](./06-emit-torch.md) — the underlying emit mechanic
- [Research code: HSiKAN](../research/hsikan.md) — what's stable, what's experimental
