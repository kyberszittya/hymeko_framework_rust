# Quickstart: Generate a PyTorch nn.Module

The `torch_dataflow` format takes a `.hymeko` declaration of a neural-network architecture and emits a runnable Python `nn.Module` class.

## Pick an architecture

The repo ships several:
- `data/nn/hsikan_mixed.hymeko` — the mixed-arity HSiKAN (signed-cycle KAN)
- `data/hsikan/arch_mixed_k34.hymeko` — k=3+k=4 variant
- `data/hsikan/arch_single_k3.hymeko` — single-arity reference

## Emit

```bash
target/release/hymeko emit \
    data/nn/hsikan_mixed.hymeko \
    --format torch_dataflow \
    --name HSiKANEmitted \
    -o /tmp/hsikan_emitted.py
```

Output:

```
Wrote 7707 bytes to /tmp/hsikan_emitted.py
```

Inspect:

```python
class HSiKANEmitted(nn.Module):
    def __init__(self):
        super().__init__()
        self.sk2 = SignedKANLayer(hidden=16, arity=2, spline_kind="catmull_rom", grid=5)
        self.sk3 = SignedKANLayer(hidden=16, arity=3, spline_kind="catmull_rom", grid=5)
        self.sk4 = SignedKANLayer(hidden=16, arity=4, spline_kind="catmull_rom", grid=5)
        self.sk5 = SignedKANLayer(hidden=16, arity=5, spline_kind="catmull_rom", grid=5)
        self.mixer = ArityMixer(hidden=16, mix_K=4)
        self.head  = SignedClassifier(d_in=16, d_out=1)

    def forward(self, x, triad_v_k2, triad_sigma_k2, ..., M_e_k5):
        cyc2_emb = self.sk2(x, triad_v_k2, triad_sigma_k2)
        cyc3_emb = self.sk3(x, triad_v_k3, triad_sigma_k3)
        ...
        edge_emb = self.mixer(cyc2_emb, cyc3_emb, cyc4_emb, cyc5_emb,
                                M_e_k2, M_e_k3, M_e_k4, M_e_k5)
        return self.head(edge_emb)
```

The `forward` signature is **derived from the `t_input` declarations** — every input tensor declared in the `.hymeko` becomes a positional parameter.

## Run it

```python
import sys; sys.path.insert(0, "/path/to/hymeko_framework_rust")
sys.path.insert(0, "/path/to/hymeko_framework_rust/python/ehk_torch_stub/src")

import importlib.util
spec = importlib.util.spec_from_file_location("hsikan_emitted", "/tmp/hsikan_emitted.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

m = mod.HSiKANEmitted()
print(f"params={sum(p.numel() for p in m.parameters())}")  # 3749
```

The emitted layers (`SignedKANLayer` etc.) come from `ehk_torch_stub`. As of the May 2026 Phase-1 codegen change, the stub **delegates to the real `signedkan_wip.src.signedkan.SignedKANLayer`** when `signedkan_wip/` is on the import path — the emitted module computes the actual signed-cycle aggregation, not a placeholder.

## End-to-end smoke test

```bash
python3 scripts/verify_hsikan_emit.py
```

Builds an IR, emits the module, constructs synthetic cycles + `M_e` incidence, runs forward + backward + an SGD step, asserts loss decreases.

## Add your own layer kind

The template expects layer types like `signedkan_layer`, `linear_layer`, `arity_mixer` to be declared in `data/nn/meta_nn.hymeko`. To add a new one:

1. Declare it in `meta_nn.hymeko` under `layers { ... }`
2. Add a per-kind collection in `transforms/torch_dataflow/queries.hymeko`
3. Add an `{{#each new_layer}}...{{/each}}` block in `transforms/torch_dataflow/template.py`
4. Implement the runtime class in `python/ehk_torch_stub/src/ehk_torch_stub/__init__.py`

See [Add a new layer kind](../recipes/add-a-layer-kind.md) for the full walk-through.

## Next

- [HyMeKo-controlled training](./09-hsikan-training.md) — once you have an emit, drive a real training cell from the dataflow
- [Build an HSiKAN architecture](./08-hsikan-architecture.md) — write a new architecture .hymeko from scratch
