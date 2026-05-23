# Recipe: Add a new layer kind

Goal: extend the framework's schema with a new layer type (e.g. `attention_layer`, `lstm_layer`). After this you can write `.hymeko` descriptions that use it, and they'll round-trip through the `torch_dataflow` codegen path.

This is the recipe for the **research-side** extension (`signedkan_wip`-flavoured KAN layers). For domain-format extensions (URDF-style layer kinds), see [Add a new format](./add-a-format.md).

## The 4 places to touch

1. **`data/nn/meta_nn.hymeko`** — declare the layer kind in the schema
2. **`transforms/torch_dataflow/queries.hymeko`** — register a per-kind collection
3. **`transforms/torch_dataflow/template.py`** — emit the layer's instantiation + spectral-weights collection
4. **`python/ehk_torch_stub/src/ehk_torch_stub/__init__.py`** — runtime class

## Step 1: declare in `meta_nn.hymeko`

Add the new kind under the `layers { ... }` block:

```hymeko
layers {
    // ... existing kinds ...

    // Multi-head attention layer.
    // Fields:
    //   d_model : token / feature dim
    //   n_heads : number of attention heads
    attention_layer: + <isa> meta_layer {}
}
```

Use a comment to document the field surface — it's the contract for the runtime class.

## Step 2: register in `queries.hymeko`

```hymeko
torch_dataflow_transform {}
context
{
    // ... existing collections ...

    // Tier-3 attention layer.
    attention_layers: attention_layer {}
}
```

The query engine binds `attention_layers` to all decls inheriting from `attention_layer`.

## Step 3: emit in `template.py`

Two places to add an `{{#each attention_layers}}` block.

**Constructor** (alongside other layer instantiations):

```python
def __init__(self):
    super().__init__()
    {{#each attention_layers}}        self.{{name}} = AttentionLayer(d_model={{field:d_model}}, n_heads={{field:n_heads}})
{{/each}}
    # ... other layers ...
```

**Spectral-weights collection** (in `spectral_weights()` method):

```python
{{#each flows}}        _m = self.{{bind:~:0}}
        # ... existing isinstance branches ...
        elif isinstance(_m, AttentionLayer):
            out.append(_m.q_proj.weight)
            out.append(_m.k_proj.weight)
            out.append(_m.v_proj.weight)
{{/each}}
```

The forward dataflow walking line works automatically — `{{bind:-:0}} = self.{{bind:~:0}}({{bind:+:all_csv}})` handles any layer kind.

## Step 4: implement in `ehk_torch_stub`

```python
class AttentionLayer(nn.Module):
    """Stub Tier-3 ``attention_layer``.

    The real layer would do scaled-dot-product or quaternion attention
    (per HSIKAN_ATTENTION_M_E env var); this stub is just three Linears
    so emitted networks construct + train.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (self.d_model ** 0.5), dim=-1)
        return attn @ v
```

For Phase-1 codegen delegation (real math), have the stub lazily import from the corresponding `signedkan_wip` module — see `SignedKANLayer` in `__init__.py` for the pattern.

## Step 5: write a `.hymeko` and emit

```hymeko
my_attn_arch {
    @"meta_nn.hymeko";
    using nn.layers as lyr;
    using nn.tensors as ten;
}

my_attn_arch: lyr, ten {
    x: ten.t_input { shape [16]; }
    y: ten.t_output { shape [16]; }

    attn: lyr.attention_layer { d_model 16; n_heads 4; }

    @flow_attn: lyr.dataflow { (+ x, ~ attn, - y); }
}
```

```bash
target/release/hymeko emit /tmp/my_attn.hymeko --format torch_dataflow -o /tmp/my_attn.py
```

The emitted module:

```python
class MyAttnArch(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AttentionLayer(d_model=16, n_heads=4)

    def forward(self, x):
        y = self.attn(x)
        return y
```

## Verification

Add a test in `hymeko_query/tests/test_torch_dataflow.rs`:

```rust
#[test]
fn emits_attention_layer_instantiation() {
    let out = render_with_attention_layer();
    assert!(out.contains("AttentionLayer(d_model=16, n_heads=4)"));
}
```

The existing torch_dataflow test suite (~10 tests) is a good model.

## Next

- [Add a new format](./add-a-format.md) — extend output targets
- [Generate a PyTorch nn.Module](../quickstart/06-emit-torch.md) — the existing emit pipeline
