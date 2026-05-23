# Quickstart: Use the PyO3 wheel from Python

`hymeko_py` builds a Python wheel with `maturin`. Install once, then use HyMeKo from any Python script.

## Build + install

```bash
cd hymeko_py
maturin build --release
pip install --force-reinstall --no-deps \
    /path/to/hymeko_framework_rust/target/wheels/hymeko-*.whl
```

Or, if `pip install hymeko` works (when published).

## Parse + emit

```python
import hymeko

src = open("data/robotics_imported/wam/wam.hymeko").read()
doc = hymeko.compile_description(src)

# Emit any format
print(doc.to_urdf("wam7")[:200])
print(doc.to_sdf("wam7")[:200])
print(doc.to_dot("wam7")[:200])

# Snapshot JSON for visualization
snap = doc.snapshot_json()
import json
print(json.loads(snap)["node_count"], "nodes,", json.loads(snap)["edge_count"], "edges")
```

## Query

```python
links = doc.query("INHERITS(link)")
print(links)  # ['base_link', 'shoulder_link', ...]

n_revolute = doc.query_count("INHERITS(rev_joint)")
print(f"{n_revolute} revolute joints")
```

## Cycle enumeration (HSiKAN research)

```python
import numpy as np

# Edges as (n_edges, 2) int array; signs as (n_edges,) +1/-1 array
edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3]], dtype=np.int64)
signs = np.array([1, 1, -1, 1], dtype=np.int64)
n_nodes = 4

# Enumerate all signed 3-cycles (triangles)
triad_v, triad_sigma = hymeko.enumerate_k_cycles_rs(
    edges, signs, n_nodes, k=3, max_cycles=10000, seed=0,
)
print(f"{triad_v.shape[0]} cycles, shape={triad_v.shape}")
```

## Tensor expansions

```python
# Compile clique-expansion to a 3D sparse COO tensor
coo = hymeko.compile_clique_tensor_expansion(doc.ir)
print(coo.shape, coo.nnz)

# Zero-copy export to PyTorch
indices, values = coo.export_to_pytorch()
import torch
t = torch.sparse_coo_tensor(indices, values, coo.shape)
```

## Walker integration

The walker (`signedkan_wip/src/hymeko_train_walker.py`) uses the wheel for `parse_hymeko_rs`. See [HyMeKo-controlled training](./09-hsikan-training.md) for the full training-cell driver.

## Where the surface is defined

- Free functions: `hymeko_py/src/lib.rs`
- The compiled-document class (`PyHypergraphIR`): `hymeko_py/src/interface_python/api.rs`
- Cycle enumeration: `hymeko_py/src/cycles.rs`

## Next

- [Use the WASM bundle in a browser](./12-wasm.md) — same API surface, different host
- [Compute structural entropy + HOSVD](./13-tensor-decomposition.md)
