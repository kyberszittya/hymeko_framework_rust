# hymeko

Python bindings for the HyMeKo hypergraph framework.

HyMeKo is a hypergraph-based modelling language for robotic systems with
multi-context state representation. This package exposes the Rust core as
a native Python extension.

## Install (from source)

    pip install maturin
    maturin develop --release -m hymeko_py/Cargo.toml

Or build a wheel:

    maturin build --release -m hymeko_py/Cargo.toml

## Quick start

```python
import hymeko

engine = hymeko.PyHypergraphEngine()
ir = engine.load_file("examples/paper/hymeko_robot.hymeko")

print(ir)                  # nodes/edges/arcs summary
print(ir.query("KIND(joint)"))                          # ['j1', 'j2', 'j3', 'j4']
print(ir.query_count("KIND(sensor) AND HASARCREF(+1, KIND(joint))"))  # 3

urdf_xml = ir.to_urdf("mini_arm")
with open("mini_arm.urdf", "w") as f:
    f.write(urdf_xml)
```

## Public surface

- `PyHypergraphEngine` — parse `.hymeko` sources, compile to IR, compute
  star / clique tensor expansions (Arrow zero-copy to PyTorch).
- `PyHypergraphIR` — compiled IR; introspection, predicate queries,
  URDF / SDF emission, CBOR serialisation.
- `PyTensorCoo3D`, `PySparseMatrix2D` — sparse tensor exports.

## Notes

- **URDF joint emission** expects the joint-subtype taxonomy
  (`rev_joint`, `conti_joint`, `fixed_joint`, `prismatic_joint`). A
  declaration like `@j1: + <isa> joint {}` is not enough — use e.g.
  `@j1: + <isa> rev_joint { ... }` to have the joint picked up.
- The query predicate language mirrors `queries/standard.qlist`:
  `KIND`, `INHERITS`, `SCOPEDIN`, `HASARCREF`, `ANY`, and `AND`.
