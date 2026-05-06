# Summary

[Introduction](./intro.md)

# Quickstarts (use-case tutorials)

- [Parse a .hymeko file](./quickstart/01-parse.md)
- [Emit URDF for ROS](./quickstart/02-emit-urdf.md)
- [Emit SDF for Gazebo](./quickstart/03-emit-sdf.md)
- [Emit MJCF for MuJoCo](./quickstart/04-emit-mjcf.md)
- [Visualize with DOT](./quickstart/05-emit-dot.md)
- [Generate a PyTorch nn.Module](./quickstart/06-emit-torch.md)
- [Query an IR](./quickstart/07-query.md)
- [Build an HSiKAN architecture](./quickstart/08-hsikan-architecture.md)
- [HyMeKo-controlled training](./quickstart/09-hsikan-training.md)
- [P-graph axiom feasibility](./quickstart/10-pgraph.md)
- [Use the PyO3 wheel from Python](./quickstart/11-pyo3-wheel.md)
- [Use the WASM bundle in a browser](./quickstart/12-wasm.md)
- [Compute structural entropy + HOSVD](./quickstart/13-tensor-decomposition.md)

# Architecture

- [Crate map](./architecture/crate-map.md)
- [Layered architecture](./architecture/layers.md)
- [Data flow: parse → IR → query → emit](./architecture/data-flow.md)
- [Extension points](./architecture/extension-points.md)

# Concepts

- [The IR](./concepts/ir.md)
- [Queries](./concepts/queries.md)
- [Templates and codegen](./concepts/templates.md)
- [The Tier system](./concepts/tier-system.md)
- [Tensor decomposition](./concepts/tensor-decomposition.md)

# Extension recipes

- [Add a new format](./recipes/add-a-format.md)
- [Add a new layer kind](./recipes/add-a-layer-kind.md)
- [Add a new query](./recipes/add-a-query.md)
- [Debug the pipeline](./recipes/debug-pipeline.md)

# Research code

- [signedkan_wip overview](./research/signedkan-overview.md)
- [HSiKAN architecture](./research/hsikan.md)
- [HyMeKo-driven training](./research/hymeko-driven.md)

---

[Reference: env vars](./reference-env-vars.md)
[Reference: CLI](./reference-cli.md)
[Rust API (cargo doc)](./reference-rust-api.md)
