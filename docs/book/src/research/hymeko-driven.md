# Research: HyMeKo-driven training

For the runnable walkthrough → [Quickstart: HyMeKo-controlled training](../quickstart/09-hsikan-training.md).

## Why drive training from a `.hymeko` file

A training run has many knobs:
- Architecture (arity set, hidden dim, n_layers, spline kind)
- Cycle enumeration policy (mode, top-K m, scorer, axiom pruner)
- Training schedule (n_epochs, lr, weight decay, grad clip)
- Loss configuration (BCE, entropy regulariser λ, kind)

Hand-coded Python smears these across env vars + CLI flags + scripts. A `.hymeko` declarative config:
- **Provenance** — every result row carries the source `.hymeko`
- **Reproducibility** — re-run is a single CLI invocation
- **Diff-able** — config changes show up as version-controlled diffs, not bash history
- **Composable** — mix-and-match arch.hymeko × training.hymeko × sweep.hymeko

## Walker pattern

`signedkan_wip/src/hymeko_train_walker.py` parses training.hymeko into FlowEdges, topo-sorts on tensor I/O dependency, dispatches each kind to a registered handler:

```python
@register("dataset")
def op_dataset(ctx: Ctx, e: FlowEdge): ...

@register("cycle_enum")
def op_cycle_enum(ctx: Ctx, e: FlowEdge): ...

@register("loss")
def op_loss(ctx: Ctx, e: FlowEdge): ...

@register("optimizer")
def op_optimizer(ctx: Ctx, e: FlowEdge): ...

@register("epoch_loop")
def op_epoch_loop(ctx: Ctx, e: FlowEdge):
    # Fires cell_signed_graph for the inner training
    ...
```

## What the dataflow walking buys

Reordering edges in `training.hymeko` changes dispatch order. Adding a new training-graph op kind = registering a new handler — no Python edits to existing code.

Today the walker's `epoch_loop` handler delegates the whole forward+loss+backward+step block to `cell_signed_graph` (a 500-line monolith). Future work: decompose into per-step ops so even inner reordering is HyMeKo-driven.

## Companion files

- `data/hsikan/arch_*.hymeko` — architecture descriptions (signedkan_layer × N + arity_mixer + signed_classifier)
- `data/hsikan/training.hymeko` — the canonical training dataflow
- `data/hsikan/sweep_grid.hymeko` — grid sweep policy
- `data/hsikan/sweep_genetic.hymeko` — GA policy (parses but currently runs as grid)
- `data/hsikan/sweep_msg.hymeko` — P-graph axiom-feasibility sweep

## Next steps for this line

`docs/plans_rl_al_hsikan_2026_05_06.md` sketches a controller (bandit / RL / GP+EI) over the HSiKAN action space, using the walker as the cell evaluator (config in, AUC out).

## See also

- [Quickstart: HyMeKo-controlled training](../quickstart/09-hsikan-training.md)
- [HSiKAN architecture](./hsikan.md)
- `signedkan_wip/src/hymeko_train_walker.py` — implementation
