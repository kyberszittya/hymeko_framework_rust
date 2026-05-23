# Quickstart: HyMeKo-controlled training

The `hymeko_train_walker` runs a complete training cell where the **dataflow ordering, optimizer config, loss config, cycle-enumeration knobs all come from `.hymeko` files**, not from hand-coded Python.

## What you need

- An architecture `.hymeko` (e.g. `data/hsikan/arch_mixed_k34.hymeko`)
- A training `.hymeko` (e.g. `data/hsikan/training.hymeko`)
- The `hymeko` PyO3 wheel installed: `pip install hymeko-*.whl` (built from `hymeko_py/`)

## Run

```bash
PYTHONPATH=signedkan_wip python3 -m src.hymeko_train_walker \
    --arch     data/hsikan/arch_mixed_k34.hymeko \
    --training data/hsikan/training.hymeko \
    --dataset  bitcoin_alpha \
    --seed 0
```

Output (truncated):

```
[walk] dataflow ordered:
  - @load_dataset         <dataset>     +[] → -[edges]
  - @enumerate_cycles     <cycle_enum>  +[edges] → -[cycles]
  - @forward              <forward>     +[edges, cycles] → -[embeddings]
  - @classify             <forward>     +[embeddings] → -[logits]
  - @compute_loss         <loss>        +[logits, edge_labels] → -[loss_value]
  - @backprop             <backward>    +[loss_value] → -[grads]
  - @optimizer_step       <optimizer>   +[grads] → -[weights_updated]
  - @train_loop           <epoch_loop>  +[cycles, edges, edge_labels] → -[metrics]
[op_dataset]    resolved name=bitcoin_alpha train_frac=0.8
[op_cycle_enum] arities=(3, 4) mode='per_vertex' m=16 scorer='fraction_negative' pruner='davis'
[op_loss]       kind=bce entropy_lambda=0.01
[op_optimizer]  kind=adam lr=0.01 wd=0.0001 clip=1.0
[op_epoch_loop] n_epochs=30 → cell_signed_graph
{"dataset":"bitcoin_alpha","model":"HSiKAN-mixed (HyMeKo-walked)","auc":0.7105, ...}
```

## What the walker did

1. **Parsed** `training.hymeko` into a list of `FlowEdge` objects (each with `+inputs`, `~op`, `-outputs`).
2. **Topologically sorted** them by tensor input/output dependency. The `@load_dataset` edge has `+[]`, so it fires first. `@enumerate_cycles` waits for `+[edges]`. And so on.
3. **Dispatched** each kind (`dataset`, `cycle_enum`, `loss`, `optimizer`, `epoch_loop`, …) to a registered handler in `signedkan_wip/src/hymeko_train_walker.py`'s `OPS` dict.
4. **The `epoch_loop` handler** fired the actual inner training kernel (`cell_signed_graph`) with the env-vars + ctx state set by prior ops.

Reordering or removing edges in `training.hymeko` changes the walker's dispatch — order is no longer hardcoded.

## Tweak the training without touching Python

Want to try a different pruner? Edit `training.hymeko`:

```hymeko
@enumerate_cycles <cycle_enum> {
    mode "per_vertex";
    m_per_vertex 64;          # was 16
    scorer "fraction_negative";
    pruner "balance";          # was "davis"
    arities [3, 4];
    (+edges, ~enumerator, -cycles);
}
```

Re-run the walker — same Python code, different config, different result.

## Add a new training step

Want to add a calibration step after training? Add an edge:

```hymeko
@calibrate <calibration> {
    method "platt_scaling";
    (+logits, +edge_labels, ~platt, -calibrated_probs);
}
```

Then in `hymeko_train_walker.py`:

```python
@register("calibration")
def op_calibration(ctx, e):
    method = _child_value(e.body, "method", "platt_scaling")
    # ... apply to ctx.logits ...
```

The walker will pick it up automatically based on tensor dependency ordering.

## Limitations

- The walker's `epoch_loop` handler currently delegates the whole forward+loss+backward+step block to `cell_signed_graph` (a 500-line monolith from `signedkan_wip/src/run_final_cell.py`). Decomposing that into per-step ops would let you reorder inner operations from HyMeKo too. Future work.
- `forward` and `backward` ops are currently no-ops (declarations only) — they exist for dataflow shape but inner training is delegated.

## Next

- [Build an HSiKAN architecture](./08-hsikan-architecture.md) — the architecture half of the picture
- [Research code: HyMeKo-driven training](../research/hymeko-driven.md) — design rationale
