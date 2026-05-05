# HyMeKo-driven HSiKAN

Express HSiKAN's architecture, training loop, and tuning policy
in HyMeKo `.hymeko` files; let a tiny driver read them and run
real training. Every result row carries the source `.hymeko`
provenance.

## TL;DR

```bash
# Single training cell driven by .hymeko config:
python3 -m signedkan_wip.src.hymeko_driver \
    --arch     data/hsikan/arch_mixed_k34.hymeko \
    --training data/hsikan/training.hymeko \
    --dataset  bitcoin_alpha

# Grid sweep:
python3 -m signedkan_wip.src.hymeko_driver \
    --sweep    data/hsikan/sweep_grid.hymeko \
    --arch     data/hsikan/arch_mixed_k34.hymeko \
    --training data/hsikan/training.hymeko \
    --dataset  bitcoin_alpha \
    --output   /tmp/sweep.jsonl
```

## Files

| path | purpose |
|---|---|
| `data/hsikan/arch_single_k3.hymeko` | single-arity HSiKAN (k=3 only) |
| `data/hsikan/arch_mixed_k34.hymeko` | mixed-arity (k=3 + k=4) with arity_mixer |
| `data/hsikan/training.hymeko` | dataset → cycle_enum → forward → loss → backward → optimizer dataflow |
| `data/hsikan/sweep_grid.hymeko` | Cartesian-product hyperparameter sweep |
| `data/hsikan/sweep_genetic.hymeko` | GA-driven search |
| `data/hsikan/sweep_msg.hymeko` | P-graph axiom-feasibility (MSG/SSG/ABB) |
| `signedkan_wip/src/hymeko_driver.py` | parse → instantiate → train |
| `hymeko_py/src/hymeko_parse.rs` | PyO3 bridge wrapping `parser::parse_description` |

## Schema

### Tensors and layers

```hymeko
HSiKAN_mixed_k34 {
    const HIDDEN = 16;
    const GRID   = 5;
}
context {
    // Tensors are nodes tagged by role.
    t_in       <input>;
    t_emb_k3   <activation>;
    t_emb_k4   <activation>;
    t_emb      <activation>;
    t_out      <output>;

    // Layers inherit from a layer-class base; sub-tags are kwargs.
    sk_k3 : signedkan_layer { hidden 16; arity 3; grid 5; spline_kind "catmull_rom"; }
    sk_k4 : signedkan_layer { hidden 16; arity 4; grid 5; spline_kind "catmull_rom"; }
    mixer : arity_mixer     { hidden 16; mix_K 2; }
    head  : signed_classifier { d_in 16; d_out 1; }

    // Dataflow: (+input_tensor[, +input_tensor], ~layer, -output_tensor).
    @flow_k3   <dataflow> { (+t_in,    ~sk_k3, -t_emb_k3); }
    @flow_k4   <dataflow> { (+t_in,    ~sk_k4, -t_emb_k4); }
    @flow_mix  <dataflow> { (+t_emb_k3, +t_emb_k4, ~mixer, -t_emb); }
    @flow_head <dataflow> { (+t_emb,    ~head,  -t_out); }
}
```

Convention matches `transforms/torch_dataflow/template.py`:
- `+` = input tensor
- `~` = layer / operation
- `-` = output tensor

Multi-input fan-in → multiple `+`-signed sources in one hyperarc.

**Important grammar constraints** (learned the hard way overnight):
- `const NAME = value;` declarations only allowed in the
  description's *header* block (between description name and the
  first context), not inside `context { ... }`.
- Tags are identifier lists only — no `key="value"` syntax inside
  `<...>`. Use child node statements (`name "value";`) instead.

### Training & feedback dataflow

```hymeko
context {
    @load_dataset <dataset> {
        name        "bitcoin_alpha";
        train_frac   0.8;
        (~dataset, -edges);
    }

    @enumerate_cycles <cycle_enum> {
        mode          "per_vertex";   // or "global"
        m_per_vertex   16;
        scorer        "fraction_negative";
        pruner        "davis";        // strategy/command pattern!
        arities       [3, 4];
        (+edges, ~enumerator, -cycles);
    }

    @forward <forward> {
        model_ref     "arch_mixed_k34";
        (+edges, +cycles, ~model, -embeddings);
    }

    @compute_loss <loss> {
        loss_kind         "bce";
        entropy_lambda    0.01;       // spectral-entropy regulariser
        entropy_kind     "spectral";
        (+logits, +edge_labels, ~bce_with_entropy, -loss_value);
    }

    @optimizer_step <optimizer> {
        kind          "adam";
        lr             0.01;
        weight_decay   0.0001;
        (+grads, ~adam, -weights_updated);
    }

    @train_loop <epoch_loop> {
        n_epochs   30;
        (+cycles, +edges, +edge_labels, ~loop, -metrics);
    }
}
```

**Strategy/command pattern**: changing `pruner "davis";` to
`pruner "balance";` swaps the axiom pruner without any code edit.
The driver maps the field to `HSIKAN_TOPK_PRUNER` env var.

### Tuning policies

#### Grid sweep

```hymeko
@sweep_topk_m <param_range> { target "topk.m_per_vertex"; values [4, 16, 64]; }
@sweep_pruner <param_range> { target "topk.pruner";       values ["none", "davis", "balance"]; }
@policy <sweep_policy> { kind "full_grid"; max_runs 50; select_metric "auc"; }
```

#### Genetic algorithm

```hymeko
@policy <sweep_policy> {
    kind             "genetic";
    population_size   16;
    generations       10;
    elite_keep         4;
    mutation_rate      0.2;
    crossover_rate     0.6;
}
```

#### P-graph axiom feasibility (MSG/SSG/ABB)

Architecture choices encoded as a **P-graph** — same axioms used in
`data/pgraph/hda.hymeko` (chemical-process synthesis), now applied
to neural-architecture search:

```hymeko
context {
    gpu_memory   <material, raw>;
    train_time   <material, raw>;
    auc_score    <material, product>;

    // Each unit consumes resources, produces quality. Cost = edge value.
    @cycle_topk_m4   <unit>  10 { (-gpu_memory, -train_time, +cycle_quality); }
    @cycle_topk_m16  <unit>  40 { (-gpu_memory, -train_time, +cycle_quality); }
    @cycle_topk_m64  <unit> 160 { (-gpu_memory, -train_time, +cycle_quality); }
    @model_h16       <unit>  60 { (-gpu_memory, -cycle_quality, +embedding_quality); }
    @train_long      <unit> 120 { (-train_time, -embedding_quality, +auc_score); }
}
```

`hymeko_pgraph::lower` reads this as a P-graph; MSG trims units
that can't reach `auc_score`; ABB picks the cost-minimum subset.
Architecture search becomes a structural-feasibility
branch-and-bound.

## Driver

The driver does three things:

1. **Parse** `.hymeko` via `hymeko.parse_hymeko_rs` → nested dict.
2. **Extract** salient knobs (arch: hidden/grid/arities/spline_kind;
   training: topk_*, lr, n_epochs, entropy_lambda).
3. **Dispatch** to the existing `cell_signed_graph` with `HSIKAN_TOPK_*`
   env vars set.

Smoke-tested: `arch_mixed_k34.hymeko + training.hymeko + bitcoin_alpha`
→ AUC = 0.7930 (reproduces hand-coded m=16+davis result exactly).

### Output schema

Each result row carries the provenance:

```json
{
  "dataset": "bitcoin_alpha",
  "model": "HSiKAN-mixed",
  "auc": 0.7930,
  "f1m": 0.4644,
  "fwd_per_call_ms": 20.6,
  "hymeko_arch": "HSiKAN_mixed_k34",
  "hymeko_training": {
    "topk_mode": "per_vertex",
    "topk_k": 16,
    "topk_scorer": "fraction_negative",
    "topk_pruner": "davis",
    "n_epochs": 30,
    "lr": 0.01
  }
}
```

### Adding a new pruner

Just add it to the `pick_pruner` switch in
`hymeko_py/src/cycles.rs` and use the new tag value in
`training.hymeko`:

```hymeko
@enumerate_cycles <cycle_enum> {
    pruner "frustration_davis_combined";  // your new tag
}
```

No driver change needed.

## Building

```bash
cd hymeko_py && maturin build --release
pip install --force-reinstall --no-deps \
    /path/to/target/wheels/hymeko-0.1.0-cp313-*-manylinux_*.whl
```

## Open work

- **GA driver branch** — currently the genetic-policy file parses
  but the driver still runs it as a grid (TODO).
- **MSG driver branch** — call `hymeko_pgraph::abb_solve` on the
  parsed P-graph and emit the optimum architecture as a synthesised
  `arch_*.hymeko`.
- **Round-trip codegen** — `hymeko compile arch_mixed_k34.hymeko
  --format torch_dataflow` should emit a `torch.nn.Module` class
  directly. The Tier-3 layer kinds are already in
  `transforms/torch_dataflow/template.py`; needs a small
  `queries.hymeko` extension to recognise our `<dataflow>` shape.
- **Driver-emit-back** — after sweep, write the best-cell config
  to a new `.hymeko` for reproducibility.
