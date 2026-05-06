# Quickstart: P-graph axiom feasibility

P-graphs (Process Graphs, Friedler 1992) are a structural framework for synthesising chemical / process systems. HyMeKo includes `hymeko_pgraph` which implements:

- **MSG** (Maximal Structure Generator): trim units that can't reach a product
- **SSG** (Solution Structure Generator): enumerate all feasible structures
- **ABB** (Accelerated Branch & Bound): cost-minimum optimum subset

The same machinery applies to **architecture search** for neural networks — pick a cost-minimum subset of layer kinds + cycle enumeration policies that reaches a target accuracy.

## A chemical-process example

```bash
target/release/hymeko emit data/pgraph/hda.hymeko --format dot -o /tmp/hda.dot
dot -Tsvg /tmp/hda.dot -o /tmp/hda.svg
```

`data/pgraph/hda.hymeko` describes Hydrodealkylation of Toluene — a classic textbook process. Each unit consumes raw materials and produces products with associated cost.

## Run MSG

```bash
target/release/hymeko pgraph data/pgraph/hda.hymeko --algorithm msg
# Eliminates structurally infeasible units
```

```bash
target/release/hymeko pgraph data/pgraph/hda.hymeko --algorithm abb
# Returns the cost-minimum production structure
```

## Architecture search via P-graph

`data/hsikan/sweep_msg.hymeko` shows the same axioms applied to NN architecture choice:

```hymeko
context {
    gpu_memory   <material, raw>;
    train_time   <material, raw>;
    auc_score    <material, product>;

    @cycle_topk_m4   <unit>  10 { (-gpu_memory, -train_time, +cycle_quality); }
    @cycle_topk_m16  <unit>  40 { (-gpu_memory, -train_time, +cycle_quality); }
    @cycle_topk_m64  <unit> 160 { (-gpu_memory, -train_time, +cycle_quality); }
    @model_h16       <unit>  60 { (-gpu_memory, -cycle_quality, +embedding_quality); }
    @train_long      <unit> 120 { (-train_time, -embedding_quality, +auc_score); }
}
```

ABB picks the cost-minimum subset of (cycle topk, hidden size, training duration) that reaches `auc_score`. The search becomes a structural-feasibility branch-and-bound, not a black-box hyperparameter sweep.

## Programmatic API

```rust
use hymeko_pgraph::{lower, msg, ssg, abb_solve};

let ir = parse_and_resolve("data/pgraph/hda.hymeko")?;
let pg = lower(&ir, &resolver);
let trimmed = msg(&pg);
let solutions = ssg(&trimmed);
let optimum = abb_solve(&trimmed)?;
println!("min cost = {}, units = {:?}", optimum.cost, optimum.units);
```

## Next

- [Add a new query](../recipes/add-a-query.md) — the predicate language used inside P-graph extraction
- [Research: HyMeKo-driven HSiKAN sweeps](../research/hymeko-driven.md) — actual sweep_msg.hymeko results
