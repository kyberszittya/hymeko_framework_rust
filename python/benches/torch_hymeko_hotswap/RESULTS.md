# End-to-end HyMeKo ↔ torch hot-swap benchmark — results

**Date:** 2026-04-21
**Script:** `run_benchmark.py`
**Binary:** `target/release/hymeko` (branch `refactor/extract-hymeko-hre`)

## What the benchmark actually measures

Does the full pipeline — `hymeko compile --format torch_dataflow` → import → train → periodic recompile + `transfer_compatible_weights` → continue training — preserve training signal compared to a no-op optimizer restart? Three arms, same data, same seed, same budget:

| Arm | Intervention at each `--swap-at` epoch |
|---|---|
| `baseline` | (none) |
| `optimizer_restart` | Rebuild Adam optimizer only |
| `hymeko_recompile` | Log structural entropy + rewrite proposal; subprocess-recompile `.hymeko` → Python; instantiate fresh model; transfer compatible weights; rebuild Adam optimizer |

## Results

### `data/nn/simple_net.hymeko` (3→2 MLP, 2 layers)

5 seeds × 200 epochs × swap-at 50 (→ 3 recompile events per seed, 15 total):

| Arm | mean val loss | ± std | wall s/run |
|---|---:|---:|---:|
| `baseline` | 0.29663 | 0.04477 | 0.11 |
| `optimizer_restart` | 0.29666 | 0.04479 | 0.11 |
| `hymeko_recompile` | 0.29666 | 0.04479 | 0.13 |

Gap `optimizer_restart` → `hymeko_recompile`: **+0.00000**. Byte-identical final loss across 15 recompile+transfer cycles.

`hymeko_recompile`: **4 / 4** state_dict keys transferred at every recompile event.

### `data/nn/disjoint_net.hymeko` (two disjoint 2-layer pipelines sharing input `x`)

3 seeds × 100 epochs × swap-at 25:

| Arm | mean val loss | ± std | wall s/run |
|---|---:|---:|---:|
| `baseline` | 0.29182 | 0.06170 | 0.07 |
| `optimizer_restart` | 0.29201 | 0.06111 | 0.06 |
| `hymeko_recompile` | 0.29201 | 0.06111 | 0.09 |

Same story: `hymeko_recompile` exactly equals `optimizer_restart`. **8 / 8** keys transferred per recompile.

## What this means

**Plumbing validation ✓.** The compile + weight-transfer cycle is deterministic enough that its final training trajectory is byte-identical to a bare optimizer restart. Across two different `.hymeko` sources, three independent seeds each, 15+ recompile events: zero detectable loss from the HyMeKo round-trip.

This is the strongest possible validation of the infrastructure we have. The pipeline can be invoked mid-training without losing training signal; when a future iteration actually *changes* the `.hymeko` source between recompiles (the real entropy-feedback scenario), any delta in the training trajectory will be attributable to the structural change, not to the recompile/transfer mechanics.

## What this does *not* measure

**Does entropy feedback improve learning?** — not tested. The `.hymeko` source doesn't change during training, so the structural entropy is static (`h_sign = ln 3 ≈ 1.0986`, `h_total = 0.533` for simple_net, `0.578` for disjoint_net — unchanged across all epochs). The `hymeko_recompile` arm's "entropy feedback" reduces to "entropy logging"; no intervention is triggered.

To test the learning claim:
- **Shape-changing rewrites** need partial-row/col weight transfer (current `transfer_compatible_weights` is exact-shape-match only, so widening wipes 3/4 keys — see `python/benches/hotswap/` benchmark).
- **Cross-cluster rewrites** need port-based rewiring so the emitted `.hymeko` round-trips the compiler (step 4.5 in the improvement plan).
- **Real research claim** (structural entropy drives better representations) needs `ehk_torch` with actual signed-incidence + GGK layers; the stub here is just `nn.Linear` underneath.

See `docs/quality/benchmark_plan.md` for the full roadmap.

## Overhead

HyMeKo recompile + model instantiation + weight transfer: ~**5–7 ms per swap event** on a warm cache, measured as the wall-time difference between `hymeko_recompile` (0.13s / run) and `optimizer_restart` (0.11s / run) for a 3-recompile run. For training loops of minutes or more, negligible.

## Reproducing

```bash
cargo build -p hymeko_cli --release
python3 python/benches/torch_hymeko_hotswap/run_benchmark.py
# or with an alternative source:
python3 python/benches/torch_hymeko_hotswap/run_benchmark.py \
    --source data/nn/disjoint_net.hymeko --name DisjointNet
```

Raw per-epoch CSVs land in `data/benchmarks/hymeko_hotswap_<timestamp>.csv`.
