# HyMeKo Structure Toy Models

Date: 2026-07-01

## Summary

Built a small parsed-HyMeKo toy source and a development harness that projects the parsed AST into tensors for four existing model families:

- HSIKAN via `MiddleHSiKAN`
- Gomb via `HymeKoGomb`
- Gomb-Soma via `WalkConvLayer`
- FSR via `FiberSpikeRotorMixer`

The harness is intentionally outside core: it lives in `scripts/dev` and consumes `hymeko.parse_hymeko_rs` without changing parser, IR, dependency, or core crates.

## Files

- `scripts/dev/hymeko_structure_toy.hymeko`
  - Six graph nodes with tier, token, and feature fields.
  - Four signed `graph.cycle` triples.
  - Four signed `graph.walk` triples.
  - Three signed `graph.link` pairs.
  - Five `graph.sequence` pairs.
- `scripts/dev/hymeko_structure_toy_models.py`
  - Parses the toy source through `hymeko.parse_hymeko_rs`.
  - Projects AST nodes and signed arcs into tensor-ready graph structure.
  - Runs HSIKAN, Gomb, Gomb-Soma, and FSR adapters with fixed seeds.
  - Writes JSON metrics with `--out`.
- `hymeko_neuro/tests/test_hymeko_structure_toy_models.py`
  - Checks parsed projection counts/shapes.
  - Checks all four adapters run on CPU and emit finite outputs.
- `reports/2026-07-01-hymeko-structure-toy-models.json`
  - Captured metrics from the current run.

## Current Toy Numbers

Input projection:

- nodes: 6
- cycles: 4
- walks: 4
- links: 3
- sequence length: 6

Model outputs:

| Model | Output shape | Params | Summary |
| --- | ---: | ---: | --- |
| HSIKAN | `[6, 8]` | 200 | sum `2.8669896126`, std `0.0923119709` |
| SA-HSIKAN proxy | `[6, 8]` | 80 | sum `3.0766038895`, std `0.0823436603` |
| Gomb | `[3]` | 5,181 | mean `-0.0407572091`; scores `[-0.0479526855, -0.0359493643, -0.0383695811]` |
| Fixed-topology Gomb | `[3]` | 5,181 | max abs delta vs Gomb `3.7252902985e-09` |
| Gomb-Soma | `[6, 8]` | 208 | sum `3.6164333820`, std `0.2212398350` |
| FSR | `[1, 6, 6]` | 194 | sum `0.2637821138`, std `0.3282744288` |

All outputs were finite.

## Performance

Measured on CPU from the local `uv --group ml` environment. The hot-path timings below parse and instantiate once, then run cached model forwards under `torch.inference_mode()` with one CPU thread. They do not represent trained-model throughput.

| Scope | Mean us | Median us | Min us | Max us |
| --- | ---: | ---: | ---: | ---: |
| HSIKAN, eager | 1,794.31 | 1,760.80 | 1,314.60 | 5,581.80 |
| SA-HSIKAN proxy, eager | 632.38 | 616.00 | 391.10 | 1,182.10 |
| Gomb, eager | 5,117.39 | 4,966.45 | 3,892.20 | 11,604.00 |
| Fixed-topology Gomb, eager | 3,965.16 | 3,910.10 | 2,896.10 | 6,933.30 |
| Gomb-Soma, eager | 470.53 | 442.45 | 328.00 | 1,867.50 |
| FSR, eager | 958.65 | 940.10 | 707.10 | 1,801.40 |

For this fixed toy graph shape, optional TorchScript tracing removes more Python dispatch:

| Scope | Mean us | Median us | Min us | Max us |
| --- | ---: | ---: | ---: | ---: |
| HSIKAN, traced | 1,067.32 | 1,032.80 | 719.60 | 1,724.40 |
| SA-HSIKAN proxy, traced | 458.71 | 433.55 | 293.40 | 950.10 |
| Gomb, traced | 2,906.69 | 2,786.00 | 2,096.70 | 5,715.90 |
| Fixed-topology Gomb, traced | 2,052.51 | 1,995.30 | 1,373.50 | 10,368.10 |
| Gomb-Soma, traced | 290.12 | 270.80 | 206.10 | 609.50 |
| FSR, traced | 650.01 | 613.55 | 447.20 | 2,424.70 |

The fixed-topology Gomb prototype caches the toy graph's cycle incidence and CPML tier subsets as linear pooling operators. It is numerically equivalent to the regular path on this fixture and improves Gomb:

- eager: `5,043.65 us -> 3,918.53 us` (`1.29x`)
- traced: `2,906.69 us -> 2,052.51 us` (`1.42x`)

The SA-HSIKAN proxy is not parity with the full HSIKAN layer; it is the fixed-topology B^L-collapse analogue from the RL actor stack. It accelerates the HSIKAN-shaped path:

- eager median: `1,760.80 us -> 616.00 us` (`2.86x`)
- traced median: `1,032.80 us -> 433.55 us` (`2.38x`)

## Optimization Findings

Spatial-tree style optimization:

- The reusable idea is not tree geometry itself; it is precomputing topology as a linear operator. This works for Gomb's repeated cycle-to-vertex scatter just as it worked for spatial pyramid pooling.
- The prototype `FixedTopologyGomb` caches full-cycle and per-tier `vertex x cycle` mean-pool matrices and avoids per-forward CPML tier filtering.
- On this six-node toy, dense pooling is faster after tracing. On larger sparse graphs, the same idea should become CSR/COO incidence matrices rather than dense matrices.

Sparse operation integration:

- Gomb-Soma already uses sparse aggregation and is the fastest model here (`~285 us` traced).
- CPML structural routing still rebuilds tier masks and scatter counts during every forward. A core-level optimization would cache `(cycles_ell, signs_ell, M_ell)` by topology/tier and use one sparse-dense matmul per tier: `H_ell = M_ell @ per_cycle`.
- The existing changelog mentions CSR tensor handoff and `TensorCsr::spmv/spmm`; those are the natural bridge for a Rust/Nagare-backed fixed-topology CPML path.

Rotor integration:

- FSR already has sparse top-k rotor transport and is sub-millisecond on this toy (`~632 us` traced).
- The next real rotor optimization is kernel fusion: precompute/cache offset indices, fuse Cayley-to-quaternion + quaternion rotation + signed weighting for fixed sequence length, and route that through the Nagare FSR kernels already started.
- For Gomb/HSIKAN, rotor integration is more about representation quality than toy speed unless the rotor transport replaces a current gather/scatter stage with a fused signed transport kernel.

SA-HSIKAN and structural generators:

- The RL implementation already encodes the launch-bound fix: precompute the signed walk-holonomy operator `B^L = (A_pos - A_neg)^L`, then run one CR/HSiKAN cell over `B^L x`.
- The toy proxy mirrors that using parsed `graph.link` signs. It validates the expected speed direction without claiming equality to the full cycle HSIKAN layer.
- Structural generators can supply the fixed topology for the same trick: chain/tree/Fano/Steiner/sunflower-like generated designs become reusable `B^L` or incidence pools.

P-graph feature enrichment and alpha-mixing:

- `hymeko_neuro/experiments/hsikan_pgraph_mapping.py` maps pgraph-selected units into HSIKAN knobs, including mixed tuples, attention, edge gates, and direct messaging.
- `hymeko_neuro/baselines/structural_features.py` already has leakage-free enriched node features: `degree`, `cycle_k3`, `cycle_k34`, `walk_k3`, and `ratios`.
- Alpha-mixing is already present in mixed-arity HSIKAN/Gomb variants. The optimization angle is to treat alpha as both a model lever and a scheduler: if learned alpha collapses onto a few branches, unused arity/topology branches can be pruned or skipped at inference.

## Supervised Toy Tasks

Added `--supervised`, a deterministic closed-form probe over node representations. It trains ridge heads, not neural training loops, and reports both full-fit and leave-one-out metrics on the six parsed graph nodes.

Classification task:

- Target: parsed node `tier`, classes `{0, 1, 2}`.
- Metric below: leave-one-out accuracy.

| Representation | Train acc | LOO acc |
| --- | ---: | ---: |
| raw features | 1.000 | 1.000 |
| HSIKAN | 0.833 | 0.000 |
| SA-HSIKAN proxy | 1.000 | 1.000 |
| Gomb-Soma | 1.000 | 0.000 |
| FSR | 0.500 | 0.000 |

Regression task:

- Target: synthetic signed-topology scalar derived from parsed node features, parsed tier, and signed `graph.link` degree.
- Metric below: RMSE.

| Representation | Train RMSE | LOO RMSE |
| --- | ---: | ---: |
| raw features | 0.0262 | 0.1631 |
| HSIKAN | 0.2115 | 0.2710 |
| SA-HSIKAN proxy | 0.0359 | 0.1242 |
| Gomb-Soma | 0.0622 | 0.1677 |
| FSR | 0.2361 | 0.2894 |

Interpretation: this is too tiny to claim model quality, but it is useful as a wiring sanity check. The fixed holonomy path keeps the tier signal and the signed-topology scalar cleanly on this graph while also being faster than full HSIKAN.

## General Graph Problem

Added `--general-problem`, which builds a deterministic family of 36 graph samples from the same parsed topology by perturbing node features. This is graph-level rather than node-level:

- Representation: per-node encoder output.
- Global pooling: `concat(mean, std, max)` over nodes.
- Class task: three bins of a latent graph scalar, with stratified `24/12` train/test split.
- Regression task: predict the latent graph scalar.
- Entropy feedback: train a first-pass classifier, append normalized predictive entropy to pooled graph features, then refit the closed-form heads.
- Timing: median forward + global-pool time per graph sample over 300 repeats.

Classification:

| Representation | Plain acc | Entropy-feedback acc |
| --- | ---: | ---: |
| raw features | 0.917 | 0.833 |
| HSIKAN | 0.667 | 0.917 |
| SA-HSIKAN proxy | 0.833 | 0.833 |
| Gomb-Soma | 0.833 | 0.917 |
| FSR | 0.667 | 0.917 |

Regression:

| Representation | Plain RMSE | Entropy-feedback RMSE |
| --- | ---: | ---: |
| raw features | 0.0031 | 0.0021 |
| HSIKAN | 0.0122 | 0.0120 |
| SA-HSIKAN proxy | 0.0242 | 0.0259 |
| Gomb-Soma | 0.0053 | 0.0063 |
| FSR | 0.0248 | 0.0236 |

Forward + global-pool timing:

| Representation | Median us |
| --- | ---: |
| raw features | 89.65 |
| HSIKAN, traced | 1,247.85 |
| Fixed-topology HSIKAN, traced | 1,197.15 |
| SA-HSIKAN proxy, traced | 623.20 |
| Gomb-Soma, traced | 418.05 |
| FSR, traced per-graph | 870.85 |
| FSR, batched amortized | 45.47 |

On this constructed task, entropy feedback is most useful for classification when the base pooled representation is uncertain; it lifts HSIKAN, Gomb-Soma, and FSR to `0.917` test accuracy. Regression remains dominated by raw features because the target is intentionally feature-derived, but the feedback path is wired and measurable.

## FSR/HSIKAN Optimization Pass

Two missing pieces showed up in profiling:

- FSR was being evaluated one generated graph at a time even though `FiberSpikeRotorMixer` natively accepts `(B, T, d)`.
- HSIKAN was not traced in the general-problem runner, so fixed-shape Python dispatch was still in the timing.

Changes:

- General-problem encoding now batches all FSR samples into one `(36, 6, 6)` tensor.
- General-problem timing emits `optimized_forward_pool_us.fsr_batched_amortized`.
- `--general-problem --trace` now traces the fixed-shape suite before encoding/timing.
- Added `FixedTopologyHSIKAN`, a parity experiment that replaces HSIKAN scatter-mean with a cached cycle-pool matrix.

Results:

| Path | Before median us | After median us | Speedup |
| --- | ---: | ---: | ---: |
| HSIKAN general forward+pool | 2,376.40 | 1,247.85 | 1.90x |
| FSR general forward+pool | 1,387.90 | 870.85 | 1.59x |
| FSR batched amortized | 1,387.90 | 45.47 | 30.53x |

Fixed-topology HSIKAN was numerically equivalent to HSIKAN on the toy fixture, but it did not materially beat traced HSIKAN at `N=6`; the dense pool itself is too small to overcome overhead. The lesson is: trace for tiny fixed CPU graphs; use sparse cached incidence for larger graphs if we move this into core.

## MSG, SSG, ABB

Added a structured `structure_search` block to the general-problem JSON so P-graph search is represented as a first-class optimization path.

Roles:

| Method | Role | Speedup angle | Selection/generation angle |
| --- | --- | --- | --- |
| MSG | Maximal Structure Generation | Precompute the full admissible candidate universe once | Generates all feasible feature/topology/arity branches before learning |
| SSG | Solution Structure Generation | Enumerates smaller feasible substructures instead of running the full MSG superset | Discrete feature selection over cycles, walks, enriched features, and structural generators |
| ABB | Accelerated Branch-and-Bound | Prunes SSG when candidate sets are large | Multi-objective selection over accuracy, latency, params, entropy, and alpha support |

Concrete targets:

- cycle/walk arity selection
- structural feature enrichment selection
- pgraph-generated topology branches
- alpha-mixing branch pruning
- cached sparse-incidence generation for fixed-topology speedups

Existing repo hooks:

- `hymeko_neuro/experiments/hsikan_pgraph_mapping.py`
- `hymeko_neuro/experiments/gomb_pgraph_mapping.py`
- `hymeko_pgraph::msg`, `ssg`, and `abb_solve` through `hymeko_pgraph_dump`

Practical interpretation: MSG creates the broad structural menu, SSG enumerates feasible model submenus, and ABB chooses the cheapest useful submenu. That can speed inference by skipping branches, improve feature selection by pruning redundant structural enrichments, and generate candidate topologies for SA-HSIKAN/HSIKAN/FSR without hand-authoring every variant.

## FSR Clifford Numbers

Added `fsr_clifford` diagnostics to the general-problem JSON. Current traced run:

| Quantity | Value |
| --- | ---: |
| Hidden input shape | `[36, 6, 6]` |
| Blocks | `2` |
| Relative offsets | `6` |
| Spike `k` | `2` |
| Dense all-pair slots | `1296` |
| Causal slots | `756` |
| Sparse top-k slots | `432` |
| Sparse / dense | `0.3333` |
| Sparse / causal | `0.5714` |
| Rotor unit-norm max error | `0.0` |
| Bivector norm max | `0.0` |
| Soft sign mean | `0.761594` |
| Input block norm mean | `1.245593` |
| Output block norm mean | `0.576071` |
| Per-graph FSR forward+pool median | `880.50 us` |
| Batched amortized FSR forward+pool median | `41.68 us` |

Interpretation: the rotor path is currently identity-initialized (`bivector_norm_max = 0`, unit rotor error `0`), so the Clifford part is algebraically well-conditioned. The speed lever is not the Cayley map yet; it is batching and sparse top-k transport. Once learned bivectors move away from zero, these same diagnostics will catch rotor drift, sign saturation, and transport-energy changes.

## Closed-Form Entropy Feedback

The entropy-feedback pass is now explicitly reported as closed form:

`W = (X^T X + lambda I)^-1 X^T Y`

No SGD loop is used. Pass 1 fits a ridge classifier, computes normalized predictive entropy, appends that scalar to pooled graph features, and pass 2 refits ridge heads.

Classification accuracy delta from entropy feedback:

| Representation | Delta |
| --- | ---: |
| raw features | `-0.0833` |
| HSIKAN | `+0.2500` |
| Fixed-topology HSIKAN | `+0.2500` |
| SA-HSIKAN proxy | `+0.0000` |
| Gomb-Soma | `+0.0833` |
| FSR | `+0.2500` |

Regression RMSE delta from entropy feedback:

| Representation | Delta |
| --- | ---: |
| raw features | `-0.0011` |
| HSIKAN | `-0.0002` |
| Fixed-topology HSIKAN | `-0.0002` |
| SA-HSIKAN proxy | `+0.0016` |
| Gomb-Soma | `+0.0009` |
| FSR | `-0.0012` |

For this graph family, closed-form entropy feedback is most useful for uncertain classifiers, especially HSIKAN and FSR. It is small but measurable for regression because the regression target is already smooth in the pooled features.

One cold CLI-style invocation through `uv run --group ml python scripts\dev\hymeko_structure_toy_models.py` took `11.03s`, dominated by Python/Torch/uv startup rather than toy model compute.

Raw timing captures:

- `reports/2026-07-01-hymeko-structure-toy-performance.json`
- `reports/2026-07-01-hymeko-structure-toy-performance-traced.json`

## Verification

Commands run:

```powershell
$env:PYTHONPATH='.'; uv run --group ml python scripts\dev\hymeko_structure_toy_models.py --out reports\2026-07-01-hymeko-structure-toy-models.json
$env:PYTHONPATH='.'; uv run --group ml python scripts\dev\hymeko_structure_toy_models.py --supervised --out reports\2026-07-01-hymeko-structure-toy-supervised.json
$env:PYTHONPATH='.'; uv run --group ml python scripts\dev\hymeko_structure_toy_models.py --general-problem --trace --out reports\2026-07-01-hymeko-structure-general-problem.json
$env:PYTHONPATH='.'; uv run --group ml python scripts\dev\hymeko_structure_toy_models.py --benchmark --repeats 1000 --warmup 50 --threads 1 --out reports\2026-07-01-hymeko-structure-toy-performance.json
$env:PYTHONPATH='.'; uv run --group ml python scripts\dev\hymeko_structure_toy_models.py --benchmark --trace --repeats 1000 --warmup 50 --threads 1 --out reports\2026-07-01-hymeko-structure-toy-performance-traced.json
$env:PYTHONPATH='.'; uv run --group ml python -m py_compile scripts\dev\hymeko_structure_toy_models.py hymeko_neuro\tests\test_hymeko_structure_toy_models.py
$env:PYTHONPATH='.'; uv run --group ml pytest -p no:randomly hymeko_neuro\tests\test_hymeko_structure_toy_models.py -q
uv run ruff check scripts/dev/hymeko_structure_toy_models.py hymeko_neuro/tests/test_hymeko_structure_toy_models.py
```

Results:

- Toy harness completed and wrote JSON metrics.
- Supervised toy probe completed and wrote JSON metrics.
- General graph problem completed and wrote JSON metrics.
- `py_compile` passed.
- `pytest`: `5 passed in 12.19s`.
- `ruff`: `All checks passed!`.

## Notes

- This is an AST projection prototype, not a resolved core IR integration.
- No dependency changes were made.
- No core files were edited.
- The sparse walk membership tensor is built locally for the Gomb-Soma adapter from parsed `graph.walk` triples.
