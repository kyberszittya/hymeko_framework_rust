# Reference: Environment variables

Most `.hymeko`-driven flow respects environment variables for runtime knobs. Two conventions: `HYMEKO_*` for framework-level, `HSIKAN_*` for research-side training kernel.

## Framework-level (`HYMEKO_*`)

| variable | values | effect |
|---|---|---|
| `HYMEKO_LOG_LEVEL` | `error` / `warn` / `info` / `debug` / `trace` | Logger filter |
| `HYMEKO_TRANSFORMS_ROOT` | path | Override workspace `transforms/` directory (used in tests / non-default install) |
| `HYMEKO_PARSER_MAX_NESTING` | int | Maximum nesting depth (default high) |

## Cycle enumeration (`HSIKAN_TOPK_*`)

| variable | values | effect |
|---|---|---|
| `HSIKAN_TOPK_MODE` | `global` / `per_vertex` | Top-K selection scope |
| `HSIKAN_TOPK_K` | int | m (per-vertex top-K) or global cap |
| `HSIKAN_TOPK_SCORER` | `fraction_negative` / `mi` / … | Cycle scorer |
| `HSIKAN_TOPK_PRUNER` | `none` / `balance` / `davis` / `unbalanced` / `frustration` | Axiom-conditioned pruner |

## Architecture / cycle structure (`HSIKAN_*`)

| variable | values | effect |
|---|---|---|
| `HSIKAN_ARITIES` | `2,3,4,5` | Comma-separated arity list |
| `HSIKAN_MAX_K2` | int | Cap on k=2 cycle count (default 1M) |
| `HSIKAN_MAX_K3` | int | Cap on k=3 cycle count (default 30k) |
| `HSIKAN_MAX_K4` | int | Cap on k=4+ cycle count (default 200k) |
| `HSIKAN_MIXED_TUPLES` | `c3,c4,w2,…` | Mixed cycles + walks (cN = closed N-cycles, wL = length-L walks) |
| `HSIKAN_CHUNK_T` | int | Chunk T-dimension to bound peak GPU memory (Epinions / large datasets) |
| `HSIKAN_CYCLE_BATCH` | int | Cycle-batch size for chunked forward (auto for slashdot/epinions) |
| `HSIKAN_STRICT_PROTOCOL` | `0` / `1` | Strict no-leakage cycle exclusion |

## Spline activations

| variable | values | effect |
|---|---|---|
| `HSIKAN_SPLINE_KIND` | `bspline` / `catmull_rom` / `kochanek_bartels` / composites | Override the .hymeko-declared spline kind |
| `HSIKAN_KB_PRESET` | `smooth` / `tense` / `cusp` / `skew` / `sharp` / `flat` | Named TCB triple for Kochanek-Bartels |
| `HSIKAN_KB_INIT_TCB` | `t,c,b` | Raw TCB triple (overridden by preset if both set) |

## Attention + messaging

| variable | values | effect |
|---|---|---|
| `HSIKAN_ATTENTION_M_E` | `dot` / `quaternion` | Per-cycle attention head kind |
| `HSIKAN_DIRECT_MESSAGING` | `0` / `1` | Enable SGCN-style direct sign-conditional W_pos / W_neg path |

## Training

| variable | values | effect |
|---|---|---|
| `HSIKAN_ENTROPY_LAMBDA` | float | Spectral-entropy regulariser strength |
| `HSIKAN_TORCH_COMPILE` | `0` / `1` | Wrap model with `torch.compile(mode='reduce-overhead')` |
| `HSIKAN_WALK_LENS` | `2,3,…` | Walk lengths (Walk-HSiKAN) |
| `HSIKAN_TOPK_MODE=per_vertex` | (see above) | Stratified top-K |

## How to set them

CLI:
```bash
HSIKAN_TOPK_K=64 HSIKAN_TOPK_PRUNER=balance python -m signedkan_wip.src.run_final_cell ...
```

Or via the walker (set automatically from `training.hymeko` body):
```hymeko
@enumerate_cycles <cycle_enum> {
    mode "per_vertex"; m_per_vertex 64; pruner "balance";
    ...
}
```

The walker reads these and exports them to the env vars before calling `cell_signed_graph`.

## See also

- [HyMeKo-controlled training](./quickstart/09-hsikan-training.md) — env vars set declaratively from `.hymeko`
- [Debug the pipeline](./recipes/debug-pipeline.md) — env-var audit framework
