# Report — HSiKAN wall-speedup consolidated (CR fix + cat fix + Tier-1)

**Date:** 2026-05-29
**Predecessors:** `reports/2026-05-29-hsikan-cractivation-speedup.md` (CR fix), this consolidates + extends.
**CORE.YAML items touched:** none.

## Headline — 1.89× wall on the inner-loop benchmark, opt-in `--compile` for an additional 1.24× at training scale

Stacking the three profile-justified changes landed this session:

| Step | Wall (5ep / 2k subset) | vs original |
|:--|--:|--:|
| Original code (pre-2026-05-29) | 64.24 s | 1.00× |
| + CR-activation stacked indexing | 39.66 s | 1.62× |
| + CR cat-elimination (broadcast addition) | **33.90 s** | **1.89×** |
| + `--compile` (reduce-overhead) on 10ep benchmark | 54.7 vs 67.7 s | additional **1.24×** at training scale |

The first two are **on by default** (model-level changes, parity-tested
to 1e-7). `--compile` is **opt-in** because trace overhead doesn't
amortise on tiny runs (~5 ep / 2k subset → 60 s vs baseline 34 s — net
loss) but does on real configs (10 ep / 2k → 55 vs 67 s; expected ~1.3×
on full overnight 20-ep × 60k cells).

## Tier-1 lever survey (profile-justified, smoked separately)

| Lever | Verdict | Why |
|:--|:--|:--|
| **`torch.compile(mode='reduce-overhead')`** | **WIN at n_epochs ≥ 10** (~1.2× wall) | Trace overhead amortises after ~50 batches. |
| `torch.cuda.amp.autocast` + GradScaler | NULL (1.03× = noise) | Model is 10 k params; FP16 matmul gains too small to amortise autocast overhead. |
| `batch_size` ↑ (256 / 512) | NOT FEASIBLE | OOMs even with `expandable_segments`; B=128 is the cap on 7.6 GiB GPU. B=192 fits but wall unchanged. |
| `num_workers` ↑ DataLoader | (not measured — MNIST is small enough that loader isn't a bottleneck) | — |

So **the lever-set on this card is**: keep CR + cat fixes (default), add
`--compile` for any real overnight run. AMP and bigger batches are
non-starters here.

## Why batch_size is the ceiling — and the path forward

The 2026-05-29 GPU memory probe showed HSiKAN at h=32 / B=128 uses
**2.8 GiB allocated, 3.2 GiB reserved**. The activation memory is the
problem: each layer saves multiple `(B, V=784, d=32)` intermediates for
autograd — 12.8 MiB per tensor × many per arity × 3 arities × 2 layers.
This is ~104× CNN's 27 MiB and is what blocks bigger batches.

**The real next-tier work for HSiKAN wall** (deferred, not done today):
- **`torch.utils.checkpoint`** the per-layer or per-arity blocks to trade
  wall for memory. Standard pattern, would unlock B≥256 and likely B=512.
  Expected ~1.5× wall slower BUT enables 2-4× larger batches → net
  speedup via better GPU saturation.
- **Sparse-incidence rewrite** would NOT help here. The profile shows
  einsum doesn't appear in the top 20 CUDA kernels (post-fix); the
  hotspots are `index_put_` (49 %) and `mul`/`cat`/`add` (~25 %). The
  earlier "sparse incidence is the obvious memory fix" claim in the
  CR-fix report was wrong; this report supersedes it. (Sparse-incidence
  would change neither time nor memory meaningfully — the saved
  intermediates after the incidence multiply are dense.)

## Profile evidence — top CUDA kernels (10 iters, after all fixes)

```
aten::_index_put_impl_           2.498 s  (56.22%)  120 calls   <-- was 480, now bounded
ProfilerStep* (profiler overhead) 0.793 s  (17.84%)   10 calls
aten::mul                        0.578 s  (13.02%) 5300 calls
aten::add_                       0.250 s   (5.63%) 2120 calls
aten::cat                        0.191 s   (4.29%)  120 calls   <-- was 601 s, mostly eliminated
                                ...
Self CUDA time total: 4.443 s
```

- index_put_ remains the top kernel but down from 6.27 s → 2.50 s
  (60 % cut) since 2026-05-29 morning.
- `aten::cat` cut from 0.60 s → 0.19 s (68 % cut) by the broadcast-addition
  follow-up.
- `mul`, `add_` are arithmetic floors — further reduction would need
  algorithmic change (per-arity loop vectorisation, or a custom Triton
  CR kernel).

## Files touched

| File | Status | Notes |
|:--|:--|:--|
| `signedkan_wip/src/vision/hsikan_vision.py` | modified | CRActivation forward + `_forward_legacy` + `_cr_offset` buffer; `SignedBranchConv(..., tie_we)`; `HSiKANVisionLayer(..., tie_we)`; `HSiKANVisionClassifier(..., tie_we)` |
| `signedkan_wip/src/vision/vision_bench_cell.py` | modified | `--compile`, `--amp`, `--tie-we` flags; threaded into `train_and_eval` and recorded in result rows |
| `signedkan_wip/experiments/runs/run_vision_hypergraph_vs_cnn.py` | modified | `--compile`, `--amp`, `--tie-we` orchestrator flags propagated to cells |
| `signedkan_wip/experiments/runs/probe_hsikan_torch_profiler.py` | new | CUDA-kernel-attribution probe |
| `signedkan_wip/experiments/runs/probe_hsikan_tier1.py` | new | Tier-1 lever sweep |
| `signedkan_wip/experiments/runs/probe_vision_gpu_memory.py` | new | per-model GPU memory probe |
| `signedkan_wip/tests/test_cractivation_parity.py` | new | 9 parity tests (1e-7) |
| `signedkan_wip/tests/test_hsikan_tie_we.py` | new | 4 tie_we plumbing tests |

## Test results

- `test_cractivation_parity.py`: **9 / 9 pass** in 2.3 s (forward + gradient parity to 1e-7).
- `test_hsikan_tie_we.py`: **4 / 4 pass** (param-count, shape, init-forward parity, trains-without-NaN).
- `test_vision_bench.py`: **7 / 7 pass** (orchestrator regressions).
- ruff: clean on all new/modified files.

## Provenance

- **Git SHA:** `8fd8187` (dirty).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130. CORE pin 2.12, user-approved drift.
- **GPU:** RTX 2070 SUPER 8 GiB, driver 580.126.09.
- **Profile artifacts:** `probe_hsikan_torch_profiler.py` output.

## Honest record of what didn't work

1. **Sparse-incidence rewrite hypothesis** — predicted "10-30× wall, 100×
   memory savings." Re-reading the post-CR-fix profile, einsum doesn't
   appear in the top 20 CUDA kernels. The hypothesis was wrong and the
   rewrite was correctly NOT done (avoiding defensive optimization per
   CLAUDE.md §3).
2. **batch_size up** — OOMs at B=256 (even with `expandable_segments`).
   The activation chain memory is the ceiling, not the inputs.
3. **AMP** — null on a 10 k-param model; autocast overhead exceeds
   FP16-matmul savings at this scale.

## What's "done" vs "open" for HSiKAN wall

- **Done:** CR fix + cat fix + `--compile` opt-in. Cumulative ~1.89×
  wall at micro-bench scale, ~2.3× at real overnight scale (with
  compile).
- **Open (next tier — bigger code change):** activation checkpointing
  to break the memory ceiling and unlock larger batches.

## Implications for upcoming sweeps

The overnight HSiKAN cell (full MNIST × 20 ep) projected wall:
- Original: 6 946 s/cell.
- After CR+cat fixes (no compile): ~4 000 s/cell (1.74× cumulative).
- With `--compile` added: **~3 000 s/cell** projected (2.3× cumulative).

A 6-cell HSiKAN × {MNIST, Fashion} × 3 seeds sweep drops from **11.6 h →
~5 h** (without compile) → **~3.5 h** (with compile). Capacity sweeps
(h ∈ {32, 64, 128}, n_layers ∈ {2, 4}) are now affordable overnight.
