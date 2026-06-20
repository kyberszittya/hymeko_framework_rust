# Triton feasibility on the rotor signed-link path — profile-driven verdict: NO (host/launch-bound)

**Date:** 2026-06-18
**Question:** is a Triton kernel a justified lever for the active rotor signed-link line?
**Verdict:** **No.** The discriminating profile (CLAUDE.md §3: no optimization without a
profile demonstrating a hot spot) shows the heaviest cell is **host/launch-bound**, with
GPU time spread across optimizer + sparse-coalesce + library GEMM/SpMM — **no dense
custom-math hotspot that Triton would address.** The real levers are framework changes
(fused optimizer, fewer launches), not kernels.

## Method
- Target: the heaviest cell `bitcoin_alpha → epinions`, `cayley_rotor_walk`, `real` arm
  (train 120 ep on bitcoin + frozen eval on epinions, 131 k nodes). The single most
  compute-heavy configuration in the transfer grid.
- Tool: `torch.profiler` (CPU + CUDA activities), torch 2.12.0+cu132, RTX 3070 Laptop.
  Artifacts: `2026-06-18-triton-profile/{gpu_optable.txt, cpu_optable.txt}`. A 157 MB
  Chrome trace was exported and inspected, then deleted (too large for the repo); the
  op-tables + the tables inlined below are the retained summary.
- **py-spy (the §10-pinned CPU profiler) could not attach** on this host —
  `Failed to find python version from target process`: py-spy 0.3.14 cannot introspect
  uv's standalone CPython build on Windows. Documented as a tool limitation (§10
  on_minor_drift); the torch.profiler CPU-side table substitutes (it carries the same
  host-time breakdown a py-spy flamegraph would).

## Measured (totals)
| metric | value |
|---|---|
| wall (profiled cell) | 7.84 s |
| **self CUDA time** | **1.40 s** (~18 % of wall) |
| **self CPU time** | **3.61 s** |
| `cudaLaunchKernel` count | **42 464** (951 ms CPU = the single largest cost) |
| `cudaStreamSynchronize` | 301 ms · `aten::empty` | 21 248 allocs / 118 ms |

## Measured (GPU self-time breakdown, self_cuda %)
| op | self CUDA | % | note |
|---|---|---|---|
| `Optimizer.step#Adam.step` | 167 ms | 27.2 % | optimizer state update, 120 steps |
| `aten::_coalesce` (sparse COO) | 161 ms | 26.3 % | per-forward signed rotor propagation |
| `aten::copy_` | 106 ms | 17.2 % | |
| `aten::addmm` (cuBLAS) | 92 ms | 15.0 % | dense linears — already library-optimal |
| cub `merge_sort` | 71 ms | 11.6 % | sort inside coalesce/index_put |
| elementwise kernels | 67 ms | 10.9 % | |
| `aten::_index_put_impl_` | 66 ms | 10.7 % | scatter |
| **`cusparse csrmm` (the signed SpMM)** | **37 ms** | **6.1 %** | the message-passing — library-optimal, small |
| `ampere_sgemm` (cuBLAS) | 31 ms | 5.1 % | |

## Analysis — why not Triton
Triton accelerates **GPU compute kernels**. Here:
1. **The cell is host-bound.** GPU self-time is 1.40 s of 7.84 s wall (18 %); the
   remaining 82 % is CPU dispatch, 42 k kernel launches, sync, and allocation. Triton
   cannot touch any of it. The dominant single cost is **kernel-launch overhead** —
   the signature of many tiny ops on a small graph (bitcoin ≈ 6 k nodes), not a
   compute-bound kernel.
2. **No Triton-shaped GPU hotspot.** Of the 1.40 s GPU time:
   - Adam (27 %) → a **fused optimizer** flag (`fused=True`), not a kernel rewrite.
   - sparse `_coalesce` (26 %) → sparse-tensor maintenance from the per-forward signed
     propagation; reduce by representation/caching, **not** a Triton kernel.
   - dense `addmm`/`mm`/`sgemm` (≈25 %) → already **cuBLAS-optimal** at these sizes;
     Triton will not beat the library.
   - the actual signed SpMM (`cusparse csrmm`) is **6 %** — library-backed and small.
   There is no dense, custom-math, fusible inner loop dominating the trace.
3. **Triton already lives where it pays** in this repo — the KAN inner-product basis
   (`triton_kernels/inner*.py`), `clifford_fir.py`, spline (`catmull_rom.py`), pooling
   (`fused_pool_scatter.py`), behind `dispatch.py` + a `triton_kernels_sensitivity.py`
   harness. The rotor/SGCN baselines are deliberately **pure-torch sparse**; that path
   has no KAN-basis / Clifford dense math for Triton to fuse.

## The real levers (ranked, if speed becomes the goal — none are Triton)
1. **Fewer kernel launches** (42 k → CUDA graphs, or larger batched ops). Biggest cost.
2. **Fused Adam** (`torch.optim.Adam(fused=True)`) — collapses the 27 % optimizer cost
   and its launches. One-line, but lives in the shared `_train` loop → user call.
3. **Cut the per-forward coalesce churn** — the signed propagation rebuilds/coalesces
   sparse intermediates each forward; a CSR-native or cached-structure representation
   would shrink the 26 % coalesce + the index_put/merge_sort tail.
4. **Scale caveat:** every number here is at *bitcoin training scale*. The only place a
   GPU kernel could plausibly dominate is **audit scale** (the queued tier-2/3 32 h run,
   or large-graph *training* rather than eval-only). The Triton candidate there is the
   **geometric triad attention head** (custom Clifford scoring; `clifford_fir.py`
   substrate exists) — but it **lost the link-head ablation** to real-bilinear and is
   defaults-off, so it is not load-bearing. Re-profile at that scale before any kernel.

## Provenance
- Git SHA `7d16ad0` (tree dirty). Cell: `cayley_rotor_walk`, `bitcoin_alpha→epinions`,
  seed 0, `n_epochs=120`, deduped eval, auc 0.8947. Device CUDA (RTX 3070 Laptop,
  torch 2.12.0+cu132). No CORE.YAML edits. `py-spy==0.3.14` installed into `.venv` via
  `uv pip install` (imperative; project manifest/lock untouched) — but non-functional
  here (see Method).

## Open / follow-up
- If the user wants the speedup regardless: levers 1–3 above, profiled before/after with
  the same harness (and a §3 ≥5-iteration benchmark, not single-shot). All touch the
  shared `_train` loop / SGCN representation → escalate before editing.
- The Triton question is **closed for the rotor line** until a profile at audit scale (or
  large-graph training) shows a dense kernel in the critical path.
