# Report: Gömb outer-shell vectorization (2026-05-12)

## Summary

Replaced the Python loop over `M` Clifford-FIR banks in `OuterFIRShell` with a single `einsum` pre-projection stack, batched Cl(0,1) coefficient multiply, and one batched scatter. Replaced the per-corner `for i in range(k)` in `scatter_mean` with a flattened `index_add` (and a batched variant for `(B, M_c, d)` inputs). The timing benchmark supports optional **`--torch-compile`** (cloned shell, Inductor) on CUDA; **`_bench`** wraps timed forwards in **`torch.inference_mode()`** for forward-only semantics. See Performance results and `signedkan_wip/src/benchmarks/gomb_outer_timing.py`.

## Files touched

| File | Change |
|------|--------|
| `signedkan_wip/src/hymeko_gomb/shells.py` | Batched `OuterFIRShell.forward`; `scatter_mean` + `_scatter_mean_flat` / `_scatter_mean_batched` |
| `signedkan_wip/tests/test_hymeko_gomb.py` | Parity tests vs corner-loop reference; outer-shell sequential equivalence |
| `signedkan_wip/src/benchmarks/gomb_outer_timing.py` | Synthetic + `--datasets` timing; **`--torch-compile`**; `_bench` uses `torch.inference_mode` |
| `reports/2026-05-12-gomb-outer-perf.md` | Perf tables + provenance (this document) |
| `docs/plans/2026-05-12-gomb-outer-perf/plan.{tex,pdf,tikz,mmd}` | Plan artifacts |

## CORE.YAML items touched

None.

## Dependencies

None added.

## Test results

- Command: `python -m pytest -p no:randomly signedkan_wip/tests/test_hymeko_gomb.py -q`
- Result: **18 passed** (re-run after `_bench` `inference_mode` change).

## Performance results

Measured with `python -m signedkan_wip.src.benchmarks.gomb_outer_timing` (warmup + multi-iteration wall time; **not** Criterion — diagnostic protocol per CLAUDE.md for Python quick benches). The timed loop runs inside **`torch.inference_mode()`** (forward-only; no autograd tape).

**Legacy baseline** in the script = original semantics: Python `for m in range(M)` over banks + per-corner `for i in range(k)` scatter. **Eager (batched)** = current `OuterFIRShell.forward`. **torch.compile** = cloned shell, `dynamic=True`, `mode="reduce-overhead"`, parity vs eager on the first `min(4096, M_c)` cycles (rtol `2e-3`), then timed with **+15** extra warmup steps on CUDA before the same `iters` timed loop.

**Host (2026-05-12 refresh, post `inference_mode` timing):** NVIDIA GeForce RTX 2070 SUPER, `torch` **2.11.0+cu130**, Linux. Inductor may still log SM / autotune notes; CUDAGraph “pending backwards” noise is **not** expected on this benchmark after `inference_mode`. Peak RSS not instrumented for this micro-bench.

| Device | Config | Warmup / iters | Legacy median | Eager median | Legacy / eager | Compiled median | Eager / compiled | Eager worst |
|--------|--------|----------------|---------------|--------------|----------------|-----------------|------------------|-------------|
| CUDA | N=2048, Mc=16384, M=8, k=3, float32 | 8 / 35 | 2.49 ms | 2.01 ms | **1.24×** | 0.35 ms | **5.76×** | 2.81 ms |

Historical snapshot (earlier run on a different day / stack; retained for rough comparison only): CPU N=512 Mc=4096 — legacy 6.38 ms, eager 4.58 ms (**1.39×**); CUDA synthetic — legacy 2.56 ms, eager 1.01 ms (**2.54×**). Do not mix with the table above when claiming regressions; re-run on one machine.

Reproduce (synthetic + compile):

```bash
python -m signedkan_wip.src.benchmarks.gomb_outer_timing --device cuda --N 2048 --Mc 16384 --M 8 --warmup 8 --iters 35 --torch-compile
python -m signedkan_wip.src.benchmarks.gomb_outer_timing --device cpu --N 512 --Mc 4096 --M 8
```

### Bitcoin Alpha / OTC (real train-split cycle pools)

Command: `python -m signedkan_wip.src.benchmarks.gomb_outer_timing --device cuda --datasets bitcoin_alpha bitcoin_otc --warmup 5 --iters 20 --topk 64 --torch-compile`

Same setup as `run_gomb_smoke`: 80/20 edge split, cycles enumerated on **train** edges only, `m_per_vertex=64`, triads `k=3`. Outer shell dims match default Gömb smoke (`d_in=32`, `d_layer=16`, `M=8`). Parity (legacy vs eager) on first 4096 cycles before timing; compile parity vs eager on same slice.

| Dataset        | \|V\| | \|E\| | Train \|E\| | Mc (cycles) | enum prep | Legacy median | Eager median | Legacy / eager | Compiled median | Eager / compiled |
|----------------|------|-------|-------------|-------------|-----------|---------------|--------------|----------------|-----------------|------------------|
| bitcoin_alpha  | 3783 | 24186 | 19349       | 13190       | ~0.06 s   | 2.22 ms       | 1.76 ms      | **1.26×**      | 0.65 ms         | **2.70×**        |
| bitcoin_otc    | 5881 | 35592 | 28474       | 19148       | ~0.08 s   | 2.24 ms       | 1.76 ms      | **1.27×**      | 0.45 ms         | **3.88×**        |

End-to-end **legacy → compiled** (approximate): alpha **~3.4×** (2.22 / 0.65), otc **~4.9×** (2.24 / 0.45). One legacy worst-sample spike (~17 ms) occurred on alpha in this run; medians above are still comparable to the eager/compiled block.

Without `--torch-compile`, the script prints legacy + eager only (same medians as the eager columns above when run back-to-back on a quiet GPU).

Enumerate wall is Rust + Python sign lookup, not included in forward median.

## Quality metrics and parameters (Gömb)

`python -m signedkan_wip.src.run_gomb_smoke` now ends with a human-readable **`[metrics]`** line and one **JSON** object that includes:

| Field | Meaning |
|-------|---------|
| `n_params` | Total trainable parameters |
| `params_by_module` | First-level breakdown (`node_embed`, `outer`, `middle`, `core`, …; mixed-arity adds `outers` / `middles` / `cores` as `nn.ModuleDict` children) |
| `val_auroc` | ROC-AUC on held-out **validation** edges (same as ranking metric in `val_auc_*`) |
| `val_average_precision` | Average precision (area under precision–recall curve; informative under class imbalance) |
| `val_recall_pos` / `val_recall_neg` | Recall at threshold **0.5** on logits for label **+1** / **−1** edges |
| `val_precision_pos` / `val_precision_neg` | Precision at 0.5 for each class |
| `val_f1_pos` / `val_f1_neg` / `val_f1_macro` | F1 per class and macro-F1 at 0.5 |

The outer-shell timing benchmark (`gomb_outer_timing`) does **not** train; it only measures `OuterFIRShell` forward. Use `run_gomb_smoke` for end-to-end AUC/recall/AP and parameter reporting.

## Static analysis

`ruff` not available in the execution environment; rely on CI / local `ruff check` on the two Python paths above.

## Open issues / follow-up

- Optional: fuse `torch.stack` of linear weights into a single `nn.Parameter` to avoid per-forward stack (would change `state_dict` keys — migration only if desired).
- Middle `SignedKANLayer` remains the dominant cost for full training; outer shell can still use **`torch.compile` on a copy** in benchmarks; wiring the **same** compiled module into autograd training needs a dedicated pass (recompile policy, dynamic `M_c`).
- Benchmark `_bench` now uses **`torch.inference_mode()`**; training forwards still use autograd as usual.

## Git / provenance

- **Git SHA:** `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (branch `refactor/extract-hymeko-hre`; **dirty** working tree — treat numbers as tied to this environment).
- **Torch / device:** `2.11.0+cu130`, `NVIDIA GeForce RTX 2070 SUPER`.
- **Commands:** synthetic and Bitcoin runs as in the “Reproduce” / Bitcoin command blocks above (2026-05-12, after `_bench` gained `torch.inference_mode`).
