# Report — vectorized signed-attention for the Phase-B baseline grid

**Date:** 2026-06-14
**Slug:** `vectorize-signed-attention`
**Plan:** `docs/plans/2026-06-14-vectorize-signed-attention/` (`plan.tex/.pdf/.tikz`+`plan-figure.pdf`/`.mmd`)
**Author:** Csaba Hajdu
**Branch:** `feature/ac-hsikan`

## Summary

Replaces the per-forward by-length Python loop in the signed-attention baselines
with a cached, vectorized **segment (CSR) attention**, removing the CPU-side
bottleneck that put the Phase-B Table-1 grid at ~80–90 GPU-h. Measured **~6×
per-epoch speedup** on the worst-case graph (Epinions), parity-verified against a
naive reference. The four attention methods (`sigat`, `sgt`, and the
`sigformer`/`sesgformer` reimpls that reuse their attention) all benefit from one
shared primitive; `encode_nodes` signatures are unchanged, so the other runners
that use `SiGATAttn`/`SGT` (`run_inference_bench`, `run_sgt_sweep`,
`test_sota_compare`) are untouched.

## Performance result (the goal)

| | per-epoch wall (Epinions, sesgformer) | GPU peak |
|---|---|---|
| before (by-length Python loop) | ~22.9 s | 1.93 GB |
| after (cached segment attention) | **~3.7 s** (4-ep wall 14.7 s incl. one-time CSR build) | 2.78 GB |

**~6× faster**, well under the 16 GB cap. AUROC 0.903 (vs 0.895 before, 2-ep) —
the model learns identically. The cost was Python-side, exactly as predicted: a
bigger GPU would not have helped, but vectorizing did.

**Grid impact:** the transformer×large-graph cells drop from ~75 min/arm to
~12 min; the full 7×5×5×2 grid drops from ~80–90 GPU-h to **~10–15 GPU-h** —
feasible locally (overnight), no cloud required.

## Approach

`segment_attention` over incidences (`E = Σ_v |N(v)|`, 1.68 M on Epinions):
flatten the Python neighbour buckets **once per run** (cached on the attention
instance, keyed by `(id(buckets), device)`) into `seg`/`nbr`(/`sign`) tensors via
NumPy `repeat`/`concatenate`; each forward is then gather + segment-softmax
(`scatter_reduce(amax)` for stability, `index_add` for the denominator and the
weighted-V sum), `O(E)` memory, no global max-degree padding. One shared
primitive removes the near-duplicate loop previously copied in both attention
modules (§6.1).

## Files touched

| Path | Action | Lines |
|---|---|---|
| `hymeko_neuro/baselines/_attention.py` | new (`build_csr` + `segment_attention`) | 99 |
| `hymeko_neuro/baselines/sigat_model.py` | modify (`MotifAttention.forward` → primitive + cache) | +25/−40 |
| `hymeko_neuro/baselines/sgt.py` | modify (`SignedAttention.forward` → primitive + cache, drop unused `F`) | +22/−41 |
| `hymeko_neuro/tests/test_vectorized_attention.py` | new (parity/boundary/determinism) | 110 |

## CORE.YAML items touched

**None.** `hymeko_neuro/` is not in `CORE.YAML`. No new dependency
(`scatter_reduce`/`index_add` are core torch, verified in 2.12).

## Test results

`pytest -p no:randomly` over the new + existing baseline tests — **19 passed**.

- **Parity (the correctness oracle):** `segment_attention` matches a naive
  per-node softmax reference to `atol=1e-5`, **with and without** the sign bias
  (covers both the `MotifAttention` and `SignedAttention` code paths).
- **Boundary:** isolated nodes yield zero rows; an all-empty graph returns zeros.
- **Determinism:** identical output on repeat (CPU).
- **No regression:** all 14 `test_baseline_audit.py` tests still pass — the 7
  baselines forward-shape and run-audit correctly through the new attention.
- **Static gate:** `ruff check` clean on all four files.

Numerical note (§3): segment-softmax reassociates the reduction, so audit AUROCs
shift at the fp level vs the old loop — acceptable (different impl, same math);
the parity test pins correctness to a *spec* reference, not the old code. Grid
numbers will be produced by this faster, correct path.

## §6.5 anti-patterns

None introduced. Positively removed a duplication: the two attention modules
shared a near-identical by-length loop, now one `segment_attention` primitive.

## Open issues / follow-ups

1. **Run the grid** — now ~10–15 GPU-h, feasible locally (the original reason for
   this work). The `--baselines` mode in `run_no_leak_benchmark` is ready.
2. Cloud fan-out is now optional (local is feasible); the turnkey GCP kit remains
   available on request.

## Provenance

Git SHA: working tree dirty (this change + prior uncommitted edits). Host:
Windows 11, RTX 3070 Laptop 8 GB, torch 2.12.0+cu132, Python 3.12.13. Measured:
sesgformer on Epinions (|V|=131,828, |E|=841,372), 4 epochs, seed 0. Before-number
from `reports/2026-06-14-phase-b-baseline-shuffle-audit.md` (same host, same day).
