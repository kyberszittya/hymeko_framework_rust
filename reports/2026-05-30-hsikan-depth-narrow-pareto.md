# Report — HSiKAN depth + narrow-breadth Pareto: narrow-deep beats wide-shallow at fewer params

**Date:** 2026-05-30
**Predecessors:**
- `reports/2026-05-29-hsikan-per-channel-h64-compound.md` (per_channel + h=64 = 62k matches h=128 at 55 % of params)
- 2026-05-29 user prompt: "schedule testing for depth and otherwise narrow-breadth HSiKANs" — prerequisite for the long-skip-wiring direction.
**Branch:** ran on `feature/pgraph_engine`; further work on `feat/fuzzy-signature-tnorm-pooling`.
**CORE.YAML items touched:** none.

## Headline — narrow + deep dominates the Pareto

| Config | params | MNIST | Fashion | sd (M) | wall |
|:--|--:|--:|--:|--:|--:|
| **h=16, L=8** | **14 538** | **0.9704 ± .0014** | **0.8661 ± .0041** | tight | 1 615 s |
| h=16, L=4 | 7 370 | 0.9604 ± .0019 | 0.8483 ± .0021 | tight | 813 s |
| h=8, L=8 | 6 570 | 0.9561 ± .0022 | 0.8425 ± .0030 | tightest | 1 358 s |
| h=8, L=4 | 3 338 | 0.9287 ± .0092 | 0.8223 ± .0080 | — | 692 s |
| h=16, L=2 | 3 786 | 0.9166 ± .0070 | 0.8118 ± .0040 | — | 412 s |
| h=8, L=2 | 1 722 | 0.8785 ± .0083 | 0.7781 ± .0166 | — | 358 s |

For comparison (prior runs):

| Reference config | params | MNIST | Fashion |
|:--|--:|--:|--:|
| h=32, L=2 baseline | 10 218 | 0.9426 | 0.8369 |
| h=32, L=2 + per_channel | 25 130 | 0.9622 | 0.8569 |
| h=64, L=2 | 32 298 | 0.9595 | 0.8539 |
| h=64, L=2 + per_channel | 62 122 | 0.9700 | 0.8580 |
| h=128, L=2 | 113 322 | 0.9679 | 0.8645 |
| CNN | 42 154 | 0.9874 | 0.9071 |

## h=16 L=8 (14 538 params) is the new HSiKAN-vision Pareto-optimal config

It beats **every** prior configuration on **both** datasets:

| vs reference | MNIST Δ | Fashion Δ | params ratio |
|:--|--:|--:|--:|
| h=128 (capacity ceiling) | +0.0025 | +0.0016 | **0.128×** (8× smaller!) |
| h=64 + per_channel (prior best Pareto) | +0.0004 | +0.0081 | 0.234× |
| h=32 + per_channel | +0.0082 | +0.0092 | 0.579× |
| h=32 L=2 baseline | +0.0278 | +0.0292 | 1.422× |
| CNN | −0.0170 | −0.0410 | 0.345× |

**HSiKAN-vision is now within 1.7 pp of CNN on MNIST and within 4.1 pp on Fashion, at 35 % of CNN's parameters.** That's a genuinely competitive operator family.

## Verdict against the predicted outcomes (Outcomes 1 + 5)

From the 2026-05-29 forecast:

- **Outcome 1 (depth helps, gradients flow)** — confirmed. Monotonic improvement with depth at every width.
- **Outcome 5 (narrow-deep beats wide-shallow per param)** — confirmed. h=16/L=8 dominates h=128/L=2 on both datasets at 8× fewer params.
- **Outcome 2 (saturation at L=4)** — **ruled out**. L=8 keeps improving over L=4 on both datasets, both widths.
- **Outcome 3 (depth doesn't help, only width)** — ruled out trivially.
- **Outcome 4 (catastrophic vanish at L=8)** — ruled out, *strongly*. The seed sd at L=8 is the *tightest* in the matrix (h=16/L=8 MNIST sd = .0014; h=8/L=8 MNIST sd = .0022) — gradient flow is stable, not marginal.

This is the best-case outcome combination from the prediction set.

## Two cross-cutting observations

### Depth strictly dominates width (per-parameter)

At matched parameter budgets, deeper-narrower wins:
- (h=16/L=4 = 7.4k) vs (h=32/L=2 = 10.2k baseline): **0.9604 vs 0.9426 = +1.8 pp Mat 70 % the params**.
- (h=8/L=8 = 6.6k) vs (h=32/L=2 = 10.2k): **0.9561 vs 0.9426 = +1.4 pp at 65 % the params**.
- (h=16/L=8 = 14.5k) vs (h=128/L=2 = 113k): **0.9704 vs 0.9679 = +0.25 pp at 13 % the params**.

### Variance shrinks with depth

| Config | MNIST sd |
|:--|--:|
| h=8/L=2 | .0083 |
| h=8/L=4 | .0092 |
| **h=8/L=8** | **.0022** (4× lower) |
| h=16/L=2 | .0070 |
| h=16/L=4 | .0019 |
| **h=16/L=8** | **.0014** (5× lower) |

Deeper HSiKAN is *more stable* across seeds, not less — consistent with the residual + CR activation chain providing a smooth loss landscape, not a chaotic one. This is a meaningful empirical hint that L=8 is a healthy operating point, not a fragile one.

## Engineering finding (bug + fix)

The first chain attempt false-positive-skipped configs 2–6 because the orchestrator's resume key was `(model, dataset, seed)` — single-config sweeps were fine, multi-config sweeps to one jsonl turned the resume gate into a "skip after first config" gate. Fixed in `signedkan_wip/experiments/runs/run_vision_hypergraph_vs_cnn.py`: `full_cell_key` now includes `hidden`, `n_layers`, `spatial_filter`, `tie_we`, `n_epochs`, `train_subset`, `compile`, `amp`. Regression test in `signedkan_wip/tests/test_vision_bench.py::test_full_cell_key_includes_config_axes`. 8 tests + 51 pre-existing pass.

## Files / tests

- Code change: `run_vision_hypergraph_vs_cnn.py` (resume fix), `vision_bench_cell.py` (already had `--n-layers` plumbing from the morning).
- New test: `test_full_cell_key_includes_config_axes`.
- 51 vision-bench tests still pass; 19 t-norm boundary tests (separate branch) pass.

## Provenance

- **Git SHA:** at the start of the sweep, `8fd8187` (dirty).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130 (CORE drift, user-approved).
- **GPU:** RTX 2070 SUPER 8 GiB.
- **Cells:** 6 configs × 2 datasets × 3 seeds = 36; **0 failures**.
- **Total wall:** ~5 h on the second chain attempt (first attempt failed the bug above; only h8_L2 ran).
- **Artifacts:** `/tmp/vision_depth_narrow/results.jsonl` (36 rows), `summary.json`.

## What this unlocks

1. **Long-skip wiring (the user's 2026-05-29 idea) is now well-motivated.** At L=8 with healthy gradient flow and depth still helping, U-Net-style early-to-late skips have somewhere to connect. The natural next architecture experiment.
2. **T-norm pooling (the Kóczy fuzzy-signature direction) gets its best base.** The depth-winner `h=16/L=8` is also the cleanest fuzzy-signature operating point (low-variance, deep hierarchy, narrow channels → matches Kóczy-style structured fuzzy aggregation more than wide-shallow does). Branch `feat/fuzzy-signature-tnorm-pooling` is built; t-norm GPU smoke is in flight now (b7yph9auo). If t-norms train at this config, the overnight follow-up is the empirical companion to the fuzzy-signature mapping.
3. **The vision push reaches a natural closing point.** Best HSiKAN-vision is now within ~2/4 pp of CNN at 35 % of CNN's params. Further gains likely need either long-skips (depth axis) or kernel-shape expressiveness (full `W[K, d_in, d_out]` filter, but at that point we're rebuilding CNN).

## Per-seed table (depth+narrow, raw)

| ds | h | L | seed | acc |
|:--|--:|--:|--:|--:|
| mnist | 8 | 2 | 0/1/2 | 0.870 / 0.880 / 0.886 |
| mnist | 8 | 4 | 0/1/2 | 0.933 / 0.918 / 0.935 |
| mnist | 8 | 8 | 0/1/2 | 0.958 / 0.954 / 0.956 |
| mnist | 16 | 2 | 0/1/2 | 0.915 / 0.924 / 0.911 |
| mnist | 16 | 4 | 0/1/2 | 0.962 / 0.958 / 0.961 |
| **mnist** | **16** | **8** | 0/1/2 | **0.970 / 0.969 / 0.972** |
| fashion | 8 | 2 | 0/1/2 | 0.772 / 0.797 / 0.766 |
| fashion | 8 | 4 | 0/1/2 | 0.827 / 0.827 / 0.813 |
| fashion | 8 | 8 | 0/1/2 | 0.839 / 0.844 / 0.845 |
| fashion | 16 | 2 | 0/1/2 | 0.807 / 0.814 / 0.814 |
| fashion | 16 | 4 | 0/1/2 | 0.847 / 0.849 / 0.849 |
| **fashion** | **16** | **8** | 0/1/2 | **0.867 / 0.870 / 0.862** |

## Follow-up (active)

- **`b7yph9auo`** (running): GPU smoke 4 t-norm pooling modes at `h=16/L=8` / MNIST / 5 ep / 2k subset / 3 seeds. Determines whether `min`/`product`/`Łukasiewicz` train sanely at the depth-winner before committing the overnight sweep.
- **If smoke green**: overnight 4-mode × 2-dataset × 3-seed = 24-cell sweep (the empirical companion to the fuzzy-signature mapping).
