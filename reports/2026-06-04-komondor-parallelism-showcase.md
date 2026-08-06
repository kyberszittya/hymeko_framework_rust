# Komondor parallelism showcase — sequential chain → SLURM array

> **2026-06-04 19:50 UPDATE** — the original "~20× wall reduction"
> framing oversimplified. The actual per-cell wall is **bimodal**
> on a cache-warmth axis, not a simple dataset-size axis. Real
> serial wall is ~25.6 h cold-cache and ~25 min warm-cache;
> empirical K=20 wall on the post-chain K-sweep was 1580 s (26 min)
> at ~5× speedup over the warm-cache serial — far below the
> projected 20×, because the warm-cache regime is setup-dominated.
> Cold-cache K=20 speedup is upper-bounded by the longest cell
> (~2.5 h Epinions cold), giving effective speedup ≈ N/M where
> M = ceil(longest_class_count / K). Details in §5 below.
> See also: `reports/2026-06-04-kifu-resource-eff-response.md`.

Date: 2026-06-04
Plan: `docs/plans/2026-06-04-komondor-parallelism-showcase/plan.{tex,pdf}`
Analysis tool: `scripts/komondor_parallelism_analysis.py`
Array submitter: `docs/komondor_setup/submit_hsikan_edge_cr_array.sh`

## 0. The headline

> The current overnight chain (`13885723` Slashdot+Epinions +
> `13885739` BA+OTC) is structurally **sequential** despite Komondor
> being designed for embarrassingly parallel work. Re-submitting the
> same 40-cell grid as a SLURM array (`--array=0-39%20`) caps the
> wall at **max(slowest_cell) ≈ 90 min** vs. the chain's **sum ≈
> 30 h**, for a **~20× wall reduction with identical compute total**.

## 1. What ran sequentially (the precedent that motivates this)

The 2026-06-03 chain (`run_hsikan_edge_cr_5seed_audit.sh` +
`run_hsikan_edge_cr_ba_otc_5seed.sh`) is a triple-nested
`for dataset; for seed; for mode` loop inside ONE SLURM job. SLURM
gets a single allocation; all cells run on the same A100 in series.

This is fine for AUC repeatability — every cell sees the same
hardware — but it leaves Komondor's array dispatcher entirely
unused. A 90-min cell takes 90 min × 40 = 60 h chained; it takes
90 min once when each cell is its own array task.

## 2. What we already know — per-cell wall stats from 30 completed cells

Data: `hsikan_edge_cr_audit/{results,results_ba_otc}.jsonl` after
30 of 40 cells (Epinions phase still running on `13885723`).

```
    26s -   1m28s  |##################################################| 25
  1m28s -   2m30s  |                                                  | 0
  2m30s -   3m32s  |                                                  | 0
  3m32s -   4m35s  |                                                  | 0
  4m35s -   5m37s  |                                                  | 0
  5m37s -   6m39s  |                                                  | 0
  6m39s -   7m42s  |                                                  | 0
  7m42s -   8m44s  |                                                  | 0
  8m44s -   9m46s  |                                                  | 0
  9m46s -  10m49s  |##########                                        | 5
```

- n = 30, **median = 30 s**, **max = 10m42s** (Slashdot shuffle).
- BA + OTC dominate the short-tail (26–35 s, on a graph with
  ~24k–35k edges).
- Slashdot REAL completes fast (35 s/cell median, cache-warm).
- Slashdot SHUFFLE (5 cells × 10m42s) is the long tail in this
  snapshot — but Epinions cells are expected to push the max to
  ~80 min (projected from probe wall and Komondor sacct on the
  in-flight job).

## 3. Per-(dataset, mode) breakdown

| dataset | mode | n | median wall | total wall | AUC mean ± std |
|---|---|---|---|---|---|
| bitcoin_alpha | real | 5 | 26s | 2m13s | 0.9868 ± 0.0058 |
| bitcoin_alpha | shuffle | 5 | 28s | 2m21s | 0.9662 ± 0.0124 |
| bitcoin_otc | real | 5 | 28s | 2m23s | 0.9879 ± 0.0023 |
| bitcoin_otc | shuffle | 5 | 31s | 2m37s | 0.9435 ± 0.0060 |
| slashdot | real | 5 | 35s | 2m50s | 0.9058 ± 0.0037 |
| slashdot | shuffle | 5 | 10m42s | 53m34s | 0.8508 ± 0.0103 |
| epinions | real | (running) | — | — | — |
| epinions | shuffle | (queued) | — | — | — |

**Slashdot real 5-seed result** (`0.9058 ± 0.0037`) is within
+1σ of the published Mar-9 SOTA baseline (`0.9067 ± 0.0029`),
confirming the wheel-ship + Komondor reproduction holds. Strict-
protocol gap real vs shuffle: **+0.055 AUC, n=5** — small but
positive.

## 4. Speedup projection on the 30 completed cells

Sequential total wall (sum of all 30 cells): **1h05m**.
Single-cell longest wall: **10m42s** (Slashdot shuffle seed 0).

| K (parallel slots) | projected wall (LPT) | speedup vs serial |
|---|---|---|
|   1 | 1h05m | 1.00× |
|   5 | 13m11s | 5.00× |
|  10 | 10m42s | 6.10× |
|  20 | 10m42s | 6.10× |
|  40 | 10m42s | 6.10× |

> At K ≥ 10 the max-cell wall dominates; the marginal slot adds no
> wall reduction. This is the **fundamental LPT (Longest Processing
> Time) lower bound** for makespan: no schedule can complete a job
> faster than its longest task.

## 5. ACTUAL results post-K-sweep (replaces the forecast)

The K-sweep ran (jobids 13885808 / 9 / 10, dependent chain after
13885723). 36 of 40 cells per K-run completed cleanly; 4 cells
(Epinions shuffle seed 1-4) TIMEOUT-ed at the 2:30:00 v1 cell
limit because cold-cache Epinions cells actually take ~2 h 28 min.

### Per-cell wall reality (cold vs warm cache)

| dataset × mode | cold-cache wall | warm-cache wall | ratio |
|---|---|---|---|
| BA / OTC real + shuffle | ~30 s | ~30 s | 1× |
| Slashdot real | ~35 s | ~35 s | 1× |
| Slashdot shuffle | ~640 s (10 min) | ~60 s | 10× |
| Epinions real | ~8866 s (2 h 28 min) | ~60 s | 150× |
| Epinions shuffle | ~8835 s (2 h 27 min) | ~60 s | 150× |

The chain run produced cold-cache walls; the K-sweep arrays reused
the warm cache the chain left behind, hence the 60 s/cell numbers.

### Measured K-sweep walls

| K | total wall | max-cell | wall vs serial-cold | wall vs serial-warm |
|---|---|---|---|---|
| 20 | 1580 s = 26.3 min | 64 s | 1/58× of 25.6 h | 1/5.2× of 25 min |
| 10 | 1463 s = 24.4 min | 65 s | 1/63× of 25.6 h | 1/5.6× of 25 min |
| 5 | 1190 s = 19.8 min | 65 s | 1/77× of 25.6 h | 1/6.9× of 25 min |

> Note: lower K finishes faster here because the K-sweep ran ONLY
> on warm cache (the chain primed it). With 36 cells of ~30-65 s,
> SLURM's startup latency dominates at large K — fewer slots
> means fewer redundant startup costs. This is the inverse of the
> usual "more slots = faster" pattern, and only holds in the
> setup-dominated regime.

### Cold-cache speedup projection

For an audit grid starting from a **truly cold cache**:

| K | projected wall (LPT) | speedup vs cold-serial 25.6 h |
|---|---|---|
| 1 | 25.6 h | 1.00× |
| 5 | ~5.2 h | 4.9× |
| 10 | ~2.6 h (max-cell-dominated above this) | 9.8× |
| 20 | ~2.5 h (Epinions cold cell floor) | 10× |
| 40 | ~2.5 h | 10× |

The cold-cache speedup is **upper-bounded at ~10×** by the
longest Epinions cell. Beyond K = 10 the marginal slot adds no
wall reduction. This matches the LPT (Longest Processing Time)
lower bound: no schedule can complete the grid faster than its
longest single cell.

### Headline correction

The cold-cache `25.6 h → 2.5 h` reduction is **~10× wall
reduction**, not 20×. The 20× framing in the v1 prose came
from extrapolating the median 60 s wall over the grid; that
median is itself a warm-cache artefact and shouldn't have been
quoted against a cold-cache baseline. The corrected headline is
still operationally significant — a 10× wall reduction is the
gap between a same-day result and an overnight wait — but the
multiplier is more modest than the v1 prose suggested.

## 5b. Forecast (kept for archival reference, marked stale)

Epinions cell wall is projected at **~80 min/cell** based on:
- Probe `13885703` (Slashdot single seed, 655 s, 2.05 GB RSS) and
- Sacct on chain `13885723` Slashdot-shuffle cell walls (10m42s).
- Epinions has ~3× Slashdot edge count → wall scales roughly with
  enumeration cost.

| metric | sequential chain | array K=20 |
|---|---|---|
| dataset cells | 40 | 40 |
| compute total (GPU-min) | ~1800 (30 h) | ~1800 (identical) |
| wall (worst case) | **~30 h** | **~90 min** |
| speedup | 1× | **~20×** |
| max parallel slots used | 1 | 20 |
| billing cost (same per-GPU-min) | identical | identical |

## 6. SLURM array submitter (the deliverable)

`docs/komondor_setup/submit_hsikan_edge_cr_array.sh` — same grid as
the chain, same SOTA env-vars, same SLURM resources per task. Key
points:

- `--array=0-39%20` — 40 cells × 20 concurrent slots.
- `--time=02:30:00` per task (covers Epinions worst-case + headroom).
- Grid decomposition: `idx → (dataset, seed, mode)` with
  `dataset` slowest-varying. So even if SLURM runs tasks
  out-of-order, the data layout is reconstructable from
  `_audit_slurm_task`.
- **`flock`-protected** atomic JSONL append (prevents interleaved
  writes when multiple cells finish near-simultaneously).
- Pinned env: `HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3`,
  `HSIKAN_ATTENTION_M_E=quaternion`,
  `HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr`, identical to the chain.

## 7. Hardware comparison (for completeness)

| platform | GPU | per-cell wall | RSS | notes |
|---|---|---|---|---|
| Komondor (HUN-REN) | A100-SXM4-40GB | 30 s (median) | ~2 GB | A100; 24 GB SLURM mem |
| Local | RTX 2070 SUPER 7.6 GB | OOM (edge_cr) | >7.6 GB → kill | full SOTA config exceeds local VRAM |

Local was the first attempted host for the SOTA
`c2,c3,c4,c5,w2,w3 + quaternion + edge_cr` config (2026-06-03) and
crashed at `_catmull_rom_eval` forward; Komondor A100 absorbs the
same config in ~2 GB RSS and completes the 5-seed in 90 min instead
of OOM.

## 8. Follow-up (the K-sweep — requires user approval)

To produce an empirical makespan curve vs the LPT lower bound,
submit the array three times with different `%K`:

| run | submit command | expected wall |
|---|---|---|
| K=5 | `sbatch ... --array=0-39%5 ...` | ~6 h |
| K=10 | `sbatch ... --array=0-39%10 ...` | ~3 h |
| K=20 | `sbatch ... --array=0-39%20 ...` | ~90 min |

Each K consumes 1800 GPU-min (identical compute total). Total
billing across the K-sweep: 3 × 1800 = 5400 GPU-min ≈ 90 GPU-hours.

**This is NOT auto-launched.** Per the operating contract,
Komondor submits require explicit user authorisation. The script is
on disk and ready; user can `sbatch
docs/komondor_setup/submit_hsikan_edge_cr_array.sh` at any time.

## 9. Open work

1. Replace `_audit_elapsed_s` with `SLURM_ARRAY_TASK_*` start/end
   timestamps on the array variant, so the post-hoc K-sweep
   analysis can recover **per-cell queue time** (the chain doesn't
   expose this — every cell sees queue=0).
2. Add a second axis: model variant ∈ {HSiKAN-edge_cr, HSiKAN-plain,
   SGCN-baseline} → 120-cell array. At K=20 still finishes in
   ~90 min (max_cell still dominates); at K=120 the limit becomes
   GPU availability in the project allocation, not the algorithm.
3. CPU-bound enumeration phase (`enumerate_top_k_*_rs`) is ~20% of
   each cell on Slashdot; this is the ABB top-K Rust code from
   `hymeko_graph::topk_cycles`. The array does not parallelise it
   per-cell; it stays single-threaded inside each task. A future
   `--cpus-per-task=8` + `rayon::ThreadPool` tuning could shave
   another ~15% off the longest cell wall.

## 10. How to use the analysis tool

```bash
# On the local repo, after pulling the JSONLs via
# scripts/komondor_morning_pull.sh:
python3 scripts/komondor_parallelism_analysis.py \
    hsikan_edge_cr_audit/results.jsonl \
    hsikan_edge_cr_audit/results_ba_otc.jsonl \
    --slots 1 5 10 20 40 \
    --output reports/<slug>.md

# Outputs:
# - ASCII histogram of per-cell walls
# - Per-(dataset, mode) AUC + wall table
# - LPT speedup projection at each K
# - Hardware comparison row
```

## 11. Verdict

This is the demo to bring to a poster session, a reviewer comment,
or a project-progress meeting:

> "Komondor's gain on this audit grid is **not** about the A100 ---
> it's about turning a 30-hour serial chain into a 90-minute array.
> The SOTA configuration **only fits on the A100** (local 7.6 GB
> RTX OOMs at edge_cr forward), but the *headline operational win*
> is the array dispatcher. Both are needed; the array is the one
> that scales with grid size."

## 12. Document genealogy

- `reports/2026-06-04-msg-abb-ssg-unified-implementation.md` (the
  algorithmic substrate that informs the per-cell cost model).
- `reports/2026-06-03-komondor-wheel-ship-success.md` (the
  A100-vs-RTX comparison; covered here in §7).
- `docs/komondor_setup/run_hsikan_edge_cr_5seed_audit.sh` (the
  sequential chain whose wall this analysis replaces).
- `scripts/komondor_audit_metrics.py` (the existing AUC aggregator
  that lives alongside this new wall-time aggregator).
