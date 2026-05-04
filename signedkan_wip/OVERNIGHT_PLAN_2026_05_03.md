# Overnight plan — 2026-05-03

Continuation from the day's HSiKAN work. Two jobs run sequentially
overnight; results land in JSONL files for cold-read analysis tomorrow.

---

## Current standing (entering the night)

| dataset | best HSiKAN AUC | std | recipe | published ref |
|---|--:|--:|---|---|
| Bitcoin Alpha | **0.9399** | 0.0085 (5 seeds) | k=3+k=4+k=5, balance λ=1.0, h=16, grid=5, cosine, 120ep | SGCN ~0.91, SiGAT ~0.93 |
| Slashdot | 0.7686 (CPU, no balance) / 0.61 (with balance λ=1.0) | 0.0013 / 0.0015 | k=3+k=4 reservoir | SGCN ~0.91 |
| SBM_n200 (synth, leaky) | ~1.0 (k=2) / ~0.95 (k=4,5) | tight | single-arity | — |
| Karate (no-leak vertex_adj) | 1.0 (k=3,5,6) | 0–0.06 | single-arity | — |

Open question after Bitcoin SOTA: **balance loss is dataset-specific.**
Bitcoin loves λ=1.0 (+0.06 AUC); Slashdot collapses at λ=1.0 (−0.16
AUC). The overnight grid maps this surface across 9 datasets.

---

## Job 1 — Slashdot λ sweep (running now, ~15 min ETA)

- `signedkan_wip/src/run_phase2_mixed_arity.run_one_mixed`
- `dataset=slashdot`, `arities=(3,4)`, `max_per_arity={3:30k, 4:30k}`,
  `n_epochs=120`, `grid=5`, `cosine`, `seed=0`
- Sweeps `balance_lambda ∈ {0.0, 0.01, 0.05, 0.1, 0.3, 1.0}`
- Log: `/tmp/hsikan_logs/slashdot_lambda_sweep.log`

**Question:** is there a small-λ sweet spot for Slashdot, or is balance
loss universally bad on this graph?

---

## Job 2 — Overnight grid + phase 9 chain (auto-launches when Job 1 completes)

The chain script (`/tmp/overnight_chain.sh`, PID `cat /tmp/overnight_chain.pid`)
runs three stages back-to-back:

  Stage 1: Slashdot λ sweep (already running, ~15 min)
  Stage 2: Overnight grid (~4-6 h)
  Stage 3: Phase 9 Slashdot SOTA chase (~1-2 h, 13 cells × 3 seeds = 39 cells)

### Stage 2 details

Driver: `signedkan_wip/src/run_phase8_overnight_grid.py`
Output: `signedkan_wip/experiments/results/phase8_overnight_grid.jsonl`
(resumable; rerun re-uses completed cells).

### Grid

```
datasets   = bitcoin_alpha, bitcoin_otc, slashdot,
             sbm_n200_k4_s0, sbm_n400_k5_s0, hier_n240_s0,
             sbmsweep_pos50_s0, sbmsweep_pos85_s0, karate
λ          = 0.0, 0.1, 1.0
arities    = (3,4) and (3,4,5)        # slashdot: (3,4) only
grid       = 3, 5
lr_sched   = cosine, fixed
seeds      = 0, 1, 2
```

Cells: ~600 total. ETA ~4-6 hours on GPU.

### Per-dataset config notes

- **slashdot** skips `(3,4,5)` because k=5 reservoir-Algorithm-R takes
  hours per cell on Slashdot. `(3,4)` only with reservoir.
- **karate** uses smaller per-arity caps (5k) since the graph is tiny.
- All other datasets use 30k per arity except synthetic SBMs (10k).

### What the grid answers

1. **Best balance λ per dataset** — does Bitcoin's λ=1.0 win generalize
   or is it dataset-specific?
2. **Best arity mix per dataset** — k=3+k=4 vs k=3+k=4+k=5; tracks the
   αₖ pattern across data regimes.
3. **Spline grid sensitivity** — grid=3 vs 5; previous Bitcoin sweep
   showed barely any difference but Slashdot may differ.
4. **LR schedule sensitivity** — cosine vs fixed.
5. **Cross-dataset patterns** — does HSiKAN consistently prefer high-
   community-structure graphs (Bitcoin/SBM/karate) over high-density
   trust networks (Slashdot/Epinions-like)?

### Resume / monitor

```sh
# Live log (covers all 3 stages):
tail -f /tmp/hsikan_logs/overnight_grid.log

# Inspect Stage 2 results so far:
wc -l signedkan_wip/experiments/results/phase8_overnight_grid.jsonl

# Inspect Stage 3 results once it starts:
wc -l signedkan_wip/experiments/results/phase9_slashdot_sota.jsonl
```

If killed mid-run, simply re-launch the chain — both stages have
resumable JSONL output that auto-skips completed cells.

### Stage 3 details (phase 9 Slashdot SOTA chase)

Driver: `signedkan_wip/src/run_phase9_slashdot_sota.py`
Output: `signedkan_wip/experiments/results/phase9_slashdot_sota.jsonl`

Targeted ablations to close the 0.77 → 0.91 SGCN gap on Slashdot:

| cell | attention | direct_msg | max_k4 | h | λ | recipe |
|---|---|---|---:|---:|--:|---|
| baseline_30k | – | – | 30k | 16 | 0 | grid=5, cosine |
| baseline_30k_g3_fix | – | – | 30k | 16 | 0 | grid=3, fixed (morning recipe) |
| baseline_100k | – | – | 100k | 16 | 0 | |
| balance_30k_l05 | – | – | 30k | 16 | 0.05 | |
| balance_100k_l05 | – | – | 100k | 16 | 0.05 | |
| attn_30k | ✓ | – | 30k | 16 | 0 | |
| attn_100k | ✓ | – | 100k | 16 | 0 | |
| direct_30k | – | ✓ | 30k | 16 | 0 | |
| direct_100k | – | ✓ | 100k | 16 | 0 | |
| attn_direct_30k | ✓ | ✓ | 30k | 16 | 0 | |
| attn_direct_100k | ✓ | ✓ | 100k | 16 | 0 | |
| h32_direct_100k | – | ✓ | 100k | 32 | 0 | |
| h32_attn_direct_100k | ✓ | ✓ | 100k | 32 | 0 | |

13 cells × 3 seeds = 39 runs.

### What stage 3 answers

- Does the **attention M_e init fix** finally help (or hurt less)?
- Does **direct messaging** (SGCN-style sign-conditional path) push past
  0.77 toward 0.91?
- Does **more cycles** (100k vs 30k) matter on Slashdot?
- Does the **morning recipe** (grid=3, fixed lr) actually reproduce its
  0.7686 baseline?
- Does the **stack of attention + direct + larger h** hit SGCN-level
  AUC?

---

## Decision tree for tomorrow

Based on the grid output, three branches:

### A. Bitcoin pattern survives broadly (HSiKAN looks like a general
    strong baseline)
**Trigger:** ≥4 datasets have a non-zero λ optimum and reach ≥0.90 AUC.
- 5-seed expansion on best per-dataset config
- SGCN/SiGAT reproduction in our codebase under same protocol
- Begin paper writeup with HSiKAN as a general-purpose signed link
  predictor

### B. HSiKAN is graph-class specific (wins on community-rich, loses
    on dense trust networks)
**Trigger:** clean split between SBM/Bitcoin/karate (HSiKAN strong) and
Slashdot/Epinions-like (HSiKAN weak).
- Reposition the paper as "structural-prior NN for community-rich
  signed graphs"
- Add scene-graph and kinematic-graph experiments (where structural
  priors matter most) — see Berge cycle generalization in earlier
  discussion
- Combine HSiKAN with SGCN via task #31 (hybrid layer) for the dense-
  graph regime

### C. Bitcoin was a one-off
**Trigger:** Bitcoin's 0.94 doesn't replicate on bitcoin_otc, and SBM
synthetics also don't show clean improvement from balance loss.
- Investigate whether the Bitcoin AUC is itself partially leak-
  amplified by some path we haven't seen
- Run pair-dedup version of the entire grid
- Reconsider whether HSiKAN's contribution is methodology (no-leak
  protocol, αₖ-mask B&B, Rust enumerator) more than architecture

---

## Open architectural threads (deferred)

| ID | Item | Status |
|---|---|---|
| #31 | Hybrid HSiKAN+SGCN direct messaging | Pending. Most likely SOTA-pushing architectural lever. |
| #30 | Attention-weighted M_e | Implemented but smoke test showed −0.07 AUC. Needs init tuning (W_q, W_k init scale, longer warmup, temperature). |
| #35 | Multi-task auxiliary heads (signed-degree) | Implemented but 3-seed showed no lift on Bitcoin. Different aux task (common-neighbor count, balance ratio per edge) might help. |
| — | Per-query σ-masking (stricter leak removal) | Not yet implemented. The most surgical no-leak protocol; relevant if grid results suggest residual σ-leak. |
| — | Berge cycle generalisation (hypergraph extension) | Theoretical hook for kinematic / scene graph datasets. Implementation deferred. |

---

## Files / artefacts

- Bitcoin SOTA result: `signedkan_wip/experiments/results/phase8_sota_chase.jsonl`
- Bitcoin 5-seed: `/tmp/hsikan_logs/bitcoin_5seed.log`
- Slashdot λ sweep (live): `/tmp/hsikan_logs/slashdot_lambda_sweep.log`
- Overnight grid driver: `signedkan_wip/src/run_phase8_overnight_grid.py`
- Overnight grid log: `/tmp/hsikan_logs/overnight_grid.log` (live)
- Overnight grid results: `signedkan_wip/experiments/results/phase8_overnight_grid.jsonl`

## Stack changes shipped this session

- Pair-dedup splits (`datasets.deduplicate_pairs`)
- Vertex-adjacency M_e (`m_e_mode="vertex_adjacency"`)
- Balance loss (`balance_lambda` param in `run_one_mixed`)
- Multi-task aux head (`multitask_lambda`)
- Attention M_e (`attention_m_e=True` config flag)
- Cycle batching (`cycle_batch_size` config)
- αₖ-mask B&B (`set_arity_mask` on `MixedAritySignedKAN`)
- LR schedule (`lr_schedule="cosine"`)
- Rust enumerator: directed mode, early_stop mode, reservoir mode
- scipy SpGEMM M_e construction (~3.4× over original Python)

All available via flags on `run_one_mixed`; defaults preserve original
behaviour.
