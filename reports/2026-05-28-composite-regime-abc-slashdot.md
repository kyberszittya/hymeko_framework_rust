# Report — Regime A/B/C 5-seed comparison on Slashdot (cross-dataset verdict)

**Date:** 2026-05-28
**Plan:** `docs/plans/2026-05-27-composite-regime-abc-5seed/` (same harness, `--dataset slashdot`)
**Parent:** `reports/2026-05-27-composite-regime-abc-5seed.md` (Bitcoin Alpha)
**CORE.YAML items touched:** none (torch drift caveat carries over).

## Why this run

The BA comparison found Canonical's larger admissible set buys zero
quality, *because* its distinguishing axis (attention/quaternion) was null
on BA. The decisive cross-dataset test: rerun on **Slashdot**, where
attention is documented to carry signal (memory
`attention_cycle_batch_compose`, 0.903 at full enumeration), to see
whether Canonical (which alone admits the quaternion branch) finally beats
No-Excess — i.e. whether the 33 %-lossless prune *stays* lossless when the
pruned axis is not null.

## Headline — same verdict, with one unavoidable caveat

| Regime | #admissible | Best architecture | 5-seed AUC |
|:--|--:|:--|:--|
| **A Canonical** | 12 | `attn=none · gate=scalar · dm=on` | **0.8192 ± 0.0142** |
| **B No-Excess ≡ Composite** | 8 | `attn=none · gate=scalar · dm=on` (same) | **0.8192 ± 0.0142** |
| **C Cost-dominance** | 2 | `attn=none · gate=scalar · dm=off` | 0.8188 ± 0.0142 |

Canonical-best vs No-Excess-best: **identical architecture**, paired Δ = 0
(σ = 0). The quaternion branch (Canonical-only) again never wins — best
quaternion `edge_cr·dm=off` = 0.8121, below global best 0.8192 (paired
Δ = +0.0070 for the non-quaternion arch, σ +0.92, 3/5 — a tie).

**So on both datasets tested, No-Excess (≡ Composite) reaches the same
optimum as Canonical with 33 % fewer candidates. The lossless cut held
twice.**

### The caveat I cannot wave away

This Slashdot run is under a **uniform per-vertex K=8 cycle cap**, forced
by the 7.6 GiB GPU: uncapped quaternion-attention OOMs (measured — 6.58 GiB
in use, 108 MiB short, even at h=8/K=8). The cap had to be uniform across
all 12 archs to avoid an attention-vs-non-attention enumeration confound.
The documented Slashdot attention win (0.903) used **full enumeration** +
full mix `c2,c3,c4,c5,w2,w3` + Highway-quaternion at h=4 — a regime this
hardware cannot run in this harness. **Therefore this result shows
attention is null/negative under a K=8 budget on Slashdot; it does NOT
refute attention helping at full enumeration.** The clean "attention
carries signal" test remains GPU-blocked.

### Slashdot lever profile (≠ Bitcoin Alpha)

Marginal paired effects (over all other axes × 5 seeds):

| Lever | Slashdot (K=8) | Bitcoin Alpha | note |
|:--|--:|--:|:--|
| attention `dot − none` | **−0.0085 (σ−2.04)** | −0.0111 (σ−0.95) | sig. negative on Slashdot |
| attention `quaternion − none` | **−0.0102 (σ−2.13)** | −0.0093 (σ−0.81) | sig. negative on Slashdot |
| gate `edge_cr − scalar` | **+0.0069 (σ+1.52)** | −0.0053 (σ−0.65) | flips sign vs BA |
| direct-msg `on − off` | **−0.0001 (σ−0.39) null** | +0.0150 (σ+2.50) | BA's winner is null here |

The datasets have genuinely different lever profiles (edge_cr flips, dm
flips), confirming the comparison is sensitive — but on neither does the
regime-distinguishing axis (attention) come out positive, so Canonical's
extra set never pays.

## Test / quality results

60 runs, **0 failures**. All under the 16 GB RSS gate
(`systemd-run -p MemoryMax=16G`), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
No new code — the parent task's harness (`run_regime_abc_5seed.py`) and its
31 unit/integration tests cover this; only `--dataset slashdot --hidden 4`
and the env cap differ.

## Performance results

| Metric | Value | Budget |
|:--|:--|:--|
| Total wall (60 runs) | 153.3 min | reconciled from 150 s/run probe (§11) |
| Mean per-run wall | 153 s | enumeration-dominated (epochs ~free) |
| Max per-run peak RSS | 2.46 GiB | ≤ 7 GiB ✓ (16 GB cap) |
| GPU OOMs | 0 | h=4 + K=8 + expandable_segments |

Production-scale smoke before launch (CLAUDE.md §3): heavy arch
(quaternion·edge_cr·dm=on) probed at h=8 (OOM) → h=4/K=8 (fit, AUC 0.80).
Wall validated at ~150 s/run before queuing the 60-run job.

## Provenance

- **Git SHA:** `8fd8187` (dirty; same tree as parent tasks — no new
  source files this round).
- **Interpreter:** miniconda3 python 3.13.5, **torch 2.11.0+cu130**
  (DRIFT vs CORE `==2.12.0`, user-approved; AUCs are relative, capped, not
  SOTA — Slashdot SOTA ≈ 0.907 at full enum, memory `edge_cr_5seed`).
- **GPU:** RTX 2070 SUPER 8 GiB, driver 580.126.09. **Seeds:** {0..4}.
  **Epochs:** 20. **Hidden:** 4. **Cap:** per-vertex top-K=8 (uniform).
- **Artifacts:** `/tmp/regime_abc_slashdot/results.jsonl` (60 rows),
  `summary.json`, per-cell logs.

## Cross-dataset conclusions

1. **The 33 %-lossless No-Excess cut held on both BA and Slashdot.** On
   both, No-Excess (≡ Composite) reached the Canonical optimum with 8
   candidates instead of 12. The canonical/Friedler regime's larger
   admissible set bought **zero quality** on both datasets I could run.
2. **It is not yet proven to hold where attention genuinely helps.** The
   one regime that could break it — full-enumeration Slashdot, where
   attention's 0.903 lives — is GPU-blocked on this 7.6 GiB card. So the
   honest status is: *lossless on everything testable here; the
   adversarial case is untested, not passed.*
3. **Cost-dominance is consistently too aggressive** (BA: dropped the
   dm=on winner; Slashdot: tied but only 2 candidates) — not a search
   regime to use.
4. **Practical recommendation unchanged:** use **No-Excess / Composite**
   for HSiKAN-mixed architecture search — same optimum, smaller search,
   on both datasets.

## Follow-ups

- **Lift the cap to settle the open question.** Wire the existing
  sparse-attention path (`HSIKAN_SPARSE_ATTN_K`, already in
  `runtime_config`/`scatter.py`) into the attention archs so quaternion
  can see a larger cycle pool within 7.6 GiB, then rerun the 4 quaternion
  Slashdot archs. That is the decisive test of whether Canonical ever
  earns its keep. (New wiring + smoke — a small task, not just a launch.)
- Or run on a larger-VRAM GPU at full enumeration.
