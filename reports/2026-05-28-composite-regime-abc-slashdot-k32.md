# Report — Slashdot regime A/B/C at K=32 + sparse focusing (decisive)

**Date:** 2026-05-28
**Plan:** `docs/plans/2026-05-27-composite-regime-abc-5seed/` (same harness, raised cap)
**Parents:** `reports/2026-05-27-composite-regime-abc-5seed.md` (BA),
              `reports/2026-05-28-composite-regime-abc-slashdot.md` (Slashdot K=8)
**CORE.YAML items touched:** none.

## Why this run

The Slashdot K=8 result said *capped Slashdot has the same verdict as BA*
(No-Excess ≡ Composite tied with Canonical, quaternion never wins), but
left an honest hole: the K=8 cycle cap might have starved attention,
which on Slashdot is documented to carry signal at full enumeration. This
run gives attention its **fairest shot this hardware allows**: 4× the
cycle budget (per-vertex top-K=32) plus the `HSIKAN_SPARSE_ATTN_K=8`
attention-focusing inductive bias.

## Headline — verdict tightens, doesn't flip

| Regime | #admissible | Best architecture | 5-seed AUC |
|:--|--:|:--|:--|
| **A Canonical** | 12 | `attn=none · gate=scalar · dm=off` | **0.8282 ± 0.0153** |
| **B No-Excess ≡ Composite** | 8 | `attn=none · gate=scalar · dm=off` (same) | **0.8282 ± 0.0153** |
| **C Cost-dominance** | 2 | `attn=none · gate=scalar · dm=off` (same) | **0.8282 ± 0.0153** |

All three regimes converge on the same non-quaternion architecture.
Canonical-best vs No-Excess-best: identical (Δ = 0). And — new at K=32 —
**Cost-dominance also lands on the winner** (its 2-arch admissible set
includes the global best, the simplest cheapest unit per axis).

**Best quaternion arch** = `edge_cr · dm=off` at 0.8159. Paired Δ vs
global best: **+0.0123, σ +2.07, 5/5 seeds**. At K=8 this was σ +0.92
(3/5). The richer budget made quaternion's loss *statistically
significant*, not less. Canonical's quaternion-only branch is definitively
not earning its keep on Slashdot at any cycle budget this 7.6 GiB GPU can
fit.

## Slashdot K=32 lever profile (vs K=8 and BA)

| Lever (paired Δ, σ) | Slashdot K=32 | Slashdot K=8 | BA |
|:--|--:|--:|--:|
| attention `dot − none` | −0.0074 (σ−1.88) | −0.0085 (σ−2.04) | −0.0111 (σ−0.95) |
| attention `quaternion − none` | **−0.0099 (σ−2.17)** | −0.0102 (σ−2.13) | −0.0093 (σ−0.81) |
| gate `edge_cr − scalar` | 0.0000 (σ−0.01) | +0.0069 (σ+1.52) | −0.0053 (σ−0.65) |
| direct-msg `on − off` | −0.0013 (σ−1.67) | −0.0001 (σ−0.39) | +0.0150 (σ+2.50) |

Across three configurations (BA uncapped, Slashdot K=8, Slashdot K=32) the
regime-distinguishing axis (attention) is **never positive**. That is
*why* No-Excess pruning is free — it prunes on an axis that doesn't
carry signal in any tested regime.

## Cross-config conclusion

The 33%-lossless No-Excess cut held in **every** test:

|  | Canonical best | No-Excess best | quaternion ever wins? | global − qbest |
|:--|:--|:--|:--|:--|
| BA (uncapped) | `none·scalar·dm=on` | same (Δ=0) | no | +0.0189, σ+0.66, 3/5 |
| Slashdot K=8 | `none·scalar·dm=on` | same (Δ=0) | no | +0.0070, σ+0.92, 3/5 |
| **Slashdot K=32 + sparse** | `none·scalar·dm=off` | same (Δ=0) | **no (sig.)** | **+0.0123, σ+2.07, 5/5** |

**Recommendation stands and strengthens:** use **No-Excess / Composite**
for HSiKAN-mixed architecture search. The 33% lossless cut is real, holds
across datasets, and holds at the fairest cycle budget this hardware
permits. The canonical/Friedler regime's larger admissible set buys
**zero** quality on every front tested.

## Files / tests / perf

| | |
|:--|:--|
| Code changes this round | `--topk-k` / `--sparse-attn-k` CLI added; `structure_to_env` made the attention cap a threaded param (default 8). |
| Tests | 31 pass (parent harness) + 1 new (`test_arch_env_topk_override`); ruff clean. |
| Total wall (60 runs) | 156.7 min (≈ 157 s/run, +3.5 s vs K=8) |
| Max per-run RSS | 2.78 GiB (≤ 7 GiB budget) |
| GPU OOMs | 0 (h=4, K=32, sparse-K=8, `expandable_segments`) |
| Failed cells | 0 |

## Provenance

- **Git SHA:** `8fd8187` (dirty: parent tasks' new files + this round's harness param change).
- **Interpreter:** miniconda3 python 3.13.5, **torch 2.11.0+cu130** (DRIFT vs CORE 2.12.0, user-approved).
- **GPU:** RTX 2070 SUPER 8 GiB, driver 580.126.09.
- **Seeds:** {0,1,2,3,4}. **Epochs:** 20. **Hidden:** 4. **TOPK_K:** 32. **SPARSE_ATTN_K:** 8.
- **Artifacts:** `/tmp/regime_abc_slash_k32/results.jsonl` (60 rows), `summary.json`, per-cell logs.

## What this does *not* settle

Full-enumeration Slashdot (the regime where attention's 0.903 was
measured) remains GPU-blocked on this 7.6 GiB card — uncapped attention
OOMs. K=32 is the ceiling I could fit. So the cross-config conclusion
above holds **for every cycle budget this hardware permits**; the
larger-VRAM "attention with the full 55 M cycle pool" case is still
open. But the trend across three increasingly-attention-favourable
configurations is *toward* No-Excess winning more decisively, not less.

## Follow-ups

- Larger-VRAM Slashdot run (when available) for the full-enumeration test.
- Vision matrix is already running (chain `bho60n04n` detected the gate
  at 04:36 and started its GPU smoke — separate report tomorrow).
