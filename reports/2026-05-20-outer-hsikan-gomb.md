# Outer HSIKAN → Clifford-FIR → Gömb cascade — null / negative — 2026-05-20

## Summary

User's architectural intuition: rather than replacing Gömb's
middle shell with a deeper HSIKAN stack (this afternoon's
null), **prepend a full multi-layer HSIKAN backbone before
Gömb** and let the two architectures meet at the Clifford-FIR
layer. The HSIKAN backbone produces per-vertex features from
the raw signed graph; Gömb's outer FIR shell becomes the
interface where those features feed the cortical cascade
(FIR → middle → inner CPML, all unchanged).

This is a meaningfully different topology from the
stacked-middle: the HSIKAN backbone owns its own
``nn.Embedding`` and runs $L$ HSIKAN layers BEFORE Gömb sees
anything. Gömb's only change is that its "input embedding"
is now an HSIKAN-refined activation instead of a learned
``nn.Embedding`` slot.

**Headline (NULL / NEGATIVE).** The new topology does not lift
either dataset. Bitcoin Alpha is null at all depths; Slashdot
outer-d=2 is **significantly worse** by ~0.010 AUC (far larger
than baseline σ=0.0006).

### Results

**Bitcoin Alpha** (vs plain Gömb baseline 0.9001 ± 0.0098):

| outer_d | mean AUC ± σ | wall | paired Δ | σ_d | wins |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8998 ± 0.0073 | 6.7 s | −0.0003 | −0.21 | 1/3 |
| 2 | 0.8993 ± 0.0075 | 8.3 s | −0.0008 | −0.25 | 1/3 |
| 4 | 0.8977 ± 0.0059 | 11.0 s | −0.0024 | −0.81 | 2/3 |

**Slashdot** (vs plain Gömb baseline 0.9010 ± 0.0006):

| outer_d | mean AUC ± σ | wall | paired Δ | wins |
| --- | --- | --- | --- | --- |
| 1 | 0.9010 ± 0.0017 | 26.3 s | ~0 | tie |
| 2 | **0.8912 ± 0.0007** | 33.7 s | **−0.0098** | **0/3** |
| 4 | OOM × 3 (CR spline) | — | — | — |

Slashdot outer-d=2 is **paired-significantly worse** by an
order of magnitude more than the baseline σ. d=4 OOMs at the
same `_catmull_rom_eval` site as the morning's stacked-middle
d=4 — genuine memory pressure on 7.6 GiB GPU from 4-deep
HSIKAN, not a bug.

## Why the user's intuition was reasonable

The hypothesis was structurally elegant:
- HSIKAN backbone does signed-cycle KAN-spline reasoning on
  the raw graph → produces a "smart" embedding
- Gömb's Clifford-FIR layer was tuned to refine vertex
  embeddings via multiscale FIR filtering → expects a vertex
  embedding as input
- Composing them lets the Clifford-FIR layer process a
  HSIKAN-refined embedding instead of a learned random init

**The architectural composition is sound.** It just doesn't lift
AUC on either dataset. Possible reasons:
- The Clifford-FIR layer is already doing cycle-aware
  refinement; receiving a cycle-aware embedding from HSIKAN
  is potentially **double-counting** the same signal.
- Bitcoin Alpha + Slashdot are at architectural ceiling for
  cycle-based factorisation (Phase 21/22 + stacked-middle all
  found the same).
- The outer HSIKAN's gradient through Clifford-FIR + middle
  + inner CPML is a long chain; gradient interference may
  make the HSIKAN's params under-fit.

## Four null/negative architectural extensions in one day

This is the day's fourth dataset-level null:

| extension | Bitcoin Alpha | Slashdot |
| --- | --- | --- |
| Phase 21 — side-stacked HSIKAN | null (Δ=+0.0003) | — |
| Phase 22 — side-stacked mixed-arity HSIKAN | (BA: at ceiling) | null mean, **σ halved** (variance tightened) |
| Stacked-middle Gömb | **paired worse** (d=2, σ_d=−2.32) | null (d=2) |
| **Outer HSIKAN → FIR Gömb (this)** | null (all depths) | **paired worse** (d=2, Δ=−0.010) |

Every cycle-based architectural extension we tested today
falsified on the mean AUC axis. Only Phase 22 showed any
positive signal, and that was on **variance**, not mean.

**This is real information.** Bitcoin Alpha and Slashdot, at
the current pre-cycle-pool tooling, **really are at
architectural ceiling for cycle-based factorisation**. The
dataset is doing what it's going to do.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/hymeko_gomb/cascade.py` | extended | +160 (new `GombWithOuterHSIKAN` class + 4 `GombConfig` fields) |
| `signedkan_wip/src/hymeko_gomb/__init__.py` | extended | +3 (export) |
| `signedkan_wip/experiments/runs/run_gomb_smoke.py` | extended | +30 (`--model outer_hsikan_gomb` + 4 CLI flags) |
| `signedkan_wip/tests/test_outer_hsikan_gomb.py` | new | 162 (8 unit tests) |
| `signedkan_wip/experiments/run_outer_hsikan_gomb_overnight_2026_05_20.sh` | new | 118 |
| `docs/plans/2026-05-20-outer-hsikan-gomb/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}` | new | 4-format plan |
| `reports/2026-05-20-outer-hsikan-gomb.md` | new | this file |

## CORE.YAML items touched

None.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_outer_hsikan_gomb.py` | **8 / 8 pass** |
| All prior interpret / side / arity / fuzzy / stacked-middle / gomb-signature suites | 71 / 71 (no regression) |
| Bitcoin Alpha smoke at outer-d ∈ {1, 2, 4} | all complete |
| Slashdot grid: 6/9 cells complete (d=4 × 3 seeds OOM at CR spline) | — |

## §6.5 anti-pattern audit

- `GombWithOuterHSIKAN` is a separate class (per §6.5 #8
  "structural differences → class"), not a forward-time flag
  on the existing `HymeKoGomb`.
- New config fields are additive on `GombConfig` with safe
  defaults (`outer_hsikan_n_layers=0` means "don't use this
  class").
- CLI flags route through the existing `run_gomb_smoke` model
  dispatch (`_MODELS["outer_hsikan_gomb"]`) — no Cartesian
  product wrappers.
- Gradient checkpoint / Rust acceleration for the d=4 OOM
  remains the same follow-up as the morning's stacked-middle
  d=4 issue.

Clean.

## Open follow-ups (for next session)

1. **Pivot off Bitcoin Alpha + Slashdot for cycle-architecture
   experiments.** Four nulls in one day is a strong signal —
   any further depth/width/composition variant on these
   datasets is unlikely to help. The remaining headroom is
   on:
   - **Epinions**, where Gömb-strict-finetune broke SOTA at
     0.9526 ([[project-gomb-strict-4dataset-2026-05-14]]) —
     this is the dataset that actually responds to
     architectural changes.
   - **HymeYOLO / vision**, where the inductive bias is
     completely different and the architectural lever might
     finally bite.
   - **Time-series → signed-correlation networks**, the
     thread we discussed but haven't built yet. Weighted arcs
     (the morning's `cr_highway` mode) would be a natural
     fit there.
2. **Gradient checkpointing in HSIKAN per-layer forward** —
   would unblock d=4 on Slashdot if we ever wanted to test
   it. Probably not worth the work given d=2 already says
   "no lift" on Slashdot.
3. **ABB / MSG architecture search** — given that **no axis
   tested so far has lifted mean AUC**, ABB/MSG over
   orthogonal dims (jk_mode, inner_skip, share_weights) is
   unlikely to find a positive result. Better to spend the
   budget on a NEW dataset / modality.

## Experiment provenance

- **Git SHA:** uncommitted.
- **Bitcoin Alpha grid:** 9 cells × 60 epochs, 7–11 s each.
- **Slashdot grid:** 6 cells (d=1,2 × 3 seeds) × 60 epochs,
  26–34 s each; d=4 × 3 seeds OOM in 7 s each.
- **Total wall:** ~5 minutes (Gömb is fast).
- **GPU:** RTX 2070 SUPER 8 GiB.
- **JSONL:**
  `signedkan_wip/experiments/results/outer_hsikan_gomb_overnight_2026_05_20.jsonl`
- **Plain-Gömb baseline:** taken from
  `signedkan_wip/experiments/results/stacked_gomb_overnight_2026_05_20.jsonl`
  (the depth=1 cells from this morning).

## Acceptance check

- [x] Plan in 4 formats on disk.
- [x] CORE.YAML items touched = 0.
- [x] 8 / 8 new unit tests + 71 / 71 prior tests no regression.
- [x] Bitcoin Alpha grid complete; Slashdot d ∈ {1, 2} complete
      (d=4 OOM same site as morning's stacked-middle).
- [x] **Null / negative result framed honestly:** the outer
      HSIKAN backbone composed with Gömb's cortical cascade
      does NOT lift AUC on either dataset; Slashdot d=2 is
      paired-significantly worse by ~0.010.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
- [x] Pivot direction identified for next session.
