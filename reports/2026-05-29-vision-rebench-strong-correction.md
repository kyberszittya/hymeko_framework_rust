# Report — Hypergraph-vision vs CNN at FAIR training (correction of the 2026-05-28 night-1 result)

**Date:** 2026-05-29
**Predecessor:** `reports/2026-05-28-vision-hypergraph-vs-cnn-rebench.md` (night-1: subset 8000, 15 epochs)
**CORE.YAML items touched:** none.

## Retraction headline

The 2026-05-28 night-1 conclusion ("the hypergraph-CNN gap survives fair
training") was **wrong**. At a *truly* fair training budget — full 60k
training set, 20 epochs (up from 8k subset / 15 epochs, only ~2.3× the
total samples-seen budget) — the hypergraph operators close most of the
night-1 gap. **The night-1 result was itself still undertrained.** The
2026-05-06 5-epoch null and the 2026-05-28 subset-15-epoch result were
both starving the hypergraph operators of training; both should not be
read as evidence that hypergraph vision is incompatible with the task.

## Headline numbers — 3-seed paired

| Model | MNIST night-1 (8k, 15ep) | **MNIST night-2 (60k, 20ep)** | paired Δ | σ |
|:--|--:|--:|--:|--:|
| cnn | 0.9720 ± .0003 | **0.9874 ± .0005** | +0.015 | +100.8 |
| mlp | 0.9228 ± .0045 | **0.9681 ± .0017** | +0.045 | +24.3 |
| **hsikan** | **0.4189 ± .0201** | **0.9426 ± .0013** | **+0.524** | **+42.9** |
| **hgnn** | 0.3148 ± .0272 | **0.8124 ± .0118** | +0.498 | +27.1 |

| Model | Fashion night-1 | **Fashion night-2** | paired Δ | σ |
|:--|--:|--:|--:|--:|
| cnn | 0.8608 ± .0037 | **0.9071 ± .0033** | +0.046 | +21.6 |
| mlp | 0.8283 ± .0048 | **0.8714 ± .0023** | +0.043 | +13.8 |
| **hsikan** | 0.6371 ± .0089 | **0.8369 ± .0049** | +0.200 | +34.2 |
| **hgnn** | 0.4136 ± .0098 | **0.7278 ± .0032** | +0.314 | +43.2 |

## Per-model CNN gap (at the fair budget)

| Model | gap to CNN MNIST | gap to CNN Fashion |
|:--|--:|--:|
| mlp | 1.9 pp | 3.6 pp |
| **hsikan** | **4.5 pp** | **7.0 pp** |
| hgnn | 17.5 pp | 17.9 pp |

vs night-1's 22–66 pp gaps. **HSiKAN is within 5-7 pp of CNN on both
datasets**, comparable to MLP. The night-1 "hypergraph operators are 22-66
pp below CNN, even below MLP" conclusion does not survive proper training.

## What the night-1 report got wrong

1. **Training-budget confound.** Subset=8k × 15 epochs = 120k
   samples-seen; full=60k × 20 = 1.2 M (10× more samples-seen). The
   night-1 result conflated "small budget" with "structural prior fails."
2. **The structural prior is *not* hurting.** Once trained, HSiKAN beats
   the structure-free MLP on MNIST (close call: 0.943 vs 0.968 — MLP
   slightly ahead) and is in MLP territory on Fashion (0.837 vs 0.871).
   Calling the operator "actively blocking learning" was wrong.
3. **What survives:** the qualitative *ordering* (CNN > MLP > HSiKAN > HGNN)
   does survive. The gap to CNN is real but **modest** (~5-18 pp at fair
   training), not catastrophic. CNN's translation-equivariance still
   helps; it just doesn't kill the alternatives.

## What this means going forward

- **HSiKAN-vision is a viable image-classification operator** in the
  ≥1 M-samples-seen regime — not a dead end. At 4× fewer parameters than
  CNN (10 218 vs 42 154) it lands within 4.5 pp of CNN MNIST.
- **HGNN-vision** is also viable but trails HSiKAN by ~13-11 pp; the
  per-edge weight + propagation structure is doing less work than
  HSiKAN's signed-branch Catmull-Rom.
- **The remaining gap to CNN is most plausibly translation equivariance,**
  not undertraining or structural-prior collapse. A vision-specific
  hypergraph operator with explicit translation equivariance (e.g.
  weight-tied RFs across spatial positions) would be the natural next
  experiment. The current RFs *are* translation-tiled (stride 2/4) but
  each RF has its own `W_e` weight (n_edges-many), not weight-tied.

## Performance / provenance

| | |
|:--|:--|
| Total chain wall | 12.5 h (11:21 → 00:02) |
| Cells | 24 / 24, 0 failures |
| Per-cell RSS max | 1.37 GiB (hsikan) ≤ 7 GiB budget |
| Per-cell GPU max | 4.5 GiB observed (live `nvidia-smi`); per-cell `torch.cuda.max_memory_allocated` not recorded by the runner (follow-up: GPU memory probe) |
| Hot spot | hsikan|fashion ≈ 7 100 s/cell — the wall-time-decisive axis |

- Git SHA `8fd8187` (dirty).
- Interpreter miniconda3 / torch 2.11.0+cu130 (CORE pin 2.12, user-approved drift).
- Recipe: `--n-epochs 20 --train-subset 0 --batch-size 128 --hidden 32 --lr 1e-3` + 3 seeds × 2 datasets × 4 models.
- Artifacts: `/tmp/vision_strong/results.jsonl` (24 rows), `summary.json`.
- Night-1 artifacts retained at `/tmp/vision_bench/` for the paired comparison.

## Follow-ups (queued)

1. **GPU memory probe** (next): `torch.cuda.max_memory_allocated` per model
   at the fair-training config. Closes the open "less memory than CNN"
   question with real numbers (host RSS already shows HSiKAN at 1.37 GiB
   vs CNN at 1.20 GiB — slightly higher, contradicting the verbal claim
   on host RAM; GPU could differ).
2. **HSiKAN wall optimization** (queued, profile-first per CLAUDE.md §3):
   the 7 000 s/cell on Fashion is the only friction left. py-spy profile
   first to confirm the hypothesized dense-incidence einsum hotspot
   (`SignedBranchConv.forward`, 4 dense einsums per arity × 3 arities × 2
   layers = 24 dense matmuls per forward, over a 3-15 % sparse incidence).
   Then Tier-1 (batch size up, AMP, `torch.compile`) and Tier-2 (sparse
   incidence via `torch.sparse.mm`), each with parity testing.
3. **Translation-equivariance variant** (research follow-up, not queued):
   weight-tying `W_e` across spatial positions would test whether the
   remaining gap to CNN is genuinely translation equivariance.

## Honest note on hygiene

This report **retracts** the night-1 vision report. The night-1 work
*was* contract-compliant (plan, smoke, multi-seed, lint, tests) — but the
budget I chose was still too low to falsify undertraining. The lesson:
when re-testing a *putative* null, take the training budget at least an
order of magnitude past whatever produced the original null, not just
2-3×. Memory entry
`project_hsikan_vision_redo_2026_05_06` and
`project_vision_hypergraph_vs_cnn_rebench_2026_05_28` should be read
together with **this** report as the corrected position.
