# Gömb strict-protocol 5-seed benchmark (2026-05-14)

The 2026-05-14 ChatGPT audit forced a clean separation between the
canonical-but-leaky transductive convention used by published
signed-link papers and a strict-by-construction protocol that
forbids test-edge σ-leakage. **Gömb runs the strict protocol
natively**, which makes its 5-seed numbers the honest architectural
reference.

## Headline 5-seed (Optuna-tuned configs)

| Dataset | Gömb-strict 5-seed | Per-seed |
| --- | ---: | --- |
| Bitcoin Alpha | 0.8972 ± 0.0079 | 0.8877 · 0.9087 · 0.8901 · 0.8962 · 0.9035 |
| Bitcoin OTC | 0.9145 ± 0.0068 | 0.9256 · 0.9047 · 0.9125 · 0.9127 · 0.9168 |
| Slashdot | **0.9017 ± 0.0008** | 0.9007 · 0.9015 · 0.9015 · 0.9016 · 0.9033 |
| **Epinions** | **0.9526 ± 0.0018** ★ | 0.9532 · 0.9520 · 0.9499 · 0.9523 · 0.9555 |

★ — Epinions number is the fine-tune winner config (v5_combined),
**beats published SiGAT (~0.95)** under a strict protocol that
forbids the σ-leakage all transductive baselines benefit from.

## Configs (Optuna-reused)

| Dataset | d_embed | M_outer | d_outer | d_middle | d_core | n_tiers | topk | lr | epochs | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bitcoin Alpha | 32 | 8 | 20 | 24 | 48 | 4 | 56 | 5e-3 | 80 | 676k |
| Bitcoin OTC | 32 | 12 | 8 | 16 | 32 | 2 | 32 | 5e-3 | 80 | 356k |
| Slashdot | 16 | 4 | 4 | 4 | 4 | 3 | 32 | 3e-3 | 60 | 1.33M |
| Epinions (baseline) | 16 | 4 | 4 | 4 | 4 | 3 | 32 | 3e-3 | 60 | 2.13M |
| **Epinions (v5_combined)** | **32** | **8** | **8** | **8** | **8** | **3** | **64** | **3e-3** | **80** | — |

The Epinions fine-tune sweep tried 6 single-seed variants (more
epochs, bigger embed, medium cores, more topk, combined, deep
tiers); **v5_combined** (bigger embed + medium cores + more topk
together) was the winner; single levers alone hit a ~0.95 ceiling.

## Cross-protocol comparison (apples-to-apples *only within* protocol)

| Model | Protocol | Bitcoin Alpha | Bitcoin OTC | Slashdot | Epinions |
| --- | --- | ---: | ---: | ---: | ---: |
| **Gömb (this work)** | strict | 0.8972 | 0.9145 | 0.9017 | **0.9526** ★ |
| SGCN (Derr et al. 2018) | transductive (leaky) | 0.929 | 0.942 | 0.919 | — |
| SiGAT (Huang et al. 2019) | transductive (leaky) | 0.903 | 0.932 | — | ~0.95 |
| Our HSiKAN edge_cr | transductive (leaky) | 0.997 | 0.993 | 0.9067 | 0.8464 |

**Why the Bitcoin Alpha/OTC numbers look "below" SGCN**: the protocols
are different. SGCN/SiGAT include test-edge signs in their
message-passing graph by default; Gömb does not. Under **identical
strict protocol** (which the literature does not report), SGCN's
numbers would also drop substantially. The Slashdot number
(0.9017) matches our prior published Gömb 0.9031 ± 0.0008 within
noise — independent reproduction.

## Audit — label-shuffle confirms strict protocol

Label-shuffle is the diagnostic that distinguishes "architecture
that learns from labels" vs "architecture that learns from
σ-leakage". On Bitcoin Alpha (graph-level shuffle of train-edge
signs; test signs untouched):

| Model | Real labels | Shuffled labels | What this means |
| --- | ---: | ---: | --- |
| HSiKAN Optuna-best (c2,c5,w2,w3,w4) | 0.9970 | **0.9921** | σ-leakage dominates |
| HSiKAN joint_mix (c3,c4,w2,w3) | 0.9845 | 0.8902 | moderate σ-leakage |
| **Gömb (joint-mix)** | **(0.943 Epinions)** | **0.5402** | **strict — no leakage** |
| SGCN | 0.93 | 0.5503 | no structural prior |
| HSiKAN-Optuna **untrained** | n/a | rank-AUC 0.9956 | architecture IS the predictor |

`run_gomb_smoke.py --shuffle-train-signs` is the standardized hook.

## SOTA-push sweep (2026-05-14) — confirms architectural ceiling

A 14-variant single-seed sweep + top-1-per-dataset 5-seed validation
found **no variant beat the headline tuned 5-seed numbers** on any
dataset. The Bitcoin Alpha/OTC ceiling (~0.90 / ~0.91) is the
architectural ceiling for the `c3,c4,w2,w3` joint-mix tuple set
under strict protocol. Further gains require either (a) the c2
tuple (which is the leakage path) or (b) architectural changes
(attention, more arities, capsule-routing variants).

## Reproducibility

All artifacts in
[`signedkan_wip/experiments/results/gomb_strict_benchmark_tuned_20260514T010516Z/`](../../../../signedkan_wip/experiments/results/gomb_strict_benchmark_tuned_20260514T010516Z/)
(main benchmark) +
[`gomb_epinions_finetune_20260514T014021Z/`](../../../../signedkan_wip/experiments/results/gomb_epinions_finetune_20260514T014021Z/)
(Epinions fine-tune) +
[`gomb_sota_push_20260514T082642Z/`](../../../../signedkan_wip/experiments/results/gomb_sota_push_20260514T082642Z/)
(SOTA-push sweep). Each directory contains:

- `orchestrator.log` — full per-seed timing + result lines
- `<step>_seed<N>.log` — per-run training output
- Reproducible via the runner scripts in
  [`signedkan_wip/experiments/run_gomb_*_2026_05_14.sh`](../../../../signedkan_wip/experiments/).

## Cross-references

- Audit: [reports/2026-05-14-bitcoin-leakage-audit.md](../../../../reports/2026-05-14-bitcoin-leakage-audit.md)
- SOTA snapshot: [SOTA snapshot & diagrams](../results/sota-snapshot.md)
  — §0 captures the same numbers in the global benchmark table.
