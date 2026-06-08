# Komondor parallelism analysis — 30 completed cells

## 1. Per-cell wall-time distribution

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

- n = 30
- min  = 26s
- p25  = 28s
- median = 30s
- p75  = 35s
- max  = 10m49s

## 2. Per-(dataset, mode) wall + AUC

| dataset | mode | n | median wall | total wall | AUC mean ± std |
|---|---|---|---|---|---|
| bitcoin_alpha | real | 5 | 26s | 2m13s | 0.9868 ± 0.0058 |
| bitcoin_alpha | shuffle | 5 | 28s | 2m21s | 0.9662 ± 0.0124 |
| bitcoin_otc | real | 5 | 28s | 2m23s | 0.9879 ± 0.0023 |
| bitcoin_otc | shuffle | 5 | 31s | 2m37s | 0.9435 ± 0.0060 |
| slashdot | real | 5 | 35s | 2m50s | 0.9058 ± 0.0037 |
| slashdot | shuffle | 5 | 10m42s | 53m34s | 0.8508 ± 0.0103 |

## 3. Parallelism speedup projection

Sequential total wall (sum of all cells):  **1h05m**
Single-cell longest wall (worst case):     **10m49s**

| K (parallel slots) | projected wall | speedup vs serial |
|---|---|---|
|   1 | 1h05m | 1.00× |
|   5 | 13m11s | 5.00× |
|  10 | 10m49s | 6.10× |
|  20 | 10m49s | 6.10× |
|  40 | 10m49s | 6.10× |

(LPT lower bound: each slot wall = max(max_cell, sum/K). max_cell = 10m49s dominates at high K.)

## 5. Hardware comparison

| platform | GPU | per-cell wall | RSS | notes |
|---|---|---|---|---|
| Komondor (HUN-REN) | A100-SXM4-40GB | 30s (median) | ~2 GB | A100; pinned 24 GB / 48 GB SLURM mem |
| Local | RTX 2070 SUPER 7.6 GB | OOM (edge_cr) | >7.6 GB → kill | full SOTA config exceeds local VRAM |

Local was the first attempted host for the SOTA `c2,c3,c4,c5,w2,w3 + quaternion + edge_cr` config (2026-06-03) and crashed at _catmull_rom_eval forward; Komondor A100 absorbs the same config in ~2 GB RSS and completes 5-seed in 90 min instead of OOM.
