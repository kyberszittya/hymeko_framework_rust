# Report — SOTA reproduction (Slashdot edge_cr kernel-ON) under current code + torch 2.11

**Date:** 2026-05-29
**Predecessor SOTA:** memory `project_kernel_on_5seed_2026_05_09` (May-9, mean AUC 0.9070 ± .0029).
**Recipe source:** `hymeko_neuro/experiments/run_slashdot_edge_cr_kernel_on_2026_05_09.sh` (bit-identical env vars, lines 41-53).
**Repro script:** `hymeko_neuro/experiments/run_slashdot_edge_cr_kernel_on_repro_2026_05_28.sh` (preserves the May-9 jsonl).
**CORE.YAML items touched:** none.

## Headline — **reproduces, with a 40 % wall speedup at identical AUC**

```
May-09 kernel-ON (preserved):  AUC 0.9070 ± .0029   wall  792 s/seed
May-28 kernel-ON (repro):      AUC 0.9062 ± .0033   wall  482 s/seed   <-- TODAY
Verdict: reproduces (within seed noise)
```

Per-seed (May-28): 0.9097, 0.9037, 0.9019, 0.9084, 0.9074.
Paired Δ (May-28 − May-09) = −0.0008 ± 0.0034, σ −0.54 — **squarely within seed noise** (the script's own gate threshold was |σ| < 2).

**The code is clean — no regression introduced since May 9.**

## The unexpected win: 1.64× wall speedup

| | May-09 | May-29 | Δ |
|---|--:|--:|--:|
| Mean wall / seed | 792 s | 482 s | **−39 %** |
| Total 5-seed wall | 3 960 s | 2 410 s | 1 550 s saved |
| AUC mean | 0.9070 | 0.9062 | within noise |
| AUC sd | .0029 | .0033 | comparable |

Best explanation: torch 2.11 + driver 580.x + 4 weeks of code churn (Triton kernel improvements, cycle-cache fixes) collectively trimmed ~40 % of the SOTA recipe's wall — at identical accuracy.

## Test / quality

- 5/5 seeds completed; 0 failures.
- Recipe is bit-identical to May-9 (verified by `diff`-ing the env-var blocks).
- Both jsonls preserved at `hymeko_neuro/experiments/results/`:
  - `slashdot_edge_cr_kernel_on_2026_05_09.jsonl` (untouched historical record)
  - `slashdot_edge_cr_kernel_on_2026_05_28_repro.jsonl` (today)

## Provenance

- **Git SHA:** `8fd8187` (dirty: prior session's regime + vision work).
- **Interpreter:** miniconda3 python 3.13.5, **torch 2.11.0+cu130** (DRIFT vs CORE 2.12.0, user-approved). The May-9 baseline ran under an earlier torch (the AUC reproduces *across the version bump*).
- **GPU:** RTX 2070 SUPER 8 GiB, driver 580.126.09.
- **Seeds:** {0,1,2,3,4}.
- **Recipe (bit-identical to May-9):** `HSIKAN_TRITON_KERNEL=1 HSIKAN_TRITON_BACKWARD=1 HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 HSIKAN_ATTENTION_M_E=quaternion HSIKAN_ATTENTION_HIGHWAY=1 HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr HSIKAN_CYCLE_BATCH=2000 HSIKAN_MAX_K2=200000 HSIKAN_MAX_K3=200000` + `--dataset slashdot --hidden 4 --n-epochs 80 --max-k4 200000`.

## Conclusion

The Slashdot edge_cr kernel-ON SOTA holds under current code + torch 2.11, with a 1.64× wall speedup as a bonus. No regression. The May-9 number remains the published SOTA reference; today's repro is a hygiene confirmation.

This also retroactively validates the May-9 published numbers as reproducible — useful for the paper trail.
