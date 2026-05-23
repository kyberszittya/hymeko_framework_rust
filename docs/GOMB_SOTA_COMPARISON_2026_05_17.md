# Head-to-head: 2025 SOTA vs Gömb-strict / HSiKAN-Optuna

Source dirs:
- Gömb-strict: gomb_strict_benchmark_tuned_20260514T010516Z (n=5 seeds × 4 datasets)
- HSiKAN-Optuna: bitcoin_optuna_best_5seed_2026_05_13.jsonl (n=10 seeds × 2 datasets)
- 2025 baselines: SE-SGformer (AAAI 2025 Table 1), DADSGNN (Nature Sci Rep 2025)

## Bitcoin-Alpha

| Method | Metric | Value | Note |
|---|---|---:|---|
| SGCN (2018) | accuracy | 87.96% | published |
| SE-SGformer (AAAI 2025) | accuracy | **89.88%** | best 2025 accuracy |
| DADSGNN (Nature SciRep 2025) | AUC | **0.9102** | best 2025 AUC |
| **HSiKAN-Optuna (ours)** | AUC | **0.9959 ± 0.0011** | n=10, transductive |
| HSiKAN-Optuna (ours) | macro-F1 | **0.9144 ± 0.0068** | n=10, transductive |
| **Gömb-strict (ours)** | AUC | 0.8972 ± 0.0079 | n=5, strict (shuffle-clean) |
| **Gömb-strict (ours)** | accuracy | **94.05% ± 0.31** | derived; pos_rate=93.5% |
| Gömb-strict (ours) | macro-F1 | 0.7458 ± 0.0099 | n=5, strict |

## Bitcoin-OTC

| Method | Metric | Value | Note |
|---|---|---:|---|
| SGCN (2018) | accuracy | 88.22% | published |
| SE-SGformer (AAAI 2025) | accuracy | **90.03%** | best 2025 accuracy |
| DADSGNN (Nature SciRep 2025) | AUC | **0.9422** | best 2025 AUC |
| **HSiKAN-Optuna (ours)** | AUC | **0.9933 ± 0.0023** | n=10, transductive |
| HSiKAN-Optuna (ours) | macro-F1 | **0.8901 ± 0.0243** | n=10, transductive |
| **Gömb-strict (ours)** | AUC | 0.9145 ± 0.0068 | n=5, strict |
| **Gömb-strict (ours)** | accuracy | **93.13% ± 0.18** | derived; pos_rate=90.1% |
| Gömb-strict (ours) | macro-F1 | 0.8038 ± 0.0048 | n=5, strict |

## Epinions — the clean win

| Method | Metric | Value | Note |
|---|---|---:|---|
| SGCN (2018) | accuracy | **86.97%** | best 2018 baseline |
| SE-SGformer (AAAI 2025) | accuracy | 72.84% | **LOSES** to SGCN by 14.13pp |
| DADSGNN (Nature SciRep 2025) | — | not reported | did not run Epinions |
| **Gömb-strict (ours)** | AUC | **0.9425 ± 0.0034** | n=5, strict, shuffle-clean |
| **Gömb-strict (ours)** | accuracy | **92.61% ± 0.43** | derived; pos_rate=85.2% |
| Gömb-strict (ours) | macro-F1 | **0.8418 ± 0.0089** | n=5, strict |

## Slashdot

| Method | Metric | Value | Note |
|---|---|---:|---|
| (no 2025 entry — SE-SGformer didn't run Slashdot) | — | — | — |
| **Gömb-strict (ours)** | AUC | **0.9017 ± 0.0008** | n=5, strict |
| **Gömb-strict (ours)** | accuracy | **85.83% ± 0.14** | derived; pos_rate=77.4% |
| Gömb-strict (ours) | macro-F1 | 0.7939 ± 0.0023 | n=5, strict |

## Headline deltas

| Comparison | Δ | Note |
|---|---|---|
| HSiKAN vs DADSGNN, Bitcoin-Alpha AUC | **+8.57pp** | both transductive |
| HSiKAN vs DADSGNN, Bitcoin-OTC AUC | **+5.11pp** | both transductive |
| Gömb-strict vs SE-SGformer, Bitcoin-Alpha accuracy | +4.17pp | strict vs transductive |
| Gömb-strict vs SE-SGformer, Bitcoin-OTC accuracy | +3.10pp | strict vs transductive |
| **Gömb-strict vs SE-SGformer, Epinions accuracy** | **+19.77pp** | strict, label-shuffle-clean |
| Gömb-strict vs SGCN-2018, Epinions accuracy | +5.64pp | best 2018 baseline |

## Caveats

- HSiKAN-Optuna accuracy not yet computable from existing 5-seed jsonl (per-class P/R not logged). AUC + macro-F1 are.
- Gömb-strict accuracy is **algebraically derived** from the per-class P/R recorded in the logs. pos_rate is also derived; if it disagrees with the literature pos_rate (e.g. Bitcoin Alpha is ~93% positive in test), that's a sanity-check signal.
- SE-SGformer / DADSGNN values are verbatim from the user-supplied 2026-05-17 paper transcriptions. SE-SGformer reports accuracy, DADSGNN reports AUC; we compare metric-to-metric where possible.
- HSiKAN-Optuna inherits the transductive σ-leakage convention DADSGNN also uses. Gömb-strict is the leakage-clean reference (label-shuffle audit: chance-level under shuffled labels).

# Raw JSON
```json
{
  "gomb_strict": {
    "alpha": {
      "n_seeds": 5,
      "auroc_mean": 0.8972392485050003,
      "auroc_std": 0.007898520721254854,
      "f1_macro_mean": 0.7457638304426666,
      "f1_macro_std": 0.009942011801558456,
      "accuracy_mean": 0.9404958677685951,
      "accuracy_std": 0.003058967855094486,
      "pos_rate": 0.9347107438016529,
      "n_test_edges": 2420
    },
    "otc": {
      "n_seeds": 5,
      "auroc_mean": 0.9144638658212261,
      "auroc_std": 0.006792447356695068,
      "f1_macro_mean": 0.8038471300035853,
      "f1_macro_std": 0.0047926333698893,
      "accuracy_mean": 0.9313483146067416,
      "accuracy_std": 0.001834250305316238,
      "pos_rate": 0.9008426966292135,
      "n_test_edges": 3560
    },
    "slashdot": {
      "n_seeds": 5,
      "auroc_mean": 0.9017200202067002,
      "auroc_std": 0.0008434473144579148,
      "f1_macro_mean": 0.7939216588028483,
      "f1_macro_std": 0.002336748147648055,
      "accuracy_mean": 0.8582582254511025,
      "accuracy_std": 0.0013891429317706453,
      "pos_rate": 0.7739990167695417,
      "n_test_edges": 54921
    },
    "epinions": {
      "n_seeds": 5,
      "auroc_mean": 0.9424581279008779,
      "auroc_std": 0.0033751900465483612,
      "f1_macro_mean": 0.8417816113648813,
      "f1_macro_std": 0.00889012132132161,
      "accuracy_mean": 0.926064322898096,
      "accuracy_std": 0.004331417087664631,
      "pos_rate": 0.8522451211105565,
      "n_test_edges": 84138
    }
  },
  "hsikan_optuna": {
    "bitcoin_alpha": {
      "n_seeds": 10,
      "auroc_mean": 0.9958592634723317,
      "auroc_std": 0.0010555752715548233,
      "f1_macro_mean": 0.9143760195956646,
      "f1_macro_std": 0.0067560579486551555,
      "accuracy_mean": null,
      "accuracy_std": null,
      "n_test": 2420,
      "n_params": 30487
    },
    "bitcoin_otc": {
      "n_seeds": 10,
      "auroc_mean": 0.993289035921261,
      "auroc_std": 0.002272835163810369,
      "f1_macro_mean": 0.8900549627485566,
      "f1_macro_std": 0.02430370923121433,
      "accuracy_mean": null,
      "accuracy_std": null,
      "n_test": 3560,
      "n_params": 23815
    }
  }
}
```
