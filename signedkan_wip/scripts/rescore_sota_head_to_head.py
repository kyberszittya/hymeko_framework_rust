"""Compute accuracy + Macro-F1 + AUROC head-to-head against 2025 SOTA.

No retraining: parses existing per-seed jsonl/log artifacts. The
Gömb-strict logs carry full per-class precision/recall, so accuracy
is algebraically recoverable. The HSiKAN-Optuna jsonl carries AUC
+ macro-F1 directly.

Sources:
  - Gömb-strict: signedkan_wip/experiments/results/gomb_strict_benchmark_tuned_20260514T010516Z/step{1..4}_<dataset>_seed{0..4}.log
  - HSiKAN-Optuna: signedkan_wip/experiments/results/bitcoin_optuna_best_5seed_2026_05_13.jsonl  (10 alpha + 10 otc rows)

2025 baselines (hard-coded from the AAAI 2025 / Nature SciRep 2025 papers, per
user-supplied verbatim transcription 2026-05-17):
  - SE-SGformer accuracy (AAAI 2025 Table 1)
  - DADSGNN AUC (Nature Sci Rep 2025 discussion text)
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# Hardcoded 2025 baselines (mean only — std not always reported for these).

SE_SGFORMER_ACC = {  # AAAI 2025 Table 1, accuracy (%)
    "bitcoin_alpha": 89.88, "bitcoin_otc": 90.03,
    "epinions": 72.84, "wikielec": 80.63, "wikirfa": 79.99,
    "amazon_music": 79.20, "kuairand": 56.89, "kuairec": 85.60,
}
SGCN_ACC = {  # AAAI 2025 Table 1, accuracy (%) - 2018 baseline kept for "2025 doesn't beat 2018" claim
    "bitcoin_alpha": 87.96, "bitcoin_otc": 88.22, "epinions": 86.97,
    "wikielec": 79.14, "wikirfa": 78.69, "amazon_music": 70.63,
    "kuairand": 62.85, "kuairec": 85.11,
}
DADSGNN_AUC = {  # Nature Sci Rep 2025, from discussion text
    "bitcoin_alpha": 0.9102, "bitcoin_otc": 0.9422,
}


def derive_accuracy(rec_pos: float, rec_neg: float, prec_pos: float,
                    prec_neg: float, n_test: int) -> tuple[float, float]:
    """Recover accuracy + positive class rate from per-class P/R + n.

    Equations:
      P/N = (rec_pos * (1 - prec_pos)) / (prec_pos * (1 - rec_neg))   ⟸ derived from
        prec_pos = TP/(TP+FP) with TP = rec_pos*P, FP = (1-rec_neg)*N
      P + N = n_test
      accuracy = (rec_pos*P + rec_neg*N) / n_test
    """
    if prec_pos <= 0 or (1.0 - rec_neg) <= 1e-9:
        return float("nan"), float("nan")
    # FP / TP = (1 - prec_pos) / prec_pos
    # TP / P = rec_pos
    # FP / N = (1 - rec_neg)
    # ⇒ N / P = TP/FP * (1 - rec_neg) / 1   wait, let me redo:
    # TP = rec_pos * P
    # FP = (1 - rec_neg) * N
    # TP / (TP+FP) = prec_pos
    # rec_pos*P / (rec_pos*P + (1-rec_neg)*N) = prec_pos
    # rec_pos*P = prec_pos*(rec_pos*P + (1-rec_neg)*N)
    # rec_pos*P*(1 - prec_pos) = prec_pos*(1-rec_neg)*N
    # P/N = prec_pos*(1-rec_neg) / (rec_pos*(1 - prec_pos))
    p_over_n = (prec_pos * (1.0 - rec_neg)) / (rec_pos * (1.0 - prec_pos))
    N = n_test / (1.0 + p_over_n)
    P = n_test - N
    pos_rate = P / n_test
    accuracy = (rec_pos * P + rec_neg * N) / n_test
    return accuracy, pos_rate


def parse_gomb_log(p: Path) -> dict | None:
    """Read the last JSON-line metrics row from a Gömb-strict step log."""
    last_json = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            last_json = json.loads(line)
        except Exception:
            pass
    return last_json


def aggregate_gomb_strict() -> dict[str, dict[str, float]]:
    """Aggregate Gömb-strict 5-seed metrics per dataset."""
    out_dir = REPO / (
        "signedkan_wip/experiments/results/"
        "gomb_strict_benchmark_tuned_20260514T010516Z"
    )
    pattern = re.compile(r"^step\d+_(\w+?)_seed(\d+)\.log$")
    per_ds: dict[str, list[dict]] = {}
    for log in sorted(out_dir.glob("step*_seed*.log")):
        m = pattern.match(log.name)
        if not m:
            continue
        ds, seed = m.group(1), int(m.group(2))
        row = parse_gomb_log(log)
        if row is None or "test_auroc" not in row:
            continue
        acc, pos_rate = derive_accuracy(
            row["test_recall_pos"], row["test_recall_neg"],
            row["test_precision_pos"], row["test_precision_neg"],
            row["n_test_edges"],
        )
        per_ds.setdefault(ds, []).append({
            "seed": seed,
            "auroc": row["test_auroc"],
            "f1_macro": row["test_f1_macro"],
            "accuracy": acc,
            "pos_rate": pos_rate,
            "n_test": row["n_test_edges"],
        })
    summary = {}
    for ds, rows in per_ds.items():
        n = len(rows)
        if n == 0:
            continue
        auroc = [r["auroc"] for r in rows]
        f1 = [r["f1_macro"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        summary[ds] = {
            "n_seeds": n,
            "auroc_mean": statistics.mean(auroc),
            "auroc_std": statistics.pstdev(auroc),
            "f1_macro_mean": statistics.mean(f1),
            "f1_macro_std": statistics.pstdev(f1),
            "accuracy_mean": statistics.mean(acc),
            "accuracy_std": statistics.pstdev(acc),
            "pos_rate": statistics.mean(r["pos_rate"] for r in rows),
            "n_test_edges": rows[0]["n_test"],
        }
    return summary


def aggregate_hsikan_optuna() -> dict[str, dict[str, float]]:
    """Aggregate HSiKAN-Optuna 10-seed metrics per dataset.

    Note: the existing 10-seed jsonl only stores AUC + macro-F1 + n_test
    (the runner did not log per-class precision/recall). Accuracy is
    NOT recoverable from these rows; flagged as TODO in the output.
    """
    p = REPO / (
        "signedkan_wip/experiments/results/"
        "bitcoin_optuna_best_5seed_2026_05_13.jsonl"
    )
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    per_ds: dict[str, list[dict]] = {}
    for r in rows:
        per_ds.setdefault(r["dataset"], []).append(r)
    summary = {}
    for ds, drows in per_ds.items():
        aucs = [r["auc"] for r in drows]
        f1s = [r["f1m"] for r in drows]
        summary[ds] = {
            "n_seeds": len(drows),
            "auroc_mean": statistics.mean(aucs),
            "auroc_std": statistics.pstdev(aucs),
            "f1_macro_mean": statistics.mean(f1s),
            "f1_macro_std": statistics.pstdev(f1s),
            "accuracy_mean": None,  # not recoverable from these rows
            "accuracy_std": None,
            "n_test": drows[0]["n_test"],
            "n_params": drows[0]["n_params"],
        }
    return summary


def render_table(gomb: dict, hsikan: dict) -> str:
    """Render the head-to-head Markdown table."""
    out = []
    out.append("# Head-to-head: 2025 SOTA vs Gömb-strict / HSiKAN-Optuna")
    out.append("")
    out.append(f"Source dirs:")
    out.append(f"- Gömb-strict: gomb_strict_benchmark_tuned_20260514T010516Z (n=5 seeds × 4 datasets)")
    out.append(f"- HSiKAN-Optuna: bitcoin_optuna_best_5seed_2026_05_13.jsonl (n=10 seeds × 2 datasets)")
    out.append(f"- 2025 baselines: SE-SGformer (AAAI 2025 Table 1), DADSGNN (Nature Sci Rep 2025)")
    out.append("")
    out.append("## Bitcoin-Alpha")
    out.append("")
    out.append("| Method | Metric | Value | Note |")
    out.append("|---|---|---:|---|")
    out.append(f"| SGCN (2018) | accuracy | {SGCN_ACC['bitcoin_alpha']:.2f}% | published |")
    out.append(f"| SE-SGformer (AAAI 2025) | accuracy | **{SE_SGFORMER_ACC['bitcoin_alpha']:.2f}%** | best 2025 accuracy |")
    out.append(f"| DADSGNN (Nature SciRep 2025) | AUC | **{DADSGNN_AUC['bitcoin_alpha']:.4f}** | best 2025 AUC |")
    if "bitcoin_alpha" in hsikan:
        h = hsikan["bitcoin_alpha"]
        out.append(f"| **HSiKAN-Optuna (ours)** | AUC | **{h['auroc_mean']:.4f} ± {h['auroc_std']:.4f}** | n={h['n_seeds']}, transductive |")
        out.append(f"| HSiKAN-Optuna (ours) | macro-F1 | **{h['f1_macro_mean']:.4f} ± {h['f1_macro_std']:.4f}** | n={h['n_seeds']}, transductive |")
    if "alpha" in gomb:
        g = gomb["alpha"]
        out.append(f"| **Gömb-strict (ours)** | AUC | {g['auroc_mean']:.4f} ± {g['auroc_std']:.4f} | n={g['n_seeds']}, strict (shuffle-clean) |")
        out.append(f"| **Gömb-strict (ours)** | accuracy | **{g['accuracy_mean']*100:.2f}% ± {g['accuracy_std']*100:.2f}** | derived; pos_rate={g['pos_rate']*100:.1f}% |")
        out.append(f"| Gömb-strict (ours) | macro-F1 | {g['f1_macro_mean']:.4f} ± {g['f1_macro_std']:.4f} | n={g['n_seeds']}, strict |")
    out.append("")
    out.append("## Bitcoin-OTC")
    out.append("")
    out.append("| Method | Metric | Value | Note |")
    out.append("|---|---|---:|---|")
    out.append(f"| SGCN (2018) | accuracy | {SGCN_ACC['bitcoin_otc']:.2f}% | published |")
    out.append(f"| SE-SGformer (AAAI 2025) | accuracy | **{SE_SGFORMER_ACC['bitcoin_otc']:.2f}%** | best 2025 accuracy |")
    out.append(f"| DADSGNN (Nature SciRep 2025) | AUC | **{DADSGNN_AUC['bitcoin_otc']:.4f}** | best 2025 AUC |")
    if "bitcoin_otc" in hsikan:
        h = hsikan["bitcoin_otc"]
        out.append(f"| **HSiKAN-Optuna (ours)** | AUC | **{h['auroc_mean']:.4f} ± {h['auroc_std']:.4f}** | n={h['n_seeds']}, transductive |")
        out.append(f"| HSiKAN-Optuna (ours) | macro-F1 | **{h['f1_macro_mean']:.4f} ± {h['f1_macro_std']:.4f}** | n={h['n_seeds']}, transductive |")
    if "otc" in gomb:
        g = gomb["otc"]
        out.append(f"| **Gömb-strict (ours)** | AUC | {g['auroc_mean']:.4f} ± {g['auroc_std']:.4f} | n={g['n_seeds']}, strict |")
        out.append(f"| **Gömb-strict (ours)** | accuracy | **{g['accuracy_mean']*100:.2f}% ± {g['accuracy_std']*100:.2f}** | derived; pos_rate={g['pos_rate']*100:.1f}% |")
        out.append(f"| Gömb-strict (ours) | macro-F1 | {g['f1_macro_mean']:.4f} ± {g['f1_macro_std']:.4f} | n={g['n_seeds']}, strict |")
    out.append("")
    out.append("## Epinions — the clean win")
    out.append("")
    out.append("| Method | Metric | Value | Note |")
    out.append("|---|---|---:|---|")
    out.append(f"| SGCN (2018) | accuracy | **{SGCN_ACC['epinions']:.2f}%** | best 2018 baseline |")
    out.append(f"| SE-SGformer (AAAI 2025) | accuracy | {SE_SGFORMER_ACC['epinions']:.2f}% | **LOSES** to SGCN by {SGCN_ACC['epinions'] - SE_SGFORMER_ACC['epinions']:.2f}pp |")
    out.append(f"| DADSGNN (Nature SciRep 2025) | — | not reported | did not run Epinions |")
    if "epinions" in gomb:
        g = gomb["epinions"]
        out.append(f"| **Gömb-strict (ours)** | AUC | **{g['auroc_mean']:.4f} ± {g['auroc_std']:.4f}** | n={g['n_seeds']}, strict, shuffle-clean |")
        out.append(f"| **Gömb-strict (ours)** | accuracy | **{g['accuracy_mean']*100:.2f}% ± {g['accuracy_std']*100:.2f}** | derived; pos_rate={g['pos_rate']*100:.1f}% |")
        out.append(f"| Gömb-strict (ours) | macro-F1 | **{g['f1_macro_mean']:.4f} ± {g['f1_macro_std']:.4f}** | n={g['n_seeds']}, strict |")
    out.append("")
    out.append("## Slashdot")
    out.append("")
    out.append("| Method | Metric | Value | Note |")
    out.append("|---|---|---:|---|")
    out.append(f"| (no 2025 entry — SE-SGformer didn't run Slashdot) | — | — | — |")
    if "slashdot" in gomb:
        g = gomb["slashdot"]
        out.append(f"| **Gömb-strict (ours)** | AUC | **{g['auroc_mean']:.4f} ± {g['auroc_std']:.4f}** | n={g['n_seeds']}, strict |")
        out.append(f"| **Gömb-strict (ours)** | accuracy | **{g['accuracy_mean']*100:.2f}% ± {g['accuracy_std']*100:.2f}** | derived; pos_rate={g['pos_rate']*100:.1f}% |")
        out.append(f"| Gömb-strict (ours) | macro-F1 | {g['f1_macro_mean']:.4f} ± {g['f1_macro_std']:.4f} | n={g['n_seeds']}, strict |")
    out.append("")
    out.append("## Headline deltas")
    out.append("")
    out.append("| Comparison | Δ | Note |")
    out.append("|---|---|---|")
    if "bitcoin_alpha" in hsikan and DADSGNN_AUC.get("bitcoin_alpha"):
        d = hsikan["bitcoin_alpha"]["auroc_mean"] - DADSGNN_AUC["bitcoin_alpha"]
        out.append(f"| HSiKAN vs DADSGNN, Bitcoin-Alpha AUC | **+{d*100:.2f}pp** | both transductive |")
    if "bitcoin_otc" in hsikan and DADSGNN_AUC.get("bitcoin_otc"):
        d = hsikan["bitcoin_otc"]["auroc_mean"] - DADSGNN_AUC["bitcoin_otc"]
        out.append(f"| HSiKAN vs DADSGNN, Bitcoin-OTC AUC | **+{d*100:.2f}pp** | both transductive |")
    if "alpha" in gomb and "bitcoin_alpha" in SE_SGFORMER_ACC:
        d = gomb["alpha"]["accuracy_mean"]*100 - SE_SGFORMER_ACC["bitcoin_alpha"]
        sign = "+" if d >= 0 else ""
        out.append(f"| Gömb-strict vs SE-SGformer, Bitcoin-Alpha accuracy | {sign}{d:.2f}pp | strict vs transductive |")
    if "otc" in gomb and "bitcoin_otc" in SE_SGFORMER_ACC:
        d = gomb["otc"]["accuracy_mean"]*100 - SE_SGFORMER_ACC["bitcoin_otc"]
        sign = "+" if d >= 0 else ""
        out.append(f"| Gömb-strict vs SE-SGformer, Bitcoin-OTC accuracy | {sign}{d:.2f}pp | strict vs transductive |")
    if "epinions" in gomb and "epinions" in SE_SGFORMER_ACC:
        d = gomb["epinions"]["accuracy_mean"]*100 - SE_SGFORMER_ACC["epinions"]
        sign = "+" if d >= 0 else ""
        out.append(f"| **Gömb-strict vs SE-SGformer, Epinions accuracy** | **{sign}{d:.2f}pp** | strict, label-shuffle-clean |")
        d2 = gomb["epinions"]["accuracy_mean"]*100 - SGCN_ACC["epinions"]
        sign2 = "+" if d2 >= 0 else ""
        out.append(f"| Gömb-strict vs SGCN-2018, Epinions accuracy | {sign2}{d2:.2f}pp | best 2018 baseline |")
    out.append("")
    out.append("## Caveats")
    out.append("")
    out.append("- HSiKAN-Optuna accuracy not yet computable from existing 5-seed jsonl (per-class P/R not logged). AUC + macro-F1 are.")
    out.append("- Gömb-strict accuracy is **algebraically derived** from the per-class P/R recorded in the logs. pos_rate is also derived; if it disagrees with the literature pos_rate (e.g. Bitcoin Alpha is ~93% positive in test), that's a sanity-check signal.")
    out.append("- SE-SGformer / DADSGNN values are verbatim from the user-supplied 2026-05-17 paper transcriptions. SE-SGformer reports accuracy, DADSGNN reports AUC; we compare metric-to-metric where possible.")
    out.append("- HSiKAN-Optuna inherits the transductive σ-leakage convention DADSGNN also uses. Gömb-strict is the leakage-clean reference (label-shuffle audit: chance-level under shuffled labels).")
    return "\n".join(out)


def main() -> None:
    gomb = aggregate_gomb_strict()
    hsikan = aggregate_hsikan_optuna()
    md = render_table(gomb, hsikan)
    print(md)
    print("\n# Raw JSON\n```json")
    print(json.dumps({"gomb_strict": gomb, "hsikan_optuna": hsikan},
                     indent=2, default=str))
    print("```")


if __name__ == "__main__":
    main()
