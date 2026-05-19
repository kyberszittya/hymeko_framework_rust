"""Aggregate the Phase 3 sweep JSONL into paper-ready tables.

Reads `signedkan_wip/experiments/results/phase3_sweep/sweep_*.log`,
groups by (hidden, lr), reports mean ± std across seeds, and
identifies the best config by macro-F1.

Run:
    python3 -m src.aggregate_phase3 --dataset bitcoin_alpha
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


SWEEP_DIR = Path("signedkan_wip/experiments/results/phase3_sweep")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate(records: list[dict]) -> dict[tuple[int, float], dict]:
    by_cfg: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for r in records:
        by_cfg[(r["hidden"], r["lr"])].append(r)
    out: dict[tuple[int, float], dict] = {}
    for cfg, group in by_cfg.items():
        aucs = [r["test_auc"] for r in group]
        f1bs = [r["test_f1_binary"] for r in group]
        f1ms = [r["test_f1_macro"] for r in group]
        n = len(group)
        out[cfg] = dict(
            n_seeds=n,
            n_params=group[0]["n_params"],
            elapsed_mean_s=stats.mean(r["elapsed_s"] for r in group),
            auc_mean=stats.mean(aucs),
            auc_std=stats.stdev(aucs) if n > 1 else 0.0,
            f1bin_mean=stats.mean(f1bs),
            f1bin_std=stats.stdev(f1bs) if n > 1 else 0.0,
            f1mac_mean=stats.mean(f1ms),
            f1mac_std=stats.stdev(f1ms) if n > 1 else 0.0,
        )
    return out


def render_markdown(agg: dict, dataset: str) -> str:
    lines = [
        f"### Phase 3 sweep — {dataset}",
        "",
        f"| hidden | lr     | n | params  | AUC               | F1 binary         | F1 macro          | wall (s) |",
        f"|-------:|-------:|--:|--------:|------------------:|------------------:|------------------:|---------:|",
    ]
    for (h, lr), v in sorted(agg.items()):
        lines.append(
            f"| {h:>6} | {lr:>6.0e} | {v['n_seeds']} | {v['n_params']:>7,} | "
            f"{v['auc_mean']:.3f} ± {v['auc_std']:.3f} | "
            f"{v['f1bin_mean']:.3f} ± {v['f1bin_std']:.3f} | "
            f"{v['f1mac_mean']:.3f} ± {v['f1mac_std']:.3f} | "
            f"{v['elapsed_mean_s']:>8.1f} |"
        )
    return "\n".join(lines)


def render_latex(agg: dict, dataset: str) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Phase~3 hyperparameter sweep on {dataset.replace('_', '~')}, "
        f"3 seeds per cell. Best by macro-F1 highlighted.}}",
        f"\\label{{tab:phase3-{dataset}}}",
        "\\begin{tabular}{rrrrrrr}",
        "\\toprule",
        "hidden & lr & params & AUC & $F_{1}^{\\text{bin}}$ & $F_{1}^{\\text{mac}}$ & wall (s) \\\\",
        "\\midrule",
    ]
    best_cfg = max(agg.items(), key=lambda kv: kv[1]["f1mac_mean"])[0]
    for (h, lr), v in sorted(agg.items()):
        bold = (h, lr) == best_cfg
        emph = ("\\textbf{", "}") if bold else ("", "")
        lines.append(
            f"{h} & {lr:.0e} & {v['n_params']:,} & "
            f"{emph[0]}{v['auc_mean']:.3f}$\\pm${v['auc_std']:.3f}{emph[1]} & "
            f"{v['f1bin_mean']:.3f}$\\pm${v['f1bin_std']:.3f} & "
            f"{emph[0]}{v['f1mac_mean']:.3f}$\\pm${v['f1mac_std']:.3f}{emph[1]} & "
            f"{v['elapsed_mean_s']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha",
                    choices=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--latex", action="store_true",
                    help="Emit LaTeX table instead of Markdown.")
    args = ap.parse_args()
    log_path = SWEEP_DIR / f"sweep_{args.dataset}.log"
    if not log_path.exists():
        print(f"No sweep log at {log_path}; run src.run_phase3_sweep first.")
        return
    records = load(log_path)
    if not records:
        print(f"Empty log at {log_path}.")
        return
    agg = aggregate(records)
    if args.latex:
        print(render_latex(agg, args.dataset))
    else:
        print(render_markdown(agg, args.dataset))
        # Headline best.
        best_cfg, best = max(agg.items(), key=lambda kv: kv[1]["f1mac_mean"])
        print(f"\n**Best by macro-F1**: hidden={best_cfg[0]}, lr={best_cfg[1]:.0e}")
        print(f"  AUC = {best['auc_mean']:.3f} ± {best['auc_std']:.3f}")
        print(f"  F1_binary = {best['f1bin_mean']:.3f} ± {best['f1bin_std']:.3f}")
        print(f"  F1_macro = {best['f1mac_mean']:.3f} ± {best['f1mac_std']:.3f}")


if __name__ == "__main__":
    main()
