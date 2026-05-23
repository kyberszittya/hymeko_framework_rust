"""Aggregate every SignedKAN ablation result into one ledger.

Reads all `experiments/results/*.json` produced by the run_*.py
scripts, computes per-(config, dataset) medians, and emits a single
markdown table tracking AUC, macro-F1, parameter count, and
median elapsed-time delta vs the L=1 EC baseline.

Run:
  python -m signedkan_wip.src.paperkit.build_ledger
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO   = Path(__file__).resolve().parents[3]
RESDIR = REPO / "signedkan_wip" / "experiments" / "results"
LEDGER = REPO / "signedkan_wip" / "docs" / "archive" / "RESULTS_LEDGER.md"


# Per-file → list of (group_key, label) extractor. group_key partitions
# the rows of the file (typically by "cfg" field); label becomes the
# row name in the ledger.
def collect():
    groups = defaultdict(list)        # (label, dataset) -> list of rows
    for path in sorted(RESDIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list) or not data:
            continue
        for r in data:
            if not isinstance(r, dict):
                continue
            if r.get("model") not in ("signedkan", "signedkan_entropy"):
                continue
            label = r.get("cfg") or r.get("model") or path.stem
            ds = r.get("dataset")
            if ds is None:
                continue
            groups[(label, ds)].append(r)
    return groups


def med(rs, k):
    vals = [r[k] for r in rs if k in r and r[k] is not None]
    return float(np.median(vals)) if vals else float("nan")


def main():
    groups = collect()

    # Pick L=1 EC baseline per dataset.
    bl = {}
    for (lbl, ds), rs in groups.items():
        if lbl == "EC" and rs and rs[0].get("n_layers", 1) == 1:
            bl[ds] = (med(rs, "test_auc"),
                      med(rs, "test_f1_macro"),
                      rs[0].get("n_params"),
                      med(rs, "elapsed_s"))
    # Fallback: any "EC" group.
    for (lbl, ds), rs in groups.items():
        if ds not in bl and lbl == "EC":
            bl[ds] = (med(rs, "test_auc"),
                      med(rs, "test_f1_macro"),
                      rs[0].get("n_params"),
                      med(rs, "elapsed_s"))

    lines = []
    lines.append("# SignedKAN results ledger")
    lines.append("")
    lines.append("Per-(recipe, dataset) medians of three seeds. "
                 "ΔAUC and ΔF1m are vs L=1 EC (full-batch + early stopping + "
                 "class-weighted BCE) on the same dataset.")
    lines.append("")
    for ds in ["bitcoin_alpha", "bitcoin_otc", "slashdot"]:
        if ds not in bl:
            continue
        bl_auc, bl_f1, bl_params, bl_t = bl[ds]
        lines.append(f"## {ds}  (L=1 EC baseline: AUC {bl_auc:.4f}, "
                     f"macro-F1 {bl_f1:.4f}, {bl_params:,} params, "
                     f"{bl_t:.1f}s/run)")
        lines.append("")
        lines.append("| recipe | AUC | ΔAUC | macro-F1 | ΔF1m | params | Δparams | seconds |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        rows = sorted(
            [(lbl, rs) for (lbl, ds_), rs in groups.items() if ds_ == ds],
            key=lambda kv: -med(kv[1], "test_auc")
        )
        for lbl, rs in rows:
            a = med(rs, "test_auc"); f = med(rs, "test_f1_macro")
            t = med(rs, "elapsed_s")
            p = rs[0].get("n_params") or 0
            da = a - bl_auc; df = f - bl_f1
            dp = p - (bl_params or 0)
            dp_str = f"{dp:+,}" if dp != 0 else "0"
            mark = " ← baseline" if lbl == "EC" else ""
            lines.append(
                f"| {lbl}{mark} | {a:.4f} | {da:+.4f} | {f:.4f} | {df:+.4f} | "
                f"{p:,} | {dp_str} | {t:.1f} |"
            )
        lines.append("")

    LEDGER.write_text("\n".join(lines) + "\n")
    print(f"wrote {LEDGER}")


if __name__ == "__main__":
    main()
