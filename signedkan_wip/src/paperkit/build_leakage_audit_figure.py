"""Build the label-shuffle audit figure for the Nature leakage paper.

Reads the Phase-B grid JSONL (``run_no_leak_benchmark --baselines`` output) and
draws, per method, the held-out AUROC under **real** vs **shuffled** training
labels (mean ± std over datasets × seeds), with the chance line at 0.5. A
protocol-clean method's shuffled bar collapses to chance; a leaky one stays high.

Tolerates partial data (the grid may still be running) — only methods with at
least one finite real and shuffled cell are drawn. Drops the figure at
``signedkan_wip/paper/figures/leakage_audit.{pdf,png,eps}`` (EPS for Springer).

Run:
  python -m signedkan_wip.src.paperkit.build_leakage_audit_figure \
      [--input signedkan_wip/experiments/results/no_leak_baselines.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO / "signedkan_wip" / "experiments" / "results" / "no_leak_baselines.jsonl"
FIG_DIR = REPO / "signedkan_wip" / "paper" / "figures"

# Display order + labels (mirrors run_no_leak_benchmark.AUDIT_METHODS).
METHOD_ORDER = ["sgcn", "sigat", "sgt", "sgcl", "sigformer", "sesgformer", "dadsgnn"]
METHOD_LABEL = {
    "sgcn": "SGCN", "sigat": "SiGAT", "sgt": "SGT", "sgcl": "SGCL",
    "sigformer": "SiGformer", "sesgformer": "SE-SGformer", "dadsgnn": "DADSGNN",
}
CHANCE = 0.5


def _finite(xs: list[float]) -> list[float]:
    return [x for x in xs if x is not None and not math.isnan(x)]


def load_audit(path: Path) -> dict[str, dict[bool, list[float]]]:
    """``{method: {False: [real aucs], True: [shuffled aucs]}}`` from the JSONL."""
    agg: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {False: [], True: []})
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        agg[r["model"]][bool(r["shuffle"])].append(r.get("auc"))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-stem", default="leakage_audit")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"no grid JSONL at {args.input} — run --baselines first")

    agg = load_audit(args.input)
    methods, real_mean, real_std, shuf_mean, shuf_std = [], [], [], [], []
    for m in METHOD_ORDER:
        real = _finite(agg.get(m, {}).get(False, []))
        shuf = _finite(agg.get(m, {}).get(True, []))
        if not real or not shuf:
            continue  # partial data: skip methods without both arms yet
        methods.append(METHOD_LABEL[m])
        real_mean.append(float(np.mean(real)))
        real_std.append(float(np.std(real)))
        shuf_mean.append(float(np.mean(shuf)))
        shuf_std.append(float(np.std(shuf)))

    if not methods:
        raise SystemExit(f"{args.input} has no method with both real+shuffle cells yet")

    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(methods)), 3.6), dpi=120)
    ax.bar(x - w / 2, real_mean, w, yerr=real_std, capsize=3,
           color="#1b6ca8", label="real labels")
    ax.bar(x + w / 2, shuf_mean, w, yerr=shuf_std, capsize=3,
           color="#9aa0a6", label="shuffled train labels")
    ax.axhline(CHANCE, ls="--", lw=1.2, color="#b02a2a", label="chance (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("held-out AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Label-shuffle audit: held-out AUROC, real vs shuffled training labels")
    ax.grid(True, axis="y", ls=":", color="#aaa", lw=0.5, alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "eps"):
        out = FIG_DIR / f"{args.out_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=140 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {FIG_DIR / args.out_stem}.{{pdf,png,eps}}  ({len(methods)} methods)")


if __name__ == "__main__":
    main()
