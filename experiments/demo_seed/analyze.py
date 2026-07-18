"""Analyze the 16 demo-seed cells: seed-level CSV, aggregate curves with 95% CIs, paired cold-vs-demo comparison,
Q-demo-vs-online curves. Answers: is the limiting issue discovery / retention / both?
Writes demo_seed_results.csv, and figures under ../../reports/figures/2026-07-18-demo-seed/.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[1] / "reports" / "figures" / "2026-07-18-demo-seed"
ARMS = ("cold", "demo_seed")
SEEDS = range(8)


def _load(arm: str, seed: int) -> dict | None:
    p = HERE / f"result_{arm}_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ci(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and 95% bootstrap-free normal CI over seeds (rows = seeds x checkpoints)."""
    med = np.median(rows, axis=0)
    n = rows.shape[0]
    se = rows.std(axis=0, ddof=1) / np.sqrt(max(n, 1))
    return med, med - 1.96 * se, med + 1.96 * se


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    data = {arm: [_load(arm, s) for s in SEEDS] for arm in ARMS}
    present = {arm: [r for r in data[arm] if r is not None] for arm in ARMS}
    n_cold, n_demo = len(present["cold"]), len(present["demo_seed"])
    print(f"[analyze] cells present: cold={n_cold}/8 demo_seed={n_demo}/8", flush=True)
    if n_cold == 0 or n_demo == 0:
        print("[analyze] not enough cells yet", flush=True)
        return 1

    # --- seed-level CSV ---
    fields = ["arm", "seed", "first_contact_step", "first_success_step", "best_success",
              "final_success", "stable_success_final5", "retention_gap", "collapse_events"]
    with (HERE / "demo_seed_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for arm in ARMS:
            for r in present[arm]:
                w.writerow({k: r.get(k) for k in fields})

    # --- aggregate summary ---
    def agg(arm: str, key: str) -> tuple[float, float]:
        v = np.array([r[key] for r in present[arm]], float)
        return float(np.median(v)), float(np.mean(v))

    summary = {}
    for arm in ARMS:
        summary[arm] = {
            "n": len(present[arm]),
            "final5_median": agg(arm, "stable_success_final5")[0],
            "final5_mean": agg(arm, "stable_success_final5")[1],
            "best_median": agg(arm, "best_success")[0],
            "retention_gap_median": agg(arm, "retention_gap")[0],
            "final5_by_seed": sorted(r["stable_success_final5"] for r in present[arm]),
            "retention_by_seed": sorted(r["retention_gap"] for r in present[arm]),
        }
    # paired (only seeds present in BOTH arms)
    paired = []
    for s in SEEDS:
        c, d = _load("cold", s), _load("demo_seed", s)
        if c and d:
            paired.append({"seed": s,
                           "cold_final5": c["stable_success_final5"], "demo_final5": d["stable_success_final5"],
                           "d_final5": round(d["stable_success_final5"] - c["stable_success_final5"], 3),
                           "cold_gap": c["retention_gap"], "demo_gap": d["retention_gap"],
                           "d_gap": round(d["retention_gap"] - c["retention_gap"], 3)})
    summary["paired"] = paired
    n_demo_better = sum(1 for p in paired if p["d_final5"] > 0)
    n_gap_smaller = sum(1 for p in paired if p["d_gap"] < 0)
    summary["verdict"] = {
        "n_paired": len(paired),
        "demo_higher_final5_seeds": n_demo_better,
        "demo_smaller_gap_seeds": n_gap_smaller,
        "median_delta_final5": round(float(np.median([p["d_final5"] for p in paired])), 3) if paired else None,
        "median_delta_gap": round(float(np.median([p["d_gap"] for p in paired])), 3) if paired else None,
    }
    (HERE / "demo_seed_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["verdict"], indent=2), flush=True)

    # --- figure 1: aggregate success curves with CI ---
    _plot_curves(present, "success_curve", "success rate", FIG / "success_curves.png")
    _plot_curves(present, "contact_curve", "contact rate", FIG / "contact_curves.png")
    _plot_q(present, FIG / "q_demo_vs_online.png")
    _plot_paired(paired, FIG / "paired_final5_retention.png")
    print(f"[analyze] figures -> {FIG}", flush=True)
    return 0


def _steps(r: dict) -> np.ndarray:
    return np.arange(1, len(r["success_curve"]) + 1) * r["eval_every"]


def _stack(present: dict, arm: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    L = min(len(r[key]) for r in present[arm])
    rows = np.array([r[key][:L] for r in present[arm]], float)
    x = np.arange(1, L + 1) * present[arm][0]["eval_every"]
    return x, rows


def _plot_curves(present: dict, key: str, ylabel: str, out: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for arm, color in (("cold", "#c44"), ("demo_seed", "#37a")):
        if not present[arm]:
            continue
        x, rows = _stack(present, arm, key)
        med, lo, hi = _ci(rows)
        plt.plot(x, med, color=color, lw=2, label=f"{arm} (n={rows.shape[0]}, median)")
        plt.fill_between(x, lo, hi, color=color, alpha=0.18)
        for row in rows:
            plt.plot(x, row, color=color, lw=0.5, alpha=0.25)
    plt.xlabel("env steps")
    plt.ylabel(ylabel)
    plt.title(f"Coffee-Push corrected SAC — {ylabel}: cold vs demo-seeded replay")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def _plot_q(present: dict, out: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for arm, c1, c2 in (("cold", "#c44", "#e99"), ("demo_seed", "#37a", "#9bd")):
        if not present[arm]:
            continue
        x, qd = _stack(present, arm, "q_demo_curve")
        _, qo = _stack(present, arm, "q_online_curve")
        plt.plot(x, np.median(qd, 0), color=c1, lw=2, label=f"{arm} Q(demo)")
        plt.plot(x, np.median(qo, 0), color=c2, lw=1.5, ls="--", label=f"{arm} Q(online-greedy)")
    plt.xlabel("env steps")
    plt.ylabel("min-critic Q (median over seeds)")
    plt.title("Critic Q on demo transitions vs online-greedy states")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def _plot_paired(paired: list, out: Path) -> None:
    if not paired:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    seeds = [p["seed"] for p in paired]
    for a, (kc, kd, title) in zip(ax, [("cold_final5", "demo_final5", "stable final-5 success"),
                                       ("cold_gap", "demo_gap", "retention gap (best − final5)")]):
        for p in paired:
            a.plot([0, 1], [p[kc], p[kd]], "-o", color="#888", alpha=0.6)
        a.plot([0] * len(paired), [p[kc] for p in paired], "o", color="#c44", label="cold")
        a.plot([1] * len(paired), [p[kd] for p in paired], "o", color="#37a", label="demo_seed")
        a.set_xticks([0, 1])
        a.set_xticklabels(["cold", "demo_seed"])
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"Paired cold vs demo-seeded (n={len(paired)} seeds: {seeds})")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
