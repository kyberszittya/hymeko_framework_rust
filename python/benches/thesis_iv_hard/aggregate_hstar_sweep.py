"""H* sweep aggregator for phase 7.

Parses /tmp/thesis_iv_views_ph7.log to map each entropy_target run to
its (--target-entropy) value, then loads the matching CSV and computes
paired stats. Renders a table indexed by (dataset, H*).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

REPO = Path(__file__).resolve().parents[3]
LOG  = Path("/tmp/thesis_iv_views_ph7.log")
BENCH = REPO / "data" / "benchmarks"


def parse_log() -> list[dict]:
    """Yield {dataset, hstar, started_at} per entropy_target START line."""
    if not LOG.exists():
        return []
    text = LOG.read_text().splitlines()
    out = []
    for i, line in enumerate(text):
        m = re.search(r"\[(\d\d:\d\d:\d\d)\] START: (\S+) entropy_target H\*=([\d.]+)", line)
        if not m:
            continue
        time_str, dataset, hstar = m.group(1), m.group(2), float(m.group(3))
        # Look for the immediately following CSV path; the run script
        # writes "Wrote N records to data/benchmarks/...csv" at the end
        # of each invocation, so we scan ahead until the next DONE.
        csv_path = None
        for j in range(i + 1, min(i + 800, len(text))):
            cm = re.search(r"Wrote \d+ records to (data/benchmarks/\S+\.csv)", text[j])
            if cm:
                csv_path = REPO / cm.group(1)
                break
            if re.search(r"DONE: ", text[j]):
                break
        if csv_path is None:
            continue
        out.append({
            "dataset": dataset,
            "hstar":   hstar,
            "started_at": time_str,
            "csv": csv_path,
        })
    return out


def analyze(entry: dict) -> dict:
    df = pd.read_csv(entry["csv"])
    sub = df[(df["dataset"] == entry["dataset"])]
    base = sub[sub["arm"] == "baseline"]["final_val_acc"].values
    treat = sub[sub["arm"] == "entropy_target"]["final_val_acc"].values
    n = min(len(base), len(treat))
    if n < 2:
        return {**entry, "delta_pp": float("nan"), "t": float("nan"), "n": n}
    base, treat = base[:n], treat[:n]
    delta = treat - base
    t = stats.ttest_rel(treat, base)
    return {
        **entry,
        "n": n,
        "delta_pp": float(delta.mean() * 100),
        "t":        float(t.statistic),
        "p":        float(t.pvalue),
        "wins":     int((delta > 0).sum()),
        "losses":   int((delta < 0).sum()),
        "ties":     int((delta == 0).sum()),
    }


def _sig(p: float) -> str:
    if np.isnan(p):       return ""
    if p < 0.001:         return "***"
    if p < 0.01:          return "** "
    if p < 0.05:          return "*  "
    if p < 0.10:          return ".  "
    return "   "


def main() -> None:
    entries = parse_log()
    if not entries:
        print("No entropy_target runs found in", LOG)
        return
    rows = [analyze(e) for e in entries]
    rows.sort(key=lambda r: (r["dataset"], r["hstar"]))

    print(f"H* sweep — phase 7 entropy_target runs ({len(rows)} total)\n")
    print(f"{'dataset':<12s} {'H*':>5s} {'n':>4s}  {'Δ pp':>8s}   {'t':>6s}  sig  W/L/T")
    print("-" * 60)
    for r in rows:
        if "delta_pp" not in r or np.isnan(r["delta_pp"]):
            print(f"{r['dataset']:<12s} {r['hstar']:>5.2f} {'?':>4s}  (no rows)")
            continue
        wlt = f"{r['wins']}/{r['losses']}/{r['ties']}"
        print(f"{r['dataset']:<12s} {r['hstar']:>5.2f} "
              f"{r['n']:>4d}  {r['delta_pp']:+8.3f}   "
              f"{r['t']:+6.2f}  {_sig(r['p'])}  {wlt}")


if __name__ == "__main__":
    main()
