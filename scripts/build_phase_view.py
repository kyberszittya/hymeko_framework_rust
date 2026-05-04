#!/usr/bin/env python3
"""Build a clickable HTML view of a phase log + its CSVs.

Each run row collapses; click to expand and see:
  - Paired Δ statistics (mean / t / W/L)
  - Per-seed accuracy + entropy + KL + wall time
  - Per-epoch validation accuracy trajectory
  - The CLI invocation that produced the CSV
  - File path links to log + CSV (open with editor)

Usage:
    python3 scripts/build_phase_view.py \
        --log /tmp/thesis_iv_views_ph12.log \
        --csv-dir data/benchmarks \
        --baseline-from <CSV-suffix-of-anchor-run> \
        --out docs/results/ph12.html

If --baseline-from is omitted, single-arm runs get no paired Δ.
The anchor's filename suffix is e.g. "234308.csv" (HHMMSS.csv portion).
"""
from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import statistics as stats
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────── log parsing ────────────────────────────────────


@dataclass
class RunRecord:
    label: str
    cmd: str
    start_time: str
    end_time: str | None = None
    status: str = "in_progress"   # "DONE" | "FAIL" | "in_progress"
    csv_path: Path | None = None


def parse_log(log_path: Path) -> list[RunRecord]:
    """Walk the phase log file in order and return one record per RUN."""
    records: list[RunRecord] = []
    cur: RunRecord | None = None
    txt = log_path.read_text(errors="replace")
    for line in txt.splitlines():
        m_start = re.match(r"\[(\d{2}:\d{2}:\d{2})\] START:\s*(.+)", line)
        m_done = re.match(r"\[(\d{2}:\d{2}:\d{2})\] DONE:\s*(.+)", line)
        m_fail = re.match(r"\[(\d{2}:\d{2}:\d{2})\] FAIL:\s*(.+)", line)
        m_cmd = re.match(r"\s*cmd:\s*(.+)", line)
        if m_start:
            cur = RunRecord(label=m_start.group(2).strip(),
                            cmd="", start_time=m_start.group(1))
            records.append(cur)
        elif m_cmd and cur and not cur.cmd:
            cur.cmd = m_cmd.group(1).strip()
        elif m_done and cur:
            cur.end_time = m_done.group(1)
            cur.status = "DONE"
        elif m_fail and cur:
            cur.end_time = m_fail.group(1)
            cur.status = "FAIL"
    return records


# ─────────────────── CSV pairing ────────────────────────────────────


def csv_mtime_secs(p: Path) -> float:
    return p.stat().st_mtime


def pair_runs_with_csvs(
    runs: list[RunRecord],
    csv_dir: Path,
    log_path: Path,
) -> None:
    """Assign each DONE run the smallest-mtime CSV after the log's run-start
    that's also after the previous run's CSV."""
    log_start_unix = log_path.stat().st_mtime - 24 * 3600  # generous window
    csvs = sorted(
        [p for p in csv_dir.glob("thesis_iv_hard_*.csv")
         if csv_mtime_secs(p) >= log_start_unix],
        key=csv_mtime_secs,
    )
    used: set[Path] = set()
    for r in runs:
        if r.status != "DONE":
            continue
        # First unused CSV in order is this run's CSV.
        for c in csvs:
            if c in used:
                continue
            r.csv_path = c
            used.add(c)
            break


# ─────────────────── stats per CSV ──────────────────────────────────


@dataclass
class RunStats:
    n_baseline: int = 0
    n_treat: int = 0
    paired_n: int = 0
    delta_pp: float = 0.0
    t_stat: float = 0.0
    wins: int = 0
    losses: int = 0
    sig: str = ""
    treat_mean: float = 0.0
    treat_sd: float = 0.0
    base_mean: float = 0.0
    base_sd: float = 0.0
    treat_arm: str = ""
    seed_rows: list[dict] = field(default_factory=list)


def sig_label(t: float) -> str:
    a = abs(t)
    if a > 3.291: return "***"
    if a > 2.576: return "**"
    if a > 1.96:  return "*"
    if a > 1.645: return "."
    return ""


def stats_for_csv(csv_path: Path, baseline_anchor: dict[int, float] | None) -> RunStats:
    rows = list(csv.DictReader(csv_path.open()))
    by_arm: dict[str, dict[int, float]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {})[int(r["seed"])] = float(r["final_val_acc"])
    out = RunStats()
    out.seed_rows = rows
    arms = sorted(by_arm)
    if "baseline" in arms:
        bs = by_arm["baseline"]
    elif baseline_anchor is not None:
        bs = baseline_anchor
    else:
        bs = {}
    treat = [a for a in arms if a != "baseline"]
    if not treat:
        out.n_baseline = len(by_arm.get("baseline", {}))
        return out
    out.treat_arm = treat[0]
    ts = by_arm[out.treat_arm]
    out.n_baseline = len(by_arm.get("baseline", bs))
    out.n_treat = len(ts)
    if bs:
        common = sorted(set(bs) & set(ts))
        diffs = [(ts[i] - bs[i]) * 100 for i in common]
        if len(diffs) >= 2:
            md = stats.mean(diffs)
            sd = stats.stdev(diffs)
            n = len(diffs)
            t = md / (sd / math.sqrt(n)) if sd > 0 else 0
            out.paired_n = n
            out.delta_pp = md
            out.t_stat = t
            out.wins = sum(d > 0 for d in diffs)
            out.losses = sum(d < 0 for d in diffs)
            out.sig = sig_label(t)
        bs_vals = [bs[i] for i in common]
        if bs_vals:
            out.base_mean = stats.mean(bs_vals) * 100
            out.base_sd = stats.stdev(bs_vals) * 100 if len(bs_vals) > 1 else 0
    if ts:
        ts_vals = list(ts.values())
        out.treat_mean = stats.mean(ts_vals) * 100
        out.treat_sd = stats.stdev(ts_vals) * 100 if len(ts_vals) > 1 else 0
    return out


# ─────────────────── HTML rendering ─────────────────────────────────


CSS = """
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1080px; margin: 1.5em auto; padding: 0 1em; color: #1f2328; }
h1 { font-size: 1.4em; margin: 0 0 0.2em 0; }
h2 { font-size: 1.05em; margin-top: 1.2em; border-bottom: 1px dotted #d0d7de; padding-bottom: 2px; }
.meta { color: #57606a; font-size: 0.92em; margin-bottom: 1em; }
table.runs { border-collapse: collapse; width: 100%; }
table.runs td, table.runs th { padding: 6px 8px; border-bottom: 1px solid #eaecef; }
table.runs th { text-align: left; background: #f6f8fa; font-weight: 600; }
.delta-pos { color: #117a3d; }
.delta-neg { color: #b62324; }
.delta-zero { color: #57606a; }
.sig { font-family: ui-monospace, monospace; font-weight: 600; }
.sig-three { color: #b62324; }
.sig-two   { color: #c1721c; }
.sig-one   { color: #966d00; }
.sig-trend { color: #57606a; }
details { margin: 0; }
details > summary { list-style: none; cursor: pointer; padding: 0; }
details > summary::-webkit-details-marker { display: none; }
details[open] > summary { background: #f6f8fa; }
.detail-block { background: #f6f8fa; padding: 10px 14px; margin: 4px 0 12px 0;
                border-left: 3px solid #cdd9e5; font-size: 13px; }
.detail-block code { font-family: ui-monospace, monospace; font-size: 12px;
                      background: #fff; padding: 1px 4px; border-radius: 3px;
                      border: 1px solid #d0d7de; }
.detail-block pre { background: #fff; padding: 8px; border: 1px solid #d0d7de;
                    border-radius: 4px; overflow-x: auto; font-size: 12px; }
.kv { display: grid; grid-template-columns: 11em 1fr; gap: 2px 12px; margin: 4px 0; }
.kv .k { color: #57606a; }
.seed-table { font-size: 12px; font-family: ui-monospace, monospace;
              border-collapse: collapse; margin-top: 6px; max-width: 100%; }
.seed-table td, .seed-table th { padding: 2px 6px; border: 1px solid #eaecef; }
.seed-table th { background: #fff; }
.path { font-family: ui-monospace, monospace; font-size: 12px; color: #2152a0; }
.label-fail { color: #b62324; font-weight: 600; }
.label-done { color: #1a7f37; }
.label-running { color: #966d00; }
.summary-row { display: grid; grid-template-columns: 4em 26em 5em 7em 6em 8em 4em 2em;
               gap: 8px; align-items: baseline; padding: 6px 4px;
               border-bottom: 1px solid #eaecef; }
.summary-row:hover { background: #f6f8fa; }
"""


def fmt_delta(s: RunStats) -> str:
    if s.paired_n == 0:
        return '<span class="delta-zero">—</span>'
    cls = "delta-pos" if s.delta_pp > 0 else ("delta-neg" if s.delta_pp < 0 else "delta-zero")
    return f'<span class="{cls}">{s.delta_pp:+.3f}</span>'


def fmt_sig(sig: str) -> str:
    if not sig: return ""
    cls = {"***": "sig-three", "**": "sig-two", "*": "sig-one", ".": "sig-trend"}[sig]
    return f'<span class="sig {cls}">{sig}</span>'


def fmt_status(status: str) -> str:
    cls = {"DONE": "label-done", "FAIL": "label-fail",
           "in_progress": "label-running"}.get(status, "label-running")
    return f'<span class="{cls}">{status}</span>'


def render_seed_table(rows: list[dict], baseline_anchor: dict[int, float] | None) -> str:
    if not rows:
        return ""
    cols = ["seed", "arm", "final_val_acc", "final_entropy", "final_kl",
            "wall_seconds", "stable_rank_mean"]
    lines = ['<table class="seed-table"><tr>']
    for c in cols:
        lines.append(f"<th>{html.escape(c)}</th>")
    if baseline_anchor:
        lines.append("<th>Δ vs anchor (pp)</th>")
    lines.append("</tr>")
    for r in rows:
        lines.append("<tr>")
        for c in cols:
            v = r.get(c, "")
            if c in ("final_val_acc", "final_entropy", "final_kl",
                     "stable_rank_mean"):
                try:
                    v = f"{float(v):.4f}"
                except ValueError:
                    pass
            elif c == "wall_seconds":
                try:
                    v = f"{float(v):.1f}s"
                except ValueError:
                    pass
            lines.append(f"<td>{html.escape(str(v))}</td>")
        if baseline_anchor and r.get("arm") != "baseline":
            try:
                seed = int(r["seed"])
                tv = float(r["final_val_acc"])
                bv = baseline_anchor.get(seed)
                if bv is not None:
                    d = (tv - bv) * 100
                    cls = "delta-pos" if d > 0 else ("delta-neg" if d < 0 else "delta-zero")
                    lines.append(f'<td class="{cls}">{d:+.3f}</td>')
                else:
                    lines.append("<td>—</td>")
            except Exception:
                lines.append("<td>—</td>")
        elif baseline_anchor:
            lines.append("<td>—</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "".join(lines)


def render_run(idx: int, r: RunRecord, s: RunStats,
               baseline_anchor: dict[int, float] | None,
               log_rel: str) -> str:
    delta_html = fmt_delta(s)
    sig_html = fmt_sig(s.sig)
    wl = f"{s.wins}/{s.losses}" if s.paired_n else "—"
    csv_link = (f'<a class="path" href="{html.escape(str(r.csv_path))}">'
                f'{html.escape(r.csv_path.name)}</a>'
                if r.csv_path else '<span class="delta-zero">—</span>')
    summary = (
        f'<div class="summary-row">'
        f'  <div>{idx + 1}.</div>'
        f'  <div><strong>{html.escape(r.label)}</strong></div>'
        f'  <div>{fmt_status(r.status)}</div>'
        f'  <div>n={s.paired_n or s.n_treat or s.n_baseline}</div>'
        f'  <div>{delta_html}</div>'
        f'  <div>t={s.t_stat:+.2f}</div>'
        f'  <div>{wl}</div>'
        f'  <div>{sig_html}</div>'
        f'</div>'
    )
    body_lines = ['<div class="detail-block">', '<div class="kv">']
    body_lines.append(f'<div class="k">log time</div><div>{r.start_time} → {r.end_time or "(running)"}</div>')
    body_lines.append(f'<div class="k">CSV</div><div>{csv_link}</div>')
    body_lines.append(f'<div class="k">log file</div><div><span class="path">{html.escape(log_rel)}</span></div>')
    if s.paired_n:
        body_lines.append(f'<div class="k">paired Δ (pp)</div><div>{s.delta_pp:+.3f} '
                          f'(t={s.t_stat:+.2f}, n={s.paired_n}, W/L={s.wins}/{s.losses}, sig={s.sig or "ns"})</div>')
        body_lines.append(f'<div class="k">baseline mean</div><div>{s.base_mean:.3f} ± {s.base_sd:.3f} pp</div>')
        body_lines.append(f'<div class="k">{html.escape(s.treat_arm)} mean</div>'
                          f'<div>{s.treat_mean:.3f} ± {s.treat_sd:.3f} pp</div>')
    body_lines.append('</div>')
    if r.cmd:
        body_lines.append(f'<pre>{html.escape(r.cmd)}</pre>')
    body_lines.append(render_seed_table(s.seed_rows, baseline_anchor))
    body_lines.append('</div>')
    body = "".join(body_lines)
    return f"<details>\n<summary>{summary}</summary>\n{body}\n</details>\n"


def render_page(title: str, log_path: Path, runs: list[RunRecord],
                stats_per_run: list[RunStats],
                baseline_anchor: dict[int, float] | None) -> str:
    log_rel = str(log_path)
    head = (
        '<table class="runs">'
        '<thead><tr>'
        '<th>#</th><th>run</th><th>status</th><th>n</th>'
        '<th>Δ pp</th><th>t</th><th>W/L</th><th>sig</th>'
        '</tr></thead><tbody>'
    )
    rows_html = "".join(
        render_run(i, r, s, baseline_anchor, log_rel)
        for i, (r, s) in enumerate(zip(runs, stats_per_run))
    )
    completed = sum(1 for r in runs if r.status == "DONE")
    failed = sum(1 for r in runs if r.status == "FAIL")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">
  log: <span class="path">{html.escape(log_rel)}</span><br>
  {len(runs)} runs · {completed} done · {failed} failed
</div>

<p style="color:#57606a;font-size:0.92em">
  Click any row to expand and see paired Δ details, the CLI invocation,
  and the per-seed CSV table. CSV paths are clickable links to the file.
</p>

<div class="summary-row" style="background:#f6f8fa;font-weight:600;color:#57606a">
  <div>#</div><div>run</div><div>status</div><div>n</div>
  <div>Δ pp</div><div>t</div><div>W/L</div><div>sig</div>
</div>
{rows_html}

</body></html>
"""


# ─────────────────── main ───────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--csv-dir", type=Path, default=Path("data/benchmarks"))
    ap.add_argument("--baseline-from", type=str, default=None,
                    help="CSV filename suffix (e.g., '234308.csv') of the "
                         "anchor run; baseline rows from this CSV are reused "
                         "as the comparison anchor for single-arm runs.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    runs = parse_log(args.log)
    if not runs:
        print(f"No RUN records parsed from {args.log}", flush=True)
        return 1
    pair_runs_with_csvs(runs, args.csv_dir, args.log)

    # Build the baseline anchor row dict if requested.
    anchor: dict[int, float] | None = None
    if args.baseline_from:
        match = list(args.csv_dir.glob(f"*{args.baseline_from}"))
        if match:
            with match[0].open() as f:
                rows = list(csv.DictReader(f))
            anchor = {int(r["seed"]): float(r["final_val_acc"])
                      for r in rows if r["arm"] == "baseline"}

    stats_per_run = []
    for r in runs:
        if r.csv_path:
            stats_per_run.append(stats_for_csv(r.csv_path, anchor))
        else:
            stats_per_run.append(RunStats())

    title = args.title or f"Phase view — {args.log.name}"
    page = render_page(title, args.log, runs, stats_per_run, anchor)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"Wrote {args.out}  ({len(runs)} runs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
