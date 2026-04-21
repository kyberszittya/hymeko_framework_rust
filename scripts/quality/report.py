#!/usr/bin/env python3
"""Turn /tmp/audit.json into docs/quality/<date>.md per the quality-audit spec."""
from __future__ import annotations

import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ─── Thresholds (from .claude/commands/quality-audit.md) ─────────────

T = {
    "fanout":  (7, 15),
    "ccn":     (10, 20),
    "nloc":    (50, 100),
    "file":    (500, 1000),
    "nesting": (4, 6),
    "dit":     (5, 7),
    "wmc":     (50, 100),
    "fog":     (12, 16),
}

def band(v: float, thresh: tuple[float, float]) -> str:
    lo, hi = thresh
    if v <= lo:
        return "good"
    if v <= hi:
        return "warn"
    return "critical"


REPO_ROOT = str(Path(".").resolve())

# Names that are too idiomatic to treat as meaningful fan-in targets —
# they appear as identifiers everywhere and inflate the count without
# indicating anything about architecture.
FANIN_NOISE = {
    "new", "len", "name", "iter", "default", "start", "end", "push",
    "get", "set", "from", "into", "to", "as", "is", "or", "and", "not",
    "map", "filter", "collect", "clone", "drop", "run", "info", "debug",
    "warn", "error", "build", "main", "init", "ok", "some", "none",
}


def strip_path(p: str) -> str:
    p = p.lstrip("./")
    if p.startswith(REPO_ROOT.lstrip("/")):
        p = p[len(REPO_ROOT.lstrip("/")):].lstrip("/")
    return p


def main() -> None:
    data = json.loads(Path("/tmp/audit.json").read_text())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("docs/quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{today.replace('-', '')}.md"
    if out_file.exists():
        for i in range(2, 99):
            candidate = out_dir / f"{today.replace('-', '')}-{i}.md"
            if not candidate.exists():
                out_file = candidate
                break

    # Baseline detection
    prior = sorted([p for p in out_dir.glob("*.md") if p != out_file])
    baseline_name = prior[-1].name if prior else "none"

    # Commit info
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"],
                                         text=True).strip()
    except Exception:
        sha, branch = "?", "?"

    # ─── Aggregate ───────────────────────────────────────────────────

    fns = data["per_fn"]
    # Filter out trivial 0-NLOC rows (e.g. trait method sigs)
    fns = [f for f in fns if f["nloc"] > 0]

    def band_counts(values: list[float], thresh: tuple[float, float]) -> dict[str, int]:
        b = Counter(band(v, thresh) for v in values)
        return {"good": b["good"], "warn": b["warn"], "critical": b["critical"]}

    fanout_b = band_counts([f["fanout"] for f in fns], T["fanout"])
    ccn_b = band_counts([f["ccn"] for f in fns], T["ccn"])
    nloc_b = band_counts([f["nloc"] for f in fns], T["nloc"])
    nesting_b = band_counts([f["nesting"] for f in fns], T["nesting"])
    file_slocs = data["file_slocs"]
    file_b = band_counts([s for _, s, _ in file_slocs], T["file"])

    dit_all = [d for _, _, d in data["py_dits"]] + [d for _, d in data["rust_dits"]]
    dit_all = [d for d in dit_all if d > 0]
    dit_b = band_counts(dit_all, T["dit"]) if dit_all else {"good": 0, "warn": 0, "critical": 0}

    wmc_all = [v for _, _, v in data["wmc_rust"]] + [v for _, _, v in data["wmc_python"]]
    wmc_b = band_counts(wmc_all, T["wmc"]) if wmc_all else {"good": 0, "warn": 0, "critical": 0}

    # Fog: use repo median as the comparator (per spec note)
    all_fog = [s["fog"] for s in data["fog_md"] + data["fog_rust"]]
    fog_median = statistics.median(all_fog) if all_fog else 0.0

    short_idents = data["idents"]["short"]
    long_idents = data["idents"]["long"]

    # ─── Build report ────────────────────────────────────────────────

    out: list[str] = []
    out.append(f"# Software quality audit — {today}")
    out.append("")
    out.append(f"**Baseline:** {baseline_name}"
               + ("" if baseline_name == "none" else " (N/A days — first report)"))
    out.append(f"**Commit:** {sha} on `{branch}`")
    out.append(f"**Scope:** full workspace · Rust + Python")
    out.append("")

    out.append("## Summary")
    out.append("")
    out.append("| Metric | Good | Warn | Critical | Δ vs baseline |")
    out.append("|---|---:|---:|---:|---:|")
    delta = " — "  # No baseline yet
    rows = [
        ("Fan-out (functions, ≤7 / ≤15)", fanout_b),
        ("Cyclomatic complexity (≤10 / ≤20)", ccn_b),
        ("Function length NLOC (≤50 / ≤100)", nloc_b),
        ("File length SLOC (≤500 / ≤1000)", file_b),
        ("Nesting depth (≤4 / ≤6)", nesting_b),
        ("DIT (≤5 / ≤7)", dit_b),
        ("WMC (≤50 / ≤100)", wmc_b),
    ]
    for label, b in rows:
        out.append(f"| {label} | {b['good']} | {b['warn']} | {b['critical']} | {delta} |")
    out.append(f"| Short identifiers (<3 chars, non-loopvar) | — | — | {len(short_idents)} | {delta} |")
    out.append(f"| Long identifiers (>30 chars) | — | — | {len(long_idents)} | {delta} |")
    # Fog: classify by deviation from median
    high_fog = [s for s in data["fog_md"] + data["fog_rust"] if s["fog"] > fog_median + 3]
    crit_fog = [s for s in data["fog_md"] + data["fog_rust"] if s["fog"] > fog_median + 5]
    out.append(f"| Fog index (paragraphs, median={fog_median:.1f}; warn >+3, crit >+5) | "
               f"{len(data['fog_md']) + len(data['fog_rust']) - len(high_fog)} | "
               f"{len(high_fog) - len(crit_fog)} | {len(crit_fog)} | {delta} |")
    out.append("")
    out.append("*Δ column is zeroed — this is the first report; future runs will diff against the most recent prior entry under `docs/quality/`.*")
    out.append("")

    # ─── Top offenders ──────────────────────────────────────────────

    out.append("## Top offenders")
    out.append("")
    out.append("Top 5 per metric. Format: `<file>:<line> <symbol> — <metric> = <value>`.")
    out.append("")

    out.append("### Fan-out")
    out.append("")
    for f in sorted(fns, key=lambda x: -x["fanout"])[:5]:
        if f["fanout"] == 0:
            break
        out.append(f"- `{strip_path(f['file'])}`:{f['start']} `{f['name']}` — fan-out = {f['fanout']}")
    out.append("")

    out.append("### Fan-in (informational — widely-used helpers)")
    out.append("")
    # Filter out idiomatic names (new/len/name/iter/…) that name-based
    # fan-in can't distinguish across types. Keep everything else.
    fanin_sorted = sorted(
        [(n, fi) for n, fi in {f["name"]: f["fanin"] for f in fns}.items()
         if n not in FANIN_NOISE],
        key=lambda x: -x[1],
    )[:10]
    for name, fi in fanin_sorted:
        if fi == 0:
            break
        out.append(f"- `{name}` — fan-in = {fi}")
    out.append("")

    out.append("### Cyclomatic complexity")
    out.append("")
    for f in sorted(fns, key=lambda x: -x["ccn"])[:5]:
        out.append(f"- `{strip_path(f['file'])}`:{f['start']} `{f['name']}` — CCN = {f['ccn']}")
    out.append("")

    out.append("### Function length (NLOC)")
    out.append("")
    for f in sorted(fns, key=lambda x: -x["nloc"])[:5]:
        out.append(f"- `{strip_path(f['file'])}`:{f['start']} `{f['name']}` — NLOC = {f['nloc']}")
    out.append("")

    out.append("### File length (SLOC)")
    out.append("")
    for path, sloc, _ in sorted(file_slocs, key=lambda x: -x[1])[:5]:
        out.append(f"- `{path}` — SLOC = {sloc}")
    out.append("")

    out.append("### Nesting depth")
    out.append("")
    for f in sorted(fns, key=lambda x: -x["nesting"])[:5]:
        out.append(f"- `{strip_path(f['file'])}`:{f['start']} `{f['name']}` — depth = {f['nesting']}")
    out.append("")

    out.append("### Identifier length (short, <3 chars)")
    out.append("")
    for r in sorted(short_idents, key=lambda x: x["len"])[:5]:
        out.append(f"- `{r['file']}`:{r['line']} `{r['name']}` — length = {r['len']}")
    out.append("")

    out.append("### Identifier length (long, >30 chars)")
    out.append("")
    for r in sorted(long_idents, key=lambda x: -x["len"])[:5]:
        out.append(f"- `{r['file']}`:{r['line']} `{r['name']}` — length = {r['len']}")
    out.append("")

    out.append(f"### Fog index (documentation paragraphs, >{fog_median + 3:.1f})")
    out.append("")
    for s in sorted(high_fog, key=lambda x: -x["fog"])[:5]:
        out.append(f"- `{s['path']}` — fog = {s['fog']:.1f} ({s['words']} words)")
    out.append("")

    out.append("### DIT")
    out.append("")
    dits = [(f, n, d) for f, n, d in data["py_dits"]] \
           + [("<rust trait>", n, d) for n, d in data["rust_dits"] if d > 0]
    for path, name, d in sorted(dits, key=lambda x: -x[2])[:5]:
        out.append(f"- `{strip_path(path)}` `{name}` — depth = {d}")
    out.append("")

    out.append("### WMC")
    out.append("")
    wmc_rows = [(f, t, v) for f, t, v in data["wmc_rust"]] \
               + [(f, t, v) for f, t, v in data["wmc_python"]]
    for path, ty, v in sorted(wmc_rows, key=lambda x: -x[2])[:5]:
        out.append(f"- `{path}` `{ty}` — WMC = {v}")
    out.append("")

    out.append("### Overloaded methods (informational)")
    out.append("")
    for ty, name, n in data["rust_overload"][:10]:
        out.append(f"- `{ty}::{name}` — {n} impl/trait definitions")
    out.append("")

    # ─── Repeat / New / Resolved ────────────────────────────────────

    out.append("## Repeat offenders (present in baseline)")
    out.append("")
    out.append("*No baseline to compare against — first report.*")
    out.append("")
    out.append("## New since baseline")
    out.append("")
    out.append("*No baseline to compare against — first report.*")
    out.append("")
    out.append("## Resolved since baseline")
    out.append("")
    out.append("*No baseline to compare against — first report.*")
    out.append("")

    # ─── Methodology ────────────────────────────────────────────────

    out.append("## Methodology")
    out.append("")
    out.append(f"- **Tools used:** `lizard 1.21.6` (CCN, NLOC, per-function metrics); "
               f"`textstat 0.7.13` (Gunning fog); Python `ast` module (Python DIT, WMC, "
               f"identifier extraction); regex-based Rust walks (trait graph, impl block "
               f"detection, identifier extraction, fan-out). `scc`/`tokei` not installed — "
               f"file SLOC computed by excluding blank + pure-comment lines.")
    out.append(f"- **Files scanned:** {data['n_rust_files']} Rust · {data['n_py_files']} Python")
    out.append("- **Skipped:** `target/`, `__pycache__/`, `generated/`, `archive.zip`, "
               "`data/`, `input/`, `steps/`. Changelog Markdown files excluded from Fog "
               "(historical entries skew the median).")
    out.append(f"- **Functions analysed:** {len(fns)} (lizard parse)")
    out.append(
        "- **Fan-in caveat:** name-based matching (no type resolution) — "
        "idiomatic names like `new`, `len`, `iter` are filtered from the "
        "informational top-10 list. Cross-type collisions are still possible "
        "for non-idiomatic names; treat the list as a hint, not a proof."
    )
    out.append("")

    # ─── Notes ──────────────────────────────────────────────────────

    out.append("## Notes")
    out.append("")
    notes = []

    # Highlight: any single critical CCN / NLOC / nesting
    top_ccn = max((f["ccn"] for f in fns), default=0)
    top_nloc = max((f["nloc"] for f in fns), default=0)
    top_nest = max((f["nesting"] for f in fns), default=0)
    top_fanout = max((f["fanout"] for f in fns), default=0)
    top_file = max((s for _, s, _ in file_slocs), default=0)

    notes.append(
        f"- **Ceiling values observed this run:** CCN={top_ccn}, NLOC={top_nloc}, "
        f"nesting={top_nest}, fan-out={top_fanout}, file SLOC={top_file}."
    )

    # Rust overload highlight — anything > 5 is notable
    big_overloads = [(t, n, c) for t, n, c in data["rust_overload"] if c >= 5][:5]
    if big_overloads:
        bits = ", ".join(f"`{t}::{n}`={c}" for t, n, c in big_overloads)
        notes.append(f"- **High-overload Rust methods (≥5 impls):** {bits}. Often legitimate — trait impls on a single type — but worth a look to confirm.")

    # Fan-in hotspots
    top_fanin = [(n, fi) for n, fi in fanin_sorted if fi >= 20][:5]
    if top_fanin:
        bits = ", ".join(f"`{n}`={fi}" for n, fi in top_fanin)
        notes.append(f"- **High fan-in helpers:** {bits}. These are likely utility functions or common names; a rename collision check is worth doing if one of them was recently introduced.")

    # Identifier warnings
    if len(short_idents) > 50:
        notes.append(f"- **{len(short_idents)} short identifiers flagged.** Many will be field names, local vars, or generic parameters; review the offender list to decide which are worth renaming.")
    if len(long_idents) > 20:
        notes.append(f"- **{len(long_idents)} long identifiers flagged.** Many will be test-function names (`foo_matches_design_note_...`) which are intentional and readable in isolation.")

    out.extend(notes)
    out.append("")

    out_file.write_text("\n".join(out))
    print(f"Wrote {out_file} ({len(out)} lines, {out_file.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
