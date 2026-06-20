"""Leakage METRICS for the Nature HSiKAN audit paper — two scalars from the label-shuffle audit.

The figures show the audit; this tabulates it. For each (method, dataset), over the 5 seeds:
  - real     = AUROC on true labels
  - shuffle  = AUROC under train-only label shuffle (strict; topo variant where available)
  - L  (leak residual) = max(0, shuffle - 0.5)   -- test-sign info SURVIVING the shuffle = direct
                                                    leakage. Clean -> ~0; leaking -> > ~0.05.
  - Δ  (audit drop)    = real - shuffle           -- how much of the score was shuffle-destructible.

Emits markdown + Springer-ready LaTeX tables (booktabs) to signedkan_wip/paper/tables/. No experiments
-- pure aggregation of the committed JSONL. Run:
    uv run python -m signedkan_wip.src.paperkit.build_leakage_metric_table
    uv run python -m signedkan_wip.src.paperkit.build_leakage_metric_table --self-check
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

_RESULTS = Path("signedkan_wip/experiments/results")
_OUT = Path("signedkan_wip/paper/tables")
_STRICT = _RESULTS / "no_leak_baselines.jsonl"
_TOPO = _RESULTS / "no_leak_baselines_topo.jsonl"
_CYCLE = _RESULTS / "cycle_reachability_grid.jsonl"
_T95_N5 = 2.776            # Student-t, 95%, 4 dof (n=5 seeds)
_CHANCE = 0.5
_LEAK_THRESHOLD = 0.55     # paper convention: shuffle AUROC above this = residual leak


@dataclass(frozen=True)
class Agg:
    """Mean + 95% CI half-width + n over the seed samples of one (key, shuffle) cell."""

    mean: float
    ci: float
    n: int


def _agg(vals: list[float]) -> Agg:
    n = len(vals)
    if n == 0:
        return Agg(float("nan"), float("nan"), 0)
    mean = sum(vals) / n
    if n == 1:
        return Agg(mean, 0.0, 1)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return Agg(mean, _T95_N5 * sem, n)


def _load(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _seed_means(rows: list[dict], key_field: str) -> dict[tuple[str, str, bool], list[float]]:
    """(key, dataset, shuffle) -> list of per-seed AUROCs. Dedup by (key,dataset,seed,shuffle) keeping
    the LAST row (the topo file re-ran seed 0, so last-wins avoids double-weighting it)."""
    last: dict[tuple[str, str, int, bool], float] = {}
    for r in rows:
        last[(r[key_field], r["dataset"], int(r["seed"]), bool(r["shuffle"]))] = float(r["auc"])
    out: dict[tuple[str, str, bool], list[float]] = {}
    for (k, ds, _seed, sh), auc in last.items():
        out.setdefault((k, ds, sh), []).append(auc)
    return out


@dataclass(frozen=True)
class Cell:
    real: Agg
    shuf: Agg

    @property
    def leak(self) -> float:                       # L = residual leak above chance
        return max(0.0, self.shuf.mean - _CHANCE)

    @property
    def drop(self) -> float:                        # Δ = audit drop
        return self.real.mean - self.shuf.mean

    @property
    def leaks(self) -> bool:
        return self.shuf.mean > _LEAK_THRESHOLD


def _cells(path: Path, key_field: str) -> dict[tuple[str, str], Cell]:
    sm = _seed_means(_load(path), key_field)
    keys = sorted({(k, ds) for (k, ds, _sh) in sm})
    out: dict[tuple[str, str], Cell] = {}
    for k, ds in keys:
        real = _agg(sm.get((k, ds, False), []))
        shuf = _agg(sm.get((k, ds, True), []))
        out[(k, ds)] = Cell(real, shuf)
    return out


def _fmt(a: Agg) -> str:
    return f"{a.mean:.3f}±{a.ci:.3f}" if a.n > 1 else f"{a.mean:.3f}"


def build_baseline_tables() -> tuple[str, str]:
    """Per-method leakage summary (mean over datasets) + per (method,dataset) breakdown. Returns
    (markdown, latex)."""
    strict = _cells(_STRICT, "model")
    topo = _cells(_TOPO, "model")
    methods = sorted({m for (m, _d) in strict})
    datasets = sorted({d for (_m, d) in strict})

    # --- summary: per method, means across datasets ---
    md = ["## Leakage metrics — baselines (5-seed, mean over datasets)",
          "",
          "`L = max(0, shuffle−0.5)` (residual leak), `Δ = real − shuffle` (audit drop). "
          "`L_topo` uses the topology-shuffle variant; `L≈L_topo≈0` ⇒ no structural leak.",
          "",
          "| method | real | shuffle (strict) | L | Δ | L_topo |",
          "|---|---|---|---|---|---|"]
    lx = [r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"method & real & shuffle & $L$ & $\Delta$ & $L_{\text{topo}}$ \\", r"\midrule"]
    for m in methods:
        real = _agg([strict[(m, d)].real.mean for d in datasets])
        shuf = _agg([strict[(m, d)].shuf.mean for d in datasets])
        ltopo = max(0.0, sum(topo[(m, d)].shuf.mean for d in datasets) / len(datasets) - _CHANCE) \
            if all((m, d) in topo for d in datasets) else float("nan")
        lk = max(0.0, shuf.mean - _CHANCE)
        dr = real.mean - shuf.mean
        md.append(f"| {m} | {_fmt(real)} | {_fmt(shuf)} | {lk:.3f} | {dr:.3f} | {ltopo:.3f} |")
        lx.append(f"{m} & {real.mean:.3f} & {shuf.mean:.3f} & {lk:.3f} & {dr:.3f} & {ltopo:.3f} "
                  r"\\")
    lx += [r"\bottomrule", r"\end{tabular}"]

    # --- per-dataset breakdown ---
    md += ["", "## Leakage metrics — by method × dataset", "",
           "| method | dataset | real | shuffle | L | Δ | leaks? |",
           "|---|---|---|---|---|---|---|"]
    for m in methods:
        for d in datasets:
            c = strict[(m, d)]
            md.append(f"| {m} | {d} | {_fmt(c.real)} | {_fmt(c.shuf)} | {c.leak:.3f} | "
                      f"{c.drop:.3f} | {'**yes**' if c.leaks else 'no'} |")
    return "\n".join(md) + "\n", "\n".join(lx) + "\n"


def build_cycle_table() -> str:
    """The cycle/HSiKAN method across the reachability lattice (strict/topo/full)."""
    cells = _cells(_CYCLE, "rule")
    rules = ["strict", "topo", "full"]
    datasets = sorted({d for (_r, d) in cells})
    md = ["## Leakage metrics — cycle/HSiKAN method across the reachability lattice",
          "",
          "Leakage (`L > 0`) appears only at `full` (held-out sign reachable); `topo` keeps real "
          "signal but shuffles to chance ⇒ cycle topology is a clean feature.",
          "",
          "| rule | dataset | real | shuffle | L | Δ | leaks? |",
          "|---|---|---|---|---|---|---|"]
    for rule in rules:
        for d in datasets:
            if (rule, d) not in cells:
                continue
            c = cells[(rule, d)]
            md.append(f"| {rule} | {d} | {_fmt(c.real)} | {_fmt(c.shuf)} | {c.leak:.3f} | "
                      f"{c.drop:.3f} | {'**yes**' if c.leaks else 'no'} |")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-check", action="store_true", help="synthetic-data sanity check, no I/O")
    a = ap.parse_args(argv)
    if a.self_check:
        # real 0.9, shuffle 0.6 over 3 seeds → L=0.1, Δ=0.3, leaks (>0.55)
        c = Cell(_agg([0.9, 0.9, 0.9]), _agg([0.6, 0.6, 0.6]))
        assert abs(c.leak - 0.1) < 1e-9 and abs(c.drop - 0.3) < 1e-9 and c.leaks
        clean = Cell(_agg([0.85, 0.85]), _agg([0.50, 0.50]))
        assert clean.leak == 0.0 and not clean.leaks
        print("self-check OK")
        return 0

    base_md, base_tex = build_baseline_tables()
    cyc_md = build_cycle_table()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "leakage_metrics.md").write_text(base_md + "\n" + cyc_md, encoding="utf-8")
    (_OUT / "leakage_metrics_baselines.tex").write_text(base_tex, encoding="utf-8")
    print(f"wrote {_OUT / 'leakage_metrics.md'} and {_OUT / 'leakage_metrics_baselines.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
