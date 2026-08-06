"""Reduce spec strategy-search + conjunct-pruning to a P-graph solution — on our ``hymeko_pgraph`` crate.

The arbiter's structure-refinement (which of the LLM's candidate predicates to keep) is a *solution-structure
search*: the minimal feasible subset of predicates that produces the success decision. That is exactly P-graph SSG
(Friedler et al.): map **signals → raw materials**, each **candidate predicate → an operating unit** that consumes
its signal and produces the **success** product; ``hymeko_pgraph``'s SSG enumerates the combinatorially-feasible
solution structures, and we rank them by F1 on the verification split. This fixes what threshold calibration cannot
(dropping a *noise-signal* conjunct — the gemma limit): SSG enumerates the subset without it, F1 selects it.

Uses our crate via its CLI (``pgraph solve … --algorithm ssg --json`` on a generated ``.hymeko``) — no external
P-graph library, no re-implementation (§6.1). Falls back to an exhaustive Python subset search if the binary is
unbuilt (numerically identical for small pools; the P-graph's combinatorial/axiom value is at scale).
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
import tempfile
from pathlib import Path

from hymeko_rl.eval.spec_bench.spec_bench import _PRED, Rollout, calibrate_thresholds, formula_f1

_PGRAPH_BIN = Path(__file__).resolve().parents[3] / "target" / "debug" / "pgraph"


def _predicates(formula: str) -> list[tuple[str, str]]:
    """The formula's predicates as ``(text, signal)`` — the candidate operating units / conjuncts."""
    return [(m.group(0), m.group(1)) for m in _PRED.finditer(formula)]


def _outer_op(formula: str) -> str:
    m = re.match(r"\s*([FG])\b", formula)
    return m.group(1) if m else "F"


def predicates_to_pgraph_hymeko(preds: list[tuple[str, str]], name: str = "spec") -> str:
    """Emit the ``.hymeko`` P-graph: signals = raw materials, ``success`` = product, each predicate = a unit
    ``(-signal, +success)``. (The reduction of predicate-subset-selection to Process Network Synthesis.)"""
    signals = sorted({sig for _t, sig in preds})
    lines = [f"{name} {{}}", "", "context", "{"]
    lines += [f"    {s} <material, raw>;" for s in signals]
    lines.append("    success <material, product>;")
    lines += [f"    @P{i} <unit> 0.0 {{ (-{sig}, +success); }}" for i, (_t, sig) in enumerate(preds)]
    lines.append("}")
    return "\n".join(lines)


def solve_pgraph(hymeko_src: str, *, algorithm: str = "ssg", binary: Path = _PGRAPH_BIN,
                 timeout: float = 60.0) -> "dict | None":
    """Run ``hymeko_pgraph solve`` on the ``.hymeko`` source; return the parsed JSON (``ssg_structures`` for ssg,
    ``abb`` for abb), or ``None`` if the binary is unavailable / the solve fails."""
    if not binary.exists():
        return None
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".hymeko", delete=False) as f:
            f.write(hymeko_src)
            path = f.name
        out = subprocess.run([str(binary), "solve", path, "--algorithm", algorithm, "--json"],
                             capture_output=True, text=True, timeout=timeout)
        data = json.loads(out.stdout)
        return data if isinstance(data, dict) else None
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def solve_ssg(hymeko_src: str, *, binary: Path = _PGRAPH_BIN, timeout: float = 60.0) -> "list[list[str]] | None":
    """The feasible SSG solution structures (unit-name subsets), or ``None``."""
    data = solve_pgraph(hymeko_src, algorithm="ssg", binary=binary, timeout=timeout)
    structures = data.get("ssg_structures") if data else None
    return structures if isinstance(structures, list) else None


def _subsets_fallback(n: int) -> list[list[str]]:
    """Exhaustive non-empty subsets as unit names (the Python fallback when the pgraph binary is unbuilt)."""
    return [[f"P{i}" for i in combo]
            for r in range(1, n + 1) for combo in itertools.combinations(range(n), r)]


def refine_via_pgraph(formula: str, verif: list[Rollout], *, calibrate: bool = True) -> str:
    """Refine ``formula`` by reducing conjunct-pruning to a P-graph SSG solution, then calibrating + F1-ranking.

    Single-predicate formulas are just calibrated (nothing to prune). # Postconditions returns a parse-valid
    formula whose verification F1 is ``>=`` the calibrated original's (pruning only helps or is a no-op)."""
    preds = _predicates(formula)
    if len(preds) < 2:
        return calibrate_thresholds(formula, verif) if calibrate else formula
    op = _outer_op(formula)
    structures = solve_ssg(predicates_to_pgraph_hymeko(preds)) or _subsets_fallback(len(preds))

    def build(struct: list[str]) -> str:
        idx = sorted(int(u[1:]) for u in struct if u[1:].isdigit())
        return f"{op}(" + " AND ".join(preds[i][0] for i in idx) + ")"

    candidates = []
    for s in structures:
        spec = build(s)
        candidates.append(calibrate_thresholds(spec, verif) if calibrate else spec)
    calibrated_original = calibrate_thresholds(formula, verif) if calibrate else formula
    candidates.append(calibrated_original)
    return max(candidates, key=lambda f: formula_f1(f, verif))


def greedy_conjunct_select(formula: str, verif: list[Rollout], *, calibrate: bool = True) -> str:
    """Forward-greedy conjunct selection — the O(n²) baseline the P-graph SSG is measured against.

    Seeds with the single best conjunct, then repeatedly adds the remaining conjunct with the largest F1 gain,
    stopping when none improves F1. Unlike the SSG (which enumerates all feasible subsets), greedy can stall in a
    local optimum when a conjunct only helps *in combination*. This is the honest comparator for "does the P-graph
    earn its keep": if greedy ties the SSG everywhere, the P-graph's value is its axioms/scale, not an F1 win.

    # Preconditions ``formula`` parses. # Postconditions parse-valid; F1 >= the best single conjunct's."""
    preds = _predicates(formula)
    if len(preds) < 2:
        return calibrate_thresholds(formula, verif) if calibrate else formula
    op = _outer_op(formula)

    def build(idxs: list[int]) -> str:
        return f"{op}(" + " AND ".join(preds[i][0] for i in sorted(idxs)) + ")"

    def score(idxs: list[int]) -> float:
        spec = build(idxs)
        return formula_f1(calibrate_thresholds(spec, verif) if calibrate else spec, verif)

    remaining = list(range(len(preds)))
    seed = max(remaining, key=lambda i: score([i]))
    chosen = [seed]
    remaining.remove(seed)
    best_f1 = score(chosen)
    while remaining:
        cand = max(remaining, key=lambda i: score([*chosen, i]))
        f1 = score([*chosen, cand])
        if f1 <= best_f1 + 1e-9:
            break
        chosen.append(cand)
        remaining.remove(cand)
        best_f1 = f1
    final = build(chosen)
    return calibrate_thresholds(final, verif) if calibrate else final
