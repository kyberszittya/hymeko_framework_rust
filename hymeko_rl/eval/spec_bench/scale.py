"""Scaled refinement — temporal-form selection over a large candidate pool via a coverage P-graph (our crate).

Conjunct-pruning (which signals) is one axis; this adds the second: for each kept signal, which **temporal form**
(`F`, `G`, late-window `G[0,4]`) and comparison. The pool is `signals × {>=,<=} × {F,G,G[0,4]}` — large — and the
task needs a genuine conjunction with a temporal component, so single primitives and wrong temporal wrappers fail
(qwen's `G(F(...))` failure mode). We express it as a **coverage** P-graph: each *aspect* (signal) needs one of its
temporal-variant units to produce its evidence, and `success` consumes all aspect evidences. SSG on our
``hymeko_pgraph`` enumerates the feasible structures (one variant per aspect — a genuinely combinatorial `V^A`
space); we calibrate + F1-rank them. This is where the P-graph does real combinatorial work, not a powerset.

Ground truth: ``F(in_place >= 0.9) AND G[0,4](obj_to_target <= 0.1)`` — placed at some point AND settled over the
final steps.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.eval.spec_bench.pgraph_refine import solve_pgraph, solve_ssg
from hymeko_rl.eval.spec_bench.spec_bench import Rollout, calibrate_thresholds, formula_f1

_SIGNALS = ("in_place", "obj_to_target", "near_object", "grasp_success")
_LATE = 4                                   # G[0,4] = the final 5 steps (HTL past-window convention)
# temporal-form × comparison variants, with a seed threshold {t} calibration will fit.
_VARIANT_TEMPLATES = ("F({s} {c} {t})", "G({s} {c} {t})", f"G[0,{_LATE}]({{s}} {{c}} {{t}})")
_CMPS = (">=", "<=")


def synth_conj_temporal(n: int, *, seed: int, steps: int = 20, noise: float = 0.05) -> list[Rollout]:
    """Balanced (~50%) rollouts whose success is ``F(in_place>=0.9) AND G[0,4](obj_to_target<=0.1)`` — placed at
    some point AND settled over the FINAL steps. The load-bearing negative is **touch-then-drift** (kind 2): the
    object briefly reaches the target mid-episode (so ``F(obj<=0.1)`` is TRUE) but drifts away by the end (so the
    late-window ``G[0,4](obj<=0.1)`` is FALSE) — this is what makes the temporal form necessary. ``near_object``
    correlates; ``grasp_success`` is noise."""
    if n < 4:
        raise ValueError("need n >= 4")
    rng = np.random.default_rng(seed)
    out: list[Rollout] = []
    for k in range(n):
        positive = k % 2 == 0
        kind = 0 if positive else 1 + (k // 2) % 3       # 1 no-place · 2 touch-then-drift · 3 never-settle
        placed = kind != 1
        peak = rng.uniform(0.94, 1.05) if placed else rng.uniform(0.3, 0.82)
        trace: list[dict[str, float]] = []
        for st in range(steps):
            frac = st / max(1, steps - 1)
            late = st >= steps - 1 - _LATE
            mid = 8 <= st <= 12
            if kind in (0, 1):                            # settled: low and stays low in the late window
                ott = rng.uniform(0.0, 0.05) if late else float(np.clip(0.9 - frac, 0.0, 1.0))
            elif kind == 2:                               # touch-then-drift: dips low mid, high at the end
                ott = 0.03 if mid else (rng.uniform(0.25, 0.4) if late else float(np.clip(0.9 - frac, 0.0, 1.0)))
            else:                                         # never settles
                ott = float(np.clip(0.6 - 0.2 * frac, 0.2, 1.0))
            trace.append({
                "in_place": float(np.clip(peak * frac + rng.normal(0, noise), 0.0, 1.2)),
                "obj_to_target": float(np.clip(ott + rng.normal(0, noise * 0.4), 0.0, 1.5)),
                "near_object": float(np.clip(0.5 + 0.4 * frac + rng.normal(0, noise), 0.0, 1.0)),
                "grasp_success": float(rng.random() < 0.6),
            })
        out.append(Rollout(trace=trace, success=positive))
    return out


def synth_single_settle(n: int, *, seed: int, steps: int = 20) -> list[Rollout]:
    """Single-signal task where temporal form is DECISIVE: success = ``G[0,4](obj_to_target<=0.1)`` (settled over
    the final steps). The negative is **touch-then-drift** — the object dips to target mid-episode (so
    ``F(obj<=0.1)`` is TRUE, a false accept) but drifts away by the end (``G[0,4]`` FALSE). No second signal, so
    ``F(A AND B)`` cannot rescue ``F`` — the late-window ``G`` is genuinely required. Balanced ~50%."""
    if n < 2:
        raise ValueError("need n >= 2")
    rng = np.random.default_rng(seed)
    out: list[Rollout] = []
    for k in range(n):
        pos = k % 2 == 0
        trace: list[dict[str, float]] = []
        for st in range(steps):
            late, mid = st >= steps - 1 - _LATE, 8 <= st <= 12
            if pos:
                ott = rng.uniform(0.0, 0.05) if late else float(np.clip(0.8 - st / steps, 0.0, 1.0))
            else:
                ott = 0.03 if mid else (rng.uniform(0.25, 0.4) if late else float(np.clip(0.8 - st / steps, 0.0, 1.0)))
            trace.append({"obj_to_target": float(np.clip(ott + rng.normal(0, 0.02), 0.0, 1.5))})
        out.append(Rollout(trace=trace, success=pos))
    return out


def temporal_variants(signal: str, *, seed_threshold: float = 0.5) -> list[str]:
    """All `{F,G,G[0,4]} × {>=,<=}` variant formulas for ``signal`` (seed threshold; calibrated downstream)."""
    return [tpl.format(s=signal, c=c, t=seed_threshold) for c in _CMPS for tpl in _VARIANT_TEMPLATES]


def coverage_pgraph_hymeko(aspects: list[str], name: str = "spec", *, costs: "dict[str, float] | None" = None,
                           ) -> "tuple[str, dict[str, str]]":
    """A coverage P-graph: each aspect's temporal variants are alternative producers of ``<aspect>_ok``; ``success``
    consumes every ``<aspect>_ok``. Per-unit ``costs`` (default 1.0) become the ``@U <unit> COST`` field that ABB
    minimises. Returns the ``.hymeko`` source + a map ``unit-name -> variant formula``."""
    costs = costs or {}
    lines = [f"{name} {{}}", "", "context", "{"]
    lines += [f"    {s} <material, raw>;" for s in aspects]
    lines += [f"    {s}_ok <material>;" for s in aspects]
    lines.append("    success <material, product>;")
    unit_formula: dict[str, str] = {}
    for ai, asp in enumerate(aspects):
        for vi, formula in enumerate(temporal_variants(asp)):
            u = f"U{ai}_{vi}"
            unit_formula[u] = formula
            lines.append(f"    @{u} <unit> {costs.get(u, 1.0):g} {{ (-{asp}, +{asp}_ok); }}")
    lines.append(f"    @SUCCESS <unit> 0.0 {{ ({', '.join(f'-{s}_ok' for s in aspects)}, +success); }}")
    lines.append("}")
    return "\n".join(lines), unit_formula


def _variant_costs(unit_formula: dict[str, str], verif: list[Rollout]) -> dict[str, float]:
    """Per-unit **cost = anti-faithfulness** = ``1 - F1`` of that (calibrated) variant on ``verif`` — so ABB's
    cost-minimisation picks the most faithful variant per aspect (and its branch-and-bound prunes the rest)."""
    return {u: round(max(0.0, 1.0 - formula_f1(calibrate_thresholds(f, verif), verif)), 4)
            for u, f in unit_formula.items()}


def refine_scaled_abb(aspects: list[str], verif: list[Rollout]) -> "tuple[str | None, dict[str, object]]":
    """Axis-2: solve the coverage P-graph with **ABB** under ``cost = anti-F1`` — the cost-optimal structure is the
    most-faithful variant per aspect, found by branch-and-bound (not full enumeration). Returns
    ``(refined_formula, abb_stats)`` where stats include ``explored`` / ``pruned_*`` (the bounding that bit)."""
    if not aspects:
        return None, {}
    _, unit_formula = coverage_pgraph_hymeko(aspects)
    costs = _variant_costs(unit_formula, verif)
    src, _ = coverage_pgraph_hymeko(aspects, costs=costs)
    data = solve_pgraph(src, algorithm="abb")
    abb = data.get("abb") if data else None
    if not isinstance(abb, dict):                       # binary/solve unavailable → SSG fallback
        return refine_scaled(aspects, verif), {"fallback": True}
    formulas = [unit_formula[u] for u in abb.get("units", []) if u in unit_formula]
    if not formulas:
        return None, dict(abb)
    spec = formulas[0] if len(formulas) == 1 else "(" + " AND ".join(formulas) + ")"
    stats = {k: abb.get(k) for k in ("cost", "explored", "pruned_by_inclusion", "pruned_by_reachability")}
    return calibrate_thresholds(spec, verif), stats


def refine_scaled(aspects: list[str], verif: list[Rollout], *, max_structures: int = 400) -> "str | None":
    """Solve the coverage P-graph (SSG on our crate), build each structure's conjunction (one variant/aspect),
    calibrate + F1-rank. Returns the best temporal-form conjunction, or ``None`` if no aspect given."""
    if not aspects:
        return None
    src, unit_formula = coverage_pgraph_hymeko(aspects)
    structures = solve_ssg(src)
    if structures is None:                  # fallback: the cartesian product = one variant per aspect
        import itertools
        per = [[u for u in unit_formula if u.startswith(f"U{ai}_")] for ai in range(len(aspects))]
        structures = [list(combo) for combo in itertools.product(*per)]
    # keep MINIMAL structures — exactly one variant unit per aspect (a feasible structure with
    # `len(aspects)` variant units must cover each aspect once); drops redundant supersets SSG also returns.
    na = len(aspects)
    minimal = [s for s in structures if sum(1 for u in s if u in unit_formula) == na]
    best, best_f1 = None, -1.0
    for struct in minimal[:max_structures]:
        formulas = [unit_formula[u] for u in struct if u in unit_formula]
        spec = formulas[0] if len(formulas) == 1 else "(" + " AND ".join(formulas) + ")"
        cal = calibrate_thresholds(spec, verif)
        f1 = formula_f1(cal, verif)
        if f1 > best_f1:
            best, best_f1 = cal, f1
    return best
