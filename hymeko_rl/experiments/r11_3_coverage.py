"""R11.3 bank coverage analysis (Phase 9): read the demonstration bank JSONL, emit a coverage report (.md + .json) and
visualizations (coin/target coverage map, failure taxonomy, planning-time + min_dtz histograms, positive actuator work).

Reports only what the bank measures — no Hamiltonian-optimality or energy-conservation claims (energy is measured:
ENERGY_LEDGER_COMPLETE / ENERGY_BALANCE_RESIDUAL_RECORDED only). Reads raw JSON so it is decoupled from the record class.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend selection)

Rec = dict[str, Any]


def load(path: Path) -> list[Rec]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _partition(records: list[Rec]) -> "tuple[list[Rec], list[Rec]]":
    """(admissible non-invalid rollouts, accepted = those with an executed reach)."""
    admissible = [r for r in records if r["admissible"] and r["split"] != "invalid"]
    accepted = [r for r in admissible if r["reach"] is not None]
    return admissible, accepted


def _best_per_scenario(admissible: list[Rec]) -> dict[str, Rec]:
    """The preferred record per scenario: the min-teacher-seed K6 if any, else the min-dtz attempt (never overwritten)."""
    by: dict[str, list[Rec]] = defaultdict(list)
    for r in admissible:
        by[r["scenario_id"]].append(r)
    best = {}
    for sid, rs in by.items():
        k6s = [r for r in rs if r["k6"]]
        best[sid] = (min(k6s, key=lambda r: r["teacher_seed"]) if k6s
                     else min(rs, key=lambda r: (r["min_dtz_mm"] if r["min_dtz_mm"] is not None else 1e9)))
    return best


def _stage_coverage(best: dict[str, Rec]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_stage: dict[str, list[Rec]] = defaultdict(list)
    for r in best.values():
        by_stage[r["curriculum_stage"]].append(r)
    for stage, rs in sorted(by_stage.items()):
        k6 = [r for r in rs if r["k6"]]
        out[stage] = {"scenarios": len(rs), "k6": len(k6), "k6_rate": round(len(k6) / len(rs), 3) if rs else 0.0}
    return out


def _headline(records: list[Rec], rejected: list[Rec], best: dict[str, Rec]) -> dict[str, Any]:
    k6 = sum(1 for r in best.values() if r["k6"])
    return {"total_attempts": len(records), "admissible_scenarios": len(best), "rejected_attempts": len(rejected),
            "rejection_reasons": dict(Counter(r["rejection_reason"] for r in rejected)), "k6_scenarios": k6,
            "k6_rate": round(k6 / len(best), 3) if best else 0.0}


def _taxonomy(admissible: list[Rec], best: dict[str, Rec]) -> dict[str, Any]:
    return {"stage_coverage": _stage_coverage(best),
            "failure_class_counts_best": dict(Counter(r["outcome_label"] for r in best.values() if not r["k6"])),
            "failure_class_counts_all_attempts": dict(Counter(r["outcome_label"] for r in admissible if not r["k6"])),
            "split_counts": dict(Counter(r["split"] for r in best.values())),
            "scenarios_without_positive_demo": sorted(sid for sid, r in best.items() if not r["k6"])}


def _reach_energy(admissible: list[Rec], accepted: list[Rec]) -> dict[str, Any]:
    return {
        "reach_found_fraction": round(sum(r["reach_found"] for r in admissible) / len(admissible), 3)
        if admissible else 0.0,
        "precontact_motion_max_mm": max((r["reach"]["coin_moved_before_capture_mm"] for r in accepted), default=0.0),
        "premature_contacts_total": sum(r["premature_contacts"] for r in accepted),
        "energy_ledger_complete_fraction": round(sum(r["energy_measurement_complete"] for r in accepted) / len(accepted), 3)
        if accepted else 0.0}


def _summ(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "min": _r(min(values)), "median": _r(float(np.median(values))), "max": _r(max(values))} \
        if values else {}


def _distributions(best: dict[str, Rec], accepted: list[Rec]) -> dict[str, Any]:
    dtz = [r["min_dtz_mm"] for r in best.values() if r["min_dtz_mm"] is not None]
    wpos = [r["energy_ledger"]["w_actuator_pos"] for r in accepted if r["energy_ledger"]]
    return {"min_dtz_mm": _summ(dtz), "planning_time_s": _summ([r["planning_time_s"] for r in accepted]),
            "w_actuator_pos": _summ(wpos)}


def metrics(records: list[Rec]) -> dict[str, Any]:
    admissible, accepted = _partition(records)
    rejected = [r for r in records if not r["admissible"]]
    best = _best_per_scenario(admissible)
    claims = {"energy_claims": ["ENERGY_LEDGER_COMPLETE", "ENERGY_BALANCE_RESIDUAL_RECORDED"],
              "non_claims": ["NO_HAMILTONIAN_OPTIMALITY_CLAIM", "NO_ENERGY_CONSERVATION_CLAIM",
                             "NO_TEACHER_FREE_DEPLOYMENT_CLAIM"]}
    return {**_headline(records, rejected, best), **_taxonomy(admissible, best),
            **_reach_energy(admissible, accepted), **_distributions(best, accepted), **claims}


def _r(x: "float | None") -> "float | None":
    return None if x is None else round(float(x), 3)


def _plot_inputs(records: list[Rec]) -> "tuple[list[Rec], list[float], list[float], list[float]]":
    admissible, accepted = _partition(records)
    best = list(_best_per_scenario(admissible).values())
    dtz = [r["min_dtz_mm"] for r in best if r["min_dtz_mm"] is not None]
    ptime = [r["planning_time_s"] for r in accepted]
    wpos = [r["energy_ledger"]["w_actuator_pos"] for r in accepted if r["energy_ledger"]]
    return best, dtz, ptime, wpos


def plots(records: list[Rec], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    best, dtz, ptime, wpos = _plot_inputs(records)
    names = [
        _plot_coverage_map(best, out_dir),
        _plot_failure_taxonomy(best, out_dir),
        _plot_hist(dtz, "min_dtz (mm)", "min_dtz distribution (best per scenario)", out_dir / "min_dtz_hist.png"),
        _plot_hist(ptime, "planning time (s)", "RRT planning-time distribution", out_dir / "planning_time_hist.png"),
        _plot_hist(wpos, "W+ (J-proxy)", "positive actuator work (measured proxy)", out_dir / "w_pos_hist.png"),
    ]
    return [n for n in names if n]


def _plot_coverage_map(best: list[Rec], out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in best:
        c, t = r["coin_pose"], r["target_pose"] or r["zone_pose"]
        col = "green" if r["k6"] else "red"
        ax.plot([c[0], t[0]], [c[1], t[1]], color=col, alpha=0.35, lw=0.8, zorder=1)
        ax.scatter(c[0], c[1], c=col, s=28, zorder=2)
        ax.scatter(t[0], t[1], c=col, marker="x", s=28, zorder=2)
    ax.set_title("coin(o)->target(x) coverage: green=K6, red=fail")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    path = out_dir / "coverage_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _plot_failure_taxonomy(best: list[Rec], out_dir: Path) -> str:
    counts = Counter(r["outcome_label"] for r in best)
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = sorted(counts, key=lambda k: counts[k])
    ax.barh(keys, [counts[k] for k in keys], color="steelblue")
    ax.set_title("outcome / failure taxonomy (best per scenario)")
    ax.set_xlabel("scenarios")
    path = out_dir / "failure_taxonomy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _plot_hist(values: list[float], xlabel: str, title: str, path: Path) -> str:
    if not values:
        return ""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(values, bins=min(20, max(5, len(values))), color="slateblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def write_report(m: dict[str, Any], plot_names: list[str], md_path: Path, json_path: Path, bank_dir: str) -> None:
    json_path.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# R11.3 coin/target demonstration bank — coverage analysis\n",
             f"**Bank:** `{bank_dir}/bank.jsonl` · generation-only (no BC/RL) · CEM = training teacher.\n",
             "## Headline",
             f"- Admissible scenarios: **{m['admissible_scenarios']}**; teacher attempts: **{m['total_attempts']}**; "
             f"rejected: **{m['rejected_attempts']}** ({m['rejection_reasons']}).",
             f"- Strict-K6 demonstrations: **{m['k6_scenarios']}/{m['admissible_scenarios']}** "
             f"(rate **{m['k6_rate']}**).",
             f"- Precontact coin motion (max): **{m['precontact_motion_max_mm']} mm**; premature contacts: "
             f"**{m['premature_contacts_total']}**; energy-ledger complete: **{m['energy_ledger_complete_fraction']}**.\n",
             "## Coverage by curriculum stage",
             "| stage | scenarios | K6 | K6 rate |", "|---|---|---|---|"]
    lines += [f"| {s} | {d['scenarios']} | {d['k6']} | {d['k6_rate']} |" for s, d in m["stage_coverage"].items()]
    lines += ["\n## Failure taxonomy (best per scenario)",
              "".join(f"\n- `{k}`: {v}" for k, v in sorted(m["failure_class_counts_best"].items())) or "\n- (none)",
              "\n## Distributions",
              f"- min_dtz (mm): {m['min_dtz_mm']}", f"- planning time (s): {m['planning_time_s']}",
              f"- W+ (measured proxy): {m['w_actuator_pos']}",
              f"\n## Splits\n- {m['split_counts']}",
              f"\n## Scenarios without a positive demonstration ({len(m['scenarios_without_positive_demo'])})",
              ", ".join(m["scenarios_without_positive_demo"]) or "(none)",
              "\n## Visualizations", "".join(f"\n![{n}]({bank_dir}/figures/{n})" for n in plot_names),
              "\n## Claims / non-claims",
              f"- Energy claims: {m['energy_claims']}", f"- Non-claims: {m['non_claims']}"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=Path("reports/2026-07-30-r11-3-coin-target-demo-bank/bank.jsonl"))
    args = ap.parse_args()
    records = load(args.bank)
    m = metrics(records)
    base = args.bank.parent
    plot_names = plots(records, base / "figures")
    write_report(m, plot_names, base.parent / "2026-07-30-r11-3-coin-target-demo-bank.md",
                 base.parent / "2026-07-30-r11-3-coin-target-demo-bank.json", base.name)
    print(f"COVERAGE_DONE k6={m['k6_scenarios']}/{m['admissible_scenarios']} plots={plot_names}", flush=True)


if __name__ == "__main__":
    main()
