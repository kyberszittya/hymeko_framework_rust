"""R11.4B — conditioned delivery BC: assimilate the 49 certified teacher theta-schedules (+7 frozen-R2 anchors) into one
coin/target/handoff-conditioned policy and closed-loop-certify it against frozen R2 and the mandatory simple baselines.

Phases (parallelizable via --offset/--limit; deterministic):
  * ``extract`` — reconstruct each scenario's certified handoff, replay the frozen teacher (R=11) to recover theta, and
    write a ``BcSample`` shard. Anchors search for their first certified+K6 seed. A seed that yields no K6 is OMITTED
    (never a fabricated label) and reported.
  * ``eval``    — fit every policy on the TRAIN split, then for each scenario roll out each policy's predicted theta
    (plus frozen R2 and the teacher-theta sanity) PHYSICALLY from the reconstructed handoff. Every eval worker fits the
    identical models (deterministic), so shards are consistent.
  * ``gate``    — the pre-registered R11.4B1 gate + the negative classifier.

Load-bearing question: can the scenario-specific schedules be assimilated into one policy that delivers strict K6 with NO
per-scenario CEM and NO oracle? RL is NOT started to mask a failed assimilation.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import (
    BcSample,
    bc_context,
    extract_sample,
    fresh_rig,
    reconstruct_capture,
    scenario_by_id,
)
from hymeko_rl.coin_delivery.delivery_bc.evaluate import r2_delivery, rollout_theta
from hymeko_rl.coin_delivery.delivery_bc.models import (
    MeanThetaPolicy,
    MlpBcPolicy,
    NearestSchedulePolicy,
    RidgePolicy,
)
from hymeko_rl.experiments.r11_5_full_coverage import R11_4A_BANK, _scenario_kind

CERT = Path("reports/2026-08-02-r11-5ppp-cert-mac/merged.json")
SIMPLE_BASELINES = ("mean_theta", "nearest_schedule", "ridge")
SPLITS = ("train", "dev", "test")


def entries(cert: Path, bank: Path) -> "list[tuple[str, str, int | None, str]]":
    """The 56 positive scenarios to assimilate: (sid, split, seed|None, source). Recovered carry their certified
    ``selected_seed``; frozen-R2 anchors carry ``None`` (their first certified+K6 seed is searched at extraction)."""
    merged = json.loads(cert.read_text())
    rec: list[tuple[str, str, "int | None", str]] = [
        (r["scenario_id"], r["split"], int(r["selected_seed"]), "recovered")
        for r in merged["rows"] if r.get("recovered")]
    by: dict[str, list] = defaultdict(list)
    for line in bank.open():
        if line.strip():
            r = json.loads(line)
            by[r["scenario_id"]].append(r)
    anchors: list[tuple[str, str, "int | None", str]] = [
        (sid, next(x["split"] for x in rs), None, "anchor")
        for sid, rs in by.items() if (_scenario_kind(sid, rs) or ("",))[0] == "r2"]
    return rec + anchors


def _extract_entry(ctx: tuple, sid: str, split: str, seed: "int | None", source: str,
                   restarts: int, anchor_seeds: int) -> "BcSample | None":
    cfg, conf, obj = ctx
    if seed is not None:
        return extract_sample(cfg, conf, obj, sid, split, seed, source, restarts)
    for s in range(anchor_seeds):                                    # anchor: first certified+K6 seed
        smp = extract_sample(cfg, conf, obj, sid, split, s, source, restarts)
        if smp is not None:
            return smp
    return None


def _load_dataset(dataset_dir: Path) -> "list[BcSample]":
    out: list[BcSample] = []
    for f in sorted(glob.glob(str(dataset_dir / "extract_*.jsonl"))):
        for line in Path(f).open():
            if line.strip():
                d = json.loads(line)
                if d.get("omitted"):
                    continue
                out.append(BcSample.from_json(d))
    return out


def _fit_policies(train: "list[BcSample]") -> list:
    X = np.array([s.x for s in train], np.float64)
    T = np.array([s.theta for s in train], np.float64)
    return [MeanThetaPolicy.fit(X, T), NearestSchedulePolicy.fit(X, T), RidgePolicy.fit(X, T, lam=1.0),
            MlpBcPolicy.fit(X, T)]


def _eval_one(ctx: tuple, policies: list, nn: NearestSchedulePolicy, smp: BcSample) -> dict[str, Any]:
    cfg, conf, obj = ctx
    scen = scenario_by_id(smp.scenario_id)
    rig = fresh_rig()                                              # per-scenario clean rig -> deterministic handoff
    rc = reconstruct_capture(rig, cfg, conf, obj, scen, smp.seed)
    if rc is None:                                                  # reconstruction must be deterministic; flag if not
        return {"scenario_id": smp.scenario_id, "split": smp.split, "error": "reconstruction_failed"}
    snap = rc.result.outcome.snapshot
    x = np.array(smp.x, np.float64)
    row: dict[str, Any] = {"scenario_id": smp.scenario_id, "split": smp.split, "source": smp.source,
                           "nn_distance": nn.nn_distance(x),
                           "teacher": rollout_theta(snap, np.array(smp.theta, np.float64))}
    for pol in policies:
        row[pol.name] = rollout_theta(snap, pol.predict(x))
    row["frozen_r2"] = r2_delivery(snap, rig["down"])
    return row


def _rate(rows: list, policy: str, split: str) -> float:
    sub = [r for r in rows if r["split"] == split and policy in r]
    return round(sum(1 for r in sub if r[policy]["k6"]) / len(sub), 3) if sub else 0.0


def _held(rows: list, policy: str) -> float:
    sub = [r for r in rows if r["split"] in ("dev", "test") and policy in r]
    return round(sum(1 for r in sub if r[policy]["k6"]) / len(sub), 3) if sub else 0.0


def _heldout_nn(rows: list, k6: bool) -> "list[float]":
    """The nearest-train-neighbour distances of the held-out scenarios where the MLP {hit, missed} strict-K6."""
    return [r["nn_distance"] for r in rows
            if r["split"] in ("dev", "test") and "mlp_bc" in r and r["mlp_bc"]["k6"] == k6]


def _classify_negative(rows: list, rates: dict) -> str:
    """Classify a BC miss. The 1-NN memorizer trivially scores train=1.0 (it copies its own theta), so it is NOT a fair
    fit target — the parametric comparator is ridge.
      * OPTIMIZATION — the MLP cannot even fit train as well as closed-form ridge (a genuine fit/tuning bug).
      * REPRESENTATION — the SMOOTH regressors reproduce K6 on <80% of TRAIN (nn distance ~0, in-distribution): the
        descriptor->theta map is not smoothly learnable at all (here: chaotic / narrow-basin teacher theta).
      * DATA_COVERAGE — smooth regression fits train but held-out misses sit far from the train manifold.
    """
    if rates["mlp_bc"]["train"] < rates["ridge"]["train"] - 1e-9:
        return "BC_OPTIMIZATION_FAILURE"
    if rates["ridge"]["train"] < 0.80:
        return "BC_REPRESENTATION_INSUFFICIENT"
    miss, hit = _heldout_nn(rows, k6=False), _heldout_nn(rows, k6=True)
    far = bool(miss) and (not hit or float(np.median(miss)) > float(np.median(hit)))
    return "BC_DATA_COVERAGE_INSUFFICIENT" if far else "BC_REPRESENTATION_INSUFFICIENT"


def _bc_passed(bc: dict, r2: dict, mlp_held: float, best_baseline_held: float, bc_safe: bool) -> bool:
    """The pre-registered PASS condition (factored out to keep ``bc_gate`` within the complexity budget)."""
    return (bc["train"] >= 0.80 and bc["dev"] >= 0.50 and bc["test"] >= 0.50
            and bc["dev"] > r2["dev"] and bc["test"] > r2["test"]
            and mlp_held >= best_baseline_held and bc_safe)


def bc_gate(rows: list) -> dict[str, Any]:
    """Pre-registered R11.4B1 gate: train>=80%, dev>=50%, test>=50%, BC>R2 on dev AND test, BC>=strongest simple
    supervised baseline on held-out; 0 safety regression on BC deliveries."""
    policies = [*SIMPLE_BASELINES, "mlp_bc", "frozen_r2"]
    rates = {p: {sp: _rate(rows, p, sp) for sp in SPLITS} for p in policies}
    held = {p: _held(rows, p) for p in policies}
    best_baseline_held = max(held[b] for b in SIMPLE_BASELINES)
    bc_safe = all(r["mlp_bc"]["safe"] for r in rows if "mlp_bc" in r)
    teacher_k6 = all(r["teacher"]["k6"] for r in rows if "teacher" in r)      # extraction/eval-harness consistency
    passed = _bc_passed(rates["mlp_bc"], rates["frozen_r2"], held["mlp_bc"], best_baseline_held, bc_safe)
    verdict = "R11_4B_CONDITIONED_DELIVERY_BC_PASS" if passed else _classify_negative(rows, rates)
    return {"n": len(rows), "rates": rates, "held_out": held, "best_simple_baseline_held": best_baseline_held,
            "bc_safe": bc_safe, "teacher_theta_reproduces_k6": teacher_k6, "verdict": verdict}


def _write(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _slice(seq: list, offset: int, limit: int) -> list:
    return seq[offset:(offset + limit if limit else len(seq))]


def _run_extract(args: argparse.Namespace) -> None:
    ctx = bc_context()
    ents = _slice(entries(args.cert, args.bank), args.offset, args.limit)
    recs: list[dict] = []
    for i, (sid, split, seed, source) in enumerate(ents, 1):
        smp = _extract_entry(ctx, sid, split, seed, source, args.restarts, args.anchor_seeds)
        recs.append(smp.to_json() if smp else {"omitted": True, "scenario_id": sid, "split": split, "source": source})
        print(f"[{i}/{len(ents)}] {sid:26s} {split:5s} {source:9s} "
              f"{'theta ok dtz=' + str(smp.dtz_mm) if smp else 'OMITTED (no certified K6)'}", flush=True)
    _write(args.dataset_dir / f"extract_{args.offset:03d}.jsonl", recs)


def _run_eval(args: argparse.Namespace) -> None:
    ctx = bc_context()
    samples = _load_dataset(args.dataset_dir)
    policies = _fit_policies([s for s in samples if s.split == "train"])
    nn = next(p for p in policies if isinstance(p, NearestSchedulePolicy))
    rows = [_eval_one(ctx, policies, nn, smp) for smp in _slice(samples, args.offset, args.limit)]
    for r in rows:
        tag = r.get("error") or "  ".join(f"{p}={r[p]['k6']:d}"
                                          for p in (*SIMPLE_BASELINES, "mlp_bc", "frozen_r2") if p in r)
        print(f"{r['scenario_id']:26s} {r['split']:5s} {tag}", flush=True)
    _write(args.eval_dir / f"eval_{args.offset:03d}.jsonl", rows)


def _run_gate(args: argparse.Namespace) -> None:
    rows: list = []
    for f in sorted(glob.glob(str(args.eval_dir / "eval_*.jsonl"))):
        rows += [json.loads(line) for line in Path(f).open() if line.strip()]
    gate = bc_gate([r for r in rows if "error" not in r])
    (args.eval_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)
    print("R11_4B_BC_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("extract", "eval", "gate"), required=True)
    ap.add_argument("--dataset-dir", type=Path, default=Path("reports/2026-08-03-r11-4b-bc/dataset"))
    ap.add_argument("--eval-dir", type=Path, default=Path("reports/2026-08-03-r11-4b-bc/eval"))
    ap.add_argument("--cert", type=Path, default=CERT)
    ap.add_argument("--bank", type=Path, default=R11_4A_BANK)
    ap.add_argument("--restarts", type=int, default=11)
    ap.add_argument("--anchor-seeds", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"extract": _run_extract, "eval": _run_eval, "gate": _run_gate}[args.phase](args)


if __name__ == "__main__":
    main()
