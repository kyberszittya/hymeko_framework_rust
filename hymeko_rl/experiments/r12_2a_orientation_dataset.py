"""R12.2-A2 — orientation-varying transportability dataset (the supervision R12.2-B learns from).

Like the R12.1 dataset (`r12_hsikan1_dataset.py`) but the handoffs VARY object yaw: for the elongated O5-R rectangle,
re-acquire handoffs on a yaw × scenario × seed grid via `reach_capture_at_yaw` (the co-rotating straddle), record the
post-grasp yaw per handoff, and apply the pooled-θ bank to each → (K6, dtz, safe). Every row carries `yaw_deg`
(commanded) and `post_grasp_yaw_deg` (measured) so R12.2-B can add orientation `(R,ω)` to the descriptor and test the
Δ_HSiKAN − Δ_MLP interaction. Only certified handoffs (a stable straddle grasp) contribute rows; the certified yield
per yaw is logged (no silent caps).

Run:  python -m hymeko_rl.experiments.r12_2a_orientation_dataset [family] [n_seeds] [n_scen] [yaws] [bank_path]

θ SOURCE = the ORIENTATION-AWARE bank (`r12_2b_orientation_bank`), NOT the pooled bank. History (2026-08-10): the
pooled bank is axis-aligned-tuned, so it delivers the rotated rectangle ONLY near yaw≈0 (17 positives all at yaw 0,
ZERO at 30/60/90) ⇒ K6 collapsed to a trivial "is-yaw-0" indicator, a DEGENERATE basis for the ranker test. The
feasibility probe (`r12_2b_theta_feasibility`, GO) then showed orientation-specific delivering θ DO exist; the bank
collects them, so applied across handoffs K6 depends non-trivially on orientation×θ. `theta_family` = each θ's
tuning-yaw provenance.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.coin_delivery.r12_orientation import object_yaw, reach_capture_at_yaw
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r12_2b_theta_feasibility import _TARGETS   # the bank's tuning scenarios (θ are target-specific)

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_DEFAULT_YAWS = (0.0, 30.0, 45.0, 60.0, 90.0)     # commanded placement yaws (cert yield uneven; kept for coverage)


def _load_bank(path: str) -> "tuple[np.ndarray, list[str]]":
    """Load the ORIENTATION-AWARE θ bank (`r12_2b_orientation_bank`). Each θ carries the yaw it was tuned for as
    provenance, so `theta_family` labels the θ's home orientation — the substrate R12.2-B needs (a θ delivers near its
    tuning-yaw, fails far from it). Falls back with a loud error if the bank is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"orientation bank not found: {path} — run `r12_2b_orientation_bank` first "
                                "(the pooled bank is degenerate for this dataset, see module docstring)")
    d = json.loads(p.read_text())
    return np.asarray(d["thetas"], np.float64), [f"yaw{int(pr['tuning_yaw'])}" for pr in d["provenance"]]


def _gen(fam: str, thetas: np.ndarray, theta_src: list[str], yaws: tuple[float, ...], scenarios: tuple[str, ...],
         n_seeds: int, cfg: Any, conf: Any, obj: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    rig = _rig(object_spec=variant(fam).object_spec)
    out_path = _OUT / f"orientation_dataset_{fam}.jsonl"
    attempted_path = _OUT / f"orientation_dataset_{fam}.attempted.tsv"
    # RESUME/EXTEND: skip EVERY prior attempt (certified→jsonl, uncertified→attempted sidecar); each new handoff is
    # written+flushed immediately (checkpointed) so a crashed or seed-extended multi-hour run continues without
    # re-acquiring anything already tried.
    _OUT.mkdir(parents=True, exist_ok=True)
    attempted: set[tuple[float, str, int]] = set()
    n_prior = 0
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            r = json.loads(ln)
            attempted.add((float(r["yaw_deg"]), r["scenario"], int(r["seed"])))
            n_prior += 1
    if attempted_path.exists():
        for ln in attempted_path.read_text().splitlines():
            y, sid, sd = ln.split("\t")
            attempted.add((float(y), sid, int(sd)))
    n_handoff = n_deliv = n_attempt = n_pairs = 0
    cert_by_yaw: dict[float, list[int]] = {y: [0, 0] for y in yaws}
    with out_path.open("a") as f, attempted_path.open("a") as af:
        for yaw_deg in yaws:
            yaw = math.radians(yaw_deg)
            for sid in scenarios:
                scen = scenario_by_id(sid)
                for seed in range(n_seeds):
                    if (yaw_deg, sid, seed) in attempted:
                        continue                              # already tried (resume/extend)
                    af.write(f"{yaw_deg}\t{sid}\t{seed}\n")
                    af.flush()
                    n_attempt += 1
                    cert_by_yaw[yaw_deg][1] += 1
                    h = reach_capture_at_yaw(rig, scen, seed, yaw, cfg, conf, obj)
                    if h.record is not None:
                        continue
                    cert_by_yaw[yaw_deg][0] += 1
                    n_handoff += 1
                    post_yaw = math.degrees(object_yaw(h.snap))
                    x = [float(v) for v in np.asarray(h.x, np.float64)]
                    for ti, th in enumerate(thetas):
                        s = _delivery_signals(h.snap, clip_theta(th))
                        n_deliv += int(s.k6)
                        n_pairs += 1
                        f.write(json.dumps({"handoff_family": fam, "scenario": sid, "seed": seed, "yaw_deg": yaw_deg,
                                            "post_grasp_yaw_deg": round(post_yaw, 4), "theta_idx": ti,
                                            "theta_family": theta_src[ti], "x": x, "theta": [float(t) for t in th],
                                            "k6": bool(s.k6), "dtz_mm": s.dtz_mm, "gap_closed": s.gap_closed,
                                            "safe": bool(s.safe)}) + "\n")
                    f.flush()
                    print(f"[{fam} y{yaw_deg:4.0f}° {sid:20s} s{seed}] handoff#{n_handoff}: {len(thetas)}θ pos={n_deliv} "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return {"family": fam, "n_attempts": n_attempt, "n_new_handoffs": n_handoff, "n_new_pairs": n_pairs,
            "n_new_positive": n_deliv, "n_prior_pairs": n_prior, "n_total_pairs": n_prior + n_pairs,
            "cert_by_yaw": {y: cert_by_yaw[y] for y in yaws}, "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O5-R"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_scen = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    yaws = tuple(float(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else _DEFAULT_YAWS
    bank_path = sys.argv[5] if len(sys.argv) > 5 else str(_OUT / f"orientation_bank_{fam}.json")
    scenarios = _TARGETS[:n_scen]           # align with the bank's tuning scenarios (θ are (yaw,target)-specific)
    thetas, theta_src = _load_bank(bank_path)
    cfg, conf, obj = bc_context()
    print(f"R12.2-A2 dataset [{fam}]: yaws {yaws}° × {len(scenarios)} scen × {n_seeds} seeds × {len(thetas)}θ", flush=True)
    meta = _gen(fam, thetas, theta_src, yaws, scenarios, n_seeds, cfg, conf, obj)
    (_OUT / f"orientation_dataset_meta_{fam}.json").write_text(json.dumps(meta, indent=1))
    cert = {y: f"{c[0]}/{c[1]}" for y, c in meta["cert_by_yaw"].items()}
    pr = round(100 * meta["n_new_positive"] / max(1, meta["n_new_pairs"]))
    print(f"\nR12.2-A2 DONE [{fam}]: +{meta['n_new_pairs']} new pairs ({meta['n_total_pairs']} total) from "
          f"{meta['n_new_handoffs']}/{meta['n_attempts']} new certified handoffs, {pr}% positive (new), cert-by-yaw "
          f"{cert}, {meta['wall_s'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
