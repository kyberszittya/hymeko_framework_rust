"""R9 R2-under-H1 multiseed reproduction — is the first learned s1 K6 a reproducible LEARNING result under the explicit handoff?

The MAIN panel of the corrected story: re-run the R2 residual training (frozen architecture / reward2 / authority α = 0.15 /
100 %-cradle curriculum / safety / K6 / champion order) but with the KINETIC policy deployed through the explicit H1
HANDOFF_RESET contract, across independent seeds — every episode is the full uninterrupted chain from the canonical cradle
(`cradle → APPROACH → HANDOFF_RESET → KINETIC clone + learning-R2 → coast → K6`), NEVER the offline frozen-entry artifact as
episode-start. Per seed: freeze on the first strict K6, then the next seed. Every claimed K6 is independently verified on the H1
deploy: canonical `delivery_success`, dwell ≥ 6, exactly one HANDOFF_RESET before the first policy step, teacher-absent,
deterministic replay, safe, clean.

Pre-declared gates (fixed before any seed): R2_H1_MULTISEED_REPRODUCTION_PASS (≥ 3 verified K6, 0 safety) and
R2_H1_RELIABLE_LEARNING_PASS (≥ 12/24, 0 safety). All `8a0c1c7b`/`41510cac` modules imported unchanged; tags never moved.

Run:  ``python -m hymeko_rl.experiments.coin_kinetic_r2_h1_multiseed --seeds 1-24``
"""
from __future__ import annotations

import json
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_r2_h1 as r2h1
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import stop_on_strict_k6
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM, deterministic_residual
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r2_rl import _clone_augs, _load_clone
from hymeko_rl.experiments.coin_kinetic_r3c_multiseed import _dist, _git_commit, _rebuild_actor, _wilson_ci
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed")
OPTIONS = 600                                    # per-seed training budget (freeze on first strict K6 stops early)


def _distill(actor: Any, cradle: Any, cf: Any, bounds: ResidualBounds, seed: int) -> None:
    from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
    distill_zero_residual(actor, _clone_augs(cradle, cf(), bounds), seed=seed)


def _train_seed(cradle: Any, cf: Any, bounds: ResidualBounds, w: krl2.Reward2Weights, seed: int, log: Any) -> "tuple[Any, list]":
    actor = make_actor("td3", AUG_DIM, ACT_DIM)
    _distill(actor, cradle, cf, bounds, seed)
    cfg = replace(krl2.PerStepConfig(), total_options=OPTIONS, eval_every=25, warmup_options=20)
    return krl2.train_perstep("td3", cradle, cf(), bounds, w, cfg, seed=seed, warm_actor=actor, log=log,
                              collect_override=r2h1.make_collect_r2_h1(cradle, cf, bounds, w),
                              champion_override=r2h1.make_eval_r2_h1(cradle, cf, bounds, w),
                              stop_when=stop_on_strict_k6)


def _scr(kin: list) -> list:
    """[stalls, clamps, reversals] over the KINETIC-policy steps."""
    vpar = [r["v_par"] for r in kin]
    return [sum(1 for v in vpar if v <= 0.0), sum(1 for r in kin if min(r["fn_l"], r["fn_r"]) > krl2.FN_CLAMP),
            sum(1 for i in range(1, len(vpar)) if vpar[i] * vpar[i - 1] < 0.0)]


def _reset_before_policy(kinds: list) -> bool:
    return bool("HANDOFF_RESET" in kinds and "KINETIC_CLONE" in kinds
               and kinds.index("HANDOFF_RESET") < kinds.index("KINETIC_CLONE"))


def verify_h1(actor_state: dict, cradle: Any, cf: Any, bounds: ResidualBounds) -> dict:
    """Independently verify the trained R2 residual on the H1 deploy from the cradle (reconstructed from the checkpoint)."""
    r2_fn = deterministic_residual(_rebuild_actor(actor_state))

    def _roll() -> "tuple[Any, dict]":
        c = HandoffResetTemporalController(cradle, cf(), r2_fn, bounds)
        return c, velocity_rollout(cradle, c, DELIVERY_CFG)
    c1, m1 = _roll()
    _c2, m2 = _roll()
    kinds = [r["kind"] for r in c1.clone_trace]
    scr = _scr([r for r in c1.clone_trace if r["kind"] == "KINETIC_CLONE"])
    return {"delivery_success": bool(delivery_success(m1, DELIVERY_CFG)), "k6_dwell": int(m1["k6_max_dwell"]),
            "min_dtz_mm": round(_min_dtz_mm(cradle, m1), 2), "n_handoff_reset": kinds.count("HANDOFF_RESET"),
            "reset_before_policy": _reset_before_policy(kinds),
            "teacher_absent": not any(hasattr(c1, x) for x in ("teacher", "theta", "cem")),
            "deterministic": bool(np.array_equal(np.asarray(m1["coin_trace"]), np.asarray(m2["coin_trace"]))),
            "safe": bool(m1["peak_qdot"] <= 3.0 and m1["peak_coin_speed"] <= 1.5),
            "stall_clamp_reversal": scr, "clean": bool(sum(scr) == 0)}


def _first_k6_option(hist: list) -> "int | None":
    return next((h["it"] for h in hist if isinstance(h, dict) and h.get("aux", {}).get("k6_strict")), None)


def run_one_seed(cradle: Any, cf: Any, bounds: ResidualBounds, w: krl2.Reward2Weights, seed: int, out: Path,
                 *, host: str, commit: str) -> dict:
    seed_dir = out / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    lines: list = []
    t0 = time.time()
    best, hist = _train_seed(cradle, cf, bounds, w, seed, lines.append)
    wall = round(time.time() - t0, 1)
    state = {k: v.tolist() for k, v in best.state_dict().items()}
    aux = r2h1.make_eval_r2_h1(cradle, cf, bounds, w)(best)[1]
    k6 = bool(aux["k6_strict"])
    ver = verify_h1(state, cradle, cf, bounds) if k6 else None
    ckpt = {"r2_actor_state": state, "alpha": r2h1.R2_ALPHA, "seed": seed, "contract": "H1_EXPLICIT_HANDOFF_RESET", "aux": aux}
    (seed_dir / "checkpoint.json").write_text(json.dumps(ckpt))
    rec = {"seed": seed, "host": host, "commit": commit, "wall_s": wall, "k6_strict": k6, "min_dtz": aux["min_dtz"],
           "first_k6_option": _first_k6_option([{**h, "aux": h.get("aux", {})} for h in hist if isinstance(h, dict)]),
           "final_aux": aux, "verification": ver}
    (seed_dir / "record.json").write_text(json.dumps({**rec, "history": lines[-30:]}, indent=1))
    tag = "✅K6" if k6 else "  ··"
    print(f"  {tag} seed {seed:02d} min_dtz {aux['min_dtz']}mm clean {aux['clean']} safe {aux['safe']} "
          f"first_k6@{rec['first_k6_option']} wall {wall}s"
          + (f"  VERIFY deliver={ver['delivery_success']} dwell={ver['k6_dwell']} reset={ver['n_handoff_reset']} "
             f"teacher_absent={ver['teacher_absent']} det={ver['deterministic']} clean={ver['clean']}" if ver else ""))
    return rec


def _verified(r: dict) -> bool:
    v = r["verification"]
    return bool(r["k6_strict"] and v and v["delivery_success"] and v["k6_dwell"] >= 6 and v["n_handoff_reset"] == 1
                and v["reset_before_policy"] and v["teacher_absent"] and v["deterministic"] and v["safe"] and v["clean"])


def _safety_fail(records: list) -> list:
    return [r["seed"] for r in records if not r["final_aux"]["safe"]
            or (r["verification"] and not r["verification"]["safe"])]


def _dist_block(vk6: list) -> dict:
    return {"options_to_first_k6": _dist([r["first_k6_option"] for r in vk6 if r["first_k6_option"]]),
            "min_dtz_mm": _dist([r["verification"]["min_dtz_mm"] for r in vk6]),
            "k6_dwell": _dist([r["verification"]["k6_dwell"] for r in vk6]),
            "stall_clamp_reversal_total": [sum(r["verification"]["stall_clamp_reversal"][i] for r in vk6) for i in range(3)]}


def _failures(records: list) -> list:
    return [{"seed": r["seed"], "min_dtz": r["min_dtz"], "clean": r["final_aux"]["clean"], "safe": r["final_aux"]["safe"]}
            for r in records if not r["k6_strict"]]


def _aggregate(records: list) -> dict:
    vk6 = [r for r in records if _verified(r)]
    n, k = len(records), len(vk6)
    sf = _safety_fail(records)
    return {"n_seeds": n, "k6_verified": k, "k6_rate": round(k / n, 4) if n else 0.0, "wilson95": _wilson_ci(k, n),
            "safety_violations": sf, "R2_H1_MULTISEED_REPRODUCTION_PASS": bool(k >= 3 and not sf),
            "R2_H1_RELIABLE_LEARNING_PASS": bool(k >= 12 and not sf), **_dist_block(vk6), "failures": _failures(records)}


def _parse_seeds(argv: list) -> list:
    for i, a in enumerate(argv):
        if a == "--seeds" and i + 1 < len(argv):
            lo, hi = (argv[i + 1].split("-") + [argv[i + 1]])[:2]
            return list(range(int(lo), int(hi) + 1))
    return list(range(1, 25))


def run(seeds: "list | None" = None, out: Path = OUT) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    seeds = seeds if seeds is not None else list(range(1, 25))
    host, commit = socket.gethostname(), _git_commit()
    print(f"R2-UNDER-H1 MULTISEED — host {host} commit {commit[:8]} seeds {seeds[0]}–{seeds[-1]} (H1 explicit handoff-reset)")
    t0 = time.time()
    model, norm = _load_clone()
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    bounds = ResidualBounds(alpha=r2h1.R2_ALPHA)
    w = krl2.Reward2Weights()

    def cf() -> CloneActor:
        return CloneActor(model, norm)
    records = [run_one_seed(cradle, cf, bounds, w, sd, out, host=host, commit=commit) for sd in seeds]
    agg = _aggregate(records)
    summary = {"contract": "R2_UNDER_H1_MULTISEED_V1", "immutable_source": "41510cac", "host": host, "commit": commit,
               "tag": "coin-r9-r2-first-learned-s1-k6-explicit-handoff-reset", "seeds": seeds,
               "records": records, "aggregate": agg, "wall_s": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"r2_h1_multiseed_{seeds[0]:02d}_{seeds[-1]:02d}.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    r = run(_parse_seeds(sys.argv))
    a = r["aggregate"]
    print(f"\nR2-H1 MULTISEED: {a['k6_verified']}/{a['n_seeds']} verified strict K6 (rate {a['k6_rate']}, "
          f"Wilson95 {a['wilson95']}); safety {a['safety_violations']}")
    print(f"  R2_H1_MULTISEED_REPRODUCTION_PASS={a['R2_H1_MULTISEED_REPRODUCTION_PASS']}  "
          f"R2_H1_RELIABLE_LEARNING_PASS={a['R2_H1_RELIABLE_LEARNING_PASS']}")
    print(f"  options-to-K6 median {a['options_to_first_k6']['median']}; best min_dtz {a['min_dtz_mm']['min']}mm; "
          f"dwell median {a['k6_dwell']['median']}; wall {r['wall_s']}s")
