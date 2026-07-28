"""COIN R9 — distributed C1 multiseed REPRODUCTION panel (is the seed-0 learned K6 reproducible?).

NOT a new optimisation campaign and NOT generalisation: it re-runs the FROZEN C1 authority-unlock contract (commit `8a0c1c7b`,
tag `coin-r9-first-learned-s1-k6-delivery`) on independent training seeds 1–24 (seed 0 is the frozen champion — never retrained),
per-seed freeze-on-first-K6, and measures the strict-K6 success rate. Every seed uses the IMMUTABLE contract imported from
`coin_kinetic_r3c_rl` (frozen K2 clone + frozen R2 residual α0 = 0.15 + zero-init β = 0.85 expansion head + TD3 + the same reward /
curriculum / safety / champion order); only critics, replay, exploration RNG and the expansion head are fresh per seed. Nothing is
tuned on the seeds' results.

Pre-declared aggregate gates (fixed BEFORE any seed is seen):
  MULTISEED_K6_REPRODUCTION_PASS : ≥ 3 independent seeds verified strict K6, 0 safety violations.
  RELIABLE_K6_LEARNING_PASS      : ≥ 12 / 24 seeds verified strict K6, 0 safety violations.

Each seed writes a self-contained directory (hostname, commit, seed, config hash, timestamps, Phase A/B history, best +
first-K6 checkpoint, per-step trace, deterministic-replay result, safety/cleanliness). Every claimed K6 is INDEPENDENTLY verified
from its saved checkpoint against the canonical `delivery_success`.

Run:  ``python -m hymeko_rl.experiments.coin_kinetic_r3c_multiseed --seeds 1-24``
      ``python -m hymeko_rl.experiments.coin_kinetic_r3c_multiseed --seeds 1-12``   (e.g. one worker)
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option import kinetic_authority_unlock as ku
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM, deterministic_residual
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r3c_rl import R2_OPTIONS, _setup, run_seed
from hymeko_rl.option_rl.agents import make_actor

PANEL_OUT = Path("reports/2026-07-28-coin-r9-r3c-multiseed")
ALL_SEEDS = list(range(1, 25))                   # seed 0 is the frozen champion — never a training seed here
PHASE_A, PHASE_B = 600, 600                       # the frozen C1 contract's Phase A / conditional Phase B budgets


def _config_hash() -> str:
    """A stable hash of the FROZEN C1 contract — any drift changes it, so a reproduction can be tied to the exact contract."""
    cfg = {"family": "C1", "beta": ku.C1_BETA, "alpha0": ku.ALPHA0, "aug_dim": AUG_DIM, "act_dim": krl2.ACT_DIM,
           "reward": krl2.Reward2Weights().__dict__, "r2_options": R2_OPTIONS, "phase_a": PHASE_A,
           "phase_b": PHASE_B, "expl_applied_noise": ku.R2_APPLIED_NOISE}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _rebuild_actor(state: dict) -> Any:
    a = make_actor("td3", AUG_DIM, krl2.ACT_DIM)
    a.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    a.eval()
    return a


def _checkpoint(actor: Any, r2_champ: Any, seed: int, aux: dict) -> dict:
    return {"expansion_state": {k: v.tolist() for k, v in actor.state_dict().items()},
            "r2_champ_state": {k: v.tolist() for k, v in r2_champ.state_dict().items()},
            "beta": ku.C1_BETA, "alpha0": ku.ALPHA0, "family": "C1", "seed": seed, "aux": aux}


def verify_k6(ckpt: dict, camp: Any) -> dict:
    """Reconstruct the policy from the checkpoint and INDEPENDENTLY re-verify strict K6 teacher-free: canonical
    `delivery_success`, dwell ≥ 6, teacher/θ/CEM absent, deterministic replay, safety + cleanliness. No state edit / hidden force
    (the frozen `velocity_rollout` physics is the only dynamics). Returns the full verification record."""
    r2 = _rebuild_actor(ckpt["r2_champ_state"])
    exp = _rebuild_actor(ckpt["expansion_state"])
    r2_fn, exp_fn = deterministic_residual(r2), deterministic_residual(exp)
    factory = ku.unlock_controller_factory(r2_fn, ckpt["beta"])

    def _roll() -> "tuple[Any, dict]":
        c = factory(camp.entry_snap, camp.cf(), exp_fn, camp.bounds)
        return c, velocity_rollout(camp.entry_snap, c, DELIVERY_CFG)
    c1, m1 = _roll()
    _c2, m2 = _roll()
    teacher_absent = not any(hasattr(c1, x) for x in ("teacher", "theta", "cem")) and c1.r2_fn is not None
    kin = [r for r in c1.clone_trace if r["kind"] == "KINETIC_CLONE"]
    fn_clamped = sum(1 for r in kin if min(r["fn_l"], r["fn_r"]) > krl2.FN_CLAMP)
    return {"delivery_success": bool(delivery_success(m1, DELIVERY_CFG)), "k6_delivered": bool(m1["k6_delivered"]),
            "k6_dwell": int(m1["k6_max_dwell"]), "min_dtz_mm": round(_min_dtz_mm(camp.entry_snap, m1), 2),
            "dtz_end_mm": round(m1["dtz_end"] * 1000, 2), "touched": bool(m1["touched"]),
            "teacher_absent": bool(teacher_absent), "clamp_frames": int(fn_clamped),
            "peak_qdot": round(float(m1["peak_qdot"]), 3), "peak_coin_speed": round(float(m1["peak_coin_speed"]), 3),
            "safe": bool(m1["peak_qdot"] <= 3.0 and m1["peak_coin_speed"] <= 1.5),
            "deterministic": bool(np.array_equal(np.asarray(m1["coin_trace"]), np.asarray(m2["coin_trace"])))}


def _trace(ckpt: dict, camp: Any) -> list:
    """The champion's per-step event/action/force/K6 trace (from the reconstructed checkpoint)."""
    exp_fn = deterministic_residual(_rebuild_actor(ckpt["expansion_state"]))
    r2_fn = deterministic_residual(_rebuild_actor(ckpt["r2_champ_state"]))
    c = ku.unlock_controller_factory(r2_fn, ckpt["beta"])(camp.entry_snap, camp.cf(), exp_fn, camp.bounds)
    velocity_rollout(camp.entry_snap, c, DELIVERY_CFG)
    kin = [r for r in c.clone_trace if r["kind"] == "KINETIC_CLONE"]
    return [{"t": r["t"], "dtz_mm": r["dtz_mm"], "v_par": r["v_par"], "fn_l": r["fn_l"], "fn_r": r["fn_r"],
             **rt} for r, rt in zip(kin, c.residual_trace)]


def _first_k6_option(history: list) -> "int | None":
    for h in history:
        if isinstance(h, dict) and h.get("aux", {}).get("k6_strict"):
            return int(h.get("it", -1))
    return None


def _parse_history(lines: list) -> list:
    """Parse the per-eval champion lines the frozen `run_seed` logs (``[td3 it N/T] champ (...) {aux} | replay R``). The aux is a
    Python-repr dict (single quotes), so `ast.literal_eval`, not `json.loads`."""
    out = []
    for ln in lines:
        if " it " not in ln or "min_dtz" not in ln or "{" not in ln:
            continue
        try:
            it = int(ln.split(" it ", 1)[1].split("/", 1)[0])
            aux = ast.literal_eval(ln[ln.index("{"):ln.index("}", ln.index("{")) + 1])
            out.append({"it": it, "aux": aux})
        except (ValueError, IndexError, SyntaxError):
            continue
    return out


def run_one_seed(camp: Any, seed: int, out: Path, *, host: str, commit: str, cfg_hash: str) -> dict:
    """Train one seed on the frozen C1 contract, save its self-contained directory, and (if it claims K6) verify independently."""
    seed_dir = out / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    lines: list = []
    t0 = time.time()
    res = run_seed(camp.entry_snap, camp.frontiers, camp.cf, camp.bounds, camp.w, camp.r2_fn, "C1", ku.C1_BETA,
                   seed=seed, phase_a=PHASE_A, phase_b=PHASE_B, log=lines.append)
    wall = round(time.time() - t0, 1)
    history = _parse_history(lines)
    aux = res["final_aux"]
    ckpt = _checkpoint(res["actor"], camp.r2_champ, seed, aux)
    (seed_dir / "checkpoint.json").write_text(json.dumps(ckpt))
    verification = verify_k6(ckpt, camp) if res["k6_strict"] else None
    if res["k6_strict"]:
        (seed_dir / "first_k6_checkpoint.json").write_text(json.dumps(ckpt))
    record = {"seed": seed, "host": host, "commit": commit, "config_hash": cfg_hash, "wall_s": wall,
              "phase": res["phase"], "k6_strict": bool(res["k6_strict"]), "min_dtz": aux["min_dtz"],
              "final_aux": aux, "phase_a_aux": res.get("phase_a_aux"), "phase_b_aux": res.get("phase_b_aux"),
              "first_k6_option": _first_k6_option(history), "verification": verification}
    (seed_dir / "record.json").write_text(json.dumps({**record, "history": history, "trace": _trace(ckpt, camp)}, indent=1))
    tag = "✅K6" if res["k6_strict"] else "  ··"
    print(f"  {tag} seed {seed:02d} [{res['phase']}] min_dtz {aux['min_dtz']}mm clean {aux['clean']} safe {aux['safe']} "
          f"first_k6@{record['first_k6_option']} wall {wall}s"
          + (f"  VERIFY delivery={verification['delivery_success']} dwell={verification['k6_dwell']} "
             f"teacher_absent={verification['teacher_absent']} det={verification['deterministic']}" if verification else ""))
    return record


def _wilson_ci(k: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """95 % Wilson score interval for a binomial proportion (closed-form; better than normal-approx at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def _is_verified_k6(r: dict) -> bool:
    """A seed counts only if it claimed K6 AND the independent checkpoint verification confirms every clause."""
    v = r["verification"]
    return bool(r["k6_strict"] and v and v["delivery_success"] and v["k6_dwell"] >= 6
                and v["teacher_absent"] and v["deterministic"] and v["safe"])


def _dist(vals: list) -> dict:
    return {"vals": vals, "median": (round(float(np.median(vals)), 1) if vals else None),
            "min": (min(vals) if vals else None), "max": (max(vals) if vals else None)}


def _safety_violations(records: list) -> list:
    return [r["seed"] for r in records if not r["final_aux"]["safe"] or (r["verification"] and not r["verification"]["safe"])]


def _failures(records: list) -> list:
    return [{"seed": r["seed"], "phase": r["phase"], "min_dtz": r["min_dtz"], "clean": r["final_aux"]["clean"],
             "safe": r["final_aux"]["safe"], "mode": _failure_mode(r)} for r in records if not r["k6_strict"]]


def _gates(k: int, safety_fail: list) -> dict:
    """The two PRE-DECLARED aggregate gates (fixed before any seed is seen)."""
    return {"MULTISEED_K6_REPRODUCTION_PASS": bool(k >= 3 and not safety_fail),
            "RELIABLE_K6_LEARNING_PASS": bool(k >= 12 and not safety_fail)}


def _distributions(vk6: list) -> dict:
    """Phase split + the reported distributions over the VERIFIED-K6 seeds."""
    return {"phase_a_k6": sum(1 for r in vk6 if r["phase"] == "A"), "phase_b_k6": sum(1 for r in vk6 if r["phase"] == "B"),
            "options_to_first_k6": _dist([r["first_k6_option"] for r in vk6 if r["first_k6_option"]]),
            "wall_to_first_k6_s": _dist([r["wall_s"] for r in vk6]),
            "min_dtz_mm": _dist([r["verification"]["min_dtz_mm"] for r in vk6]),
            "k6_dwell": _dist([r["verification"]["k6_dwell"] for r in vk6])}


def _clean_totals(records: list) -> dict:
    return {kind: sum(1 for r in records if r["final_aux"][f"{kind}_pen"] > 0) for kind in ("stall", "clamp", "reversal")}


def _aggregate(records: list) -> dict:
    """Pre-declared gates + the full reproduction report over the verified seeds."""
    vk6 = [r for r in records if _is_verified_k6(r)]
    n, k = len(records), len(vk6)
    safety_fail = _safety_violations(records)
    return {"n_seeds": n, "k6_verified": k, "k6_rate": round(k / n, 4) if n else 0.0, "wilson95": _wilson_ci(k, n),
            "safety_violations": safety_fail, **_gates(k, safety_fail), **_distributions(vk6),
            "stall_clamp_reversal_totals": _clean_totals(records), "failures": _failures(records)}


def _failure_mode(r: dict) -> str:
    a = r["final_aux"]
    if not a["safe"]:
        return "UNSAFE"
    if a["clamp_pen"] > 0 or a["stall_pen"] > 0 or a["reversal_pen"] > 0:
        return "NOT_CLEAN_stall/clamp/reversal"
    if a["min_dtz"] <= krl2.CORRIDOR_MM:
        return "CLEAN_CORRIDOR_NO_K6"                              # reached ≤30 mm cleanly but not the strict zone
    return "CLEAN_SHORT_OF_CORRIDOR"


def run(seeds: "list | None" = None, out: Path = PANEL_OUT) -> dict:
    seeds = seeds if seeds is not None else ALL_SEEDS
    host, commit, cfg_hash = socket.gethostname(), _git_commit(), _config_hash()
    print(f"C1 MULTISEED REPRODUCTION — host {host} commit {commit[:8]} config {cfg_hash} seeds {seeds[0]}–{seeds[-1]}")
    t0 = time.time()
    camp = _setup()
    print(f"  frozen R2 champion + frontiers {[f.dtz_mm for f in camp.frontiers]}mm ready; running {len(seeds)} seeds")
    records = [run_one_seed(camp, sd, out, host=host, commit=commit, cfg_hash=cfg_hash) for sd in seeds]
    agg = _aggregate(records)
    summary = {"contract": "COIN_KINETIC_R3C_C1_MULTISEED_V1", "host": host, "commit": commit, "config_hash": cfg_hash,
               "immutable_source_tag": "coin-r9-first-learned-s1-k6-delivery", "seeds": seeds,
               "records": [{k: v for k, v in r.items() if k != "trace"} for r in records], "aggregate": agg,
               "wall_s": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"multiseed_{seeds[0]:02d}_{seeds[-1]:02d}.json").write_text(json.dumps(summary, indent=1))
    return summary


def _parse_seeds(argv: list) -> "list | None":
    for i, a in enumerate(argv):
        if a == "--seeds" and i + 1 < len(argv):
            lo, hi = (argv[i + 1].split("-") + [argv[i + 1]])[:2]
            return list(range(int(lo), int(hi) + 1))
    return None


if __name__ == "__main__":
    r = run(_parse_seeds(sys.argv))
    a = r["aggregate"]
    print(f"\nC1 MULTISEED: {a['k6_verified']}/{a['n_seeds']} verified strict K6 (rate {a['k6_rate']}, "
          f"Wilson95 {a['wilson95']}); safety violations {a['safety_violations']}")
    print(f"  MULTISEED_K6_REPRODUCTION_PASS = {a['MULTISEED_K6_REPRODUCTION_PASS']}  |  "
          f"RELIABLE_K6_LEARNING_PASS = {a['RELIABLE_K6_LEARNING_PASS']}")
    print(f"  options-to-first-K6 median {a['options_to_first_k6']['median']}; best min_dtz {a['min_dtz_mm']['min']}mm; "
          f"Phase A/B K6 {a['phase_a_k6']}/{a['phase_b_k6']}; wall {r['wall_s']}s")
