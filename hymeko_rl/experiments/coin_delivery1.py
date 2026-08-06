"""COIN-DELIVERY-1 — delivery-specific candidate GENERATION (not scorer changes).

The delivery re-score showed the bottleneck is candidate generation, not selection: the frozen (acquisition-trained)
decoder pool + 2-step horizon delivers only 0.233; ~77% of states have no deliverable candidate. This module adds
SEMANTICALLY delivery-directed candidate PRIMITIVES — scripted, geometry-based, no training — and measures whether they
raise the offline delivery ceiling (in_zone) under exact replay.

Action semantics (CooperativeContactController, 6-d): a[0:2] translate the grasp MIDPOINT (x,y); a[2] aperture; a[3]
squeeze both tips toward the coin (grasp); a[4] balance shift along the finger axis; a[5] rotate. So delivery is
scriptable: translate the midpoint along (zone − coin) while holding the coin. The catch: the env TERMINATES on handoff
(4 consecutive both-contact steps), and midpoint translation is _STEP_MID=0.012/step, so a sustained grasp-carry can move
the coin only ~0.048 before the episode ends — coins sit ~0.08–0.16 from the zone. The primitives below therefore
include handoff-AVOIDING transport (pulse-release, push) so transport can exceed 4 steps.

Do NOT run RL. No env/reward/CORE change. Handoff is NOT task success. delivery_success = in_zone (disk_to_zone<=zone_half).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from hymeko_rl.experiments.pedc_selection import _env

C = Path("experiments/2026_07_20_coin_delivery1")
_DELIVERY_CORPUS = Path("experiments/2026_07_20_delivery_rescore/corpus/h0_delivery.pkl")
_HELD = range(64_000, 64_090)


# ── geometry ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _dir_to_zone(inner) -> tuple:
    """Backward-compat delegator to the canonical `PlanarGraspEnv.direction_to_zone` (geometry now lives on the env)."""
    return inner.direction_to_zone()


# ── Stage 1: delivery-directed candidate primitives (scripted, 6-d actions) ──────────────────────────────────────
# p_grasp_carry now lives in the library (hymeko_rl.train.coin_delivery_rl) so production RL code no longer imports a
# scripted primitive from this experiment module; re-exported here for the local PRIMS table + downstream experiments.
from hymeko_rl.train.coin_delivery_rl import p_grasp_carry  # noqa: E402  (re-export; kept after the module docstring)


def p_carry_pulse(inner, t):
    """B+E: carry toward zone but RELEASE every 4th step (reset the both-contact streak) so transport can exceed the
    4-step handoff horizon; re-grasp and continue."""
    d, _n = _dir_to_zone(inner); sq = 0.8 if (t % 4 != 3) else -0.7
    return np.array([d[0], d[1], 0.0, sq, 0.0, 0.0], np.float32)


def p_push(inner, t):
    """A/B push: translate midpoint toward the zone with a light, slightly-open asymmetric contact (balance shift) so
    the coin is nudged rather than clamped — avoids sustained both-contact handoff."""
    d, _n = _dir_to_zone(inner); return np.array([d[0], d[1], -0.3, 0.35, 0.6, 0.0], np.float32)


def p_carry_settle(inner, t):
    """C overshoot/settle: carry hard toward zone while far, ease off (settle) once close so the coin stays in-zone."""
    d, n = _dir_to_zone(inner); far = n > inner._zone_half * 1.5
    return np.array([d[0], d[1], 0.0, 0.8 if far else 0.3, 0.0, 0.0], np.float32) if far \
        else np.array([0.3 * d[0], 0.3 * d[1], 0.0, 0.6, 0.0, 0.0], np.float32)


PRIMS = {"grasp_carry": p_grasp_carry, "carry_pulse": p_carry_pulse, "push": p_push, "carry_settle": p_carry_settle}


def _rand_prim(rng):
    """Random-action control (same step budget): NOT delivery-directed — the sanity control for §2."""
    def f(inner, t):
        return rng.uniform(-1, 1, 6).astype(np.float32)
    return f


# ── Stage 2: offline delivery-ceiling test (exact replay, scripted primitives) ───────────────────────────────────
def rollout(env, seed: int, prim, max_steps: int = 40) -> dict:
    """Roll one scripted primitive from `seed`; capture DELIVERY (in_zone ever), progress, handoff, distances."""
    env.reset(seed=int(seed)); inner = env._env; m = inner._planar_metrics
    start_dtz = float(m.disk_to_zone); deliv = handoff = False; min_dtz = start_dtz; final_dtz = start_dtz
    for t in range(max_steps):
        _o, _r, term, trunc, info = env.step(np.clip(prim(inner, t), -1, 1).astype(np.float32))
        mm = inner._planar_metrics; dtz = float(mm.disk_to_zone)
        min_dtz = min(min_dtz, dtz); final_dtz = dtz; deliv = deliv or bool(mm.in_zone)
        handoff = handoff or bool(info.get("handoff_ready"))
        if term or trunc:
            break
    return {"deliv": deliv, "handoff": handoff, "start_dtz": round(start_dtz, 4), "min_dtz": round(min_dtz, 4),
            "final_dtz": round(final_dtz, 4), "progress": round(start_dtz - min_dtz, 4)}


def stage2_ceiling(seeds=_HELD) -> dict:
    """Per state, the best delivery over the scripted delivery primitives + a random control; report the pool ceiling
    and per-primitive delivery, vs the old-pool 0.233."""
    env = _env(); rng = np.random.default_rng(0); rows = {name: [] for name in PRIMS}; rows["random_ctrl"] = []
    n = 0
    for sd in seeds:
        n += 1
        for name, prim in PRIMS.items():
            rows[name].append(rollout(env, sd, prim))
        rows["random_ctrl"].append(rollout(env, sd, _rand_prim(rng)))
        if n % 20 == 0:
            deliv = {k: sum(r["deliv"] for r in v) for k, v in rows.items()}
            print(f"  [stage2] {n}/{len(list(seeds))} | deliv Σ {deliv}", flush=True)
    per = {name: _summ(rows[name], n) for name in rows}
    ceil = [int(any(rows[name][i]["deliv"] for name in PRIMS)) for i in range(n)]      # best over delivery primitives
    per["delivery_primitives_CEILING"] = {"delivery_success": round(sum(ceil) / n, 4), "n": n}
    return {"per_primitive": per, "old_pool_ceiling": 0.233, "n": n, "rows": rows}


def _summ(rs: list, n: int) -> dict:
    return {"delivery_success": round(sum(r["deliv"] for r in rs) / n, 4),
            "handoff_success": round(sum(r["handoff"] for r in rs) / n, 4),
            "grasp_no_delivery_rate": round(sum(r["handoff"] and not r["deliv"] for r in rs) / n, 4),
            "progress_med": round(float(np.median([r["progress"] for r in rs])), 4),
            "final_dtz_med": round(float(np.median([r["final_dtz"] for r in rs])), 4),
            "moved_coin_rate": round(sum(r["progress"] > 0.01 for r in rs) / n, 4)}


# ── Stage 0: delivery failure analysis (from the cached delivery re-score corpus) ────────────────────────────────
def stage0_classify() -> dict:
    if not _DELIVERY_CORPUS.exists():
        return {"error": f"{_DELIVERY_CORPUS} missing — run delivery_rescore first"}
    trees = pickle.loads(_DELIVERY_CORPUS.read_bytes()); cls = {}
    for t in trees:
        pairs = [{"h": fr["e1"]["handoff"], "d": fr["e1"]["deliv"], "mdtz": fr["e1"]["min_dtz"]} for fr in t["firsts"]] + \
                [{"h": s["handoff"], "d": s["deliv"], "mdtz": s["min_dtz"]} for fr in t["firsts"] for s in fr["seconds"]]
        any_deliv = any(p["d"] for p in pairs); any_handoff = any(p["h"] for p in pairs)
        best_prog = t["start_dtz"] - min(p["mdtz"] for p in pairs)
        cls[t["sid"].seed if hasattr(t["sid"], "seed") else id(t)] = _class1(any_deliv, any_handoff, best_prog, t["start_dtz"])
    hist = {}
    for c in cls.values():
        hist[c] = hist.get(c, 0) + 1
    return {"n": len(trees), "histogram": hist}


def _class1(any_deliv, any_handoff, best_prog, start_dtz) -> str:
    if any_deliv:
        return "1_already_deliverable"
    if any_handoff and best_prog < 0.02:
        return "2_grasped_no_transport"
    if best_prog >= 0.02 * start_dtz and best_prog < start_dtz - 0.04:
        return "3_transport_starts_misses_zone"
    if not any_handoff and best_prog < 0.02:
        return "4_no_stable_acquisition"
    return "6_unknown"


def run() -> dict:
    C.mkdir(parents=True, exist_ok=True); (C / "manifests").mkdir(exist_ok=True)
    s0 = stage0_classify()
    print(f"=== Stage 0 delivery failure analysis: {s0} ===", flush=True)
    s2 = stage2_ceiling()
    old = s2["old_pool_ceiling"]; new = s2["per_primitive"]["delivery_primitives_CEILING"]["delivery_success"]
    gain = round(new - old, 4)
    verdict = ("STAGE2_PASS_generation_was_the_bottleneck" if new >= 0.40 or gain >= 0.10
               else "STAGE2_FAIL_action_abstraction_or_horizon_insufficient")
    out = {"stage0": s0, "stage2": {k: v for k, v in s2.items() if k != "rows"},
           "old_pool_ceiling": old, "delivery_primitives_ceiling": new, "delivery_ceiling_gain": gain, "verdict": verdict}
    (C / "manifests" / "coin_delivery1.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n=== Stage 2 delivery ceiling (scripted delivery primitives, exact replay) ===", flush=True)
    for name, m in s2["per_primitive"].items():
        print(f"  {name:28s} {m}", flush=True)
    print(f"[COIN-DELIVERY-1] old-pool ceiling {old} → delivery-primitives ceiling {new} (gain {gain}) → {verdict}",
          flush=True)
    return out


if __name__ == "__main__":
    run()
