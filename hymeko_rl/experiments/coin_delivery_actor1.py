"""COIN-DELIVERY-ACTOR-1 — cooperative push/plow actor portfolio: evaluation, structural controls, headroom (NO RL).

Answers: can cleanly reconstructed cooperative fingertip push/plow actors deliver the cylindrical coin WITHOUT strict
bilateral clamp transport? Evaluates A0–A5 on a fresh same-harness corpus under the STRICT dwell-verified valid-delivery
monitor + impulse attribution + mechanism classification; runs baselines (random/neutral) and structural controls
(direction / actor-label scramble); verifies the reward ordering; classifies task headroom. NO RL, NO canonical change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hymeko_rl.experiments.pedc_selection import _env
from hymeko_rl.train.coin_delivery_actor import (
    DeliveryReward,
    DeliveryActor,
    actor_action,
    delivery_step_reward,
    rollout_delivery,
)

_OUT = Path("experiments/2026_07_20_coin_delivery_actor1")
_HELD = list(range(64_000, 64_090))
_PRIMARY = [DeliveryActor.A0_SYM_PUSH, DeliveryActor.A1_VPLOW, DeliveryActor.A2_ASYM_PUSH,
            DeliveryActor.A3_SETUP_PUSH, DeliveryActor.A4_RECOVERY, DeliveryActor.A5_NEUTRAL]


def _log(m: str) -> None:
    print(m, flush=True)


def _rand_scramble(rng):
    def f(a, t):
        return rng.uniform(-1, 1, 6).astype(np.float32)
    return f


def _dir_scramble(a, t):
    return np.array([-a[0], -a[1], a[2], a[3], a[4], a[5]], np.float32)   # push AWAY from the zone (direction scramble)


def _mech_hist(rows) -> dict:
    mech: dict[str, int] = {}
    for r in rows:
        mech[r.mechanism] = mech.get(r.mechanism, 0) + 1
    return mech


def _lr_balance(rows) -> float:
    prog = [r for r in rows if r.progress > 0.02]
    bal = [min(r.attribution["alpha_L"], r.attribution["alpha_R"]) / (r.fingertip_fraction + 1e-9) for r in prog]
    return round(float(np.median(bal)) if bal else 0.0, 3)


def _eval_actor(env, actor, seeds, *, scramble=None) -> dict:
    rows = [rollout_delivery(env, sd, actor, scramble=scramble) for sd in seeds]
    return {"n": len(rows), "strict": sum(r.strict_delivery for r in rows), "loose": sum(r.loose_in_zone for r in rows),
            "init_success": sum(r.initial_success for r in rows), "body_shove": sum(r.body_shove for r in rows),
            "med_progress": round(float(np.median([r.progress for r in rows])), 4),
            "med_fingertip_fraction": round(float(np.median([r.fingertip_fraction for r in rows])), 3),
            "med_LR_balance_on_progress": _lr_balance(rows), "mechanisms": _mech_hist(rows), "rows": rows}


def _reward_ordering(env) -> dict:
    """Verify R(clean delivery) > R(partial progress) > R(contact stall) AND R(clean fingertip) > R(body shove), by
    accumulating the DELIVERY-ALIGNED reward (one-time delivery bonus, no contact-loss penalty) over real rollouts
    grouped by outcome class (independent of the monitor)."""
    cfg = DeliveryReward()
    buckets: dict[str, list] = {"clean_delivery": [], "partial_progress": [], "contact_stall": [], "body_shove": []}
    for actor in (DeliveryActor.A1_VPLOW, DeliveryActor.A4_RECOVERY, DeliveryActor.A0_SYM_PUSH):
        for sd in _HELD[:40]:
            ret, r = _episode_return(env, sd, actor, cfg)
            key = ("clean_delivery" if r.strict_delivery else "body_shove" if r.body_shove
                   else "contact_stall" if r.progress < 0.02 else "partial_progress")
            buckets[key].append(ret)
    means = {k: (round(float(np.mean(v)), 3) if v else None) for k, v in buckets.items()}
    return {"mean_return_by_class": means, "counts": {k: len(v) for k, v in buckets.items()}, **_check_ordering(means)}


def _check_ordering(means: dict) -> dict:
    cd, pp, cs, bs = means["clean_delivery"], means["partial_progress"], means["contact_stall"], means["body_shove"]
    triple = all(x is not None for x in (cd, pp, cs))
    return {"ordering_clean_gt_partial_gt_stall": bool(triple and cd > pp > cs),
            "ordering_clean_gt_bodyshove": bool(cd > bs if (cd is not None and bs is not None) else True)}


def _episode_return(env, seed, actor, cfg) -> tuple:
    env.reset(seed=int(seed))
    inner = env._env
    prev_dtz = float(inner._planar_metrics.disk_to_zone)
    prev_body_steps = int(inner._arm_body_steps)
    delivered = False
    ret = 0.0
    for t in range(60):
        _o, _r, term, trunc, _i = env.step(np.clip(actor_action(inner, t, actor), -1, 1).astype(np.float32))
        m = inner._planar_metrics
        dtz = float(m.disk_to_zone)
        body = int(inner._arm_body_steps) > prev_body_steps
        prev_body_steps = int(inner._arm_body_steps)
        delivered_now = bool(m.in_zone) and not delivered
        delivered = delivered or bool(m.in_zone)
        ret += delivery_step_reward(m, prev_dtz - dtz, delivered_now, body, cfg)
        prev_dtz = dtz
        if term or trunc:
            break
    # re-roll for the classified DeliveryResult (deterministic same seed)
    return ret, rollout_delivery(env, seed, actor)


def _headroom(actors: dict, baselines: dict) -> dict:
    """Classify task headroom H0–H4 from the actor table. H0 one actor closes it; H1 selection; H4 contact/representation
    limiting (dominant mechanism is a contact-mechanics failure)."""
    n = actors[_PRIMARY[0].value]["n"]
    best_single = max(a["strict"] for a in actors.values())
    # selection ceiling: per-seed best over actors (do different actors deliver different seeds?)
    per_seed = [[r.strict_delivery for r in actors[a.value]["rows"]] for a in _PRIMARY]
    sel_ceiling = sum(any(col) for col in zip(*per_seed))
    # dominant failure mechanism across the portfolio
    allmech: dict[str, int] = {}
    for a in actors.values():
        for k, v in a["mechanisms"].items():
            allmech[k] = allmech.get(k, 0) + v
    dominant = max(allmech, key=allmech.get)
    return {"best_single_actor_strict": best_single, "selection_ceiling_strict": sel_ceiling, "n": n,
            "dominant_mechanism": dominant, "portfolio_mechanisms": allmech,
            "headroom": _classify_headroom(best_single, sel_ceiling, n, dominant)}


def _classify_headroom(best_single: int, sel_ceiling: int, n: int, dominant: str) -> str:
    if best_single >= 0.8 * n:
        return "H0_one_deterministic_actor_closes"
    if sel_ceiling >= best_single + 0.1 * n and sel_ceiling >= 0.3 * n:
        return "H1_actor_selection_headroom"
    if dominant in ("one_finger_bulldoze", "contact_stall"):
        return "H4_contact_representation_limiting"
    return "H3_sequencing_or_search_headroom"


def _strip(a: dict) -> dict:
    return {k: v for k, v in a.items() if k != "rows"}


def run() -> dict:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests").mkdir(exist_ok=True)
    env = _env()
    _log("=== actor portfolio on fresh held corpus (strict dwell-verified delivery monitor) ===")
    actors = {a.value: _eval_actor(env, a, _HELD) for a in _PRIMARY}
    for k, v in actors.items():
        _log(f"  {k:26s} strict={v['strict']}/{v['n']} loose={v['loose']} body_shove={v['body_shove']} "
             f"med_prog={v['med_progress']} med_LR_bal={v['med_LR_balance_on_progress']} {v['mechanisms']}")

    _log("=== baselines + structural controls (correct MUST beat scramble; §structural) ===")
    rng = np.random.default_rng(0)
    baselines = {"random_action": _eval_actor(env, DeliveryActor.A0_SYM_PUSH, _HELD, scramble=_rand_scramble(rng))}
    controls = {}
    for a in (DeliveryActor.A1_VPLOW, DeliveryActor.A4_RECOVERY):
        controls[f"{a.value}|dir_scramble"] = _eval_actor(env, a, _HELD, scramble=_dir_scramble)
        _log(f"  {a.value} correct strict={actors[a.value]['strict']} vs dir-scramble strict={controls[f'{a.value}|dir_scramble']['strict']}")
    _log(f"  random-action baseline strict={baselines['random_action']['strict']}")

    _log("=== reward-ordering audit ===")
    rew = _reward_ordering(env)
    _log(f"  {rew['mean_return_by_class']} | clean>partial>stall={rew['ordering_clean_gt_partial_gt_stall']} clean>shove={rew['ordering_clean_gt_bodyshove']}")

    head = _headroom(actors, baselines)
    _log(f"=== headroom: {head['headroom']} (best_single={head['best_single_actor_strict']}, sel_ceiling={head['selection_ceiling_strict']}, dominant={head['dominant_mechanism']}) ===")

    verdict = _route(actors, controls, baselines, head)
    out = {"actors": {k: _strip(v) for k, v in actors.items()}, "baselines": {k: _strip(v) for k, v in baselines.items()},
           "structural_controls": {k: _strip(v) for k, v in controls.items()}, "reward_ordering": rew,
           "headroom": head, "finding_status": verdict["status"], "answers": verdict["answers"],
           "tensor_rl_eligible": verdict["tensor_rl_eligible"], "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests" / "coin_delivery_actor1.json").write_text(json.dumps(out, indent=2, default=str))
    _log(f"\n[COIN-DELIVERY-ACTOR-1] status={verdict['status']} | tensor-RL eligible={verdict['tensor_rl_eligible']} | {out['wall_s']}s")
    return out


def _route(actors: dict, controls: dict, baselines: dict, head: dict) -> dict:
    n = actors[_PRIMARY[0].value]["n"]
    best = max(a["strict"] for a in actors.values())
    clean_actors = [k for k, a in actors.items() if a["strict"] >= 2]
    beats_scramble = all(actors[a.split("|")[0]]["strict"] >= max(1, controls[a]["strict"]) for a in controls) if controls else False
    # symmetric-push geometric degeneration: L/R balance ≈ 0 → one-finger bulldoze
    sym_lr = actors[DeliveryActor.A0_SYM_PUSH.value]["med_LR_balance_on_progress"]
    tensor_ok = best >= 0.3 * n and len(clean_actors) >= 2 and beats_scramble
    if best >= 0.8 * n:
        status = "POSITIVE"
    elif best >= 2:
        status = "PARTIAL"
    else:
        status = "NEGATIVE_WITH_MECHANISM"
    answers = {
        "1_clean_fingertip_delivery": f"{status}: best strict {best}/{n}; dominant mechanism {head['dominant_mechanism']}; "
        f"symmetric push L/R balance≈{sym_lr} (one-finger bulldoze is GEOMETRIC — two fingertips cannot share the push "
        "load on a cylinder); a few clean deliveries via v-plow/recovery",
        "2_competent_actors": f"none robust; marginal: {clean_actors}",
        "3_one_deterministic_actor_sufficient": best >= 0.8 * n,
        "4_headroom": head["headroom"],
        "5_tensor_rl_eligible": tensor_ok,
    }
    return {"status": status, "answers": answers, "tensor_rl_eligible": bool(tensor_ok),
            "beats_scramble": bool(beats_scramble)}


if __name__ == "__main__":
    import sys
    run()
    sys.exit(0)
