"""Post-run strict-delivery gap diagnostic (read-only; no reward/actor/replay/env/predicate/architecture change).
Reproduces the eval#1 strict event, sweeps the saved checkpoints on DEMO+VAL states through the canonical rollout(),
and decomposes the strict predicate per trajectory. Emits strict_gap_results.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_two_arm_sac import _DEMO_SEEDS, _VAL_SEEDS, direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import (
    _BODY_SHOVE_MAX,
    _DWELL_STEPS,
    _ONE_FINGER_MAX,
    _SETTLE_VEL,
    _attribution_from_trace,
    rollout,
)
from hymeko_rl.train.sac import build_sac

_RUN = Path("experiments/2026_07_20_coin_two_arm_sac_100k")
_ATTR_MIN = 0.60


def load_actor(path: Path):
    actor, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()
    return actor


def greedy_of(actor):
    def g(inner, t, obs):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return g


def _best_dwell_interval(steps):
    """Return (max_consec, speeds_in_that_interval) for the longest consecutive in-zone run."""
    best_len, best_speeds, run, run_speeds = 0, [], 0, []
    for s in steps:
        if s.in_zone:
            run += 1
            run_speeds.append(s.disk_vel_norm)
            if run > best_len:
                best_len, best_speeds = run, list(run_speeds)
        else:
            run, run_speeds = 0, []
    return best_len, best_speeds


def _state_hash(env) -> str:
    q = np.asarray(env.inner.data.qpos, np.float64)
    return hashlib.sha256(q.tobytes()).hexdigest()[:16]


def decompose(env, actor, seed: int) -> dict:
    env.reset(seed=int(seed))
    shash = _state_hash(env)
    tr = rollout(env, greedy_of(actor), max_steps=60)
    att = _attribution_from_trace(tr)
    max_consec, speeds = _best_dwell_interval(tr.steps)
    ff = att.fingertip_fraction
    clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= _ONE_FINGER_MAX
    dwell_pass = max_consec >= _DWELL_STEPS
    settle_pass = tr.settle_vel <= _SETTLE_VEL
    attr_pass = ff >= _ATTR_MIN
    body_pass = att.alpha_body <= _BODY_SHOVE_MAX
    mech_pass = bool(clean)
    return dict(
        seed=int(seed), state_hash=shash,
        initially_successful=tr.initial_success, entered_zone=tr.loose,
        max_consecutive_in_zone=int(max_consec),
        dwell_interval_max_speed=float(max(speeds)) if speeds else None,
        dwell_interval_mean_speed=float(np.mean(speeds)) if speeds else None,
        settle_vel=float(tr.settle_vel), fingertip_attribution=float(ff),
        body_shove_fraction=float(att.alpha_body), clean_mechanism=mech_pass,
        bilateral_contact=float(tr.both_frac), progress=float(tr.progress),
        strict_delivery=bool(policy_strict(tr)),
        dwell_pass=bool(dwell_pass), settle_pass=bool(settle_pass), attribution_pass=bool(attr_pass),
        body_pass=bool(body_pass), mechanism_pass=mech_pass,
        dwell_margin=int(max_consec - _DWELL_STEPS), settle_margin=float(_SETTLE_VEL - tr.settle_vel),
        attribution_margin=float(ff - _ATTR_MIN), body_margin=float(_BODY_SHOVE_MAX - att.alpha_body),
    )


def main() -> None:
    checkpoints = {"eval1_best": _RUN / "sac_actor_best.pt", "final": _RUN / "sac_actor_final.pt"}
    # dedup identical files by content hash
    seen, unique = {}, {}
    for label, p in checkpoints.items():
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        (unique.setdefault(h, {"labels": [], "path": str(p)})["labels"].append(label))
        seen[label] = h
    env = direct_env()
    all_states = list(_DEMO_SEEDS) + list(_VAL_SEEDS)

    # ── §2 eval#1 reproduction: find the strict VAL state under the eval1 checkpoint, reproduce 10x ──────────────
    best_actor = load_actor(_RUN / "sac_actor_best.pt")
    strict_states = [s for s in _VAL_SEEDS if decompose(env, best_actor, s)["strict_delivery"]]
    repro = None
    if strict_states:
        sseed = strict_states[0]
        reps = [decompose(env, best_actor, sseed) for _ in range(10)]
        n_strict = sum(r["strict_delivery"] for r in reps)
        hashes = {r["state_hash"] for r in reps}
        strict_all_equal = all(r["strict_delivery"] == reps[0]["strict_delivery"] for r in reps)
        cls = ("REPRODUCIBLE_STRICT" if n_strict == 10 else
               "ISOLATED_VALID_STRICT" if (n_strict >= 1 and len(hashes) == 1 and strict_all_equal) else
               "MONITOR_BOUNDARY_EVENT" if 1 <= n_strict < 10 else
               "NONDETERMINISTIC_OR_INVALID")
        # deterministic single-state => if 10 identical reps all agree, it is reproducible-or-not by that value
        if len(hashes) == 1 and n_strict in (0, 10):
            cls = "REPRODUCIBLE_STRICT" if n_strict == 10 else "MONITOR_BOUNDARY_EVENT"
        repro = dict(strict_state_seed=int(sseed), n_reps=10, n_strict=int(n_strict),
                     unique_state_hashes=len(hashes), classification=cls, reps=reps)
    else:
        repro = dict(strict_state_seed=None, note="eval1 checkpoint produced no strict VAL state on deterministic replay",
                     classification="LOGGING_OR_CHECKPOINT_MISMATCH")

    # ── §3/§4 checkpoint sweep + strict decomposition on DEMO+VAL ────────────────────────────────────────────────
    sweep = {}
    for h, info in unique.items():
        actor = load_actor(Path(info["path"]))
        rows = [decompose(env, actor, s) for s in all_states]
        loose = [r for r in rows if r["entered_zone"] and not r["initially_successful"]]
        conds = ["dwell_pass", "settle_pass", "attribution_pass", "body_pass", "mechanism_pass"]
        n_loose = max(1, len(loose))
        pass_rates = {c: sum(r[c] for r in loose) / n_loose for c in conds}
        # joint failure combinations among loose-success trajectories
        joint = {}
        for r in loose:
            fails = tuple(c.replace("_pass", "").upper() for c in conds if not r[c])
            key = " + ".join(fails) if fails else "NONE(all pass)"
            joint[key] = joint.get(key, 0) + 1
        margins = {c: {"median": float(np.median([r[f"{c}_margin"] for r in loose])) if loose else None,
                       "best": float(np.max([r[f"{c}_margin"] for r in loose])) if loose else None}
                   for c in ("dwell", "settle", "attribution", "body")}
        sweep[h] = dict(labels=info["labels"], path=info["path"], content_hash=h,
                        n_states=len(rows), n_loose_success=len(loose),
                        n_strict=sum(r["strict_delivery"] for r in rows),
                        zone_rate=sum(r["entered_zone"] for r in rows) / len(rows),
                        mean_progress=float(np.mean([r["progress"] for r in rows])),
                        cond_pass_rate_given_zone=pass_rates, joint_failure_combos=joint,
                        margins=margins, rows=rows)

    # ── §5 blocker classification (from the eval1_best sweep, the checkpoint with any strict signal) ─────────────
    key = next((h for h, v in sweep.items() if "eval1_best" in v["labels"]), next(iter(sweep)))
    pr = sweep[key]["cond_pass_rate_given_zone"]
    contact_group_ok = min(pr["attribution_pass"], pr["body_pass"], pr["mechanism_pass"])
    stab_group_ok = min(pr["dwell_pass"], pr["settle_pass"])
    # Evidence-based (not a fixed 0.5 cut): the blocker is the group carrying the failures. If stabilization is
    # (near-)clean while the contact group fails, the gap is contact strategy; the symmetric case is stabilization;
    # only genuine two-sided failure is MIXED.
    _STRONG, _GAP = 0.8, 0.15
    if stab_group_ok >= _STRONG and contact_group_ok < stab_group_ok - _GAP:
        blocker = "CONTACT_STRATEGY"
    elif contact_group_ok >= _STRONG and stab_group_ok < contact_group_ok - _GAP:
        blocker = "POST_ENTRY_STABILIZATION"
    else:
        blocker = "MIXED"
    # for MIXED tie-break: component with largest normalized median deficit
    loose_all = [r for r in sweep[key]["rows"] if r["entered_zone"] and not r["initially_successful"]]
    deficits = {}
    if loose_all:
        norms = {"dwell": 6.0, "settle": _SETTLE_VEL, "attribution": _ATTR_MIN, "body": _BODY_SHOVE_MAX}
        for c, nrm in norms.items():
            deficits[c] = float(np.median([-r[f"{c}_margin"] for r in loose_all]) / nrm)

    out = dict(run_dir=str(_RUN), checkpoint_hashes=seen, unique_checkpoints=len(unique),
               eval1_reproduction=repro, checkpoint_sweep=sweep,
               blocker={"classification": blocker, "contact_group_min_pass": contact_group_ok,
                        "stabilization_group_min_pass": stab_group_ok, "normalized_median_deficits": deficits})
    (_RUN / "strict_gap_results.json").write_text(json.dumps(out, indent=1, default=float))
    print("=== eval#1 reproduction ===")
    print(json.dumps({k: repro[k] for k in repro if k != "reps"}, indent=1, default=float))
    print("=== checkpoint sweep ===")
    for h, v in sweep.items():
        print(f"  {v['labels']} n_loose={v['n_loose_success']} n_strict={v['n_strict']} zone={v['zone_rate']:.2f} "
              f"prog={v['mean_progress']:.4f} pass(given zone)={ {k: round(x,2) for k,x in v['cond_pass_rate_given_zone'].items()} }")
        print(f"     joint_failures={v['joint_failure_combos']}")
    print(f"=== BLOCKER: {blocker} | contact_ok={contact_group_ok:.2f} stab_ok={stab_group_ok:.2f} deficits={ {k: round(x,2) for k,x in deficits.items()} }")
    print(f"saved {_RUN/'strict_gap_results.json'}")


if __name__ == "__main__":
    main()
