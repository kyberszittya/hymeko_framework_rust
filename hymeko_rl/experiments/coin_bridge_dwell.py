"""Dwell-relay iteration — correct the bridge train/eval mismatch: the bridge is now trained to HOLD readiness for 3
consecutive steps (the relay's handoff condition), and the relay fallback no longer punishes the transport policy for
leaving the ready region (the plumbing defect the one-step diagnostic exposed). Progressive curriculum with rehearsal
(retention), lexicographic checkpoint selection, and the §9 5-arm causal controls. Reuses :mod:`coin_bridge_relay`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.planar_snapshot import snapshot_planar
from hymeko_rl.experiments.coin_bridge_relay import (
    build_basin, load_transport_policy, make_reverse_curriculum, relay_rollout, train_bridge,
)
from hymeko_rl.experiments.coin_clearance_curriculum import _CURDIR, _clearance
from hymeko_rl.experiments.coin_generator_exp import _greedy, _restore_generated, direct_env
from hymeko_rl.experiments.coin_problem_generator import load_configs
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import rollout

_STRONG = "04870b0e0357ecb5"
_DWELL = 3
_BUDGETS = {"B0_ready": 25_000, "B1_near": 30_000, "B2_mid": 40_000, "B3_far": 50_000, "B4_clear_start": 50_000}


def _find_strong(env, held):
    for c in held:
        _restore_generated(env, c.snapshot)
        if snapshot_hash(snapshot_planar(env.inner)).startswith(_STRONG[:12]):
            return c
    return None


def _relay_metrics(env, bridge, transport, det, snaps, *, dwell=_DWELL) -> dict:
    """Relay eval with the explicit chain log (§8): strict / handoff / dwell / post-handoff completion."""
    strict = handoff = dwell_ok = post = ready_entry = 0
    for s in snaps:
        tr, log = relay_rollout(env, bridge, transport, det, s, hysteresis=dwell)
        st = bool(policy_strict(tr))
        strict += int(st)
        ready_entry += int(log.ready_step >= 0)
        dwell_ok += int(log.max_ready_streak >= dwell)
        if log.handoffs > 0:
            handoff += 1
            post += int(st)                                        # relay-valid success: went through a real handoff
    n = max(1, len(snaps))
    return dict(n=len(snaps), strict=strict, handoff=handoff, dwell_reached=dwell_ok, post_handoff_strict=post,
                ready_entry=ready_entry, handoff_rate=round(handoff / n, 3), dwell_rate=round(dwell_ok / n, 3))


def _rehearsal_pool(bands: dict, band: str, rng: np.random.Generator) -> list:
    """§6 rehearsal: >=30% B0/B1 readiness-boundary samples mixed into every later band (retention)."""
    cur = bands[band]
    if band in ("B0_ready", "B1_near"):
        return cur
    earlier = bands["B0_ready"] + bands["B1_near"]
    k = max(1, int(0.30 / 0.70 * len(cur)))                        # ~30% of the mixed pool from earlier bands
    idx = rng.integers(0, max(1, len(earlier)), size=min(k, len(earlier)))
    return cur + [earlier[i] for i in idx]


def _causal(env, bridge, bridge0, transport, det, snaps) -> dict:
    """§9 five-arm control on identical states (dwell=3 relay, plus the 1-step diagnostic relay for bridge0)."""
    def relay_strict(br, dwell):
        h = 0
        for s in snaps:
            tr, _ = relay_rollout(env, br, transport, det, s, hysteresis=dwell)
            h += int(bool(policy_strict(tr)))
        return h

    def plain_strict(fn):
        h = 0
        for s in snaps:
            _restore_generated(env, s)
            h += int(bool(policy_strict(rollout(env, fn, max_steps=60))))
        return h

    def zero(_i, _t, _o):
        return np.zeros(6, np.float32)
    return dict(n=len(snaps), transport_alone=plain_strict(_greedy(transport)),
                prev_bridge_3step=relay_strict(bridge0, 3), prev_bridge_1step=relay_strict(bridge0, 1),
                dwell_bridge_3step=relay_strict(bridge, 3), zero_action=plain_strict(zero))


def _classify(c1, c2, best) -> str:
    """§10 taxonomy: RELAY_POSITIVE / RELAY_HANDOFF_POSITIVE / DETECTOR_MISMATCH / NO_EFFECT / BLOCKED."""
    ta = c1["transport_alone"] + c2["transport_alone"]
    dwell_relay = c1["dwell_bridge_3step"] + c2["dwell_bridge_3step"]
    handoff = best["handoff"]
    post = best["post_handoff_strict"]
    if handoff > 0 and post == 0:
        return "DETECTOR_MISMATCH"
    if dwell_relay >= 8 and dwell_relay > ta and handoff > 0:
        return "RELAY_POSITIVE"
    if handoff > 0 and post > 0 and dwell_relay >= ta:
        return "RELAY_HANDOFF_POSITIVE"
    return "NO_EFFECT"


def run(steps_scale: float, seed: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = direct_env()
    tp = load_transport_policy()
    held1 = load_configs(_CURDIR / "STAGE1_held.pkl")
    held2 = load_configs(_CURDIR / "STAGE2_held.pkl")
    strong = _find_strong(env, held1)
    labels, det = build_basin(env, tp, [strong.snapshot] + [c.snapshot for c in held1], stride=2, n_robust=3)
    ready_n = sum(lab.label == "TRANSPORT_READY" for lab in labels)
    print(f"[basin] READY={ready_n} enter={det.enter_thresh:.3f} exit={det.exit_thresh:.3f} "
          f"labels={dict(Counter(lab.label for lab in labels))}", flush=True)
    tr0 = load_configs(_CURDIR / "STAGE0_train.pkl")
    tr1 = load_configs(_CURDIR / "STAGE1_train.pkl")
    bands = make_reverse_curriculum(labels, det, [c.snapshot for c in (tr0 + tr1)])
    print(f"[curriculum] { {k: len(v) for k, v in bands.items()} }", flush=True)

    from hymeko_rl.train.sac import build_sac
    bridge0, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    bridge0.load_state_dict(torch.load("experiments/2026_07_21_coin_bridge_relay/run_s0/bridge_B0_ready.pt",
                                       map_location="cpu"))
    rng = np.random.default_rng(seed)
    bridge = crit = None
    band_reports, best = [], {"key": (-1, -1, -1, -1), "ckpt": None, "handoff": 0, "post_handoff_strict": 0}
    he1 = [c.snapshot for c in held1]
    he2 = [c.snapshot for c in held2]
    for band in ("B0_ready", "B1_near", "B2_mid", "B3_far", "B4_clear_start"):
        if not bands.get(band):
            continue
        budget = max(3_000, int(_BUDGETS[band] * steps_scale))
        pool = _rehearsal_pool(bands, band, rng)
        print(f"[train {band}] pool={len(pool)} ({len(bands[band])} band + rehearsal) steps={budget}", flush=True)
        init = None if bridge is None else (bridge, crit)
        bridge, crit = train_bridge(det, tp, pool, steps=budget, seed=seed, init_actor=init, log_every=budget)
        m1 = _relay_metrics(env, bridge, tp, det, he1)
        m2 = _relay_metrics(env, bridge, tp, det, he2)
        band_reports.append(dict(band=band, budget=budget, held1=m1, held2=m2))
        print(f"[eval {band}] held1 strict={m1['strict']}/24 handoff={m1['handoff']} dwell={m1['dwell_rate']} "
              f"post={m1['post_handoff_strict']} | held2 strict={m2['strict']}/24 handoff={m2['handoff']}", flush=True)
        torch.save(bridge.state_dict(), out / f"dwell_bridge_{band}.pt")
        # lexicographic checkpoint selection (§6): hardest-band relay strict, handoff, dwell, B0 retention
        key = (m2["strict"], m1["handoff"], m1["dwell_reached"], m1["strict"])
        if key > best["key"]:
            best.update(key=key, ckpt=str(out / f"dwell_bridge_{band}.pt"), band=band,
                        handoff=m1["handoff"] + m2["handoff"],
                        post_handoff_strict=m1["post_handoff_strict"] + m2["post_handoff_strict"])
            torch.save(bridge.state_dict(), out / "dwell_bridge_best.pt")

    best_bridge, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    best_bridge.load_state_dict(torch.load(best["ckpt"] or (out / "dwell_bridge_B0_ready.pt"), map_location="cpu"))
    causal1 = _causal(env, best_bridge, bridge0, tp, det, he1)
    causal2 = _causal(env, best_bridge, bridge0, tp, det, he2)
    best_metrics = {k: _relay_metrics(env, best_bridge, tp, det, he1)[k] + _relay_metrics(env, best_bridge, tp, det, he2)[k]
                    for k in ("handoff", "post_handoff_strict")}
    cls = _classify(causal1, causal2, best_metrics)
    # max certified clearance among relay-strict held states
    max_clr = -9.9
    for s in he1 + he2:
        tr, log = relay_rollout(env, best_bridge, tp, det, s, hysteresis=_DWELL)
        if policy_strict(tr):
            _restore_generated(env, s)
            max_clr = max(max_clr, _clearance(env.inner))
    result = dict(seed=seed, steps_scale=steps_scale, ready_states=ready_n,
                  thresholds=[det.enter_thresh, det.exit_thresh], band_sizes={k: len(v) for k, v in bands.items()},
                  band_reports=band_reports, best_band=best.get("band"), best_ckpt=best["ckpt"],
                  causal_held1=causal1, causal_held2=causal2, max_certified_clearance=round(float(max_clr), 4),
                  classification=cls)
    (out / "dwell_result.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[causal held1] {causal1}\n[causal held2] {causal2}\nmax_cert_clearance={max_clr:+.4f}\n"
          f"=== CLASSIFICATION: {cls}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_bridge_dwell/run")
    a = ap.parse_args()
    run(a.steps_scale, a.seed, Path(a.out))


if __name__ == "__main__":
    main()
