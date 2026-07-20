"""Bridge-relay driver — build the empirical TRANSPORT_READY basin, a reverse curriculum, train the BRIDGE_POLICY
progressively, and run the §7 causal relay evaluation. Bounded budgets (overnight); everything reuses
:mod:`coin_bridge_relay`. The frozen transport policy, delivery-v2b reward and strict predicate are never changed.
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
    build_basin, eval_relay, load_transport_policy, make_reverse_curriculum, relay_rollout, train_bridge,
)
from hymeko_rl.experiments.coin_clearance_curriculum import _CURDIR
from hymeko_rl.experiments.coin_generator_exp import _greedy, _restore_generated, direct_env
from hymeko_rl.experiments.coin_problem_generator import load_configs
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import rollout

_STRONG = "04870b0e0357ecb5"
_BUDGETS = {"B0_ready": 8_000, "B1_near": 12_000, "B2_mid": 15_000, "B3_far": 15_000, "B4_clear_start": 25_000}


def _find_strong(env, held):
    for c in held:
        _restore_generated(env, c.snapshot)
        if snapshot_hash(snapshot_planar(env.inner)).startswith(_STRONG[:12]):
            return c
    return None


def _causal(env, bridge, bridge0, transport, det, snaps) -> dict:
    """§7 causal comparison on the identical states: transport-alone / untrained-relay / trained-relay / zero-action."""
    def _strict_pool(action_or_relay, is_relay):
        hits = 0
        for s in snaps:
            if is_relay:
                tr, _ = relay_rollout(env, action_or_relay, transport, det, s)
            else:
                _restore_generated(env, s)
                tr = rollout(env, action_or_relay, max_steps=60)
            hits += int(bool(policy_strict(tr)))
        return hits
    def zero(_i, _t, _o):
        return np.zeros(6, np.float32)
    return dict(n=len(snaps),
                transport_alone=_strict_pool(_greedy(transport), False),
                untrained_relay=_strict_pool(bridge0, True),
                trained_relay=_strict_pool(bridge, True),
                zero_action=_strict_pool(zero, False))


def run(steps_scale: float, seed: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = direct_env()
    tp = load_transport_policy()
    held1 = load_configs(_CURDIR / "STAGE1_held.pkl")
    held2 = load_configs(_CURDIR / "STAGE2_held.pkl")
    tr0 = load_configs(_CURDIR / "STAGE0_train.pkl")
    tr1 = load_configs(_CURDIR / "STAGE1_train.pkl")
    tr2 = load_configs(_CURDIR / "STAGE2_train.pkl")
    strong = _find_strong(env, held1)

    # ── Phase 2: basin from many seeds (strong state + STAGE1 held) ──
    seeds = [strong.snapshot] + [c.snapshot for c in held1]
    labels, det = build_basin(env, tp, seeds, stride=2, n_robust=3)
    ready_n = sum(lab.label == "TRANSPORT_READY" for lab in labels)
    print(f"[basin] {len(labels)} candidates | labels={dict(Counter(lab.label for lab in labels))} | "
          f"READY={ready_n} | enter={det.enter_thresh:.3f} exit={det.exit_thresh:.3f}", flush=True)

    # ── Phase 3: reverse curriculum (near→far + clear-start) ──
    clear_start = [c.snapshot for c in (tr0 + tr1 + tr2)]
    bands = make_reverse_curriculum(labels, det, clear_start)
    print(f"[curriculum] band sizes: { {k: len(v) for k, v in bands.items()} }", flush=True)

    # ── untrained-relay baseline actor (warm-started transport clone, NO bridge training) ──
    from hymeko_rl.train.sac import build_sac
    bridge0, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    bridge0.load_state_dict(torch.load("experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt",
                                       map_location="cpu"))

    # ── Phase 6: progressive training ──
    bridge = _critics = None
    band_reports = []
    for band, snaps in bands.items():
        if not snaps:
            continue
        budget = max(2_000, int(_BUDGETS[band] * steps_scale))
        print(f"[train {band}] {len(snaps)} states, {budget} steps", flush=True)
        init = None if bridge is None else (bridge, _critics)
        bridge, _critics = train_bridge(det, tp, snaps, steps=budget, seed=seed, init_actor=init, log_every=budget)
        m1 = eval_relay(env, bridge, tp, det, [c.snapshot for c in held1])
        m2 = eval_relay(env, bridge, tp, det, [c.snapshot for c in held2])
        band_reports.append(dict(band=band, budget=budget, held1=m1, held2=m2))
        print(f"[eval {band}] held1 strict={m1['strict']}/{m1['n']} ready_entry={m1['ready_entry_rate']} "
              f"handoff={m1['handoff_rate']} | held2 strict={m2['strict']}/{m2['n']} "
              f"ready_entry={m2['ready_entry_rate']}", flush=True)
        torch.save(bridge.state_dict(), out / f"bridge_{band}.pt")

    # ── Phase 7: causal comparison on held1 + held2 ──
    causal1 = _causal(env, bridge, bridge0, tp, det, [c.snapshot for c in held1])
    causal2 = _causal(env, bridge, bridge0, tp, det, [c.snapshot for c in held2])
    cls = _classify(causal1, causal2, band_reports)
    result = dict(seed=seed, steps_scale=steps_scale, ready_states=ready_n, thresholds=[det.enter_thresh, det.exit_thresh],
                  band_sizes={k: len(v) for k, v in bands.items()}, band_reports=band_reports,
                  causal_held1=causal1, causal_held2=causal2, classification=cls)
    (out / "bridge_result.json").write_text(json.dumps(result, indent=1, default=float))
    torch.save(bridge.state_dict(), out / "bridge_final.pt")
    print(f"[causal held1] {causal1}\n[causal held2] {causal2}\n=== CLASSIFICATION: {cls}", flush=True)
    return result


def _classify(c1: dict, c2: dict, bands: list[dict]) -> str:
    """§7 taxonomy: BRIDGE_POSITIVE / BRIDGE_CONTACT_POSITIVE / NO_EFFECT / BLOCKED."""
    tp_all = c1["transport_alone"] + c2["transport_alone"]
    relay_all = c1["trained_relay"] + c2["trained_relay"]
    ready_entry = max((b["held1"]["ready_entry"] + b["held2"]["ready_entry"]) for b in bands) if bands else 0
    if relay_all >= 8 and relay_all > tp_all and c1["zero_action"] + c2["zero_action"] == 0:
        return "BRIDGE_POSITIVE"
    if ready_entry > 0 and relay_all >= tp_all:
        return "BRIDGE_CONTACT_POSITIVE"
    return "NO_EFFECT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-scale", type=float, default=1.0, help="scale all band budgets (1.0 = full)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_bridge_relay/run")
    a = ap.parse_args()
    run(a.steps_scale, a.seed, Path(a.out))


if __name__ == "__main__":
    main()
