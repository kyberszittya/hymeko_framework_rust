"""Option-chain driver — train APPROACH + CAPTURE separately, evaluate the APPROACH→CAPTURE→frozen-TRANSPORT chain by
clearance band, diagnose failures by option, and run the §9 causal controls. Reuses :mod:`coin_option_chain` +
:mod:`coin_bridge_relay`. Frozen transport policy / delivery-v2b / strict predicate unchanged.
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
from hymeko_rl.experiments.coin_bridge_relay import build_basin, load_transport_policy
from hymeko_rl.experiments.coin_clearance_curriculum import _CURDIR, _clearance
from hymeko_rl.experiments.coin_generator_exp import _greedy, _restore_generated, direct_env
from hymeko_rl.experiments.coin_option_chain import (
    ApproachRewardEnv, CaptureRewardEnv, chain_rollout, option_banks, train_option,
)
from hymeko_rl.experiments.coin_problem_generator import load_configs
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import rollout

_STRONG = "04870b0e0357ecb5"
_TP_CKPT = "experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt"
_BANDS = [("+0.018-0.030", 0.018, 0.030), ("+0.030-0.045", 0.030, 0.045),
          ("+0.045-0.060", 0.045, 0.060), ("+0.060-0.080", 0.060, 0.080)]


def _find_strong(env, held):
    for c in held:
        _restore_generated(env, c.snapshot)
        if snapshot_hash(snapshot_planar(env.inner)).startswith(_STRONG[:12]):
            return c
    return None


def _clr(env, snap) -> float:
    _restore_generated(env, snap)
    return float(_clearance(env.inner))


def _chain_eval(env, ap, cap, tp, det, snaps) -> dict:
    """Full-chain metrics + §8 per-option failure diagnosis on held states."""
    fc = ce = ho = st = phc = 0
    fails = Counter()
    for s in snaps:
        tr, log = chain_rollout(env, ap, cap, tp, det, s)
        strict = bool(policy_strict(tr))
        fc += int(log.first_contact_step >= 0)
        ce += int(log.capture_step >= 0)
        ho += int(log.handoffs > 0)
        st += int(strict)
        if log.handoffs > 0:
            phc += int(strict)
        if not strict:
            if log.first_contact_step < 0:
                fails["APPROACH_FAILURE"] += 1
            elif log.handoffs == 0:
                fails["CAPTURE_FAILURE"] += 1
            else:
                fails["TRANSPORT_FAILURE"] += 1
    n = max(1, len(snaps))
    return dict(n=len(snaps), first_contact=fc, capture_entry=ce, handoff=ho, strict=st, post_handoff_strict=phc,
                first_contact_rate=round(fc / n, 3), handoff_rate=round(ho / n, 3), strict_rate=round(st / n, 3),
                failures=dict(fails))


def _by_band(env, snaps) -> dict:
    out = {name: [] for name, _lo, _hi in _BANDS}
    for s in snaps:
        c = _clr(env, s)
        for name, lo, hi in _BANDS:
            if lo <= c < hi:
                out[name].append(s)
                break
    return out


def _causal(env, ap, cap, tp, det, snaps) -> dict:
    def zero(_i, _t, _o):
        return np.zeros(6, np.float32)

    def plain(fn):
        return sum(int(bool(policy_strict(rollout(env, fn, max_steps=60)))) for s in snaps for _ in [_restore_generated(env, s)])

    def chain(a, c):
        return sum(int(bool(policy_strict(chain_rollout(env, a, c, tp, det, s)[0]))) for s in snaps)
    # APPROACH+CAPTURE without transport = feed a zero "transport" (never actually delivers) — measured as chain w/ tp=zero
    return dict(n=len(snaps), transport_alone=plain(_greedy(tp)),
                approach_only=plain(_greedy(ap)), full_chain=chain(ap, cap), zero_action=plain(zero))


def _classify(bands_eval: dict, causal: dict) -> str:
    """§12 taxonomy: CHAIN_POSITIVE / CAPTURE_POSITIVE / APPROACH_POSITIVE / NO_EFFECT / BLOCKED."""
    hard = bands_eval.get("+0.030-0.045", {})
    hard_strict = hard.get("strict", 0)
    hard_handoff = hard.get("handoff", 0)
    fc_rate = max((b.get("first_contact_rate", 0) for b in bands_eval.values()), default=0)
    total_handoff = sum(b.get("handoff", 0) for b in bands_eval.values())
    ta = causal.get("transport_alone", 0)
    full = causal.get("full_chain", 0)
    if hard_strict >= 2 and hard_handoff > 0 and full > ta:
        return "CHAIN_POSITIVE"
    if total_handoff > 0 and full >= ta:
        return "CAPTURE_POSITIVE"
    if fc_rate >= 0.5:
        return "APPROACH_POSITIVE"
    return "NO_EFFECT"


def run(steps_scale: float, seed: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = direct_env()
    tp = load_transport_policy()
    held1 = load_configs(_CURDIR / "STAGE1_held.pkl")
    held2 = load_configs(_CURDIR / "STAGE2_held.pkl")
    strong = _find_strong(env, held1)
    labels, det = build_basin(env, tp, [strong.snapshot] + [c.snapshot for c in held1], stride=2, n_robust=3)
    banks = option_banks(labels)
    print(f"[basin] READY={sum(lab.label == 'TRANSPORT_READY' for lab in labels)} | banks="
          f"{ {k: len(v) for k, v in banks.items()} } | enter={det.enter_thresh:.3f}", flush=True)

    tr_all = load_configs(_CURDIR / "STAGE0_train.pkl") + load_configs(_CURDIR / "STAGE1_train.pkl") + \
        load_configs(_CURDIR / "STAGE2_train.pkl")
    clear_start = [c.snapshot for c in tr_all]
    capture_pool = banks["A1_first_contact"] + banks["C1_bilateral"]
    if not capture_pool:
        capture_pool = banks["T0_ready"]

    ap_steps = max(3_000, int(45_000 * steps_scale))
    cap_steps = max(3_000, int(45_000 * steps_scale))
    print(f"[train APPROACH] pool={len(clear_start)} steps={ap_steps}", flush=True)
    ap, _ = train_option(lambda i, r: ApproachRewardEnv(i, clear_start, r), steps=ap_steps, seed=seed,
                         warm_from=_TP_CKPT, log_every=ap_steps)
    torch.save(ap.state_dict(), out / "approach.pt")
    print(f"[train CAPTURE] pool={len(capture_pool)} steps={cap_steps}", flush=True)
    cap, _ = train_option(lambda i, r: CaptureRewardEnv(i, det, capture_pool, r), steps=cap_steps, seed=seed,
                          warm_from=_TP_CKPT, log_every=cap_steps)
    torch.save(cap.state_dict(), out / "capture.pt")

    all_held = [c.snapshot for c in (held1 + held2)]
    band_snaps = _by_band(env, all_held)
    bands_eval = {name: _chain_eval(env, ap, cap, tp, det, snaps) for name, snaps in band_snaps.items() if snaps}
    for name, m in bands_eval.items():
        print(f"[band {name}] n={m['n']} first_contact={m['first_contact_rate']} handoff={m['handoff']} "
              f"strict={m['strict']}/{m['n']} post_handoff={m['post_handoff_strict']} fails={m['failures']}", flush=True)
    causal = _causal(env, ap, cap, tp, det, [c.snapshot for c in held1])
    max_clr = -9.9
    for s in all_held:
        tr, _ = chain_rollout(env, ap, cap, tp, det, s)
        if policy_strict(tr):
            max_clr = max(max_clr, _clr(env, s))
    cls = _classify(bands_eval, causal)
    result = dict(seed=seed, steps_scale=steps_scale, banks={k: len(v) for k, v in banks.items()},
                  bands_eval=bands_eval, causal_held1=causal, max_certified_clearance=round(float(max_clr), 4),
                  classification=cls)
    (out / "chain_result.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[causal] {causal}\nmax_cert_clearance={max_clr:+.4f}\n=== CLASSIFICATION: {cls}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_option_chain/run")
    a = ap.parse_args()
    run(a.steps_scale, a.seed, Path(a.out))


if __name__ == "__main__":
    main()
