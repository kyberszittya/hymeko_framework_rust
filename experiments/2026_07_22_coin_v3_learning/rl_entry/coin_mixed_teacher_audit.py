"""Item 8 of CHUNK_SUPERVISED_M1_FEEDBACK_V1 — mixed-teacher-averaging audit. The M=1 diagnostic ruled out the
execution horizon (M=1 was WORSE than M=2), so this audits the labels: are the two teachers (exact pi_0 fallback vs
planner improvement) far apart in nearby states, and does the reproduced V2 chunk actor learn their AVERAGE first action
(worse than both)? No TD3/SAC. Reuses the canonical V2-actor reproduction + teacher-annotated dataset."""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_chunk_td3 import ACT_DIM, K  # noqa: E402
from hymeko_rl.coin_delivery.coin_feedback_chunk_v2 import build_teacher_annotated_dataset, reproduce_v2_actor  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_mixed_teacher_audit import mixed_teacher_metrics  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
TDCFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/mixed_teacher_audit.json"


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def main():
    cfg = json.load(open(TDCFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True); log = lambda *a: print(*a, flush=True)
    train = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["train"][m])]
    horizon = cfg["horizon"]

    log(f"[{time.strftime('%H:%M:%S')}] reproducing V2 actor (seed 0)...")
    t = time.time(); actor, _X, _Y, _stats = reproduce_v2_actor(pi0, train, horizon=horizon, log=log)
    log(f"  ({time.time()-t:.0f}s) building teacher-annotated dataset (both teacher first-actions per state)...")
    X, pi0_first, planner_first, label_first, prov, flags = build_teacher_annotated_dataset(pi0, train, horizon=horizon)
    log(f"  {len(X)} states ({sum(p=='planner' for p in prov)} planner / {sum(p=='pi0_fallback' for p in prov)} pi0_fallback)")

    with torch.no_grad():
        learned_first = actor(torch.tensor(X)).numpy().reshape(len(X), K, ACT_DIM)[:, 0, :].astype(np.float32)

    m = mixed_teacher_metrics(X, pi0_first, planner_first, label_first, prov, flags, learned_first, k=8)
    out = {"contract": "MIXED_TEACHER_AVERAGING_AUDIT (item 8)", "date": "2026-07-23", "pi0_sha": cfg["pi0_sha"],
           "no_td3": True, "trigger": "M1_FEEDBACK_NO_GAIN (M=1 worse than M=2 => horizon not the limiter)", **m}
    json.dump(out, open(OUT, "w"), indent=1, default=float)

    log(f"\n  teacher gap (pi0 vs planner first-action): median {m['teacher_gap']['median']:.3f} p90 {m['teacher_gap']['p90']:.3f}"
        f"  (planner states {m['teacher_gap']['mean_planner_states']:.3f})")
    log(f"  local mode disagreement (kNN): {m['nn_mode_disagreement_mean']:.3f}   cond-var first label: {m['conditional_variance_first_label_mean']:.4f}")
    log(f"  first-action error: mean {m['first_action_error']['mean']:.3f}   err~mode-mix corr: {m['error_vs_mode_disagreement_corr']:+.3f}")
    log(f"  error by admissibility: {m['error_by_admissibility_boundary']}")
    log(f"  learned BETWEEN teachers: {m['between_teachers_frac']:.3f}  (between & off-label {m['between_and_off_label_frac']:.3f})")
    log(f"  planner-state segment pos (1.0=on planner target): mean {m['planner_states_mean_segment_pos']}  pulled-to-pi0 frac {m['planner_states_pulled_toward_pi0_frac']}")
    log(f"\n→ {m['verdict']}\nwrote {OUT}\nMIXED_TEACHER_AUDIT_DONE")


if __name__ == "__main__":
    main()
