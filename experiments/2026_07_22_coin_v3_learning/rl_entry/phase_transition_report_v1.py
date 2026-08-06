"""§10/§11 phase-transition report on the frozen Stage-1 banks. Collects deterministic (explore=False, pi_late=pi_0
copy) late episodes and counts DYNAMIC phase_t → phase_tp1 transitions + real per-phase coverage. No training."""
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_phase_conditioning import PHASES, make_phase_actor_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_phase_stage1c import collect_late_episode_phase  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
CFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_baseline_v1_config.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/phase_transition_report_v1.json"


def main():
    cfg = json.load(open(CFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True)
    fams = tuple(cfg["stage1"]["families"]); horizon = cfg["stage1"]["horizon"]
    pl = make_phase_actor_from_pi0(pi0, trainable=True)                 # zero-init ⇒ actions == pi_0 (phase-blind rollout)
    starts = []
    for key in ("late_train", "late_dev"):
        for r in cfg["banks"][key]["rows"]:
            if r[2] in fams:
                starts.append(LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]))

    trans = Counter(); phase_t_counts = Counter(); n_ep = 0
    for ls in starts:
        trs = collect_late_episode_phase(pi0, pl, ls, None, horizon=horizon, explore=False)
        if not trs:
            continue
        n_ep += 1
        for t in trs:
            phase_t_counts[t["phase_t"]] += 1
            if t["phase_t"] != t["phase_tp1"]:
                trans[(t["phase_t"], t["phase_tp1"])] += 1

    expected = [("target_entry", "braking"), ("braking", "settling_dwell"), ("settling_dwell", "braking")]
    exp_counts = {f"{a}->{b}": trans.get((a, b), 0) for a, b in expected}
    unexpected = {f"{a}->{b}": c for (a, b), c in sorted(trans.items(), key=lambda kv: -kv[1])
                  if (a, b) not in expected}
    occurring = sorted(phase_t_counts)
    missing_stage1 = [f for f in fams if f not in phase_t_counts]

    out = {"n_episodes": n_ep, "n_starts": len(starts), "horizon": horizon, "stage1_families": list(fams),
           "expected_transition_counts": exp_counts, "unexpected_transition_counts": unexpected,
           "phase_t_step_counts": dict(phase_t_counts),
           "phases_occurring_dynamically": occurring, "stage1_phases_never_occurring": missing_stage1,
           "all_phases": list(PHASES)}
    json.dump(out, open(OUT, "w"), indent=1)
    print("expected transitions:", exp_counts)
    print("unexpected transitions:", unexpected)
    print("phase_t step counts:", dict(phase_t_counts))
    print("stage-1 phases never occurring dynamically:", missing_stage1 or "none")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
