"""§9 pre-training report for DYNAMIC_PHASE_CURRICULUM_SCOPE_V1 (no training). Rebuilds banks by the LIVE persistent
control-phase, and measures control-phase occupancy, transition matrix, and detector-predicate overlaps with the
REWORKED detector (contact demoted to a flag). Emits STAGE1_DYNAMIC_PHASE_BANK_UNDERPOWERED if a Stage-1 phase lacks
enough persistent starts."""
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_dynamic_phase_scope import (  # noqa: E402
    CONTROL_PHASES,
    PRECEDENCE,
    STAGE1_CONTROL,
    AuthPhaseDetector,
    rebuild_control_phase_bank,
)
from hymeko_rl.coin_delivery.coin_late_start import late_start_bank_manifest  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.coin_stable_engagement import StableEngagementConfig, StableEngagementGate, stable_engagement_signals  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/dynamic_phase_scope_report_v1.json"
MIN_STARTS = 6                                                    # per Stage-1 phase to NOT be underpowered
REBUILD_SEEDS = range(6000, 6300)
OCC_SEEDS = range(6000, 6160)


def _base(pi0, o):
    import numpy as np, torch
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.as_tensor(o[None], dtype=torch.float32))[0].numpy(), -4, 4).astype(np.float32)


def occupancy(pi0, seeds, horizon=360):
    occ = Counter(); trans = Counter(); overlap = Counter(); n_multi = 0; n_states = 0
    for s in seeds:
        rl = CoinRL4Dof(horizon=horizon); o = rl.reset(int(s))
        gate = StableEngagementGate(StableEngagementConfig()); det = AuthPhaseDetector(); prev_cp = None
        for _k in range(horizon):
            pr = det.predicates_of(rl)                            # read predicates (advances context inside state_of below)
            cp, _cf = det.state_of(rl)
            g = gate.gate == 1.0
            if g:
                occ[cp] += 1; n_states += 1
                if len(pr["control_predicates"]) > 1:
                    n_multi += 1; overlap["+".join(sorted(pr["control_predicates"]))] += 1
                if prev_cp is not None and prev_cp != cp:
                    trans[(prev_cp, cp)] += 1
                prev_cp = cp
            a = _base(pi0, o); o2, _r, term, trunc, _ = rl.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            o = o2
            if term or trunc:
                break
    return occ, trans, overlap, n_multi, n_states


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    out = {"detector": "control_phase(4) + contact_flag(2); contact demoted to orthogonal flag", "precedence": PRECEDENCE,
           "control_phases": list(CONTROL_PHASES), "stage1_control": list(STAGE1_CONTROL), "min_starts": MIN_STARTS}
    # §1/§2 banks by live persistent control-phase, min_persist 2 and 4
    for mp in (2, 4):
        banks, counts = rebuild_control_phase_bank(pi0, REBUILD_SEEDS, min_persist=mp, per_phase=40)
        shas = {p: late_start_bank_manifest(banks[p])["sha16"] for p in STAGE1_CONTROL}
        out[f"persistent_start_counts_min{mp}"] = counts
        out[f"bank_shas_min{mp}"] = shas
        if mp == 2:
            out["banks_min2"] = {p: late_start_bank_manifest(banks[p]) for p in STAGE1_CONTROL}
    # occupancy + transitions + predicate overlaps (reworked detector)
    occ, trans, overlap, n_multi, n_states = occupancy(pi0, OCC_SEEDS)
    out["control_phase_occupancy_gate_on"] = dict(occ)
    out["fraction_in_stage1_control"] = round(sum(occ[p] for p in STAGE1_CONTROL) / max(n_states, 1), 3)
    out["transition_matrix"] = {f"{a}->{b}": c for (a, b), c in sorted(trans.items(), key=lambda kv: -kv[1])}
    out["overlapping_predicate_counts"] = dict(overlap)
    out["states_with_multiple_control_predicates"] = n_multi
    # underpowered decision
    under = [p for p in STAGE1_CONTROL if out["persistent_start_counts_min2"][p] < MIN_STARTS]
    out["underpowered_phases_min2"] = under
    out["verdict"] = "STAGE1_DYNAMIC_PHASE_BANK_UNDERPOWERED" if under else "STAGE1_DYNAMIC_PHASE_BANK_OK"
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("persistent starts min2:", out["persistent_start_counts_min2"], "min4:", out["persistent_start_counts_min4"])
    print("gate-on control occupancy:", dict(occ), "| frac in stage1:", out["fraction_in_stage1_control"])
    print("transitions:", out["transition_matrix"])
    print("multi-predicate states:", n_multi, "overlaps:", dict(overlap))
    print("VERDICT:", out["verdict"], "underpowered:", under, "\nwrote", OUT)


if __name__ == "__main__":
    main()
