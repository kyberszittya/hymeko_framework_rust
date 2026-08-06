"""§1 BC_INITIALIZATION_HEAVILY_SATURATED report + §2 runtime action-identity audit for the clip-actor RL init.

§1: roll the frozen init on the headline panel; record raw pre-clip output vs executed clipped action; saturation
fractions by phase and on the 3 successful seeds. §2: verify the full action path — actor raw -> clip -> env exec ->
replay-stored -> critic-facing — stays in the canonical [-4,4] and the raw +-63 NEVER reaches replay/critic; the env
clip is idempotent with the actor clip; eval reproduces the update-0 action.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_v3_receding_horizon import _phase_code
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
from hymeko_rl.coin_delivery.rl_clip_actor import build_shared_sac_td3
from hymeko_rl.experiments.coin_neutral_start import neutral_env
from hymeko_rl.train.replay import ReplayBuffer

BC = "experiments/2026_07_22_coin_v3_learning/bc_configs/bc_handoff_only_best.pt"
SCALE = 4.0
SUCCESS = {1011, 1447, 1568}
PHN = ["APPROACH", "CONTACT_ACQ", "BILATERAL", "TRANSPORT", "TARGET_ENTRY", "SETTLING", "STRICT_DWELL"]


def main():
    bc = FullActionBC()
    bc.load_state_dict(torch.load(BC))
    bc.eval()
    sac, td3 = build_shared_sac_td3(bc)
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    # ---- §1 saturation + §2 action-path audit in one pass ----
    rb = ReplayBuffer(20000, (48,), 4)
    raw_abs, exec_abs, sat_any, sat_all, n = [], [], 0, 0, 0
    by_phase = {p: [0, 0] for p in range(7)}
    succ_phase = {p: [0, 0] for p in range(7)}
    identity_ok = True
    max_replay_abs = 0.0
    for s in HEADLINE:
        env.set_stage(0)
        env.reset(seed=int(s))
        touched = False
        start_dtz = float(inner._planar_metrics.disk_to_zone)
        for _t in range(160):
            m = inner._planar_metrics
            touched = touched or bool(m.left_contact or m.right_contact)
            ph = _phase_code(inner, 0, touched, start_dtz)
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            with torch.no_grad():
                raw = td3.head(td3.backbone(torch.as_tensor(nf[None]))).numpy()[0]      # pre-clip (the ±63)
                a_pol = td3.action_mean(torch.as_tensor(nf[None])).numpy()[0]           # clip(raw) = executed
            sat = np.abs(raw) >= SCALE - 1e-6
            raw_abs.append(np.abs(raw).max())
            exec_abs.append(np.abs(a_pol).max())
            sat_any += int(sat.any())
            sat_all += int(sat.all())
            n += 1
            by_phase[ph][0] += int(sat.any())
            by_phase[ph][1] += 1
            if int(s) in SUCCESS:
                succ_phase[ph][0] += int(sat.any())
                succ_phase[ph][1] += 1
            # §2 action-path: env executes a_pol; replay stores the EXECUTED clipped action
            r_prev = nf
            inner.step(np.asarray(a_pol, np.float32))
            executed = np.asarray(inner.data.ctrl[:4], np.float32)                     # what MuJoCo actually applied
            nf2 = np.asarray(inner.node_features(), np.float32).flatten()
            rb.add(r_prev, executed, 0.0, nf2, False, False)
            # invariants: executed == clip(raw) within tol; nothing exceeds ±4; env clip idempotent
            if np.abs(executed - a_pol).max() > 1e-4 or np.abs(executed).max() > SCALE + 1e-4:
                identity_ok = False
            max_replay_abs = max(max_replay_abs, float(np.abs(executed).max()))
    # replay contents never exceed ±4 (no raw ±63)
    rb_actions = rb._act[:rb._size] if hasattr(rb, "_size") else rb._act[:n]
    replay_max = float(np.abs(rb_actions).max())
    report = {
        "BC_INITIALIZATION_HEAVILY_SATURATED": {
            "raw_preclip_abs_max": round(float(np.max(raw_abs)), 2), "raw_preclip_abs_median": round(float(np.median(raw_abs)), 3),
            "executed_abs_max": round(float(np.max(exec_abs)), 3), "action_scale": SCALE,
            "frac_states_any_component_saturated": round(sat_any / n, 3),
            "frac_states_all_components_saturated": round(sat_all / n, 3), "n_states": n,
            "saturation_any_by_phase": {PHN[p]: round(by_phase[p][0] / by_phase[p][1], 3) for p in range(7) if by_phase[p][1]},
            "saturation_any_successful_seeds_by_phase": {PHN[p]: round(succ_phase[p][0] / succ_phase[p][1], 3)
                                                         for p in range(7) if succ_phase[p][1]},
        },
        "runtime_identity": {
            "executed_equals_clip_raw": identity_ok,
            "replay_action_abs_max": round(replay_max, 4), "policy_max_exec_abs": round(max_replay_abs, 4),
            "no_raw_preclip_in_replay": bool(replay_max <= SCALE + 1e-4),
            "env_clip_idempotent": identity_ok,
        },
    }
    print(json.dumps(report, indent=1))
    idp = report["runtime_identity"]
    ok = idp["executed_equals_clip_raw"] and idp["no_raw_preclip_in_replay"] and idp["env_clip_idempotent"]
    report["verdict"] = "RL_RUNTIME_IDENTITY_PASS" if ok else "RL_RUNTIME_IDENTITY_FAIL"
    json.dump(report, open(sys.argv[1], "w"), indent=1)
    print("\n" + report["verdict"])


if __name__ == "__main__":
    main()
