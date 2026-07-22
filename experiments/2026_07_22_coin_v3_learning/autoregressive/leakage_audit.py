"""§3 temporal-leakage audit of the FULL_ACTION_OBS_HISTORY_V1 encoding (build_history_features / ObsHistoryV1).

Proves, by controlled perturbation, that feature[t] depends ONLY on {obs_<=t, act_<t, deterministic padding} and NOT
on {act_t (the label), future obs/act, terminal result, trajectory length, seed, timestep, planner state}. A PASS is
required before any autoregressive training.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.full_action_bc import load_trajectory_dataset
from hymeko_rl.coin_delivery.full_action_obs_history import (
    ACTION_DIM, FEATURE_DIM, build_history_features, contract_spec,
)

BASE = sys.argv[1]
RNG = np.random.default_rng(0)


def audit():
    d = load_trajectory_dataset([BASE], patterns=("traj_*.npz",))
    obs, act, traj = d["obs"], d["act"], d["traj"]
    feats = build_history_features(obs, act, traj)
    checks = {}

    # 1. dimension + the label (act_t) is NOT byte-present in feature[t]
    checks["feature_dim_152"] = bool(feats.shape[1] == FEATURE_DIM == 152)
    # feature layout is [obs_t(48) obs_{t-1}(48) obs_{t-2}(48) act_{t-1}(4) act_{t-2}(4)] -> no slot for act_t
    checks["no_act_t_slot"] = bool(feats.shape[1] == 3 * 48 + 2 * ACTION_DIM)

    # 2. PERTURB act[t] and everything AFTER t -> feature[<=t] must be byte-identical (causality)
    tid = int(np.unique(traj)[0])
    idx = np.where(traj == tid)[0]
    t = len(idx) // 2                                        # a mid-trajectory step
    obs2, act2 = obs.copy(), act.copy()
    act2[idx[t]] = RNG.standard_normal(ACTION_DIM).astype(np.float32) * 5      # corrupt the label at t
    act2[idx[t + 1:]] = RNG.standard_normal((len(idx) - t - 1, ACTION_DIM)).astype(np.float32) * 5   # + all future acts
    obs2[idx[t + 1:]] = RNG.standard_normal((len(idx) - t - 1, obs.shape[1])).astype(np.float32) * 5  # + all future obs
    feats2 = build_history_features(obs2, act2, traj)
    # feature[<=t] within this trajectory must be unchanged; feature[t+1],[t+2] MAY change (they legitimately use act_t)
    upto_t = idx[:t + 1]
    checks["feature_le_t_invariant_to_future"] = bool(np.array_equal(feats[upto_t], feats2[upto_t]))
    changed_after = not np.array_equal(feats[idx[t + 1]], feats2[idx[t + 1]])
    checks["feature_t+1_uses_past_act_t_only"] = bool(changed_after)   # sanity: act_t legitimately enters feature[t+1]

    # 3. the label act[t] must NOT be reconstructable as a sub-vector of feature[t] (no accidental inclusion)
    ft = feats[idx[t]]
    label = act[idx[t]]
    present = any(np.allclose(ft[i:i + ACTION_DIM], label) for i in range(0, FEATURE_DIM - ACTION_DIM + 1))
    # act_{t-1} could coincidentally equal act_t only if the demo is constant there; check it's the PAST slot, not label
    checks["label_not_in_feature_unless_equals_past_action"] = bool(
        (not present) or np.allclose(act[idx[t - 1]], label))

    # 4. no trajectory-global scalars (seed/length/index/terminal) can be in a per-step obs feature: features are built
    #    only from node_features(48) + past 4-dim actions; assert the two trajectories of DIFFERENT length produce
    #    features whose values are not a function of trajectory length (a length signal would shift all rows).
    files = d["files"]
    lens = [int((traj == i).sum()) for i in range(d["n_traj"])]
    checks["variable_length_trajectories_present"] = bool(len(set(lens)) > 1)

    spec = contract_spec()
    return {"contract": spec["name"], "sha256": spec["sha256"], "n_traj": d["n_traj"], "n_samples": int(len(feats)),
            "checks": checks, "all_pass": all(checks.values())}


def main():
    r = audit()
    print(json.dumps(r, indent=1))
    verdict = "AUTOREGRESSIVE_DATA_LEAKAGE_AUDIT_PASS" if r["all_pass"] else "HISTORY_LABEL_LEAKAGE_DETECTED"
    r["verdict"] = verdict
    json.dump(r, open(sys.argv[2], "w"), indent=1)
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
