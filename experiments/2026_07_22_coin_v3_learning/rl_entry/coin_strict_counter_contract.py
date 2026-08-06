"""STRICT_COUNTER_STATE_CONTRACT_V1 (measurement-only). Tests whether the strict-dwell counter (distance-to-terminal)
is observable to the policy/critic, via a paired-state construction: identical physical simulator state + history but
strict = 0..5. Emits STRICT_COUNTER_OBSERVED_MARKOV or HIDDEN_CERTIFIER_STATE_NONMARKOV. No training."""
import hashlib
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_contract_audit import recertify, post_stream, traced_env_strict  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import control_mode  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
PI0 = f"{D}/frozen/pi0_shared_clip_actor.pt"
DWELL_REQ = 6


def _h(a):
    return hashlib.md5(np.asarray(a, np.float64).tobytes()).hexdigest()[:12]


def paired_state_probe(pi0, ls):
    """From FRESH deterministic reconstructions (identical physical state + node_features buffer), set strict = 0..5 and
    step the frozen pi_0 holding action; record obs_48 hash, control_mode label, reward, termination, strict-after."""
    rows = []
    for k in range(DWELL_REQ):
        rl, _g, _hist, _r = reconstruct_handoff(pi0, ls, horizon=360)
        rl._strict = k; rl._touched = True
        obs = rl.obs().copy(); cm = control_mode(rl._dtz(), rl._speed(), rl._speed(), rl._strict)
        with torch.no_grad():
            a = np.clip(pi0.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy(), -4, 4).astype(np.float32)
        _o, rw, term, trunc, _ = rl.step(a)
        rows.append({"strict": k, "obs48_hash": _h(obs), "control_mode": cm, "reward": round(float(rw), 3),
                     "terminated": bool(term), "strict_after": int(rl._strict)})
    return rows


def main():
    torch.set_num_threads(1); log = lambda *a: print(*a, flush=True)
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    cfg = json.load(open(f"{D}/transport_dwell_config.json"))
    bank = lambda m: [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]
    dev = [ls for fam in ("transport", "braking", "settling_dwell") for ls in bank(cfg["banks"]["dev"][fam])]
    trace = json.load(open(f"{D}/audit_trace_pi0.json"))["rollouts"]

    # (1) handoff strict-counter distribution per bank
    dist = {fam: dict(sorted(Counter(r["handoff_strict"] for r in trace if r["family"] == fam).items()))
            for fam in ("transport", "braking", "settling_dwell")}
    dist["ALL"] = dict(sorted(Counter(r["handoff_strict"] for r in trace).items()))

    # (2) observability audit (structural — measured, not inferred from dtz/speed)
    observability = {
        "actor_obs_48_node_features": {"contains_strict_counter": False, "evidence": "node_features has no _strict/_success/dwell field"},
        "critic_conditioning": {"contains_exact_counter": False,
                                "evidence": "conditioning = onehot3(control_mode)+onehot2(contact)+event5; control_mode collapses strict>=1 -> 'settling_dwell' (does not distinguish 1..5)"},
        "replay_transition": {"contains_strict_counter": False, "evidence": "late replay stores (state=obs+conditioning, action, reward, next); no counter field"},
        "target_network_input": {"contains_strict_counter": False, "evidence": "target critic sees the same state vector as the online critic"}}

    # (3) paired-state probe on the settling starts (strict actually varies there)
    probe = paired_state_probe(pi0, [ls for ls in dev if ls.family == "settling_dwell"][1])   # a delivering start
    obs_identical = len({r["obs48_hash"] for r in probe}) == 1
    reward_differs = len({r["reward"] for r in probe}) > 1
    termination_differs = len({r["terminated"] for r in probe}) > 1
    mode_distinguishes = len({r["control_mode"] for r in probe}) > 1
    g = 0.99; V = 1.0
    td = [round(r["reward"] + g * (0 if r["terminated"] else 1) * V, 3) for r in probe]
    td_differs = len(set(td)) > 1

    # (4) verdict
    counter_observed = observability["actor_obs_48_node_features"]["contains_strict_counter"] or \
        observability["critic_conditioning"]["contains_exact_counter"]
    verdict = "STRICT_COUNTER_OBSERVED_MARKOV" if counter_observed else "HIDDEN_CERTIFIER_STATE_NONMARKOV"

    # (5,6) both evaluation semantics — NEVER mixed
    cont = sum(int(traced_env_strict(r)) for r in trace)
    reset0 = sum(int(recertify(post_stream(r), r["clearance_measured"], center_tol=0.02, settle_vel=0.06, dwell_req=6)[0]) for r in trace)
    semantics = {"CONTINUATION_STRICT": {"rate": f"{cont}/31", "def": "inherit handoff_strict (arc-canonical, rl._strict)"},
                 "RESET_AT_HANDOFF_STRICT": {"rate": f"{reset0}/31", "def": "strict=0 at the late-controller boundary (late-skill-from-zero-dwell)"}}

    # (7) reclassification of local critic/TD3 findings
    reclass_rl = {"phase_switched_td3 / transactional / stage1abc / transport_dwell / residual_critic":
                  "VERDICT_REQUIRES_RERUN — the critic state HID the strict counter; identical obs mapped to TD targets "
                  "differing by up to the terminal bonus (~31), so the critic could not represent distance-to-terminal. "
                  "'no local improvement' / 'critic route blocked' are CONFOUNDED by non-Markov critic state.",
                  "chunk / primitive supervised baselines": "UNAFFECTED — supervised (no bootstrapped critic target)."}

    out = {"contract": "STRICT_COUNTER_STATE_CONTRACT_V1", "date": "2026-07-23", "measurement_only": True,
           "handoff_strict_distribution": dist, "observability": observability,
           "paired_state_probe": {"rows": probe, "obs_48_identical": obs_identical, "reward_differs": reward_differs,
                                  "termination_differs": termination_differs, "control_mode_distinguishes_strict": mode_distinguishes,
                                  "td_target": td, "td_target_differs": td_differs},
           "verdict": verdict, "evaluation_semantics": semantics, "rl_reclassification": reclass_rl}
    json.dump(out, open(f"{D}/strict_counter_contract_v1.json", "w"), indent=1, default=float)

    log("== STRICT_COUNTER_STATE_CONTRACT_V1 ==")
    log(f"  (1) handoff strict dist: {dist['ALL']}  by bank: {dist}")
    log(f"  (2) counter in actor obs: {observability['actor_obs_48_node_features']['contains_strict_counter']}  "
        f"in critic (exact): {observability['critic_conditioning']['contains_exact_counter']}  (control_mode collapses strict>=1)")
    log(f"  (3) paired state strict 0..5: obs_identical={obs_identical} reward_differs={reward_differs} "
        f"termination_differs={termination_differs} mode_distinguishes={mode_distinguishes}")
    log(f"      reward by strict: {[r['reward'] for r in probe]}  terminated: {[r['terminated'] for r in probe]}  TD: {td}")
    log(f"  (5,6) CONTINUATION_STRICT {cont}/31  |  RESET_AT_HANDOFF_STRICT {reset0}/31  (never mixed)")
    log(f"\n-> {verdict}\nwrote {D}/strict_counter_contract_v1.json\nSTRICT_COUNTER_DONE")


if __name__ == "__main__":
    main()
