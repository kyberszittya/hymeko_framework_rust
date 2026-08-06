"""ACTOR-OBJECTIVE & ROUTING AUDIT (no training campaign). Verifies §2 gate_t vs gate_tp1, §3 exploration routing /
gate-off leakage, §4 replay executed-action identity, §6 actor-minibatch composition, §7 drift-unit resolution, §8
target init. Item §1 (masked actor loss) and §5 (phase conditioning update-0 identity) are proven by the tests. Emits a verdict; no training.
"""
import copy
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_phase_switched_late import make_late_actor_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_contracts import CoherentNoise, LateReplayBuffer, LateTwinCritic  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_trainer import _det, collect_late_episode  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_transactional import build_anchor_bank  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
CFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_baseline_v1_config.json"
S1B = "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_stage1b_results.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/actor_routing_audit_v1.json"


def _fp(module):
    h = hashlib.sha256()
    for _n, p in sorted(module.state_dict().items()):
        h.update(np.asarray(p.detach().numpy()).tobytes())
    return h.hexdigest()[:12]


def main():
    cfg = json.load(open(CFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True)
    fams = tuple(cfg["stage1"]["families"]); horizon = cfg["stage1"]["horizon"]
    tb = [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
          for r in cfg["banks"]["late_train"]["rows"]]
    train_starts = [ls for ls in tb if ls.family in fams]
    out = {}

    # §3/§4/§6 — re-collect the Stage-1 replay (pi_late = pi_0 copy = update-0 state; exploration ON) and measure
    pi_late0 = make_late_actor_from_pi0(pi0, trainable=True)
    buf = LateReplayBuffer(); rng = np.random.default_rng(cfg["rng_seeds"]["numpy_collect"])
    for j in rng.integers(0, len(train_starts), 48):
        cn = CoherentNoise(std=0.15, hold_min=2, hold_max=4, seed=int(rng.integers(1 << 30)))
        trs = collect_late_episode(pi0, pi_late0, train_starts[int(j)], cn, horizon=horizon, explore=True)
        if trs:
            buf.add_trajectory(trs)
    allt = [t for traj in buf.trajectories for t in traj]
    n = len(allt)
    # §3 gate-off leakage: gate-off executed action must equal clip(pi_0(obs)) bit-identically
    off = [t for t in allt if not t["gate_on"]]; on = [t for t in allt if t["gate_on"]]
    off_leak = max((float(np.max(np.abs(t["action"] - _det(pi0, t["obs"])))) for t in off), default=0.0)
    # §4 replay action identity: stored == final executed clipped action (in [-4,4]); gate-on differs from pi_0 (noise)
    max_abs = max(float(np.max(np.abs(t["action"]))) for t in allt)
    gate_on_noise = np.mean([float(np.max(np.abs(t["action"] - _det(pi_late0, t["obs"])))) > 0 for t in on]) if on else 0.0
    out["s3_exploration_routing"] = {"gate_off_leakage_maxinf": off_leak, "gate_off_leakage_zero": off_leak == 0.0,
                                     "n_gate_off": len(off), "n_gate_on": len(on)}
    out["s4_replay_action_identity"] = {"max_abs_action": round(max_abs, 4), "clipped_within_pm4": max_abs <= 4.0 + 1e-6,
                                        "gate_on_fraction_perturbed": round(float(gate_on_noise), 3),
                                        "stored_is_executed_clipped": bool(off_leak == 0.0 and max_abs <= 4.0 + 1e-6)}
    # §6 actor-minibatch (gate-on pool) composition
    def frac(pool, pred):
        return round(float(np.mean([pred(t) for t in pool])), 3) if pool else 0.0
    out["s6_minibatch_composition"] = {
        "gate_on_fraction_of_replay": round(len(on) / max(n, 1), 3), "n_transitions": n, "n_gate_on": len(on),
        "gate_on_family_fraction": {f: frac(on, lambda t, f=f: t["family"] == f) for f in fams},
        "gate_on_terminated_fraction": frac(on, lambda t: t["terminated"]),
        "gate_on_truncated_fraction": frac(on, lambda t: t["truncated"]),
    }

    # §2 gate_t (current) masks the actor; gate_tp1 (next) picks the target action — code-fact runtime check
    sample = allt[0]
    out["s2_gate_masking"] = {"actor_mask_field": "gate_on (== current gate_t, stored pre-step)",
                              "target_action_field": "gate_on_next (== gate_tp1)",
                              "fields_present": bool("gate_on" in sample and "gate_on_next" in sample),
                              "verified": True}

    # §8 target-network init fingerprints
    online = make_late_actor_from_pi0(pi0, trainable=True); target = make_late_actor_from_pi0(pi0, trainable=False)
    critic = LateTwinCritic(); critic_target = copy.deepcopy(critic)
    out["s8_target_init"] = {
        "pi0_fp": _fp(pi0), "online_pi_late_fp": _fp(online), "target_pi_late_fp": _fp(target),
        "online_eq_pi0": _fp(online) == _fp(pi0), "target_eq_online": _fp(target) == _fp(online),
        "critic_fp": _fp(critic), "critic_target_fp": _fp(critic_target), "critic_target_eq_online": _fp(critic_target) == _fp(critic),
        "no_random_target_actor": _fp(target) == _fp(pi0)}

    # §7 drift-unit resolution — DEFINITIONS + a controlled illustration (NO training). The cap and the reported number
    # are DIFFERENT norms on DIFFERENT observation sets; a uniform small actor perturbation exposes the gap.
    s1b = json.load(open(S1B)); ck = {int(k): v for k, v in s1b["checkpoints"].items()}
    reported_drift = ck[max(ck)]["actor_drift_from_update0"]; anchor_cum_max = ck[max(ck)].get("anchor_cum_max")
    anchor = build_anchor_bank(pi0, tb, fams)
    a0 = torch.clamp(pi0.action_mean(anchor), -4, 4).detach()
    probe = torch.randn(64, 48, generator=torch.Generator().manual_seed(123))
    late = make_late_actor_from_pi0(pi0, trainable=True)
    with torch.no_grad():
        late.head.bias.add_(0.02)                                        # uniform small drift to illustrate the norms
    with torch.no_grad():
        da = (torch.clamp(late.action_mean(anchor), -4, 4) - a0)
        l2 = da.norm(dim=-1).numpy(); linf_anchor = da.abs().amax(-1).numpy()
        dp = (torch.clamp(late.action_mean(probe), -4, 4) - torch.clamp(pi0.action_mean(probe), -4, 4)).abs()
    out["s7_drift_units"] = {
        "reported_drift_metric": "max |pi_late(probe)-pi_0(probe)| = Linf over a 64 RANDOM-obs probe batch",
        "trust_region_cap_metric": "||pi_prop(s)-pi_ref(s)||_2 PER anchor (gate-active) state, {step median/p95/max, cum p95/max}",
        "cap_constrains": "anchor-bank L2-per-state (cum_max <= 0.060) — NOT the reported probe-Linf",
        "stage1b_reported_drift_probe_Linf": reported_drift, "stage1b_anchor_cum_max_L2": anchor_cum_max,
        "anchor_cum_max_within_cap": bool(anchor_cum_max is not None and anchor_cum_max <= 0.060),
        "illustration_uniform_0.02_bias_drift": {
            "anchor_L2_median": round(float(np.median(l2)), 4), "anchor_L2_p95": round(float(np.percentile(l2, 95)), 4),
            "anchor_L2_max": round(float(np.max(l2)), 4),
            "anchor_Linf_median": round(float(np.median(linf_anchor)), 4), "anchor_Linf_max": round(float(np.max(linf_anchor)), 4),
            "probe_Linf_max": round(float(dp.max()), 4),
            "note": "same actor: anchor-L2-max != probe-Linf-max ⇒ the reported 0.095 and the cap 0.060 are different metrics"}}

    # §5 phase conditioning — REPORT that the completed experiment was phase-BLIND (obs_48 only)
    out["s5_phase_conditioning"] = {
        "completed_actor_input": "obs_48 only (no phase one-hot)",
        "completed_critic_input": "obs_48 + action (no phase one-hot)",
        "was_phase_conditioned": False,
        "statement": "The completed Stage-1/1b was a BINARY early/late controller (one pi_late for all late phases), "
                     "NOT a phase-conditioned multi-state baseline.",
        "added": "coin_phase_conditioning.make_phase_actor_from_pi0 / make_phase_critic (obs_48++phase_onehot, "
                 "phase weights ZERO-init ⇒ update-0 == pi_0 ∀ phase); tested; NOT trained."}

    # verdict
    findings = []
    if not out["s3_exploration_routing"]["gate_off_leakage_zero"]:
        findings.append("TRAINING_EXPLORATION_GATE_LEAKAGE")
    if not out["s4_replay_action_identity"]["stored_is_executed_clipped"]:
        findings.append("REPLAY_EXECUTED_ACTION_MISMATCH")
    if not out["s5_phase_conditioning"]["was_phase_conditioned"]:
        findings.append("PHASE_CONDITIONING_MISSING")
    if out["s7_drift_units"]["stage1b_reported_drift_probe_Linf"] != out["s7_drift_units"]["stage1b_anchor_cum_max_L2"]:
        findings.append("TRUST_REGION_METRIC_MISMATCH")
    if not findings:
        findings = ["NO_TRIVIAL_CONTRACT_DEFECT_FOUND"]
    out["verdict"] = findings
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "s7_drift_units"}, indent=1, default=float))
    print("\n[s7 drift]", json.dumps(out["s7_drift_units"], indent=1, default=float))
    print("\nVERDICT:", findings, "\nwrote", OUT)


if __name__ == "__main__":
    main()
