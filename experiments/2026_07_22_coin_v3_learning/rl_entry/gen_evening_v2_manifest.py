"""Generate the §2 evening-v2 manifest with SHAs computed from the actual artifacts (not hand-typed).
Writes reports/2026-07-23-coin-push-delivery-evening-v2-manifest.json. No env rollouts."""
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, ".")
import mujoco  # noqa: E402
import torch  # noqa: E402

from hymeko_rl.coin_delivery.coin_counterfactual_labels import MAGNITUDES  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_critic import CompositeTwinCritic, encoder_fingerprint  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_critic_causal import CausalCompositeTwinCritic  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_critic_state import residual_critic_state_v2_contract  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2  # noqa: E402
from hymeko_rl.coin_delivery.coin_v3_seed_banks import manifest as seed_manifest  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


def _sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


sm = seed_manifest()
banks = sm["banks"]
critic_banks = {"std_train": (6000, 6100), "std_dev": (6100, 6200), "std_sealed": (6200, 6250),
                "adv_train": (6300, 6400), "adv_dev": (6400, 6500), "adv_sealed": (6500, 6550)}
manifest = {
    "run": "COIN_PUSH_DELIVERY_EVENING_V2",
    "generated_from_commit": _git("rev-parse", "--short", "HEAD"),
    "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    "start_commit": "6f36cf3",
    "host": "Hajdus-MacBook-Pro (Apple Silicon)",
    "note_host": "kato14 not used (SSH not established this session); Mac has correct source+venv, all coin stages ran "
                 "here; §3 update-0 reproduction is the parity anchor (3/9,2/30,9/9 + forced-residual leakage 0).",
    "python": sys.version.split()[0], "torch": torch.__version__, "mujoco": mujoco.__version__,
    "controller": "PHASE_GATED_LEARNED_RESIDUAL_CONTROLLER",
    "frozen_artifacts": {
        "pi0_file": PI0, "pi0_file_sha256": _sha_file(PI0), "pi0_prefix": _sha_file(PI0)[:8],
        "pi0_headline": 3, "pi0_validation": 2, "pi0_grasp": 9, "pi0_delivered": [1011, 1447, 1568],
        "gate_controller_state_v2_sha_expected": "7633dd3c", "replay_schema_sha_expected": "0b11b60e",
        "smoothing_contract_sha_expected": "d558443d",
        "encoder_fingerprint": encoder_fingerprint()[:8],
        "bundle_hash": sm["bundle_hash"], "semantic_graph_fp": sm["semantic_graph_fp"],
        "reward_file": "data/robotics/galambos_task_deliver_v3.hymeko",
    },
    "contracts": {
        "obs": sm["obs_contract"], "action": sm["action_contract"], "residual": "[-0.25,0.25]",
        "horizon": 360, "gamma": 0.99, "K6": "6 consecutive centered(dtz<=0.02)+settled(speed<0.06)+robot-attributed",
        "critic_state_v2": residual_critic_state_v2_contract()["sha256"][:12],
        "critic_state_v2_dim": residual_critic_state_v2_contract()["dim"],
        "instant_critic_contract": CompositeTwinCritic().contract_sha256()[:12],
        "causal_critic_contract": CausalCompositeTwinCritic().contract_sha256()[:12],
        "counterfactual_magnitudes": list(MAGNITUDES),
        "target_smoothing": {"residual_bound": 0.25, "target_noise_std": 0.05, "target_noise_clip": 0.125},
    },
    "evaluation_banks": {k: {"n": v["n"], "sha16": v["sha16"]} for k, v in banks.items()},
    "critic_banks": {k: {"range": list(v), "n": v[1] - v[0],
                         "sha16": hashlib.sha256(json.dumps(list(range(*v))).encode()).hexdigest()[:16]}
                     for k, v in critic_banks.items()},
    "policy_final_test_bank": {"range": [8000, 8050], "n": 50, "status": "UNOPENED tonight — not authorized"},
    "ledger": {
        "DEPLOYABLE": ["pi_0 (1902454c)", "StableObjectEngagement gate V2 (7633dd3c)",
                       "CompositeResidualController", "scale-correct smoothing (d558443d)"],
        "REPLAY_ONLY": ["gated-residual rollouts", "grouped counterfactual continuations",
                        "pi_0 successes/failures", "zero/boundary/basis/isotropic residual controls"],
        "PLANNER_ONLY": ["H=30 receding-horizon trajectories (coverage/counterfactual only, NOT a feedback policy)"],
        "HISTORICAL_CONTROL": ["prior v0-v28 coin-toss lines", "seed_stabilized v2", "coin_neutral_start deliveries"],
        "QUARANTINED": ["SAC (SAC_HARD_CLIP_LOGPROB_MISMATCH) — not touched",
                        "rejected V1 OR-contact gate (d739e8af)", "old Python-reward campaigns"],
        "INVALIDATED_DIAGNOSTIC": [
            "coin_residual_critic_dev.py (first-pass): ungated full-action noise, instantaneous-only state, "
            "cross-state advantage ranking, dropped-truncated bootstrap, 40-step truncated labels, ad-hoc gate — "
            "committed at 327afa6, withdrawn at d44f1bb (RESIDUAL_CRITIC_DEVELOPMENT_HARNESS_INVALIDATED)."],
    },
}
out = "reports/2026-07-23-coin-push-delivery-evening-v2-manifest.json"
json.dump(manifest, open(out, "w"), indent=1)
print("wrote", out)
print("pi0 sha", manifest["frozen_artifacts"]["pi0_prefix"], "| causal_state_v2",
      manifest["contracts"]["critic_state_v2"], "| encoder", manifest["frozen_artifacts"]["encoder_fingerprint"])
print("critic banks disjoint from policy:",
      not (set().union(*[set(range(*v)) for v in critic_banks.values()]) & (set(range(7000, 7030)) | set(range(8000, 8050)))))
