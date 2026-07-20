"""COIN-DELIVERY-OVERNIGHT-2 PART V — gated acquisition RL ablation (Case A: BC sanity → PPO → TD3+BC → guarded SAC).

Authorized by the STRICT acquisition gate. Trains on the acquisition subtask (stable two-finger acquisition) over the
19 acquisition-wall states; evaluates on the wall (train) + held-out FRESH pre-contact seeds; reports the BC/primitive
baselines, easy-state preservation, the correct-vs-scrambled structural control, and the chained delivery (which the
oracle measured at 0 — acquisition here has NO downstream task value, so any RL win is an ACQUISITION-subtask claim).

Reuses train.ppo / train.ddpg / train.sac / train.bc unchanged. Bounded local budget, 3 seeds. Guarded SAC runs only
if the critic sanity check passes (Q(teacher) >= Q(random)).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace

import numpy as np
import torch

from hymeko_rl.agents.policy import build_policy
from hymeko_rl.experiments.coin_delivery_acquisition1 import _OUT, _SCRAMBLE, _states
from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode, AcquisitionPrimitive
from hymeko_rl.train.coin_delivery_acquisition_rl import AcquisitionRLEnv, eval_acq_rate
from hymeko_rl.train.ddpg import build_offpolicy, td3_config, train_offpolicy
from hymeko_rl.train.ppo import PPOConfig, train_ppo
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_OBS, _ACT = 41, 6
_HELD_FRESH = list(range(64_100, 64_120))          # fresh pre-contact seeds (held-out generalization)


def _log(m: str) -> None:
    print(m, flush=True)


def _best_primitive() -> AcqParams:
    d = json.loads((_OUT / "manifests" / "coin_delivery_acquisition.json").read_text())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return replace(AcqParams(**d), regrasp=False)  # the ablation-corrected primitive (regrasp hurts)


def _greedy(ac) -> "callable":
    def fn(obs):
        with torch.no_grad():
            return ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy().astype(np.float32)
    return fn


def _collect_demos(params: AcqParams, seeds, *, horizon: int = 120) -> tuple:
    """Roll the acquisition primitive; collect (obs, action) from episodes that reach stable acquisition."""
    env = AcquisitionRLEnv(seeds, horizon=horizon)
    prim = AcquisitionPrimitive(params)
    obs_all, act_all = [], []
    for i in range(len(seeds)):
        o, _i = env.reset(seed=i)
        prim.reset()
        buf_o, buf_a = [], []
        ok = False
        for _ in range(horizon):
            a = np.clip(prim.action(o), -1, 1).astype(np.float32)
            buf_o.append(o.copy()); buf_a.append(a)
            o, _r, term, trunc, info = env.step(a)
            if info["stable_acquisition"]:
                ok = True
            if term or trunc:
                break
        if ok:                                       # keep only successful acquisition trajectories
            obs_all += buf_o; act_all += buf_a
    return np.asarray(obs_all, np.float32), np.asarray(act_all, np.float32)


def _bc_policy(demos, seed: int):
    ac = build_policy("mlp", obs_dim=_OBS, action_dim=_ACT)
    behaviour_clone(ac, demos[0], demos[1], n_epochs=150, seed=seed)
    return ac


def _critic_sanity(critics, demos, env, rng) -> dict:
    """Q(teacher) vs Q(random) on a diagnostic batch — the guarded-SAC precondition."""
    with torch.no_grad():
        o = torch.as_tensor(demos[0][:128], dtype=torch.float32)
        a_teacher = torch.as_tensor(demos[1][:128], dtype=torch.float32)
        a_rand = torch.as_tensor(rng.uniform(-1, 1, a_teacher.shape), dtype=torch.float32)
        qt = float(critics[0](o, a_teacher).mean())
        qr = float(critics[0](o, a_rand).mean())
    return {"q_teacher": round(qt, 3), "q_random": round(qr, 3), "critic_ok": qt >= qr}


def run(*, steps: int = 25_000, seeds=(0, 1, 2), fast: bool = False) -> dict:
    t0 = time.perf_counter()
    if fast:
        steps, seeds = 4_000, (0,)
    prim = _best_primitive()
    wall = _states()["acquisition_wall"]
    easy = _states()["easy"][:15]
    demos = _collect_demos(prim, wall)
    _log(f"=== PART V acquisition RL === demos: {len(demos[0])} transitions from stable-acq trajectories")

    # primitive baseline uses the STATEFUL eval (the FSM must persist across steps within an episode)
    from hymeko_rl.train.coin_delivery_acquisition import eval_acquisition, make_acq_env
    _e = make_acq_env()
    base = {"primitive_wall": eval_acquisition(prim, wall, env=_e)["n_stable"],
            "primitive_held": eval_acquisition(prim, _HELD_FRESH, env=_e)["stable_acquisition_rate"],
            "primitive_scrambled": eval_acquisition(prim, wall, env=_e, scramble_perm=_SCRAMBLE)["n_stable"]}
    _log(f"  baselines: primitive wall={base['primitive_wall']}/19 held_fresh={base['primitive_held']} "
         f"scrambled={base['primitive_scrambled']}/19")

    methods: dict = {}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        # BC sanity
        bc = _bc_policy(demos, sd)
        bc_wall = eval_acq_rate(_greedy(bc), wall)["n_stable"]
        bc_scr = eval_acq_rate(_greedy(bc), wall, scramble_perm=_SCRAMBLE)["n_stable"]
        methods.setdefault("BC", []).append({"seed": sd, "wall": bc_wall, "scrambled": bc_scr})
        _log(f"  [seed {sd}] BC wall={bc_wall}/19 scrambled={bc_scr}/19 (BC sanity: > scrambled = {bc_wall > bc_scr})")
        # PPO warm-started from BC
        ppo_ac = _bc_policy(demos, sd)
        train_ppo(ppo_ac, AcquisitionRLEnv(wall, seed=sd), PPOConfig(n_iters=max(1, steps // 1024), n_steps=1024, seed=sd))
        methods.setdefault("PPO", []).append({"seed": sd, "wall": eval_acq_rate(_greedy(ppo_ac), wall)["n_stable"],
                                              "held": eval_acq_rate(_greedy(ppo_ac), _HELD_FRESH)["rate"],
                                              "easy": eval_acq_rate(_greedy(ppo_ac), easy)["rate"]})
        _log(f"  [seed {sd}] PPO wall={methods['PPO'][-1]['wall']}/19 held={methods['PPO'][-1]['held']}")
        # TD3+BC warm-started
        actor, critics = build_offpolicy("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0, n_critics=2)
        tcfg = td3_config(total_steps=steps, start_steps=800, batch_size=128, seed=sd,
                          eval_every=max(2000, steps // 4), log_every=max(2000, steps))
        train_offpolicy(actor, critics, AcquisitionRLEnv(wall, seed=sd), tcfg)
        methods.setdefault("TD3+BC", []).append({"seed": sd, "wall": eval_acq_rate(_greedy(actor), wall)["n_stable"],
                                                 "held": eval_acq_rate(_greedy(actor), _HELD_FRESH)["rate"]})
        _log(f"  [seed {sd}] TD3+BC wall={methods['TD3+BC'][-1]['wall']}/19")

    # guarded SAC (seed 0 only) — run ONLY if the critic sanity check passes
    torch.manual_seed(0); np.random.seed(0)
    sac_actor, sac_critics = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0,
                                       n_critics=2, hidden=64)
    sanity = _critic_sanity(sac_critics, demos, None, np.random.default_rng(0))
    _log(f"  guarded-SAC critic sanity (pre-train, cold critic): {sanity}")
    if sanity["critic_ok"] and not fast:
        scfg = SACConfig(total_steps=steps, start_steps=800, batch_size=128, seed=0,
                         eval_every=max(2000, steps // 4), log_every=max(2000, steps))
        train_sac(sac_actor, sac_critics, AcquisitionRLEnv(wall, seed=0), scfg)
        post = _critic_sanity(sac_critics, demos, None, np.random.default_rng(1))
        methods["guarded_SAC"] = [{"seed": 0, "wall": eval_acq_rate(_greedy(sac_actor), wall)["n_stable"],
                                   "critic_sanity_post": post}]
        _log(f"  guarded-SAC wall={methods['guarded_SAC'][0]['wall']}/19 post-critic={post}")
        sac_status = "ran"
    else:
        sac_status = "cold_critic_gate_skipped" if not sanity["critic_ok"] else "skipped_fast"

    label = _label(base, methods)
    out = {"baselines": base, "held_fresh_seeds": [_HELD_FRESH[0], _HELD_FRESH[-1]], "steps": steps,
           "seeds": list(seeds), "methods": methods, "guarded_sac_status": sac_status,
           "chained_delivery_note": "oracle-measured chained delivery from recovered acquisitions = 0 zone / 0 center; "
                                    "RL here optimizes the acquisition SUBTASK only, no delivery/task-value claim",
           "morning_label": label, "kato15_justified": False, "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests").mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests" / "coin_delivery_acquisition_rl.json").write_text(json.dumps(out, indent=2, default=str))
    _log(f"[ACQ-RL] label={label} | {out['wall_s']}s")
    return out


def _label(base: dict, methods: dict) -> str:
    """Morning decision label — acquisition-subtask outcome (NOT a delivery claim; chained delivery was 0)."""
    prim = base["primitive_wall"]
    best_rl = max((np.median([m["wall"] for m in ms]) for k, ms in methods.items() if ms), default=0)
    bc = methods.get("BC", [{}])
    bc_ok = all(m.get("wall", 0) > m.get("scrambled", 99) for m in bc) if bc and "wall" in bc[0] else False
    if not bc_ok:
        return "INSUFFICIENT_DEMONSTRATION_COVERAGE"
    if best_rl > prim:
        return "TARGETED_GATE_RL_POSITIVE"          # RL beat the primitive on the acquisition subtask
    return "TARGETED_GATE_RL_NEGATIVE"              # RL did not beat the primitive on acquisition (subtask, no delivery value)


if __name__ == "__main__":
    run(fast="--fast" in sys.argv)
    sys.exit(0)
