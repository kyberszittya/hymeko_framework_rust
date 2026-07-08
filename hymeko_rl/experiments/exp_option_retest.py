"""Option-level coin-toss improvement — the gated, mechanism-aware overnight driver (ONE file, ``--stage``).

Pipeline (see ``docs/plans/2026-07-08-option-level-retest/``): Branch A sustained-PUSH audit → Branch B/C option-
parameter CEM (skill-level, no per-step residual) → expert selection → Branch D imitation (BC fine-tune warm-
started from the frozen DAgger, escalating to a lightweight on-policy DAgger) → acceptance vs the frozen baseline.
TaskMonitor stays the frozen verifier; ledgers active; v2b reward unchanged; selected MLP+DAgger baseline frozen;
NO per-step motor-residual / scalar TD3/SAC/CQL. Verdict: POSITIVE / NEGATIVE_WITH_MECHANISM /
INCONCLUSIVE_WITH_NEXT_FIX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.eval.evaluate import greedy_action_fn, render_episode_gif
from hymeko_rl.eval.push_audit import PushAuditResult, audit_policy
from hymeko_rl.experiments.exp_galambos_coord_ab import make_env
from hymeko_rl.experiments.exp_vector_retest import (
    _CKPT,
    _DIFFICULTY,
    _EVAL_SEED,
    certify_v2b,
    load_frozen_actor,
    measure_policy,
    wire_ledgers,
)
from hymeko_rl.experiments.galambos_bc import collect_galambos_demos
from hymeko_rl.experiments.galambos_demo import PhasePushController, PushControllerParams, PushDemonstrator
from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.train.option_search import OptionSearchConfig, option_cem
from hymeko_rl.viz.render_planar_gifs import demonstrator_action_fn

_ACCEPT_TOL = 0.02
_HEADLINE = ("ft_dom", "monitor_pass", "monitor_score")


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class OptionRetestConfig:
    stage: str = "full"
    out_dir: str = "experiments/2026_07_08_option_retest"
    n_eval: int = 24
    audit_eps: int = 24
    k_sustained: int = 5
    option: OptionSearchConfig = field(default_factory=OptionSearchConfig)
    n_demos: int = 400            # broader than the original DAgger's coverage
    bc_epochs: int = 200
    bc_batch: int = 256
    dagger_rounds: int = 2        # lightweight on-policy escalation if BC alone doesn't beat baseline
    dagger_eps: int = 40

    @classmethod
    def smoke(cls) -> "OptionRetestConfig":
        c = cls(stage="smoke", out_dir="experiments/2026_07_08_option_retest_smoke", n_eval=6, audit_eps=6)
        c.option = OptionSearchConfig(pop=6, elite=2, iters=2, n_eval_eps=3, max_steps=200, log_every=1)
        c.n_demos = 40
        c.bc_epochs = 40
        c.dagger_rounds = 1
        c.dagger_eps = 8
        return c

    @classmethod
    def full(cls) -> "OptionRetestConfig":
        c = cls(stage="full")
        c.dagger_rounds = 3          # broader on-policy coverage than the original DAgger (4×12 eps) to close the gap
        c.dagger_eps = 40
        return c


# --------------------------------------------------------------------------- helpers

def _md5_actor(actor: Any) -> str:
    h = hashlib.md5()
    for _k, v in sorted(actor.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _fresh_actor(env: Any, warm_start: bool = True) -> Any:
    actor = build_collaborative_offpolicy(env, kind="mlp", hidden=64)[0]
    if warm_start:
        actor.load_state_dict(torch.load(_CKPT, map_location="cpu"))
    return actor


def _audit(name: str, make_action_fn, cfg: OptionRetestConfig) -> PushAuditResult:
    return audit_policy(lambda: make_env(coord=False, difficulty=_DIFFICULTY), make_action_fn,
                        name=name, n_episodes=cfg.audit_eps, seed0=_EVAL_SEED, k_sustained=cfg.k_sustained)


# demonstrator-OBJECT factories (return a controller with .reset()/.action(env)) — used by collect_demos + DAgger
def _push_demo(e):
    return PushDemonstrator(e)


def _phase_default_demo(e):
    return PhasePushController(e)


def _option_demo_factory(theta):
    params = PushControllerParams.from_vector(np.asarray(theta, dtype=np.float64))
    return lambda e: PhasePushController(e, params=params)


def _demo_af(demo_factory):
    """Adapt a demonstrator-object factory to a per-episode-reset ``action_fn(env,obs)`` factory (for audit / GIF).
    ``record_trajectory`` resets the env but NOT the demonstrator, so reset here to avoid FSM leakage across eps."""
    def make(e):
        d = demo_factory(e)
        if hasattr(d, "reset"):
            d.reset()
        return demonstrator_action_fn(d)
    return make


# --------------------------------------------------------------------------- Branch D: imitation

def bc_finetune(env: Any, expert: Any, cfg: OptionRetestConfig, *, seed: int = 0,
                log=_log) -> tuple[Any, np.ndarray, np.ndarray]:
    """BC fine-tune a warm-started (frozen-DAgger-initialised) MLP on held-only demos of ``expert``.

    # Preconditions: ``expert`` exposes ``reset()``/``action(env)``; ``env`` is the v2 scene the ckpt trained on.
    # Postconditions: returns (fine-tuned actor, demo obs, demo acts); no critic-gradient, base weights only
      fine-tuned. The demos are returned so DAgger can seed its aggregation with them (proper DAgger)."""
    obs, acts = collect_galambos_demos(env, cfg.n_demos, seed, only_success=True, demonstrator=expert)
    log(f"[imit/bc] collected {len(obs)} held-only demo samples from {type(expert).__name__}")
    actor = _fresh_actor(env, warm_start=True)
    behaviour_clone(actor, obs, acts, n_epochs=cfg.bc_epochs, batch=cfg.bc_batch, seed=seed, log_every=0, device="cpu")
    actor.eval()
    return actor, obs, acts


def dagger_escalate(make_env_fn, student: Any, expert_factory, cfg: OptionRetestConfig, *,
                    seed_demos: "tuple[np.ndarray, np.ndarray] | None" = None, seed0: int = 1000, log=_log) -> Any:
    """Lightweight on-policy DAgger: SEED the aggregation with the expert's held demos, then roll the STUDENT and
    label every visited state with the closed-loop expert (recomputed from that state), aggregate, re-BC from the
    frozen init. No library sanity-gate, no critic. Seeding with the BC demos is what keeps early rounds on good
    states (an aggregation of only student-visited states early on trains on bad states → collapse)."""
    agg_o: list[np.ndarray] = list(seed_demos[0]) if seed_demos is not None else []
    agg_a: list[np.ndarray] = list(seed_demos[1]) if seed_demos is not None else []
    actor = student
    for rnd in range(cfg.dagger_rounds):
        for ep in range(cfg.dagger_eps):
            env = make_env_fn()
            obs, _ = env.reset(seed=seed0 + rnd * 1000 + ep)
            expert = expert_factory(env)
            if hasattr(expert, "reset"):
                expert.reset()
            for _t in range(env.max_steps):
                a_expert = np.asarray(expert.action(env), dtype=np.float32)   # label at the student-visited state
                agg_o.append(obs.astype(np.float32))
                agg_a.append(a_expert)
                with torch.no_grad():
                    a_student = actor.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
                obs, _r, term, trunc, _i = env.step(a_student.astype(np.float32))
                if term or trunc:
                    break
        actor = _fresh_actor(make_env_fn(), warm_start=True)
        behaviour_clone(actor, np.asarray(agg_o, np.float32), np.asarray(agg_a, np.float32),
                        n_epochs=cfg.bc_epochs, batch=cfg.bc_batch, seed=0, log_every=0, device="cpu")
        actor.eval()
        log(f"[imit/dagger] round {rnd + 1}/{cfg.dagger_rounds} | aggregated {len(agg_o)} labeled states")
    return actor


# --------------------------------------------------------------------------- acceptance

def _accept(base_m: dict, base_a: PushAuditResult, cand_m: dict, cand_a: PushAuditResult) -> dict:
    """A candidate passes iff headline preserved/improved AND sustained-contact coverage strictly increased AND no
    exploit/body/arm-body rise (the user's acceptance contract)."""
    headline = {m: cand_m[m] >= base_m[m] - _ACCEPT_TOL for m in _HEADLINE}
    coverage = {
        "sustained_push_up": cand_a.sustained_push_per_ep > base_a.sustained_push_per_ep + 0.05,
        "ft_in_contact_up": cand_a.ft_progress_in_contact > base_a.ft_progress_in_contact + 1e-4,
        "both_contact_up": cand_a.both_contact_frac > base_a.both_contact_frac + 1e-3,
    }
    no_exploit = {
        "no_exploit_rise": cand_m["body_driven_exploit"] <= base_m["body_driven_exploit"] + _ACCEPT_TOL,
        "no_arm_body_rise": cand_a.arm_body_rate <= base_a.arm_body_rate + _ACCEPT_TOL,
        "no_body_progress_rise": cand_a.body_progress_in_contact <= base_a.body_progress_in_contact + 1e-3,
    }
    ft_improved = cand_m["ft_dom"] > base_m["ft_dom"] + _ACCEPT_TOL
    # the primary coverage signal (sustained two-finger PUSH windows) is required; the others are reported
    accepted = all(headline.values()) and coverage["sustained_push_up"] and all(no_exploit.values())
    return {"accepted": bool(accepted), "ft_dom_improved": bool(ft_improved),
            "headline": headline, "coverage": coverage, "no_exploit": no_exploit}


# --------------------------------------------------------------------------- plots / gif

def _plots(out: Path, audits: dict, option_hist: list, imit_track: list, log=_log) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log(f"[plots] matplotlib unavailable: {e}")
        return []
    figs = []
    names = list(audits)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].bar(names, [audits[n]["sustained_push_per_ep"] for n in names])
    ax[0].set_title("sustained PUSH windows / ep")
    ax[1].bar(names, [audits[n]["both_contact_frac"] for n in names])
    ax[1].set_title("two-finger contact fraction")
    ax[2].bar(names, [audits[n]["ft_progress_in_contact"] for n in names])
    ax[2].set_title("ft progress in contact")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.suptitle("Branch A/D — sustained-contact coverage by policy")
    fig.tight_layout()
    p = out / "coverage.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    figs.append(str(p))
    if option_hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(option_hist, marker="o")
        ax.set_title("Branch B/C — option-CEM best objective")
        ax.set_xlabel("iter")
        fig.tight_layout()
        p = out / "option_cem.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        figs.append(str(p))
    return figs


def _gif(out: Path, actor: Any, expert_factory, log=_log) -> list[str]:
    gifs = []
    try:
        render_episode_gif(make_env(coord=False, difficulty=_DIFFICULTY), greedy_action_fn(actor),
                           str(out / "learned_policy.gif"), seed=_EVAL_SEED, width=640)
        gifs.append(str(out / "learned_policy.gif"))
        e = make_env(coord=False, difficulty=_DIFFICULTY)
        render_episode_gif(e, expert_factory(e), str(out / "expert.gif"), seed=_EVAL_SEED, width=640)
        gifs.append(str(out / "expert.gif"))
    except Exception as ex:  # noqa: BLE001
        log(f"[gif] skipped: {ex}")
    return gifs


# --------------------------------------------------------------------------- main protocol

def run(cfg: OptionRetestConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    frozen = load_frozen_actor(env)
    guards = wire_ledgers(env, frozen)
    reward = certify_v2b()
    baseline = measure_policy(frozen, cfg.n_eval)
    base_audit = _audit("frozen_DAgger", lambda e: greedy_action_fn(frozen), cfg)
    _log(f"[option] guards pipeline={guards['pipeline_schema']} provenance={guards['policy_provenance']} "
         f"| v2b {reward.get('delivers')} | BASELINE ft_dom={baseline['ft_dom']:.3f} mon_pass={baseline['monitor_pass']:.3f} "
         f"mon_score={baseline['monitor_score']:.3f} sustainedPUSH/ep={base_audit.sustained_push_per_ep:.2f}")

    # Branch A — audit the reference policies
    _log("[option] Branch A: sustained-PUSH audit ...")
    audits = {
        "frozen_DAgger": base_audit.as_dict(),
        "scripted_PushDemonstrator": _audit("scripted", _demo_af(_push_demo), cfg).as_dict(),
        "PhasePushController_default": _audit("phase_default", _demo_af(_phase_default_demo), cfg).as_dict(),
    }
    _log(f"[option] audit sustainedPUSH/ep: DAgger={audits['frozen_DAgger']['sustained_push_per_ep']:.2f} "
         f"scripted={audits['scripted_PushDemonstrator']['sustained_push_per_ep']:.2f} "
         f"phase_default={audits['PhasePushController_default']['sustained_push_per_ep']:.2f}")

    # Branch B/C — option-parameter CEM
    _log("[option] Branch B/C: option-parameter CEM over PhasePushController θ∈ℝ⁵ ...")
    opt = option_cem(lambda: make_env(coord=False, difficulty=_DIFFICULTY), cfg.option, log=_log)
    _option_df = _option_demo_factory(opt.theta)
    audits["PhasePushController_tuned"] = _audit("phase_tuned", _demo_af(_option_df), cfg).as_dict()
    _log(f"[option] option-CEM improved_over_scripted={opt.improved_over_scripted} θ={opt.theta_named} "
         f"| tuned sustainedPUSH/ep={audits['PhasePushController_tuned']['sustained_push_per_ep']:.2f}")

    # expert selection — the strongest sustained-contact expert (by delivery × sustained coverage).
    # tuple = (demonstrator object for collect_demos, demonstrator-object factory for DAgger/GIF, audit dict)
    experts = {
        "scripted_PushDemonstrator": (_push_demo(env), _push_demo, audits["scripted_PushDemonstrator"]),
        "PhasePushController_tuned": (_option_df(env), _option_df, audits["PhasePushController_tuned"]),
    }
    def _expert_score(a):
        return a["delivery_rate"] * (1.0 + a["sustained_push_per_ep"]) * (a["exploit_rate"] < 0.05)
    expert_name = max(experts, key=lambda k: _expert_score(experts[k][2]))
    expert_obj, expert_factory, _ea = experts[expert_name]
    _log(f"[option] selected imitation expert: {expert_name}")

    # Branch D — imitation (BC fine-tune, escalate to lightweight DAgger)
    _log("[option] Branch D: BC fine-tune (warm-started from frozen DAgger) ...")
    learned, demo_o, demo_a = bc_finetune(env, expert_obj, cfg, log=_log)
    n_samples = len(demo_o)
    bc_metrics = measure_policy(learned, cfg.n_eval)
    bc_audit = _audit("bc_finetune", lambda e: greedy_action_fn(learned), cfg)
    bc_accept = _accept(baseline, base_audit, bc_metrics, bc_audit)
    _log(f"[option] BC ft_dom={bc_metrics['ft_dom']:.3f} mon_score={bc_metrics['monitor_score']:.3f} "
         f"sustainedPUSH/ep={bc_audit.sustained_push_per_ep:.2f} accepted={bc_accept['accepted']}")

    candidates = [("bc_finetune", learned, bc_metrics, bc_audit, bc_accept, n_samples)]
    if not bc_accept["accepted"]:
        _log("[option] BC did not pass → lightweight on-policy DAgger escalation ...")
        dag = dagger_escalate(lambda: make_env(coord=False, difficulty=_DIFFICULTY), learned, expert_factory, cfg,
                              seed_demos=(demo_o, demo_a), log=_log)
        dag_metrics = measure_policy(dag, cfg.n_eval)
        dag_audit = _audit("dagger", lambda e: greedy_action_fn(dag), cfg)
        dag_accept = _accept(baseline, base_audit, dag_metrics, dag_audit)
        _log(f"[option] DAgger ft_dom={dag_metrics['ft_dom']:.3f} mon_score={dag_metrics['monitor_score']:.3f} "
             f"sustainedPUSH/ep={dag_audit.sustained_push_per_ep:.2f} accepted={dag_accept['accepted']}")
        candidates.append(("dagger", dag, dag_metrics, dag_audit, dag_accept, len(candidates)))

    # pick the best passing candidate (or the best-effort one for the report)
    passing = [c for c in candidates if c[4]["accepted"]]
    best = (passing[0] if passing else candidates[0])
    best_name, best_actor, best_m, best_a, best_acc, _ = best

    if passing:
        verdict = "POSITIVE"
        reason = (f"{best_name} (imitation from {expert_name}) raised sustained-contact coverage "
                  f"({base_audit.sustained_push_per_ep:.2f}→{best_a.sustained_push_per_ep:.2f} windows/ep) "
                  f"and {'improved' if best_acc['ft_dom_improved'] else 'preserved'} the headline metrics without exploit")
        ckpt = out / f"{best_name}_option.pt"
        torch.save(best_actor.state_dict(), ckpt)
        _log(f"[option] POSITIVE → saved {ckpt} (md5 {_md5_actor(best_actor)})")
    else:
        # distinguish "coverage couldn't be created" (INCONCLUSIVE) from "imitation dropped headline" (NEGATIVE)
        headline_held = all(best_acc["headline"].values())
        if headline_held and not any(best_acc["coverage"].values()):
            verdict = "NEGATIVE_WITH_MECHANISM"
            reason = ("imitation preserved the headline but did NOT transfer sustained-contact coverage from the "
                      "expert to the learned MLP — the clone still delivers via brief touches; the coverage gap is "
                      "an imitation-capacity/representation limit, not a lack of expert demonstrations")
        elif not headline_held:
            verdict = "NEGATIVE_WITH_MECHANISM"
            reason = ("imitation from the sustained-contact expert dropped a headline metric below the frozen "
                      "baseline (see acceptance.headline) — trading delivery for contact is not an improvement")
        else:
            verdict = "INCONCLUSIVE_WITH_NEXT_FIX"
            reason = ("demonstration coverage insufficient: next fix = more held-only sustained-contact demos / "
                      "more DAgger rounds / stronger expert (tuned option) before re-imitating")

    return _finalize(cfg, out, guards, reward, baseline, base_audit, audits, opt, expert_name,
                     candidates, best, verdict, reason, best_actor, expert_factory, t0)


def _cand_row(c) -> dict:
    name, actor, m, a, acc, ns = c
    return {"name": name, "n_samples_or_round": ns, "metrics": m, "audit": a.as_dict(),
            "acceptance": acc, "actor_md5": _md5_actor(actor)}


def _finalize(cfg, out, guards, reward, baseline, base_audit, audits, opt, expert_name,
              candidates, best, verdict, reason, best_actor, expert_factory, t0) -> dict:
    figs = _plots(out, audits, opt.history, [])
    gifs = _gif(out, best_actor, _demo_af(expert_factory))
    result = {
        "verdict": verdict, "reason": reason, "stage": cfg.stage, "wall_s": round(time.perf_counter() - t0, 1),
        "guards": {k: guards[k] for k in ("pipeline_schema", "policy_provenance", "actor_checkpoint_hash")},
        "provenance_fields": guards.get("provenance_fields", {}), "v2b_reward": reward,
        "baseline_frozen_dagger": baseline, "baseline_audit": base_audit.as_dict(),
        "branch_A_audit": audits,
        "branch_BC_option": {"improved_over_scripted": opt.improved_over_scripted, "theta_named": opt.theta_named,
                             "best_objective": opt.best_objective, "baseline_objective": opt.baseline_objective,
                             "best_metrics": opt.best_metrics, "history": opt.history},
        "imitation_expert": expert_name,
        "branch_D_candidates": [_cand_row(c) for c in candidates],
        "figures": figs, "gifs": gifs,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    _log(f"\n[option] VERDICT: {verdict}\n[option] {reason}\n[option] wrote {out / 'results.json'} "
         f"({result['wall_s']}s); figs={len(figs)} gifs={len(gifs)}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Option-level coin-toss improvement (gated protocol)")
    ap.add_argument("--stage", choices=["smoke", "full"], default="full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = OptionRetestConfig.smoke() if args.stage == "smoke" else OptionRetestConfig.full()
    if args.out:
        cfg.out_dir = args.out
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
