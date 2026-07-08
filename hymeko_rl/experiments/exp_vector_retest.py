"""Fair vector-critic retest — the gated, diagnostic-first overnight protocol driver (ONE file, ``--stage``).

Pipeline (see ``docs/plans/2026-07-08-fair-vector-critic-retest/``): action-diverse replay → measured MC component
critics → per-component calibration → long-horizon gradient-alignment probe → projected-direction GATE →
[gate open] bounded phase-gated residual smoke driven by the projected direction ONLY, else [gate shut]
monitor-directed CEM fallback. TaskMonitor stays the frozen external verifier; PipelineSchemaLedger +
PolicyProvenanceLedger are active and recorded; v2b reward is unchanged (reported + oracle-certified only). No
scalar TD3/SAC/CQL actor update anywhere. Emits a single verdict: POSITIVE / NEGATIVE_WITH_MECHANISM /
INCONCLUSIVE_WITH_NEXT_FIX.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.agents.residual_actor import ResidualActor, build_residual_net, contact_gate
from hymeko_rl.eval.component_calibration import calibrate_all, calibration_table
from hymeko_rl.eval.evaluate import greedy_action_fn, render_episode_gif
from hymeko_rl.eval.gradient_probe import ProbeConfig, evaluate_gate, gradient_alignment_probe
from hymeko_rl.eval.task_monitor import TaskMonitor
from hymeko_rl.eval.task_monitor.pipeline import PipelineSchemaLedger, TransitionSchema
from hymeko_rl.eval.task_monitor.provenance import (
    PolicyProvenanceLedger,
    PolicyRole,
    canonical_obs_batch,
    file_md5,
)
from hymeko_rl.experiments.exp_galambos_coord_ab import _coordination_metrics, make_env
from hymeko_rl.train.action_diverse_replay import DiverseReplayConfig, generate_action_diverse_replay
from hymeko_rl.train.cem_residual import CEMConfig, cem_optimize, make_residual_action_fn
from hymeko_rl.train.search_objective import COMPONENTS
from hymeko_rl.train.vector_critic import (
    VectorCriticConfig,
    action_gradient,
    build_vector_critics,
    projected_gradient,
    train_vector_critics_mc,
)

_CKPT = "experiments/v2_dagger/FROZEN_selected/mlp_s1_selected_d3.pt"
_REWARD_V2B = "data/robotics/galambos_task_deliver_v2b.hymeko"
_DIFFICULTY = 0.3
_EVAL_SEED = 9000
_ACCEPT_TOL = 0.02          # tolerance for "preserves baseline" acceptance


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class RetestConfig:
    stage: str = "full"
    out_dir: str = "experiments/2026_07_08_vector_retest"
    n_eval: int = 24
    replay: DiverseReplayConfig = field(default_factory=DiverseReplayConfig)
    critic: VectorCriticConfig = field(default_factory=lambda: VectorCriticConfig(steps=3000, batch_size=256, log_every=500))
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    n_probe_states: int = 40
    residual_epsilon: float = 0.4
    residual_steps: int = 800
    cem: CEMConfig = field(default_factory=CEMConfig)

    @classmethod
    def smoke(cls) -> "RetestConfig":
        c = cls(stage="smoke", out_dir="experiments/2026_07_08_vector_retest_smoke", n_eval=6)
        c.replay = DiverseReplayConfig(n_visit_episodes=6, max_steps=200, n_targets=120, branch_horizon=40, log_every=0)
        c.critic = VectorCriticConfig(steps=300, batch_size=128, log_every=150)
        c.probe = ProbeConfig(probe_horizon=60, n_best_sampled=3)
        c.n_probe_states = 8
        c.residual_steps = 100
        c.cem = CEMConfig(pop=8, elite=3, iters=3, n_eval_eps=3, max_steps=200, log_every=1)
        return c

    @classmethod
    def full(cls) -> "RetestConfig":
        c = cls(stage="full")
        c.replay = DiverseReplayConfig(n_visit_episodes=60, max_steps=300, n_targets=1800, branch_horizon=120,
                                       n_ood_per_state=1, log_every=300)
        return c


# ----------------------------------------------------------------------------- setup / ledgers / baseline

def load_frozen_actor(env: Any) -> Any:
    actor = build_collaborative_offpolicy(env, kind="mlp", hidden=64)[0]
    actor.load_state_dict(torch.load(_CKPT, map_location="cpu"))
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor


def wire_ledgers(env: Any, actor: Any) -> dict[str, Any]:
    """Activate + verify the minimum safety stack (pipeline schema + policy provenance) on the frozen anchor."""
    schema = TransitionSchema.from_env(env, action_dim=int(env.n_actions), priv_enabled=True)
    canon = [(f.name, f.dim) for f in schema.fields]
    ledger = PipelineSchemaLedger(schema)
    for st in ("rollout", "replay_serialize", "replay_load"):
        ledger.record(st, canon)
    ledger.record("critic_train", [("obs", canon[0][1]), ("action", canon[1][1]), ("priv", int(env.privileged_dim))])
    ledger.record("eval", [("obs", canon[0][1]), ("action", canon[1][1])])
    pipe_v = ledger.verify_or_abort()

    obs_batch = canonical_obs_batch(lambda: make_env(coord=False, difficulty=_DIFFICULTY), n=32, seed0=12345)
    prov = PolicyProvenanceLedger(obs_batch)
    prov.register_checkpoint("selected", PolicyRole.DAGGER_VAL_SELECTED, _CKPT, actor, arch="mlp", seed=1, dagger_stage="d3")
    prov.register_checkpoint("anchor", PolicyRole.DAGGER_VAL_SELECTED, _CKPT, actor, arch="mlp", seed=1, dagger_stage="d3")
    prov.expect_checkpoint_matches("selected", file_md5(_CKPT))
    prov_v = prov.verify_or_abort()
    fields = prov.report_fields(actor_name="selected", anchor_name="anchor", selected_name="selected",
                                reward_file=_REWARD_V2B, env_file="PlanarGraspEnv(robot=None,max_steps=300)")
    return {"pipeline_schema": "PASS" if pipe_v.passed else "FAIL",
            "policy_provenance": "PASS" if prov_v.passed else "FAIL",
            "stage_hashes": ledger.stage_hashes(), "provenance_fields": fields,
            "actor_checkpoint_hash": file_md5(_CKPT)}


def certify_v2b() -> dict[str, Any]:
    """Certify the FROZEN v2b training reward (report identity + delivers). On-record: delivers=True, return≈25.40."""
    try:
        from hymeko_rl.eval.reward_oracle import certify
        from hymeko_rl.env.reward import RewardSpec
        rep = certify(RewardSpec.from_hymeko(_REWARD_V2B))
        return {"reward_file": _REWARD_V2B, "delivers": bool(rep.delivers),
                "optimal_return": round(float(rep.optimal_return), 3)}
    except Exception as e:  # noqa: BLE001 — certification is a report field, not a hard gate for a diagnostic run
        return {"reward_file": _REWARD_V2B, "delivers": None, "certify_error": f"{type(e).__name__}: {e}",
                "on_record": "delivers=True, optimal_return=25.40 (reports/2026-07-07-v2b-reward-fix-ablation.md)"}


def measure_policy(actor_or_fn: Any, n_eval: int, *, is_action_fn: bool = False,
                   seed0: int = _EVAL_SEED) -> dict[str, float]:
    """Canonical eval: ft_dom + contact tiers (``_coordination_metrics``) + monitor pass/score/violation.

    ``is_action_fn`` routes an env-aware ``action_fn(env, obs)`` (residual / CEM policy) through both the coord
    metric (via the ``action_fn`` param) and the monitor (via ``record_trajectory``). ``seed0`` selects the eval
    episode seed batch (default the canonical 9000) — vary it for multi-seed evaluation."""
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    if is_action_fn:
        coord = _coordination_metrics(env, None, n_eval, seed0, action_fn=actor_or_fn)
        def make_action(e):
            return (actor_or_fn)
    else:
        coord = _coordination_metrics(env, actor_or_fn, n_eval, seed0)
        def make_action(e):
            return (greedy_action_fn(actor_or_fn))
    mon = TaskMonitor.from_env(env).evaluate_policy(
        lambda: make_env(coord=False, difficulty=_DIFFICULTY), make_action, n_eval, seed0=seed0)
    return {
        "ft_dom": coord["fingertip_dominant_delivery"], "raw_delivery": coord["raw_delivery"],
        "body_driven_exploit": coord["body_driven_exploit_delivery"], "arm_body_rate": coord["arm_body_rate"],
        "coin_vel_to_zone": coord["coin_vel_to_zone"], "dist_delta": coord["dist_delta"],
        "total_reward": coord.get("total_reward", float("nan")),
        "monitor_pass": mon["monitor_pass_rate"], "monitor_score": mon["monitor_score_mean"],
        "violation_reason": mon.get("top_violation", "n/a"),
    }


# ----------------------------------------------------------------------------- stage 2: q_total scalar critic

def fit_q_total(env: Any, data: dict[str, np.ndarray], cfg: VectorCriticConfig) -> Any:
    """A single scalar critic fit to the measured objective return (delivery+progress) — the 'naive scalar' whose
    action-gradient the probe compares against the projected vector direction."""
    critic = build_collaborative_offpolicy(env, kind="mlp", hidden=64, n_critics=1, privileged=True)[1][0]
    target = (data["mc_r_delivery"] + data["mc_r_progress"]).astype(np.float32)
    dev = torch.device(cfg.device)
    obs, act, z = (torch.as_tensor(data[k], device=dev) for k in ("obs", "action", "z"))
    tgt = torch.as_tensor(target, device=dev)
    opt = torch.optim.Adam(critic.to(dev).parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    n = int(obs.shape[0])
    for step in range(1, cfg.steps + 1):
        idx = torch.as_tensor(rng.integers(0, n, size=min(cfg.batch_size, n)), device=dev)
        loss = torch.nn.functional.mse_loss(critic(obs[idx], act[idx], z[idx]), tgt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not np.isfinite(float(loss.detach())):
            raise FloatingPointError(f"q_total loss non-finite at step {step}")
    return critic


# ----------------------------------------------------------------------------- stage 6: gated residual smoke

def projected_residual_smoke(env: Any, actor: Any, critics: dict[str, Any], data: dict[str, np.ndarray],
                             baseline: dict[str, float], cfg: RetestConfig, log=_log) -> dict[str, Any]:
    """Fit a bounded, phase-gated, zero-init residual to IMITATE the projected-direction-improved action on engaged
    states (projected/vector gradient only — no scalar TD). Accept iff it preserves/improves the baseline."""
    lo, hi = env._ctrl_lo.astype(np.float32), env._ctrl_hi.astype(np.float32)
    eta = cfg.probe.probe_eta
    eng = np.nonzero((data["is_ood"] < 0.5) & (data["both_contact"] > 0.5))[0]
    if eng.size < 16:
        return {"ran": False, "reason": "insufficient engaged states for residual imitation"}
    eng = eng[:400]
    log(f"[smoke] building projected targets on {eng.size} engaged states (eta={eta}) ...")
    targets = np.zeros((eng.size, int(env.n_actions)), dtype=np.float32)
    for j, i in enumerate(eng):
        base_a = _greedy_np(actor, data["obs"][i])
        grads = {c: action_gradient(critics[c], data["obs"][i], base_a, data["z"][i]) for c in COMPONENTS}
        d_star, _ = projected_gradient(grads, normalize=cfg.probe.normalize_projection)
        targets[j] = np.clip(base_a + eta * d_star, lo, hi)

    residual = ResidualActor(actor, build_residual_net(int(np.prod(env.observation_space.shape)), int(env.n_actions)),
                             epsilon=cfg.residual_epsilon, action_lo=lo, action_hi=hi)
    opt = torch.optim.Adam(residual.residual.parameters(), lr=3e-4)
    obs_t = torch.as_tensor(data["obs"][eng])
    z_t = torch.as_tensor(data["z"][eng])
    tgt_t = torch.as_tensor(targets)
    gate = contact_gate(z_t)
    for step in range(1, cfg.residual_steps + 1):
        a_res = residual.action_mean(obs_t, gate)
        loss = torch.nn.functional.mse_loss(a_res, tgt_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg.residual_steps // 4) == 0:
            log(f"  [smoke] residual step {step}/{cfg.residual_steps} imit_loss {float(loss.detach()):.4f} "
                f"sat {residual.saturation(obs_t):.3f}")

    def residual_fn(e: Any, obs: np.ndarray) -> np.ndarray:
        z = torch.as_tensor(e.privileged_state()[None])
        with torch.no_grad():
            a = residual.action_mean(torch.as_tensor(obs[None], dtype=torch.float32), contact_gate(z))
        return a[0].numpy().astype(np.float32)

    after = measure_policy(residual_fn, cfg.n_eval, is_action_fn=True)
    sat = float(residual.saturation(obs_t))
    accept = _smoke_accept(baseline, after, sat)
    return {"ran": True, "epsilon": cfg.residual_epsilon, "residual_saturation": sat,
            "baseline": baseline, "after": after, "acceptance": accept,
            "accepted": bool(accept["accepted"])}


_HEADLINE = ("ft_dom", "monitor_pass", "monitor_score")


def _strict_improvement(base: dict[str, float], after: dict[str, float], tol: float = _ACCEPT_TOL) -> tuple[bool, list[str]]:
    """A fallback's purpose is to IMPROVE: strictly raise ≥1 headline metric beyond ``tol`` while preserving the
    others and not raising exploit/arm-body. A no-op that merely reproduces the baseline is NOT an improvement."""
    preserved = all(after[m] >= base[m] - tol for m in _HEADLINE)
    no_exploit = (after["body_driven_exploit"] <= base["body_driven_exploit"] + tol
                  and after["arm_body_rate"] <= base["arm_body_rate"] + tol)
    improved = [m for m in _HEADLINE if after[m] > base[m] + tol]
    return bool(improved and preserved and no_exploit), improved


def _smoke_accept(base: dict[str, float], after: dict[str, float], sat: float) -> dict[str, Any]:
    checks = {
        "ft_dom_preserved": after["ft_dom"] >= base["ft_dom"] - _ACCEPT_TOL,
        "monitor_pass_preserved": after["monitor_pass"] >= base["monitor_pass"] - _ACCEPT_TOL,
        "monitor_score_preserved": after["monitor_score"] >= base["monitor_score"] - _ACCEPT_TOL,
        "no_exploit_rise": after["body_driven_exploit"] <= base["body_driven_exploit"] + _ACCEPT_TOL,
        "no_arm_body_rise": after["arm_body_rate"] <= base["arm_body_rate"] + _ACCEPT_TOL,
        "residual_not_saturated": sat < 0.9,
    }
    return {"accepted": all(checks.values()), "checks": checks}


# ----------------------------------------------------------------------------- fallback: monitor-directed CEM

def run_cem_fallback(env: Any, actor: Any, baseline: dict[str, float], cfg: RetestConfig, log=_log) -> dict[str, Any]:
    log("[fallback] gate shut → monitor-directed CEM over bounded phase-gated residual (no critic) ...")
    res = cem_optimize(env, actor, cfg.cem, log=log)
    lo, hi = env._ctrl_lo.astype(np.float32), env._ctrl_hi.astype(np.float32)
    fn = make_residual_action_fn(actor, res.theta, cfg.cem.epsilon, lo, hi,
                                 progress_eps=cfg.cem.progress_eps, near_coin=cfg.cem.near_coin)
    after = measure_policy(fn, cfg.n_eval, is_action_fn=True)
    accept = _smoke_accept(baseline, after, sat=0.0)
    return {"kind": "monitor_directed_cem", "theta": res.theta.tolist(),
            "search_objective_base": res.baseline_objective, "search_objective_best": res.best_objective,
            "history": res.history, "baseline": baseline, "after": after, "acceptance": accept,
            "accepted": bool(accept["accepted"]), "action_fn": fn}


# ----------------------------------------------------------------------------- helpers

def _greedy_np(actor: Any, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy().astype(np.float32)


def _select_probe_states(replay: Any, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    pool = replay.probe_pool
    if not pool:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [(pool[i].obs, pool[i].z, pool[i].snap) for i in idx]


def _plots(out: Path, cal_rows: list[dict], probe: dict, log=_log) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log(f"[plots] matplotlib unavailable: {e}")
        return []
    figs: list[str] = []
    names = [r["name"] for r in cal_rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["spearman_q_mc"] for r in cal_rows], 0.4, label="spearman(Q,MC)")
    ax.bar(x + 0.2, [r["within_state_rank"] for r in cal_rows], 0.4, label="within-state action rank")
    ax.axhline(0.5, ls="--", c="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Component critic calibration (measured MC targets)")
    ax.legend()
    fig.tight_layout()
    p1 = out / "calibration.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    figs.append(str(p1))
    if probe.get("per_candidate"):
        pc = probe["per_candidate"]
        cands = list(pc)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for k, metric in enumerate(("two_finger_rate", "ft_progress", "monitor_score")):
            ax[k].bar(cands, [pc[c][metric] for c in cands])
            ax[k].set_title(metric)
            ax[k].tick_params(axis="x", rotation=30)
        fig.suptitle("Gradient-alignment probe: candidate branch outcomes")
        fig.tight_layout()
        p2 = out / "probe_candidates.png"
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        figs.append(str(p2))
    return figs


def _gif(out: Path, actor: Any, best_fn: Any, log=_log) -> list[str]:
    gifs: list[str] = []
    try:
        env = make_env(coord=False, difficulty=_DIFFICULTY)
        render_episode_gif(env, greedy_action_fn(actor), str(out / "dagger_baseline.gif"), seed=_EVAL_SEED, width=640)
        gifs.append(str(out / "dagger_baseline.gif"))
        if best_fn is not None:
            env2 = make_env(coord=False, difficulty=_DIFFICULTY)
            render_episode_gif(env2, best_fn, str(out / "best_candidate.gif"), seed=_EVAL_SEED, width=640)
            gifs.append(str(out / "best_candidate.gif"))
    except Exception as e:  # noqa: BLE001 — GIF is presentation sugar, not a gate
        log(f"[gif] render skipped: {e}")
    return gifs


# ----------------------------------------------------------------------------- main protocol

def run(cfg: RetestConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    actor = load_frozen_actor(env)

    _log(f"[retest] stage={cfg.stage} out={out}")
    guards = wire_ledgers(env, actor)
    _log(f"[retest] ledgers: pipeline={guards['pipeline_schema']} provenance={guards['policy_provenance']} "
         f"actor_hash={guards['actor_checkpoint_hash']}")
    reward = certify_v2b()
    _log(f"[retest] v2b reward: {reward}")
    baseline = measure_policy(actor, cfg.n_eval)
    _log(f"[retest] BASELINE (frozen DAgger): ft_dom={baseline['ft_dom']:.3f} monitor_pass={baseline['monitor_pass']:.3f} "
         f"monitor_score={baseline['monitor_score']:.3f} exploit={baseline['body_driven_exploit']:.3f} "
         f"arm_body={baseline['arm_body_rate']:.3f}")

    # stage 1
    replay = generate_action_diverse_replay(env, actor, cfg.replay, log=_log)
    # stage 2
    critics = build_vector_critics(env)
    _log("[retest] fitting MC component critics ...")
    train_vector_critics_mc(critics, replay.data, cfg.critic)
    q_total = fit_q_total(env, replay.data, cfg.critic)
    # stage 3
    cal = calibrate_all(critics, replay.data)
    cal_rows = calibration_table(cal)
    _log("[retest] calibration: " + " | ".join(f"{r['name']}:sp={r['spearman_q_mc']:+.2f},ws={r['within_state_rank']:.2f}"
                                               f"{'*deg' if r['degenerate'] else ''}" for r in cal_rows))
    # only the components the projected direction actually uses need to be calibrated (approach is unused)
    _proj_comps = ("delivery", "progress", "contact", "antiexploit")
    critics_calibrated = all(cal[c].calibrated for c in _proj_comps if not cal[c].degenerate)
    # stage 4 + 5
    probe_states = _select_probe_states(replay, cfg.n_probe_states, seed=1)
    if not probe_states:
        return _finalize(cfg, out, guards, reward, baseline, cal_rows, {}, {}, None, None,
                         "INCONCLUSIVE_WITH_NEXT_FIX", "no engaged (CONTACT/PUSH) probe states found — raise "
                         "n_visit_episodes / difficulty to reach contact", t0)
    _log(f"[retest] gradient-alignment probe over {len(probe_states)} engaged states "
         f"(horizon {cfg.probe.probe_horizon}) ...")
    monitor = TaskMonitor.from_env(env)
    probe = gradient_alignment_probe(env, actor, critics, q_total, probe_states, monitor, cfg.probe, log=_log)
    gate = evaluate_gate(probe)
    _log(f"[retest] GATE: VECTOR_PROJECTED_PROMISING={gate['VECTOR_PROJECTED_PROMISING']} "
         f"proj_vs_dagger={gate['projected_vs_dagger']} | critics_calibrated={critics_calibrated}")

    # stage 6 or fallback
    branch = smoke = fallback = None
    best_fn = None
    if gate["VECTOR_PROJECTED_PROMISING"] and critics_calibrated:
        branch = "vector_residual_smoke"
        _log("[retest] GATE OPEN → residual actor smoke (projected gradient only) ...")
        smoke = projected_residual_smoke(env, actor, critics, replay.data, baseline, cfg, log=_log)
        # the gated smoke's bar is "preserve or improve" (the prior SCALAR smoke degraded 0.75→0.542)
        if smoke.get("accepted"):
            imp, imp_metrics = _strict_improvement(baseline, smoke["after"])
            verdict = "POSITIVE"
            reason = (f"projected-gradient residual {'IMPROVED ' + str(imp_metrics) if imp else 'PRESERVED'} the "
                      "frozen DAgger baseline (the scalar-gradient residual degraded it) — vector direction is "
                      "non-damaging where the scalar was damaging")
        else:
            verdict = "NEGATIVE_WITH_MECHANISM"
            reason = ("gate opened but the projected residual did not preserve the baseline: "
                      + str(smoke["acceptance"]["checks"]))
    else:
        branch = "monitor_directed_cem_fallback"
        fallback = run_cem_fallback(env, actor, baseline, cfg, log=_log)
        best_fn = fallback.pop("action_fn", None)
        improved, imp_metrics = _strict_improvement(baseline, fallback["after"])
        if improved:
            verdict = "POSITIVE"
            reason = f"monitor-directed CEM IMPROVED {imp_metrics} over the frozen DAgger baseline without exploit rise"
        else:
            verdict = "NEGATIVE_WITH_MECHANISM"
            reason = ("vector-projected gradient not monitor-aligned (gate shut) AND CEM found no improving bounded "
                      "monitor-directed residual — the frozen DAgger is a local optimum for this residual class; "
                      "mechanism: " + _mechanism(gate, cal_rows))
    return _finalize(cfg, out, guards, reward, baseline, cal_rows, probe.as_dict(), gate, smoke, fallback,
                     verdict, reason, t0, branch=branch, replay=replay, actor=actor, best_fn=best_fn)


def _mechanism(gate: dict, cal_rows: list[dict]) -> str:
    uncal = [r["name"] for r in cal_rows if not r["calibrated"] and not r["degenerate"]]
    deg = [r["name"] for r in cal_rows if r["degenerate"]]
    bits = []
    if uncal:
        bits.append(f"uncalibrated component critics {uncal}")
    if deg:
        bits.append(f"degenerate (constant target) components {deg}")
    pv = gate.get("projected_vs_scalar", {})
    bits.append(f"projected−scalar Δ={pv}")
    return "; ".join(bits)


def _finalize(cfg, out, guards, reward, baseline, cal_rows, probe, gate, smoke, fallback, verdict, reason, t0,
              *, branch="n/a", replay=None, actor=None, best_fn=None) -> dict[str, Any]:
    figs = _plots(out, cal_rows, probe) if cal_rows else []
    gifs = _gif(out, actor, best_fn) if actor is not None else []
    result = {
        "verdict": verdict, "reason": reason, "branch": branch, "stage": cfg.stage,
        "wall_s": round(time.perf_counter() - t0, 1),
        "guards": {k: guards[k] for k in ("pipeline_schema", "policy_provenance", "actor_checkpoint_hash")},
        "provenance_fields": guards.get("provenance_fields", {}),
        "v2b_reward": reward,
        "baseline_frozen_dagger": baseline,
        "replay": None if replay is None else {
            "n_visited": replay.n_visited, "n_targets": replay.n_targets, "n_rows": int(replay.data["obs"].shape[0]),
            "n_ood": int(replay.data["is_ood"].sum()), "n_diverged": replay.n_diverged,
            "phase_counts": replay.phase_counts, "eps_used": list(replay.eps_used)},
        "calibration": cal_rows,
        "probe": probe, "gate": gate,
        "residual_smoke": smoke, "fallback": fallback,
        "figures": figs, "gifs": gifs,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    _log(f"\n[retest] VERDICT: {verdict}\n[retest] {reason}\n[retest] wrote {out/'results.json'} "
         f"({result['wall_s']}s); figs={len(figs)} gifs={len(gifs)}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fair vector-critic retest (gated diagnostic protocol)")
    ap.add_argument("--stage", choices=["smoke", "full"], default="full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = RetestConfig.smoke() if args.stage == "smoke" else RetestConfig.full()
    if args.out:
        cfg.out_dir = args.out
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
