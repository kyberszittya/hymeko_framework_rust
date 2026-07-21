"""coin_toss_v3 — handoff-conditioned option mixture. Orchestrator (stages: smoke|audit|library|oracle_gate|eval|full).

Reduce V1's non-delivery tail by a gate g(z_handoff)->option_id over a DISCRETE option library {theta_k}. V1 is frozen
and only read (E_valselect_v2.pt + phase_gated_theta_v1.json). No per-step RL, no reward change, no CORE edit; the
option library is CEM-derived (a ManifoldSearchObjective, separate from the frozen TaskMonitor verifier), the gate is
a supervised classifier over the discrete library. The centerpiece is the DISCRIMINATING TEST (Part C): the per-handoff
oracle (best-of-K safe theta) vs global V1 — because theta only affects the post-handoff push, z_handoff is
theta-independent and each theta_k is evaluated by re-rolling the same seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.eval.handoff_audit import full_metrics_multiseed
from hymeko_rl.eval.handoff_tensor import (
    FAILURE_CLASSES, HANDOFF_FIELDS, HandoffSample, classify_failure, handoff_flat, handoff_index,
    handoff_node_matrix, handoff_schema_hash, roll_with_actions)
from hymeko_rl.eval.push_audit import sustained_push_windows
from hymeko_rl.eval.semantic_tensor import SemanticSchema
from hymeko_rl.eval.task_monitor import TaskMonitor
from hymeko_rl.eval.task_monitor.context import MonitorContext
from hymeko_rl.eval.task_monitor.contract import MonitorContract
from hymeko_rl.eval.task_monitor.provenance import file_md5
from hymeko_rl.experiments.exp_galambos_coord_ab import grade_delivery, make_env
from hymeko_rl.experiments.exp_option_retest import _fresh_actor
from hymeko_rl.train.manifold_option_search import ManifoldSearchConfig, make_gated, pareto_front, theta7_named
from hymeko_rl.train.option_gate import HSiKANGate, LogisticGate, MLPGate, NearestCentroidGate
from hymeko_rl.train.option_library import OptionLibrary, build_library

_E_CKPT = "experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt"
_V1 = "experiments/2026_07_09_hmco_v1/phase_gated_theta_v1.json"
_OUT = Path("experiments/2026_07_09_v3_handoff_gate")
_DIFF = 0.3
_MAX_STEPS = 300
_ZONE_HALF = 0.055
_K_SUSTAINED = 5
_METRIC_KEYS = ("ft_dom", "monitor_pass", "monitor_score", "sustained_push_per_ep", "both_contact_frac",
                "ft_progress_in_contact", "body_progress_in_contact", "arm_body_rate", "body_driven_exploit")


def _log(m: str) -> None:
    print(m, flush=True)


def _roll_env(fingertip_geometry: str = "POINT") -> Any:
    # coin-toss env uses the SOUND v2b delivery reward (2026-07-09 reward-identity fix; was falling through to the
    # galambos_task farming default). Reward-independent metrics are unaffected; only env.step's reward scalar changes.
    return make_env(coord=False, difficulty=_DIFF, deliver=True, fingertip_geometry=fingertip_geometry)


def _mirror_env() -> Any:
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    return PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=_DIFF, task_graph=True)


def _load_e() -> Any:
    a = _fresh_actor(_roll_env(), warm_start=False)
    a.load_state_dict(torch.load(_E_CKPT, map_location="cpu"))
    a.eval()
    return a


def _v1_theta7() -> np.ndarray:
    return np.asarray(json.load(open(_V1))["theta7"], np.float64)


def _gated_factory(mlp: Any, theta: np.ndarray):
    def make(env: Any):
        pg = make_gated(env, theta, mlp)
        pg.reset()
        return lambda e, _o: np.asarray(pg.action(e), np.float32)
    return make


def _metrics(mlp: Any, theta: np.ndarray, seeds: tuple[int, ...], n_eval: int, contract: Any) -> dict:
    st = full_metrics_multiseed(_roll_env, _gated_factory(mlp, theta), contract, seeds=seeds, n_eval=n_eval,
                                k=_K_SUSTAINED)
    return {k: float(st.mean.get(k, 0.0)) for k in _METRIC_KEYS}


# --------------------------------------------------------------------------- handoff sampling (shared A + C + D)

def _handoff_sample(e: Any, theta7: np.ndarray, seed: int, contract: Any, mon: Any,
                    roll: Any, mirror: Any) -> "HandoffSample | None":
    """Roll V1 on ``seed`` (reusable roll+mirror envs); extract handoff flat + node matrix + failure class + V1
    outcome labels. z_handoff is theta-independent (the approach up to first two-finger contact is frozen)."""
    pg = make_gated(roll, theta7, e)
    pg.reset()
    traj, actions = roll_with_actions(roll, lambda env, _o: np.asarray(pg.action(env), np.float32), seed)
    if not traj:
        return None
    ctx = MonitorContext.build(traj, contract)
    verdict = mon.evaluate(traj)
    zone = np.array([float(roll._zone_x), float(roll._zone_y)])
    idx = handoff_index(ctx)
    z = handoff_flat(ctx, idx, zone)
    nm = handoff_node_matrix(mirror, actions, idx, seed)
    delivered = bool(ctx.max_dwell >= contract.success_steps)
    grade = grade_delivery(held=delivered, body_progress=float(ctx.body_prog), fingertip_progress=float(ctx.ft_prog))
    labels = {"monitor_score": float(verdict.monitor_score), "monitor_pass": float(verdict.monitor_pass),
              "ft_dom": 1.0 if grade == "fingertip_dominant" else 0.0,
              "sustained": 1.0 if len(sustained_push_windows(ctx, contract, _K_SUSTAINED)) > 0 else 0.0,
              "delivered": 1.0 if delivered else 0.0, "final_dist": float(ctx.dfin)}
    return HandoffSample(seed=seed, z_flat=z, node_matrix=nm, failure_class=classify_failure(ctx, contract, zone),
                         labels=labels)


def _collect_handoffs(e: Any, v1_theta7: np.ndarray, contract: Any, mon: Any, seeds, *, tag: str) -> list:
    roll, mirror = _roll_env(), _mirror_env()
    out = []
    for i, s in enumerate(seeds):
        smp = _handoff_sample(e, v1_theta7, int(s), contract, mon, roll, mirror)
        if smp is not None:
            out.append(smp)
        if (i + 1) % 20 == 0:
            _log(f"  [{tag}] handoff {i + 1}/{len(seeds)}")
    return out


def _incidence() -> tuple:
    a_pos, a_neg = _mirror_env().hg.dense_signed_adj()
    return a_pos, a_neg


# --------------------------------------------------------------------------- Part A: failure-tail audit

def _predict_accuracy(gate, X_tr, y_tr, X_te, y_te) -> float:
    gate.fit(X_tr, y_tr)
    return float((gate.predict(X_te) == y_te).mean())


def _per_class_means(samples: list) -> dict:
    """Mean handoff-feature vector per failure class — a signature of each failure mode."""
    out = {}
    for c in FAILURE_CLASSES:
        zs = [s.z_flat for s in samples if s.failure_class == c]
        if zs:
            out[c] = {f: round(float(v), 4) for f, v in zip(HANDOFF_FIELDS, np.mean(zs, 0))}
    return out


def _separability(Z, STRUCT, NODE, y, ntr: int, a_pos, a_neg) -> dict:
    """Held-out accuracy of failure-class prediction from the handoff state (flat / structured-MLP / HSiKAN)."""
    nc, nd = len(FAILURE_CLASSES), NODE.shape[2]
    return {
        "mlp_flat": round(_predict_accuracy(MLPGate(Z.shape[1], nc, hidden=48, epochs=400),
                                            Z[:ntr], y[:ntr], Z[ntr:], y[ntr:]), 4),
        "mlp_structured": round(_predict_accuracy(MLPGate(STRUCT.shape[1], nc, hidden=48, epochs=400),
                                                  STRUCT[:ntr], y[:ntr], STRUCT[ntr:], y[ntr:]), 4),
        "hsikan": round(_predict_accuracy(HSiKANGate(nd, nc, a_pos, a_neg, hidden=48, epochs=400),
                                          NODE[:ntr], y[:ntr], NODE[ntr:], y[ntr:]), 4),
    }


def part_a_audit(e, v1_theta7, contract, mon, *, seeds, a_pos, a_neg, log=_log) -> dict:
    log(f"[v3A] failure-tail audit on {len(seeds)} V1 rollouts ...")
    samples = _collect_handoffs(e, v1_theta7, contract, mon, seeds, tag="v3A")
    n = len(samples)
    counts = Counter(s.failure_class for s in samples)
    dist = {c: counts.get(c, 0) for c in FAILURE_CLASSES}
    fc_to_id = {c: i for i, c in enumerate(FAILURE_CLASSES)}
    y = np.array([fc_to_id[s.failure_class] for s in samples], np.int64)
    NODE = np.stack([s.node_matrix for s in samples]).astype(np.float32)
    Z = np.stack([s.z_flat for s in samples]).astype(np.float32)
    sep = _separability(Z, NODE.reshape(n, -1), NODE, y, int(0.7 * n), a_pos, a_neg)
    majority = max(dist.values()) / n
    log(f"[v3A] failure dist: {dict((c, k) for c, k in dist.items() if k)} | "
        f"separability(test) flat {sep['mlp_flat']:.2f} struct {sep['mlp_structured']:.2f} hsikan {sep['hsikan']:.2f} "
        f"(majority {majority:.2f})")
    return {"n": n, "failure_distribution": dist, "nondeliver_frac": round(1 - dist["delivered"] / n, 4),
            "per_class_handoff_means": _per_class_means(samples), "separability_test_acc": sep,
            "majority_class_frac": round(majority, 4)}


# --------------------------------------------------------------------------- Part B: option library

def part_b_library(e, v1_theta7, contract, cfg: ManifoldSearchConfig, *, k: int, log=_log) -> tuple[OptionLibrary, dict]:
    log("[v3B] building option library (CEM archive + scalarization) ...")
    v1_metrics = _metrics(e, v1_theta7, cfg.search_seeds, cfg.search_n_eval, contract)
    res = build_library(_roll_env, e, v1_theta7, v1_metrics, contract, cfg, k=k, log=log)
    front = pareto_front(res.cem.candidates)
    info = {"k": res.library.k, "names": res.library.names, "n_candidates": res.n_candidates,
            "n_feasible": res.n_feasible, "v1_search_metrics": {kk: round(v, 4) for kk, v in v1_metrics.items()},
            "library": res.library.as_dict(),
            "pareto_front": [{"theta7_named": theta7_named(np.asarray(c.theta)),
                              "ft_dom": round(c.metrics["ft_dom"], 4),
                              "monitor_score": round(c.metrics["monitor_score"], 4),
                              "sustained": round(c.metrics["sustained_push_per_ep"], 4)} for c in front]}
    (_OUT / "library.json").write_text(json.dumps(info, indent=2))
    log(f"[v3B] library K={res.library.k} {res.library.names}; pareto {len(front)} pts; wrote library.json")
    return res.library, info


# --------------------------------------------------------------------------- Part C: oracle + gate (discriminating)

def _safe_score(m: dict, ref: dict) -> float:
    """Outcome score for oracle/gate ranking (higher=better delivery+monitor); unsafe options are vetoed."""
    if (m["body_driven_exploit"] > 0.02 or m["arm_body_rate"] > ref["arm_body_rate"] + 0.02
            or m["body_progress_in_contact"] > ref["body_progress_in_contact"] + 1e-3):
        return -1e9
    return m["ft_dom"] + 0.5 * m["monitor_score"] + 0.25 * m["monitor_pass"]


def _mean_metrics(rows: list[dict]) -> dict:
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in _METRIC_KEYS}


def _label_grid(e, lib: OptionLibrary, samples: list, n_label: int, contract, log) -> list:
    """grid[i][k] = mean metrics of option theta_k re-rolled on samples[i]'s seed (the per-handoff option outcomes)."""
    grid: list[list[dict]] = []
    for i, s in enumerate(samples):
        grid.append([_metrics(e, np.asarray(lib.thetas[k]), (s.seed,), n_label, contract) for k in range(lib.k)])
        if (i + 1) % 10 == 0:
            log(f"  [v3C] labeled {i + 1}/{len(samples)} seeds")
    return grid


def _train_gate_family(Z, NODE, oracle_lbl, tr: slice, te: slice, k: int, a_pos, a_neg) -> dict:
    """Fit each gate on (handoff state, oracle label) over the train slice; return test-slice predictions."""
    flat: dict[str, Any] = {"nearest_centroid": NearestCentroidGate(k), "logistic": LogisticGate(Z.shape[1], k, epochs=400),
                            "mlp": MLPGate(Z.shape[1], k, hidden=64, epochs=400)}
    preds = {}
    for name, g in flat.items():
        g.fit(Z[tr], oracle_lbl[tr])
        preds[name] = g.predict(Z[te])
    hg = HSiKANGate(NODE.shape[2], k, a_pos, a_neg, hidden=48, epochs=400).fit(NODE[tr], oracle_lbl[tr])
    preds["hsikan"] = hg.predict(NODE[te])
    return preds


def _gate_eval(preds: dict, grid: list, te_idx: list, oracle_lbl, te: slice, v1_ref: dict, k: int) -> dict:
    out = {}
    for name, p in preds.items():
        rows = [grid[te_idx[j]][int(p[j])] for j in range(len(te_idx))]
        out[name] = {"metrics": _mean_metrics(rows),
                     "score": round(float(np.mean([_safe_score(r, v1_ref) for r in rows])), 4),
                     "oracle_agreement": round(float((p == oracle_lbl[te]).mean()), 4),
                     "option_usage": {int(kk): int((p == kk).sum()) for kk in range(k)}}
    return out


def part_c_oracle_gate(e, lib: OptionLibrary, contract, mon, *, seeds, n_label, a_pos, a_neg,
                       v1_ref: dict, log=_log) -> dict:
    log(f"[v3C] oracle labeling: {len(seeds)} handoff states x K={lib.k} options x {n_label} eps ...")
    samples = _collect_handoffs(e, np.asarray(lib.thetas[0]), contract, mon, seeds, tag="v3C")
    Z = np.stack([s.z_flat for s in samples]).astype(np.float32)
    NODE = np.stack([s.node_matrix for s in samples]).astype(np.float32)
    grid = _label_grid(e, lib, samples, n_label, contract, log)
    n = len(samples)
    scores = np.array([[_safe_score(grid[i][k], v1_ref) for k in range(lib.k)] for i in range(n)])
    oracle_lbl = scores.argmax(1).astype(np.int64)
    ntr = int(0.7 * n)
    tr, te = slice(0, ntr), slice(ntr, n)
    preds = _train_gate_family(Z, NODE, oracle_lbl, tr, te, lib.k, a_pos, a_neg)
    te_idx = list(range(ntr, n))
    v1_rows = [grid[i][0] for i in te_idx]
    oracle_rows = [grid[i][int(oracle_lbl[i])] for i in te_idx]
    rand_rows = [{kk: float(np.mean([grid[i][j][kk] for j in range(lib.k)])) for kk in _METRIC_KEYS} for i in te_idx]
    gate_eval = _gate_eval(preds, grid, te_idx, oracle_lbl, te, v1_ref, lib.k)
    v1_score = float(np.mean([_safe_score(r, v1_ref) for r in v1_rows]))
    oracle_score = float(np.mean([_safe_score(r, v1_ref) for r in oracle_rows]))
    best_gate = max(gate_eval, key=lambda nm: gate_eval[nm]["score"])
    log(f"[v3C] oracle score {oracle_score:.3f} vs V1 {v1_score:.3f} (gap {oracle_score - v1_score:+.3f}); "
        f"best gate '{best_gate}' score {gate_eval[best_gate]['score']:.3f} "
        f"agree {gate_eval[best_gate]['oracle_agreement']:.2f}")
    out = {"n_seeds": n, "n_train": ntr, "n_test": len(te_idx), "k": lib.k, "seeds": [s.seed for s in samples],
           "oracle_label_distribution": {int(kk): int((oracle_lbl == kk).sum()) for kk in range(lib.k)},
           "v1_score": round(v1_score, 4), "oracle_score": round(oracle_score, 4),
           "oracle_gap": round(oracle_score - v1_score, 4),
           "v1_test_metrics": _mean_metrics(v1_rows), "oracle_test_metrics": _mean_metrics(oracle_rows),
           "random_test_metrics": _mean_metrics(rand_rows), "gates": gate_eval, "best_gate": best_gate,
           # persist training data so the eval stage can refit the chosen gate without re-labeling
           "_train": {"Z": Z.tolist(), "NODE": NODE.tolist(), "oracle_label": oracle_lbl.tolist(), "ntr": ntr}}
    (_OUT / "oracle_gate.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- Part D: fresh eval + gifs + plots

def _fit_best_gate(oc: dict, lib: OptionLibrary, a_pos, a_neg):
    Z = np.asarray(oc["_train"]["Z"], np.float32)
    NODE = np.asarray(oc["_train"]["NODE"], np.float32)
    y = np.asarray(oc["_train"]["oracle_label"], np.int64)
    name = oc["best_gate"]
    if name == "nearest_centroid":
        return NearestCentroidGate(lib.k).fit(Z, y), "flat"
    if name == "logistic":
        return LogisticGate(Z.shape[1], lib.k, epochs=400).fit(Z, y), "flat"
    if name == "mlp":
        return MLPGate(Z.shape[1], lib.k, hidden=64, epochs=400).fit(Z, y), "flat"
    return HSiKANGate(NODE.shape[2], lib.k, a_pos, a_neg, hidden=48, epochs=400).fit(NODE, y), "node"


def _delivery_precision(rows_final: list[float]) -> dict:
    fin = np.asarray(rows_final, np.float64)
    return {"final_dist_median": round(float(np.median(fin)), 4), "final_dist_mean": round(float(fin.mean()), 4),
            "within_zone_frac": round(float((fin < _ZONE_HALF).mean()), 4)}


def part_d_eval(e, lib: OptionLibrary, oc: dict, contract, mon, *, seeds, n_eval, a_pos, a_neg, log=_log) -> dict:
    gate, mode = _fit_best_gate(oc, lib, a_pos, a_neg)
    log(f"[v3D] fresh eval of best gate '{oc['best_gate']}' vs V1 on {len(seeds)} seeds ...")
    roll, mirror = _roll_env(), _mirror_env()
    v1_metric_rows, gate_metric_rows = [], []
    v1_fin, gate_fin, v1_fc, gate_fc = [], [], [], []
    chosen_hist: Counter = Counter()
    per_seed = []
    for i, s in enumerate(seeds):
        s = int(s)
        smp = _handoff_sample(e, np.asarray(lib.thetas[0]), s, contract, mon, roll, mirror)   # the V1 roll on this seed
        if smp is None:
            continue
        x = smp.node_matrix[None] if mode == "node" else smp.z_flat[None]
        k = int(gate.predict(np.asarray(x, np.float32))[0])
        gks = _handoff_sample(e, np.asarray(lib.thetas[k]), s, contract, mon, roll, mirror)   # the gate-chosen roll
        if gks is None:
            continue
        chosen_hist[k] += 1
        v1_metric_rows.append(_metrics(e, np.asarray(lib.thetas[0]), (s,), n_eval, contract))
        gate_metric_rows.append(_metrics(e, np.asarray(lib.thetas[k]), (s,), n_eval, contract))
        v1_fin.append(smp.labels["final_dist"])
        gate_fin.append(gks.labels["final_dist"])
        v1_fc.append(smp.failure_class)
        gate_fc.append(gks.failure_class)
        per_seed.append({"seed": s, "option": k, "v1_delivered": smp.labels["delivered"],
                         "gate_delivered": gks.labels["delivered"]})
        if (i + 1) % 10 == 0:
            log(f"  [v3D] {i + 1}/{len(seeds)} eval'd")
    v1m, gm = _mean_metrics(v1_metric_rows), _mean_metrics(gate_metric_rows)
    v1_nd = round(1 - float(np.mean([r == "delivered" for r in v1_fc])), 4)
    gate_nd = round(1 - float(np.mean([r == "delivered" for r in gate_fc])), 4)
    out = {"n_seeds": len(v1_metric_rows), "best_gate": oc["best_gate"], "option_usage": dict(chosen_hist),
           "v1_metrics": v1m, "gate_metrics": gm,
           "v1_delivery_precision": _delivery_precision(v1_fin), "gate_delivery_precision": _delivery_precision(gate_fin),
           "v1_nondeliver_frac": v1_nd, "gate_nondeliver_frac": gate_nd,
           "tail_reduction": round(v1_nd - gate_nd, 4),
           "v1_failure_dist": {c: v1_fc.count(c) for c in FAILURE_CLASSES if v1_fc.count(c)},
           "gate_failure_dist": {c: gate_fc.count(c) for c in FAILURE_CLASSES if gate_fc.count(c)},
           "per_seed": per_seed}
    _render_gifs(e, lib, per_seed, log=log)
    _plots(oc, log=log)
    (_OUT / "eval.json").write_text(json.dumps(out, indent=2))
    return out


def _render_gifs(e, lib: OptionLibrary, per_seed: list[dict], *, log=_log) -> None:
    from hymeko_rl.eval.evaluate import render_episode_gif

    def render(theta, seed, path):
        env = _roll_env()
        pg = make_gated(env, np.asarray(theta), e)
        pg.reset()
        try:
            render_episode_gif(env, lambda ev, _o: np.asarray(pg.action(ev), np.float32), path, seed=seed, width=640)
            log(f"[gif] {path} (seed {seed})")
        except Exception as ex:                                            # gifs never gate the verdict
            log(f"[gif] {path} failed: {ex}")

    succ = next((p for p in per_seed if p["v1_delivered"]), None)
    fail = next((p for p in per_seed if not p["v1_delivered"]), None)
    rescue = next((p for p in per_seed if not p["v1_delivered"] and p["gate_delivered"]), None)
    if succ:
        render(lib.thetas[0], succ["seed"], str(_OUT / "v1_success.gif"))
    if fail:
        render(lib.thetas[0], fail["seed"], str(_OUT / "v1_failure.gif"))
    if rescue:
        render(lib.thetas[rescue["option"]], rescue["seed"], str(_OUT / "v3_rescue.gif"))
        log(f"[gif] rescue found: seed {rescue['seed']} option {rescue['option']}")
    else:
        log("[gif] no V1-fail->gate-deliver rescue on eval seeds")


def _plots(oc: dict, *, log=_log) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        log(f"[plot] matplotlib unavailable: {ex}")
        return
    Z = np.asarray(oc["_train"]["Z"], np.float64)
    y = np.asarray(oc["_train"]["oracle_label"], np.int64)
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc / (Z.std(0) + 1e-6), full_matrices=False)      # PCA via SVD (no sklearn)
    proj = (Zc / (Z.std(0) + 1e-6)) @ vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = ax[0].scatter(proj[:, 0], proj[:, 1], c=y, cmap="tab10", s=16)
    ax[0].set_title("handoff-state PCA (colour = oracle option)")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    fig.colorbar(sc, ax=ax[0], label="option")
    names = list(oc["gates"].keys())
    ax[1].bar(names, [oc["gates"][n]["oracle_agreement"] for n in names], color="steelblue")
    ax[1].axhline(1.0 / oc["k"], ls="--", c="grey", label="chance")
    ax[1].set_title(f"gate agreement with oracle (V1 vs oracle gap {oc['oracle_gap']:+.3f})")
    ax[1].set_ylabel("agreement")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(_OUT / "oracle_vs_gate.png", dpi=110)
    plt.close(fig)
    log(f"[plot] wrote {_OUT}/oracle_vs_gate.png")


# --------------------------------------------------------------------------- verdict + run

def _accept_v3(v1: dict, gate: dict, v1_nd: float, gate_nd: float, tol: float = 0.02) -> dict:
    improved_delivery = gate["ft_dom"] > v1["ft_dom"] + tol
    mon_pass_ok = gate["monitor_pass"] >= v1["monitor_pass"] - 1e-9
    mon_score_ok = gate["monitor_score"] >= v1["monitor_score"] - tol
    tail_reduced = gate_nd < v1_nd - 0.01
    no_regress = (gate["sustained_push_per_ep"] >= v1["sustained_push_per_ep"] - tol
                  and gate["ft_progress_in_contact"] >= v1["ft_progress_in_contact"] - 1e-3
                  and gate["body_driven_exploit"] <= 0.02
                  and gate["arm_body_rate"] <= v1["arm_body_rate"] + tol)
    beats = improved_delivery and mon_pass_ok and mon_score_ok and tail_reduced and no_regress
    return {"improved_delivery": bool(improved_delivery), "mon_pass_ok": bool(mon_pass_ok),
            "mon_score_ok": bool(mon_score_ok), "tail_reduced": bool(tail_reduced), "no_regress": bool(no_regress),
            "beats_v1": bool(beats)}


def _verdict(oracle_gap: float, accept: dict, gap_thresh: float = 0.03) -> str:
    if accept["beats_v1"]:
        return "V3_ACCEPTED"
    if oracle_gap < gap_thresh:
        return "NEGATIVE_WITH_MECHANISM"                                   # oracle ~ V1: mixture is not the lever
    if accept["tail_reduced"] or accept["improved_delivery"]:
        return "PROMISING_ORACLE_ONLY"                                     # headroom exists, gate partly realises it
    return "PROMISING_BUT_GATE_WEAK"                                       # headroom exists, gate fails to realise it


@dataclass
class V3Config:
    audit_seeds: range = field(default_factory=lambda: range(60000, 60120))
    oracle_seeds: range = field(default_factory=lambda: range(62000, 62100))
    fresh_seeds: range = field(default_factory=lambda: range(64000, 64040))
    n_label: int = 4
    n_eval: int = 16
    k: int = 5
    lib_cfg: ManifoldSearchConfig = field(default_factory=lambda: ManifoldSearchConfig(
        pop=14, elite=4, iters=5, sigma0=0.35, search_seeds=(41000, 43000), search_n_eval=6))


def run(cfg: V3Config, *, stages=("audit", "library", "oracle_gate", "eval")) -> dict:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    e = _load_e()
    contract = MonitorContract.from_env(_roll_env())
    mon = TaskMonitor.from_env(_roll_env())
    v1t = _v1_theta7()
    a_pos, a_neg = _incidence()
    v1_ref = _metrics(e, v1t, cfg.lib_cfg.search_seeds, cfg.lib_cfg.search_n_eval, contract)
    schema_hashes = {"semantic": SemanticSchema().field_order_hash, "handoff": handoff_schema_hash()}
    out: dict = {"experiment": "coin_toss_v3_handoff_conditioned_option_mixture",
                 "provenance": {"base_mlp_md5": file_md5(_E_CKPT), "v1_option_md5": file_md5(_V1)},
                 "schema_hashes": schema_hashes, "v1_stays_deployable": True}

    if "audit" in stages:
        out["part_a_audit"] = part_a_audit(e, v1t, contract, mon, seeds=list(cfg.audit_seeds),
                                           a_pos=a_pos, a_neg=a_neg)
    if "library" in stages:
        lib, out["part_b_library"] = part_b_library(e, v1t, contract, cfg.lib_cfg, k=cfg.k)
    else:
        info = json.loads((_OUT / "library.json").read_text())
        lib = OptionLibrary(names=info["library"]["names"], thetas=info["library"]["thetas"],
                            metrics=info["library"]["metrics"])
    if "oracle_gate" in stages:
        oc = part_c_oracle_gate(e, lib, contract, mon, seeds=list(cfg.oracle_seeds), n_label=cfg.n_label,
                                a_pos=a_pos, a_neg=a_neg, v1_ref=v1_ref)
    else:
        oc = json.loads((_OUT / "oracle_gate.json").read_text())
    out["part_c_oracle_gate"] = {kk: vv for kk, vv in oc.items() if kk != "_train"}
    if "eval" in stages:
        ev = part_d_eval(e, lib, oc, contract, mon, seeds=list(cfg.fresh_seeds), n_eval=cfg.n_eval,
                         a_pos=a_pos, a_neg=a_neg)
        out["part_d_eval"] = ev
        accept = _accept_v3(ev["v1_metrics"], ev["gate_metrics"], ev["v1_nondeliver_frac"], ev["gate_nondeliver_frac"])
        verdict = _verdict(oc["oracle_gap"], accept)
        out["acceptance"] = accept
        out["verdict"] = verdict
        if verdict == "V3_ACCEPTED":
            (_OUT / "v3_gate.json").write_text(json.dumps({"best_gate": oc["best_gate"],
                                                           "library": lib.as_dict()}, indent=2))
            out["deployable_v3"] = str(_OUT / "v3_gate.json")

    out["wall_s"] = round(time.perf_counter() - t0, 1)
    (_OUT / "results.json").write_text(json.dumps(out, indent=2))
    _log(f"[v3] {out.get('verdict', '(stages incomplete)')}; wrote {_OUT}/results.json ({out['wall_s']}s)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="coin_toss_v3 — handoff-conditioned option mixture")
    ap.add_argument("--stage", choices=("smoke", "audit", "library", "oracle_gate", "eval", "full"), default="full")
    args = ap.parse_args(argv)
    if args.stage == "smoke":
        cfg = V3Config(audit_seeds=range(60000, 60006), oracle_seeds=range(62000, 62008),
                       fresh_seeds=range(64000, 64004), n_label=2, n_eval=2, k=5,
                       lib_cfg=ManifoldSearchConfig(pop=6, elite=2, iters=2, sigma0=0.35,
                                                    search_seeds=(41000,), search_n_eval=2))
        run(cfg)
    elif args.stage == "full":
        run(V3Config())
    else:
        run(V3Config(), stages=(args.stage,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
