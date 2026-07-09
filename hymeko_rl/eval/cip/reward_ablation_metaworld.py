"""MetaWorld reward-ablation Stage A — the coin Stage-A intervention transferred to MetaWorld / pick-place.

Ablate a declared HyMeKo reward term (`mw_grasp`), recompute the reward **offline** on fixed scripted rollouts
(the policy is unchanged), re-run CIP / LiNGAM-SH, and test whether the `grasp_fraction → total_reward` loading
collapses. Because the reward is now a HyMeKo `Σ weight·term` (`data/robotics/metaworld_reward.hymeko`), removing a
term is a deterministic reweighting — no env re-stepping, no training.

Task: **pick-place** (grasp actually fires, so `mw_grasp`↔`grasp_fraction` is in-frame and meaningful; coffee-push
never grasps). Doctrine: this is a **reward-computation-level** intervention — the policy is fixed, so no
policy-learning claim. All mechanism hypergraphs are HyMeKo-declared and engine cross-view-verified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metaworld_reward import _DEFAULT_SPEC, _TERM_TO_COMPONENT


@dataclass(frozen=True)
class AblatedRewardSpec:
    """A deterministic ablation of the HyMeKo reward SoT (drop → weight 0; downweight → scale). Original untouched."""

    source: str
    original: tuple[tuple[str, float], ...]
    ablated: tuple[tuple[str, float], ...]
    dropped: tuple[str, ...]
    downweighted: Mapping[str, float]

    def term_kinds(self) -> list[str]:
        return [k for k, _w in self.original]

    def original_weights(self) -> list[float]:
        return [w for _k, w in self.original]

    def ablated_weights(self) -> list[float]:
        return [w for _k, w in self.ablated]

    def active_terms(self) -> "list[tuple[str, float]]":
        """The ablated terms with a non-zero weight (the dropped term is absent)."""
        return [(k, w) for k, w in self.ablated if w != 0.0]


def ablate_reward_spec(reward_sot_path: str = _DEFAULT_SPEC, *, drop: "Sequence[str]" = (),
                       downweight: "Mapping[str, float] | None" = None) -> AblatedRewardSpec:
    """Load the HyMeKo reward SoT and produce an ablated term list (does **not** mutate the file)."""
    from .metaworld_reward import hymeko_reward_terms
    original = tuple(hymeko_reward_terms(reward_sot_path))
    kinds = {k for k, _w in original}
    drop_set = tuple(dict.fromkeys(drop))
    dw = {k: float(v) for k, v in (downweight or {}).items()}
    unknown = sorted(({*drop_set, *dw}) - kinds)
    if unknown:
        raise ValueError(f"unknown reward term(s) to ablate: {unknown}; declared: {sorted(kinds)}")
    ds = set(drop_set)
    ablated = tuple((k, 0.0 if k in ds else round(w * dw.get(k, 1.0), 6)) for k, w in original)
    return AblatedRewardSpec(str(reward_sot_path), original, ablated, drop_set, dw)


def _record_ablation_rollouts(task: str, policy_name: str, term_kinds: "list[str]", n: int, seed: int,
                              noise_max: float, max_steps: int = 180) -> "dict[str, Any]":
    """Roll ``n`` scripted episodes → per-step components + reward (for the weight fit) and per-rollout
    component totals + CIP frame variables (near/grasp/dist) + task-progress score."""
    import warnings

    import metaworld.policies as mp
    comp_keys = [_TERM_TO_COMPONENT[k] for k in term_kinds]
    x_step: list[list[float]] = []
    y_step: list[float] = []
    totals: list[np.ndarray] = []
    near: list[float] = []
    grasp: list[float] = []
    dist: list[float] = []
    task_score: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
        noises = np.random.default_rng(seed).uniform(0.0, noise_max, n)
        for i in range(n):
            env: Any = ENVS[f"{task}-v3-goal-observable"](render_mode=None)
            policy = getattr(mp, policy_name)()
            obs, _ = env.reset(seed=seed + i)
            rng = np.random.default_rng(seed + i)
            csum = np.zeros(len(term_kinds))
            n_near = n_grasp = steps = 0
            d0: "float | None" = None
            d_final = ip_sum = 0.0
            for _ in range(max_steps):
                act = np.clip(np.asarray(policy.get_action(obs), np.float32)
                              + rng.normal(0, noises[i], 4).astype(np.float32), -1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(act)
                cvec = [float(sign) * float(info.get(key, 0.0)) for key, sign in comp_keys]
                x_step.append(cvec)
                y_step.append(float(info.get("unscaled_reward", reward)))
                csum += np.asarray(cvec)
                steps += 1
                n_near += int(info.get("near_object", 0))
                n_grasp += int(info.get("grasp_success", 0))
                d_final = float(info.get("obj_to_target", 0.0))
                ip_sum += float(info.get("in_place_reward", 0.0))
                if d0 is None:
                    d0 = d_final
                if terminated or truncated:
                    break
            totals.append(csum)
            near.append(n_near / max(1, steps))
            grasp.append(n_grasp / max(1, steps))
            dist.append(float((d0 or 0.0) - d_final))
            task_score.append(ip_sum / max(1, steps))
    progress = np.asarray(task_score)
    return {"x_step": np.asarray(x_step), "y_step": np.asarray(y_step), "totals": np.asarray(totals),
            # progress_score (mean in_place) is IN the frame so the dominant reward term (mw_in_place) maps —
            # otherwise near_fraction proxies it and mediates the smaller grasp term away.
            "cip": {"near_fraction": np.asarray(near), "grasp_fraction": np.asarray(grasp),
                    "obj_to_target_delta": np.asarray(dist), "progress_score": progress},
            "task_score": progress}


def _relative_change(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 change ``‖a − b‖ / ‖a‖`` — how much ablating a term moved the reward (0 = unchanged)."""
    na = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / na) if na > 0 else 0.0


def _reward_decomposition(cip: "dict[str, np.ndarray]", total_reward: np.ndarray, tail: "list[str]",
                          ) -> "tuple[list[tuple[str, str, float]], dict[str, float], float]":
    """Per-tail loadings as the reward's DIRECT decomposition: regress ``total_reward`` on the tail (with intercept,
    controlling for collinearity), so a small grasp term is not mediated away. Returns (edges, loadings, R²)."""
    design = np.column_stack([cip[t] for t in tail] + [np.ones(len(total_reward))])
    coef, *_ = np.linalg.lstsq(design, total_reward, rcond=None)
    pred = design @ coef
    denom = float(np.sum((total_reward - total_reward.mean()) ** 2))
    r2 = float(1.0 - np.sum((total_reward - pred) ** 2) / denom) if denom > 0 else 0.0
    edges = [(t, "total_reward", float(coef[i])) for i, t in enumerate(tail)]   # tail → reward, coefficient = loading
    return edges, {t: round(float(coef[i]), 4) for i, t in enumerate(tail)}, r2


def _condition(cip: "dict[str, np.ndarray]", total_reward: np.ndarray, task_score: np.ndarray,
               reward_prop: Any, tag: str, out_dir: Path) -> "dict[str, Any]":
    """One reward: its per-tail decomposition loadings, the (cross-view-verified) mechanism graph, disagreement."""
    from hymeko_rl.eval.causal import (
        DirectLiNGAM,
        cross_view_verify,
        fit_loadings_least_squares,
        proposals_to_causal_hypergraph,
    )
    from hymeko_rl.eval.task_monitor.consistency import RewardConsistencyMonitor

    tail = [t for t in reward_prop.tail if t in cip and float(np.std(cip[t])) > 1e-9]
    edges, loadings, reward_r2 = _reward_decomposition(cip, total_reward, tail)
    names = [*tail, "total_reward"]
    fac = fit_loadings_least_squares(names, edges, [reward_prop])        # loadings ≡ the decomposition coefficients
    cg = proposals_to_causal_hypergraph(names, [reward_prop], name=f"PickPlaceReward{tag.title()}")
    xview = cross_view_verify(cg, out_dir / f"reward_mechanism_{tag}.hymeko")
    order = DirectLiNGAM().fit(np.column_stack([cip[t] for t in tail] + [total_reward]), names).ordered_names()
    rows = [{"policy": f"{tag}_{i}", "total_reward": float(total_reward[i]), "monitor_score": float(task_score[i])}
            for i in range(len(total_reward))]
    disagreement = round(1.0 - float(RewardConsistencyMonitor().check_reward_alignment(rows).score), 6)
    return {"tail": list(tail), "loadings": loadings, "reward_reconstruction_r2": round(reward_r2, 4),
            "weighted_explained_energy": fac.metrics["explained_energy"], "causal_order": order,
            "reward_monitor_disagreement": disagreement,
            "cross_view_agree": bool(xview.agree), "acyclic": bool(cg.check_acyclicity().acyclic)}


def run_reward_ablation_stage_a(task: str = "pick-place", spec_path: str = _DEFAULT_SPEC, *,
                                drop: "Sequence[str]" = ("mw_grasp",), n: int = 60, seed: int = 0,
                                out_dir: "Path | None" = None, noise_max: float = 0.7) -> "dict[str, Any]":
    """Stage A: ablate ``drop`` from the HyMeKo reward, recompute offline, re-fit CIP for original vs ablated."""
    from hymeko_rl.eval.evaluate import experiment_dir
    from .metaworld_generic_cip import GENERIC_TASKS
    from .metaworld_reward import fit_reward_weights, reward_mechanism_proposal

    out = out_dir or experiment_dir("reports/figures", "cip_reward_ablation_stageA")
    out.mkdir(parents=True, exist_ok=True)
    spec = ablate_reward_spec(spec_path, drop=drop)
    kinds = spec.term_kinds()
    rec = _record_ablation_rollouts(task, GENERIC_TASKS[task], kinds, n, seed + 9_000, noise_max)
    fitted, r2 = fit_reward_weights(rec["x_step"], rec["y_step"])       # calibrate the HyMeKo reward to the task

    drop_set = set(drop)
    ablated_w = np.array([0.0 if k in drop_set else fitted[i] for i, k in enumerate(kinds)])
    reward_orig = rec["totals"] @ fitted
    reward_ablt = rec["totals"] @ ablated_w
    reward_change = _relative_change(reward_orig, reward_ablt)          # how much the reward moved

    # positive control: ablate the DOMINANT term (max |fitted weight|) — the reward should move a lot vs grasp
    dominant = kinds[int(np.argmax(np.abs(fitted)))]
    dom_w = np.array([0.0 if k == dominant else fitted[i] for i, k in enumerate(kinds)])
    reward_change_dominant = _relative_change(reward_orig, rec["totals"] @ dom_w)

    # same declared reward mechanism (tail incl grasp) on both rewards → the loading comparison
    prop = reward_mechanism_proposal(spec_path, available=[*rec["cip"], "total_reward"])
    cond_orig = _condition(rec["cip"], reward_orig, rec["task_score"], prop, "original", out)
    cond_ablt = _condition(rec["cip"], reward_ablt, rec["task_score"], prop, "grasp_off", out)
    # the ablated-spec mechanism re-parents onto the remaining declared terms (grasp absent from the tail)
    prop_reparent = _drop_tail(
        reward_mechanism_proposal(spec_path, available=[*rec["cip"], "total_reward"], name="hymeko_reward_reparented"),
        "grasp_fraction")
    cond_reparent = _condition(rec["cip"], reward_ablt, rec["task_score"], prop_reparent, "reparented", out)

    g_orig = float(cond_orig["loadings"].get("grasp_fraction", 0.0))
    g_ablt = float(cond_ablt["loadings"].get("grasp_fraction", 0.0))
    verdict, reason = _verdict(g_orig, g_ablt, reward_change, dominant, dict(zip(kinds, fitted)),
                               cond_orig["cross_view_agree"] and cond_ablt["cross_view_agree"])
    summary = {
        "task": task, "spec": spec_path, "dropped": list(drop), "n": n, "seed": seed,
        "reward_fidelity_r2": round(r2, 4), "fitted_weights": {k: round(float(w), 4) for k, w in zip(kinds, fitted)},
        "reward_change_fraction": round(reward_change, 4),
        "reward_change_fraction_dominant_term": {dominant: round(reward_change_dominant, 4)},
        "grasp_loading_original": round(g_orig, 4), "grasp_loading_ablated": round(g_ablt, 4),
        "original": cond_orig, "grasp_off": cond_ablt, "reparented": cond_reparent,
        "verdict": verdict, "verdict_reason": reason,
        "_disclaimer": "Reward-computation-level only (policy fixed; NO policy-learning claim). Cross-view proves "
                       "representation, not causal truth.",
    }
    (out / "reward_ablation_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[reward-abl] {task} drop={list(drop)} | grasp loading {g_orig:+.3f} -> {g_ablt:+.3f} | "
          f"disagree {cond_orig['reward_monitor_disagreement']:.3f} -> {cond_ablt['reward_monitor_disagreement']:.3f} "
          f"| xview {cond_orig['cross_view_agree']}/{cond_ablt['cross_view_agree']} | verdict={summary['verdict']}",
          flush=True)
    return summary


def _variant_metrics(cond: "dict[str, Any]") -> "dict[str, Any]":
    """The per-variant metrics extracted from a `_condition` result (+ the dominant reward driver by |loading|)."""
    loadings = cond["loadings"]
    dominant = max(loadings, key=lambda k: abs(loadings[k])) if loadings else None
    return {"loadings": loadings, "dominant_driver": dominant,
            "reward_reconstruction_r2": cond["reward_reconstruction_r2"],
            "weighted_explained_energy": cond["weighted_explained_energy"],
            "reward_monitor_disagreement": cond["reward_monitor_disagreement"],
            "cross_view_agree": cond["cross_view_agree"]}


def run_reward_ablation_comparison(task: str = "pick-place", spec_path: str = _DEFAULT_SPEC, *,
                                   drops: "Sequence[Sequence[str]]" = (("mw_grasp",), ("mw_in_place",), ("mw_dist",)),
                                   n: int = 60, seed: int = 0, out_dir: "Path | None" = None,
                                   noise_max: float = 0.7) -> "dict[str, Any]":
    """Ablate several reward terms (offline) and compare original vs each — the positive/negative-control panel."""
    from hymeko_rl.eval.evaluate import experiment_dir
    from .metaworld_generic_cip import GENERIC_TASKS
    from .metaworld_reward import _TERM_TO_CIP_VARIABLE, fit_reward_weights, hymeko_reward_terms, reward_mechanism_proposal

    out = out_dir or experiment_dir("reports/figures", "cip_reward_ablation_poscontrol")
    out.mkdir(parents=True, exist_ok=True)
    kinds = [k for k, _w in hymeko_reward_terms(spec_path)]
    rec = _record_ablation_rollouts(task, GENERIC_TASKS[task], kinds, n, seed + 9_000, noise_max)
    fitted, r2 = fit_reward_weights(rec["x_step"], rec["y_step"])
    reward_orig = rec["totals"] @ fitted
    prop = reward_mechanism_proposal(spec_path, available=[*rec["cip"], "total_reward"])

    variants: dict[str, Any] = {"original": {"drop": [], "reward_change": 0.0,
                                             **_variant_metrics(_condition(rec["cip"], reward_orig, rec["task_score"],
                                                                          prop, "original", out))}}
    for drop in drops:
        tag = "_".join(drop) + "_off"
        ablated_w = np.array([0.0 if k in set(drop) else fitted[i] for i, k in enumerate(kinds)])
        reward_ablt = rec["totals"] @ ablated_w
        cond = _condition(rec["cip"], reward_ablt, rec["task_score"], prop, tag, out)
        variants[tag] = {"drop": list(drop), "reward_change": round(_relative_change(reward_orig, reward_ablt), 4),
                         **_variant_metrics(cond)}

    verdict, reason = _poscontrol_verdict(variants, _TERM_TO_CIP_VARIABLE)
    summary = {"task": task, "spec": spec_path, "n": n, "seed": seed, "reward_fidelity_r2": round(r2, 4),
               "fitted_weights": {k: round(float(w), 4) for k, w in zip(kinds, fitted)}, "variants": variants,
               "verdict": verdict, "verdict_reason": reason,
               "_disclaimer": "Reward-computation-level only (policy fixed; NO policy-learning claim)."}
    (out / "reward_ablation_comparison.json").write_text(json.dumps(summary, indent=2, default=float))
    for name, v in variants.items():
        print(f"[reward-abl-cmp] {name:16s} reward_change={v['reward_change']:.3f} "
              f"dominant={v['dominant_driver']} xview={v['cross_view_agree']}", flush=True)
    print(f"[reward-abl-cmp] verdict={verdict} :: {reason}", flush=True)
    return summary


def _poscontrol_verdict(variants: "dict[str, Any]", term_to_var: "Mapping[str, str]") -> "tuple[str, str]":
    """SUPPORTED iff mw_in_place_off collapses the progress loading, moves the reward a lot, and beats mw_grasp."""
    orig = variants["original"]["loadings"]
    pos = variants.get("mw_in_place_off")
    if pos is None:
        return "NOT_SUPPORTED", "mw_in_place_off variant not present"
    var = term_to_var.get("mw_in_place", "progress_score")
    l_orig = abs(float(orig.get(var, 0.0)))
    l_pos = abs(float(pos["loadings"].get(var, 0.0)))
    collapsed = l_orig > 1e-6 and l_pos < 0.5 * l_orig
    stronger = pos["reward_change"] > variants.get("mw_grasp_off", {}).get("reward_change", 0.0)
    ok = collapsed and pos["reward_change"] > 0.3 and pos["cross_view_agree"] and stronger
    if ok:
        return "SUPPORTED_at_reward_computation_level", (
            f"dropping mw_in_place moves the reward {pos['reward_change']:.1%} (vs "
            f"{variants.get('mw_grasp_off', {}).get('reward_change', 0.0):.1%} for mw_grasp) and collapses the "
            f"{var} loading {l_orig:.1f} → {l_pos:.1f}")
    return "NOT_SUPPORTED", f"{var} loading {l_orig:.1f} → {l_pos:.1f}; reward_change {pos['reward_change']:.1%}"


def _verdict(g_orig: float, g_ablt: float, reward_change: float, dominant: str, fitted: "Mapping[str, float]",
             cross_view_ok: bool) -> "tuple[str, str]":
    """Decision rule: SUPPORTED iff the grasp loading collapses AND the reward moved materially AND cross-view ok."""
    collapsed = abs(g_orig) > 1e-6 and (abs(g_ablt) < 0.5 * abs(g_orig) or abs(g_ablt) < 0.05)
    if collapsed and reward_change > 0.10 and cross_view_ok:
        return "SUPPORTED_at_reward_computation_level", "grasp loading collapsed under mw_grasp ablation"
    return "NOT_SUPPORTED", (
        f"mw_grasp is a minor reward term (fitted {fitted.get('mw_grasp', 0.0):.2f} vs dominant {dominant} "
        f"{fitted.get(dominant, 0.0):.2f}); ablation moves the reward only {reward_change:.1%} and the grasp "
        f"loading does not collapse.")


def _drop_tail(proposal: Any, tail_var: str) -> Any:
    """A copy of ``proposal`` with ``tail_var`` removed from the tail (the ablated-spec re-parented mechanism)."""
    from hymeko_rl.eval.causal import MechanismProposal
    tail = tuple(t for t in proposal.tail if t != tail_var)
    ev = {**dict(proposal.evidence or {})}
    if isinstance(ev.get("loadings"), dict):
        ev["loadings"] = {k: v for k, v in ev["loadings"].items() if k != tail_var}
    return MechanismProposal(name=proposal.name, tail=tail, head=proposal.head, strength=proposal.strength,
                             sign=proposal.sign, confidence=proposal.confidence, source=proposal.source, evidence=ev)


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="MetaWorld reward-ablation Stage A (HyMeKo reward SoT)")
    parser.add_argument("--task", default="pick-place")
    parser.add_argument("--drop", nargs="+", default=["mw_grasp"])
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_reward_ablation_stage_a(args.task, drop=tuple(args.drop), n=int(args.n), seed=int(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
