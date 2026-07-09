"""HyMeKo-declared MetaWorld reward — reconstruct MetaWorld's dense reward as ``Σ weight·term``, and ablate it.

Makes HyMeKo the source of truth for the MetaWorld *reward* (not only the physics). ``data/robotics/
metaworld_reward.hymeko`` declares the reward as a weighted bundle of named terms over MetaWorld's exposed reward
components (``in_place_reward``, ``grasp_reward``, ``near_object``, ``obj_to_target``). This module:

* parses that ``.hymeko`` with the existing :func:`~hymeko_rl.env.reward.read_reward_terms` (§6.1 — no re-parse);
* records the per-step components of a scripted rollout and reconstructs the HyMeKo reward ``Σ weight·term``;
* reports **fidelity** (R² vs MetaWorld's ``unscaled_reward``) for the declared (seed) weights and for a
  deterministic least-squares refit — the fit is "the HyMeKo reward that best matches MetaWorld";
* exposes :func:`ablate_reward` — recompute with a term dropped / downweighted, exactly the coin Stage-A
  intervention, now on a reward HyMeKo reverse-engineered rather than one it originally authored.

No training; read-only scripted rollouts. The declared reward is a *faithful decomposition* (R² ≈ 0.79–0.95), not
bit-exact (MetaWorld's reward is computed inside the env with an un-exposed reach-tolerance nonlinearity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# HyMeKo MetaWorld reward-term kind → (MetaWorld info key, sign) for the recorded per-step component.
_TERM_TO_COMPONENT: dict[str, tuple[str, float]] = {
    "mw_in_place": ("in_place_reward", 1.0),
    "mw_grasp": ("grasp_reward", 1.0),
    "mw_near": ("near_object", 1.0),
    "mw_dist": ("obj_to_target", -1.0),   # -weight·distance
}
# HyMeKo reward-term kind → the CIP rollout-frame variable it corresponds to (for the LiNGAM-SH mechanism bridge).
_TERM_TO_CIP_VARIABLE: dict[str, str] = {
    "mw_near": "near_fraction",
    "mw_grasp": "grasp_fraction",
    "mw_in_place": "progress_score",       # in_place ≈ the monitor's task-progress score
    "mw_dist": "obj_to_target_delta",
}
_DEFAULT_SPEC = "data/robotics/metaworld_reward.hymeko"


def hymeko_reward_terms(spec_path: str = _DEFAULT_SPEC) -> "list[tuple[str, float]]":
    """The declared ``(term_kind, weight)`` pairs of the HyMeKo MetaWorld reward (via ``read_reward_terms``)."""
    from hymeko_rl.env.reward import read_reward_terms
    return list(read_reward_terms(spec_path))


def reward_mechanism_proposal(spec_path: str = _DEFAULT_SPEC, *, available: "Sequence[str] | None" = None,
                              output: str = "total_reward", name: str = "hymeko_reward_mechanism") -> Any:
    """Build the ``{reward terms} → {output}`` mechanism proposal from the HyMeKo reward declaration.

    Extracts the declared terms, maps each to its CIP frame variable (:data:`_TERM_TO_CIP_VARIABLE`), and (if
    ``available`` is given) keeps only the terms whose variable is present in the frame. The declared weights carry
    through to the proposal. # Preconditions ``output`` and at least one mapped term are usable.
    """
    from hymeko_rl.eval.causal import propose_reward_terms
    mapped = [(_TERM_TO_CIP_VARIABLE[k], float(w)) for k, w in hymeko_reward_terms(spec_path)
              if k in _TERM_TO_CIP_VARIABLE]
    if available is not None:
        allow = set(available)
        mapped = [(v, w) for v, w in mapped if v in allow]
    if not mapped:
        raise ValueError(f"no HyMeKo reward term maps to an available variable ({available})")
    tail = [v for v, _w in mapped]
    return propose_reward_terms(tail, output=output, weights={v: w for v, w in mapped}, name=name)


@dataclass(frozen=True)
class RewardFidelity:
    """How faithfully the HyMeKo ``Σ weight·term`` reward reconstructs MetaWorld's dense reward."""

    task: str
    terms: tuple[str, ...]
    declared_weights: tuple[float, ...]
    fitted_weights: tuple[float, ...]
    r2_declared: float
    r2_fitted: float
    relative_error_fitted: float
    n_steps: int

    def as_dict(self) -> dict[str, Any]:
        return {"task": self.task, "terms": list(self.terms),
                "declared_weights": [round(w, 4) for w in self.declared_weights],
                "fitted_weights": [round(w, 4) for w in self.fitted_weights],
                "r2_declared": round(self.r2_declared, 4), "r2_fitted": round(self.r2_fitted, 4),
                "relative_error_fitted": round(self.relative_error_fitted, 4), "n_steps": self.n_steps}


def hymeko_reward(components: np.ndarray, weights: "np.ndarray | list[float]") -> np.ndarray:
    """Per-step HyMeKo reward ``Σ weight·term`` from a component matrix ``(n_steps, n_terms)``."""
    return np.asarray(components, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - y_hat) ** 2) / denom) if denom > 0 else 0.0


def fit_reward_weights(components: np.ndarray, target: np.ndarray) -> "tuple[np.ndarray, float]":
    """Deterministic least-squares weights (no bias — the reward_spec is a pure ``Σ weight·term``) + R²."""
    w, *_ = np.linalg.lstsq(components, target, rcond=None)
    return w, _r2(target, components @ w)


def ablate_reward(components: np.ndarray, terms: "list[str]", weights: "list[float]",
                  *, drop: "list[str] | None" = None, scale: "Mapping[str, float] | None" = None) -> np.ndarray:
    """Recompute the HyMeKo reward with some terms dropped (``drop``) or downweighted (``scale``) — the ablation."""
    drop_set = set(drop or ())
    w = np.array([0.0 if t in drop_set else weights[i] * float((scale or {}).get(t, 1.0))
                  for i, t in enumerate(terms)], dtype=np.float64)
    return hymeko_reward(components, w)


def _record_reward_components(task: str, policy_name: str, kinds: "list[str]", n_ep: int, noise: float,
                              max_steps: int) -> "tuple[np.ndarray, np.ndarray]":
    """Roll ``n_ep`` scripted episodes → (component matrix aligned to ``kinds``, MetaWorld ``unscaled_reward``)."""
    import warnings

    import metaworld.policies as mp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
        env: Any = ENVS[f"{task}-v3-goal-observable"](render_mode=None)
        policy = getattr(mp, policy_name)()
        rows: list[list[float]] = []
        target: list[float] = []
        for ep in range(n_ep):
            obs, _ = env.reset(seed=ep)
            rng = np.random.default_rng(ep)
            for _ in range(max_steps):
                act = np.clip(np.asarray(policy.get_action(obs), np.float32)
                              + rng.normal(0, noise, 4).astype(np.float32), -1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(act)
                rows.append([float(sign) * float(info.get(key, 0.0))
                             for key, sign in (_TERM_TO_COMPONENT[k] for k in kinds)])
                target.append(float(info.get("unscaled_reward", reward)))
                if terminated or truncated:
                    break
    return np.asarray(rows, dtype=np.float64), np.asarray(target, dtype=np.float64)


def evaluate_reward_fidelity(task: str, policy_name: str, *, spec_path: str = _DEFAULT_SPEC, n_ep: int = 8,
                             noise: float = 0.4, max_steps: int = 150) -> RewardFidelity:
    """Record components, reconstruct the HyMeKo reward, and score fidelity vs MetaWorld's reward (declared + fit)."""
    from hymeko_rl.env.reward import read_reward_terms

    terms = read_reward_terms(spec_path)
    kinds = [k for k, _w in terms]
    declared = [w for _k, w in terms]
    unknown = [k for k in kinds if k not in _TERM_TO_COMPONENT]
    if unknown:
        raise ValueError(f"{spec_path}: term kind(s) {unknown} have no MetaWorld component mapping")
    components, target = _record_reward_components(task, policy_name, kinds, n_ep, noise, max_steps)
    fitted, r2_fitted = fit_reward_weights(components, target)
    y_declared = hymeko_reward(components, declared)
    resid = float(np.linalg.norm(target - components @ fitted))
    rel = resid / float(np.linalg.norm(target)) if np.linalg.norm(target) > 0 else 0.0
    return RewardFidelity(task=task, terms=tuple(kinds), declared_weights=tuple(declared),
                          fitted_weights=tuple(round(float(w), 6) for w in fitted),
                          r2_declared=_r2(target, y_declared), r2_fitted=r2_fitted,
                          relative_error_fitted=rel, n_steps=len(target))


# task → scripted policy (reuse the generic-runner registry so the two stay in sync).
def _policy_for(task: str) -> str:
    from hymeko_rl.eval.cip.metaworld_generic_cip import GENERIC_TASKS
    return GENERIC_TASKS[task]


def run_reward_fidelity_sweep(tasks: "list[str]", out_dir: "Any", *, n_ep: int = 8) -> "dict[str, Any]":
    """Evaluate HyMeKo-reward fidelity across tasks; write a summary. Returns the fidelity-per-task dict."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_task = {}
    for task in tasks:
        fid = evaluate_reward_fidelity(task, _policy_for(task), n_ep=n_ep)
        per_task[task] = fid.as_dict()
        print(f"[mw-reward] {task:12s} R2_fitted={fid.r2_fitted:.3f} (declared={fid.r2_declared:.3f}) "
              f"fitted_weights={dict(zip(fid.terms, [round(w, 2) for w in fid.fitted_weights]))}", flush=True)
    summary = {"kind": "HyMeKo MetaWorld reward fidelity (Σ weight·term vs unscaled_reward)",
               "spec": _DEFAULT_SPEC, "tasks": tasks, "per_task": per_task}
    (out / "reward_fidelity.json").write_text(json.dumps(summary, indent=2))
    print(f"[mw-reward] wrote reward_fidelity.json -> {out}", flush=True)
    return summary


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from hymeko_rl.eval.cip.metaworld_generic_cip import GENERIC_TASKS
    from hymeko_rl.eval.evaluate import experiment_dir
    parser = argparse.ArgumentParser(description="HyMeKo MetaWorld reward fidelity (declare + reconstruct)")
    parser.add_argument("--task", choices=[*GENERIC_TASKS, "sweep"], default="sweep")
    parser.add_argument("--n-ep", type=int, default=8)
    parser.add_argument("--out", type=str, default="reports/figures")
    args = parser.parse_args(argv)
    out_dir = experiment_dir(args.out, "cip_metaworld_reward")
    tasks = list(GENERIC_TASKS) if args.task == "sweep" else [args.task]
    run_reward_fidelity_sweep(tasks, out_dir, n_ep=args.n_ep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
