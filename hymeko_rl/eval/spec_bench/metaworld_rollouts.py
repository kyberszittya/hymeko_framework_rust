"""Real MetaWorld coffee-push rollouts for spec_bench — the LLM→augmenter+arbiter pipeline on real data.

Rolling needs ``metaworld`` (kato15, not the Mac): ``roll_coffee_push`` captures per-step info-signal traces
(``near_object``, ``grasp_success``, ``in_place``, ``obj_to_target``) + the episode's native success, across noise
levels so both classes appear. ``save_rollouts``/``load_rollouts`` are a tiny JSON (no metaworld needed) so the
traces rsync to the Mac and the LLM+gate+pgraph pipeline runs there. ``run_bench`` runs raw-vs-gate for a set of
models on ANY rollouts (real or synthetic), against a formal ceiling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.eval.spec_bench.spec_bench import ChatModel, Rollout, formula_f1, propose_and_gate, score_raw

_COFFEE_SIGNALS = ("near_object", "grasp_success", "in_place", "obj_to_target")


def roll_coffee_push(n: int, *, seed: int = 0, noise_levels: "tuple[float, ...]" = (0.0, 0.4, 0.8, 1.2),
                     max_steps: int = 180) -> list[Rollout]:
    """Roll ``n`` coffee-push episodes (scripted policy + varied action noise) → labelled per-step traces.

    # Preconditions ``metaworld`` importable. # Postconditions ``len == n``; each ``Rollout.trace`` has the four
      coffee signals per step; ``success`` is the episode's native MetaWorld success flag."""
    import warnings

    import metaworld.policies as mp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]
    out: list[Rollout] = []
    for i in range(n):
        noise = noise_levels[i % len(noise_levels)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env: Any = ENVS["coffee-push-v3-goal-observable"](render_mode=None)
            policy = mp.SawyerCoffeePushV3Policy()
            obs, _ = env.reset(seed=seed + i)
        rng = np.random.default_rng(seed + i)
        trace: list[dict[str, float]] = []
        success = 0
        for _ in range(max_steps):
            act = np.clip(np.asarray(policy.get_action(obs), np.float32)
                          + rng.normal(0, noise, 4).astype(np.float32), -1.0, 1.0)
            obs, _r, term, trunc, info = env.step(act)
            trace.append({
                "near_object": float(info.get("near_object", 0.0)),
                "grasp_success": float(info.get("grasp_success", 0.0)),
                "in_place": float(info.get("in_place_reward", 0.0)),
                "obj_to_target": float(info.get("obj_to_target", 0.0)),
            })
            success = max(success, int(info.get("success", 0)))
            if term or trunc:
                break
        env.close()
        out.append(Rollout(trace=trace, success=bool(success)))
    return out


def save_rollouts(rollouts: list[Rollout], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"trace": r.trace, "success": r.success} for r in rollouts]))
    return p


def load_rollouts(path: str | Path) -> list[Rollout]:
    return [Rollout(trace=d["trace"], success=bool(d["success"])) for d in json.loads(Path(path).read_text())]


def run_bench(models: "dict[str, ChatModel]", verif: list[Rollout], test: list[Rollout], *, prompt: str,
              system: str, formal: str, k: int = 4, retries: int = 2) -> dict[str, object]:
    """Raw vs gate F1 (vs native success) for each named model on given rollouts, + the formal-expert ceiling."""
    rows: list[dict[str, object]] = []
    for name, model in models.items():
        try:
            raw, n_valid, n_att = score_raw(model, prompt, k=k, system=system)
            gate = propose_and_gate(model, prompt, verif, k=k, retries=retries, system=system)
            rows.append({
                "model": name, "raw_f1": round(formula_f1(raw, test) if raw else 0.0, 4),
                "gate_f1": round(formula_f1(gate.formula, test) if gate.formula else 0.0, 4),
                "raw_parse_rate": round(n_valid / max(1, n_att), 3),
                "gate_formula": gate.formula,
            })
        except Exception as e:
            rows.append({"model": name, "error": f"{type(e).__name__}: {e}"})
    return {"formal": formal, "formal_f1": round(formula_f1(formal, test), 4),
            "n_verif": len(verif), "n_test": len(test),
            "test_balance": sum(r.success for r in test), "models": rows}
