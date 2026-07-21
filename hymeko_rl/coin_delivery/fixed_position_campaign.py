"""Fixed-position Coin Delivery replay orchestration — the deterministic-problem and exact-state entry points plus
the P0/P1/P4 causal controls, wired for the canonical command layer (``python -m hymeko_rl.campaign run --domain
coin ...``).

Deterministic problem replay: a ``seed`` regenerates the exact problem through ``reset(seed)``. Exact-state replay:
a validated :class:`~hymeko_rl.coin_delivery.fixed_position.CoinInitialState` JSON sets the physics EXACTLY. Both
paths run the same traced two-phase rollout, so a run yields the same problem hash, reachability report, per-rep
causal table (first/bilateral/targetward/zone/strict + completion time + failure taxonomy) and trajectory hashes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from hymeko_rl.coin_delivery.fixed_position import (CoinInitialState, apply_initial_state, env_fingerprint,
                                                    extract_initial_state)
from hymeko_rl.coin_delivery.fixed_position_replay import (analyze_reachability, assert_replayable, build_actors,
                                                           composed_rollout)

# the causal controls (§3): P0 zero action, P1 frozen transport, P4 E-approach + handoff-matched learned transport
_CHAIN_POLICIES = {
    "zero": ["P0_ZERO_ACTION"],
    "frozen": ["P1_FROZEN_TRANSPORT"],
    "e_handoff": ["P4_E_APPROACH_HANDOFF"],
    "causal": ["P4_E_APPROACH_HANDOFF", "P1_FROZEN_TRANSPORT", "P0_ZERO_ACTION"],
}


def _build_point_env() -> tuple[Any, Any]:
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    return neutral_env(prefix_steps=0, geom="POINT")


def _problem_from_seed(env: Any, cf: Any, seed: int) -> CoinInitialState:
    env.set_stage(0)
    env.reset(seed=int(seed))
    return extract_initial_state(env, cf)


def run_fixed_replay(*, mode: str, embodiment: str = "POINT", seed: int | None = None,
                     initial_state: dict[str, Any] | None = None, policy_chain: str = "e_handoff",
                     repetitions: int = 10, neutral_start: bool = True, seed_hint: int = 0,
                     out: Path | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run a fixed-position replay and its causal controls; returns the full result (and writes artifacts if ``out``).

    # Preconditions ``mode`` in {"seed", "exact"}; for "seed" ``seed`` is set, for "exact" ``initial_state`` is a
      valid schema dict. ``policy_chain`` in :data:`_CHAIN_POLICIES`. # Errors loud on an unknown mode/chain or an
      invalid / unreachable / in-contact / target-overlapping state (never a silent fallback).
    """
    if policy_chain not in _CHAIN_POLICIES:
        raise ValueError(f"policy_chain {policy_chain!r} unknown; valid {sorted(_CHAIN_POLICIES)}")
    if embodiment != "POINT":
        raise ValueError(f"fixed-position replay is POINT-only in this release; got {embodiment!r}")
    env, cf = _build_point_env()

    if mode == "seed":
        if seed is None:
            raise ValueError("mode='seed' requires a seed")
        state = _problem_from_seed(env, cf, seed)
        state = CoinInitialState(**{**state.to_dict(), "provenance": {"source": "seed", "seed": int(seed)}})
    elif mode == "exact":
        if initial_state is None:
            raise ValueError("mode='exact' requires an initial_state dict")
        state = CoinInitialState.from_dict(initial_state)
    else:
        raise ValueError(f"mode {mode!r} unknown; expected 'seed' or 'exact'")

    apply_initial_state(env, cf, state, seed_hint=seed_hint)
    gate = assert_replayable(env, cf, state, embodiment=embodiment, neutral_start=neutral_start)
    reach = analyze_reachability(env, cf)
    problem = {"problem_hash": state.problem_hash(), "env_fingerprint": env_fingerprint(cf),
               "initial_state": state.to_dict(), "reachability": reach.to_dict(),
               "checkpoint_hashes": gate["checkpoint_hashes"]}
    log(f"[replay] mode={mode} problem_hash={problem['problem_hash']} "
        f"geom_fp={problem['env_fingerprint']['geom_fp']} clearance={reach.signed_clearance:+.5f} "
        f"coin=({state.coin_position[0]:.4f},{state.coin_position[1]:.4f}) reachable={reach.coin_reachable}")

    causal: dict[str, Any] = {}
    for policy in _CHAIN_POLICIES[policy_chain]:
        approach, tfn = build_actors(policy)
        reps = []
        for _r in range(repetitions):
            apply_initial_state(env, cf, state, seed_hint=seed_hint)     # exact same start every rep (deterministic)
            reps.append(composed_rollout(env, cf, approach, tfn, grasp_hold=1, contact_window=20, policy=policy))
        delivered = sum(1 for t in reps if t.strict_delivered)
        causal[policy] = {
            "strict": f"{delivered}/{repetitions}",
            "first_contact": sum(t.first_contact for t in reps),
            "bilateral_contact": sum(t.bilateral_contact for t in reps),
            "targetward_motion": sum(t.targetward_motion for t in reps),
            "zone_entry": sum(t.zone_entry for t in reps),
            "median_completion_time_s": _median([t.completion_time_s for t in reps if t.strict_delivered]),
            "failure_reasons": _tally([t.failure_reason for t in reps if not t.strict_delivered]),
            "trajectory_hashes": sorted({t.trajectory_hash() for t in reps}),
        }
        log(f"[replay] {policy}: strict={causal[policy]['strict']} first={causal[policy]['first_contact']} "
            f"bilateral={causal[policy]['bilateral_contact']} zone={causal[policy]['zone_entry']} "
            f"traj_hashes={causal[policy]['trajectory_hashes']}")

    result = {"domain": "coin_fixed_position", "mode": mode, "policy_chain": policy_chain,
              "repetitions": repetitions, "problem": problem, "causal_table": causal}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "problem.json").write_text(json.dumps(problem, indent=1))
        (out / "causal_table.json").write_text(json.dumps(causal, indent=1))
        (out / "initial_state.json").write_text(json.dumps(state.to_dict(), indent=1))
    return result


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def _tally(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out
