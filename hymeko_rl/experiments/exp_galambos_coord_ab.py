"""Coordination-reward A/B on the collaborative off-policy galambos scenario.

Baseline (``galambos_task.hymeko``) vs Coordination (``galambos_task_coord.hymeko`` = baseline +
``both_approach 4.0``). Identical collab CTDE off-policy setup on both arms
(:func:`build_collaborative_offpolicy`, ``sa_hsikan``, TD3+BC, best-checkpoint on delivery).

Hypothesis (measured): the baseline peaks at delivery 0.40 with ``both_contact ≈ 0.019`` — the two arms
almost never grip the coin *simultaneously*, and the two-arm ``coin_frictionloss`` needs simultaneous
force. ``both_approach = -max(left,right)`` penalises the lagging arm, the coordination gradient the
compensable mean ``grasp_approach`` lacks. Discriminating question: does ``both_contact`` (→ delivery) rise?

    python -m hymeko_rl.experiments.exp_galambos_coord_ab            # full overnight (3 seeds × 200k × 2)
    python -m hymeko_rl.experiments.exp_galambos_coord_ab --smoke    # 1 seed × ~3k, path check
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.eval.evaluate import greedy_action_fn
from hymeko_rl.experiments.galambos_bc import collect_galambos_demos, eval_delivery
from hymeko_rl.train.campaign import Campaign, CampaignConfig

_COORD_HYMEKO = "data/robotics/galambos_task_coord.hymeko"
_DELIVER_V2B = "data/robotics/galambos_task_deliver_v2b.hymeko"   # the sound coin-toss DELIVERY reward (v2b)
_PLANAR_TASK_DEFAULT = "data/robotics/galambos_task.hymeko"        # PlanarGraspEnv's default (the A/B baseline reward)
_MAX_STEPS = 300
_EVAL_SEED = 9_000
# Below this much coin→zone progress attributed to a body-only push (metres), a v2 delivery is graded
# "fingertip_dominant" (the fingertips did the work; any arm-body touch was incidental). See grade_delivery.
_NEGLIGIBLE_BODY_EPS = 0.005


def grade_delivery(*, held: bool, body_progress: float, fingertip_progress: float,
                   clean_eps: float = _NEGLIGIBLE_BODY_EPS) -> "str | None":
    """Grade a v2 delivery by PROGRESS ATTRIBUTION — how much of the coin's toward-zone motion came from a
    body-only push (``body_progress``) versus a fingertip grasp (``fingertip_progress``).

    Returns ``None`` for a non-delivery (not held), else one of:
      - ``"fingertip_dominant"``   : negligible body-only progress (``<= clean_eps``) — the fingertips did the
                                     work; **an incidental distal-link graze is allowed** (this is NOT the same
                                     as zero arm-body contact — see ``zero_body_contact_delivery`` in
                                     :func:`_coordination_metrics`, which requires NO arm-body contact at all);
      - ``"body_driven_exploit"``  : the body did most of the moving (``body_progress > fingertip_progress``) — a shove;
      - ``"body_assisted"``        : some meaningful body contribution but the fingertips dominated.

    Deliberately does NOT use the word "clean": with a high arm-body contact rate, "clean" reads as
    "no contact", which is a stronger claim than fingertip-dominant progress attribution.

    # Preconditions ``body_progress``, ``fingertip_progress`` ``>= 0``; ``clean_eps >= 0``.
    # Postconditions returns ``None`` iff ``not held``; the three grades are mutually exclusive and partition raw.
    """
    if not held:
        return None
    if body_progress <= clean_eps:
        return "fingertip_dominant"
    if body_progress > fingertip_progress:
        return "body_driven_exploit"
    return "body_assisted"


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


_FLAT_PAD_SIZE = "0.004 0.016 0.02"     # box half-extents: THIN in the contact-normal x (a flat face), wide in y/z


def make_env(*, coord: bool, difficulty: float, treatment_hymeko: str = _COORD_HYMEKO,
             deliver: bool = False, fingertip_geometry: str = "POINT") -> PlanarGraspEnv:
    """A planar grasping env, tagging the active reward file on ``env.reward_file``.

    - ``coord=True``  → the coordination treatment reward (``treatment_hymeko``).
    - ``deliver=True`` → the sound coin-toss DELIVERY reward **v2b** (``galambos_task_deliver_v2b.hymeko``). This is
      the coin-toss task's intended reward (2026-07-09 reward-identity fix: the coin-toss env previously fell through
      to the ``galambos_task`` default — a dense-annuity farming reward — which the audit flagged).
    - otherwise → the PlanarGraspEnv default (``galambos_task``), the original coord-A/B **baseline** (unchanged).

    ``coord`` and ``deliver`` are mutually exclusive; ``coord`` wins if both are set. Reward-in-hymeko (no in-memory
    term surgery). # Postconditions: ``env.reward_file`` names the active reward's ``.hymeko`` source.
    """
    # fingertip_contact_geometry: POINT (sphere = the existing golden model, no-op) vs FLAT_PAD (box = a finite contact
    # patch). Only the fingertip geom type/size changes; joint/actuator/arm/coin/target/reward/predicate are untouched.
    if fingertip_geometry not in ("POINT", "FLAT_PAD"):
        raise ValueError(f"fingertip_geometry must be POINT or FLAT_PAD; got {fingertip_geometry!r}")
    _fg = {} if fingertip_geometry == "POINT" else {"fingertip_shape": "box", "fingertip_size": _FLAT_PAD_SIZE}
    env = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=difficulty, **_fg)
    if coord:
        env.reward_spec = RewardSpec.from_hymeko(treatment_hymeko)
        env.reward_file = treatment_hymeko
    elif deliver:
        env.reward_spec = RewardSpec.from_hymeko(_DELIVER_V2B)
        env.reward_file = _DELIVER_V2B
    else:
        env.reward_file = _PLANAR_TASK_DEFAULT
    return env


def _coordination_metrics(env: PlanarGraspEnv, actor: Any, n_episodes: int, seed: int,
                          *, action_fn: "Any | None" = None) -> "dict[str, float]":
    """One greedy rollout pass → the four physics/task validation metrics (all from the same episodes, §6.1):
      - ``both_contact``      : fraction of steps BOTH fingertips touch the coin;
      - ``fingertip_contact`` : fraction of fingertip-steps in contact (``(left+right)/2`` per step);
      - ``coin_vel_to_zone``  : mean coin velocity · unit(zone−coin) (``>0`` = coin moving toward the target);
      - ``dist_delta``        : mean ``d0 − d_final`` per episode (``>0`` = coin ended closer to the zone).
    These are the sanity gate to require of any policy BEFORE trusting a delivery number (the coin must
    actually move toward the target, with real contact).

    Under v2 contact legality the same rollout also yields the graded contact-quality metrics. Deliveries are
    graded by **progress attribution** — how much of the coin's motion TOWARD the zone happened under a fingertip
    grasp versus a body-only push (an arm link touching the coin with no fingertip contact). The tiers are
    reported SEPARATELY and named precisely (no "clean", which reads as "no contact"):
      - ``raw_delivery``               : held the zone (``>= success_steps``), any contact;
      - ``fingertip_dominant_delivery``: held with negligible body-only progress — fingertips did the work,
                                         an incidental distal-link graze IS allowed (so this is NOT zero-contact);
      - ``zero_body_contact_delivery`` : held with NO arm-body↔coin contact at all (the strict "paper-clean"
                                         number; a subset of fingertip-dominant);
      - ``body_assisted_delivery``     : held with a meaningful body contribution, fingertips still dominant;
      - ``body_driven_exploit_delivery``: held with the body doing most of the moving (a shove exploit);
      - ``arm_body_rate``              : fraction of episodes with any arm-body↔coin contact;
      - ``arm_body_steps`` / ``arm_body_impulse`` : mean per-episode arm-body contact duration / summed impulse.
    ``raw = fingertip_dominant + body_assisted + body_driven_exploit``; ``zero_body_contact <= fingertip_dominant``.
    In v1 (no legality) the arm-body columns are zero and ``fingertip_dominant == zero_body_contact == raw``."""
    act = action_fn if action_fn is not None else greedy_action_fn(actor)  # env-aware policies pass action_fn
    steps = both = ft = 0
    vel_sum = 0.0
    deltas: "list[float]" = []
    n_raw = n_fingertip_dominant = n_zero_body = n_body_assisted = n_body_exploit = n_arm_body = 0
    body_steps_sum = body_impulse_sum = 0.0
    success_steps = int(getattr(env, "success_steps", 1))
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        d0 = d = prev_d = None
        dwell = dwell_max = 0
        ft_progress = body_progress = 0.0                      # coin→zone progress under fingertip vs body-only push
        info: dict[str, Any] = {}
        while not done:
            obs, _r, term, trunc, info = env.step(act(env, obs))
            m = getattr(env, "_planar_metrics", None)
            steps += 1
            dwell = dwell + 1 if info.get("in_zone") else 0     # held-delivery dwell (matches the env success rule)
            dwell_max = max(dwell_max, dwell)
            if m is not None:
                both += int(m.left_contact and m.right_contact)
                ft += int(m.left_contact) + int(m.right_contact)
                to_zone = np.array([env._zone_x - float(m.disk_pos[0]),
                                    env._zone_y - float(m.disk_pos[1])], dtype=np.float32)
                d = float(np.hypot(to_zone[0], to_zone[1]))
                if d0 is None:
                    d0 = d
                if d > 1e-6:                                   # skip when already at the zone (undefined direction)
                    vel_sum += float(np.dot(m.disk_vel, to_zone / d))
                if prev_d is not None:                         # attribute toward-zone progress to the contact kind
                    toward = max(0.0, prev_d - d)
                    if info.get("fingertip_contact"):
                        ft_progress += toward
                    elif info.get("arm_body_contact_this_step"):
                        body_progress += toward               # body-only push (no fingertip this step)
                prev_d = d
            done = bool(term or trunc)
        if d0 is not None and d is not None:
            deltas.append(d0 - d)                              # >0 = coin ended closer to the zone
        raw = dwell_max >= success_steps
        n_raw += int(raw)
        grade = grade_delivery(held=raw, body_progress=body_progress, fingertip_progress=ft_progress)
        n_fingertip_dominant += int(grade == "fingertip_dominant")
        n_body_assisted += int(grade == "body_assisted")
        n_body_exploit += int(grade == "body_driven_exploit")
        arm_body_ever = bool(info.get("arm_body_contact"))
        n_zero_body += int(raw and not arm_body_ever)          # held with NO arm-body contact at all (strict clean)
        n_arm_body += int(arm_body_ever)
        body_steps_sum += float(info.get("arm_body_contact_steps", 0))
        body_impulse_sum += float(info.get("arm_body_impulse_sum", 0.0))
    n = max(1, steps)
    ne = max(1, n_episodes)
    return {"both_contact": both / n, "fingertip_contact": ft / (2 * n),
            "coin_vel_to_zone": vel_sum / n,
            "dist_delta": float(np.mean(deltas)) if deltas else 0.0,
            "raw_delivery": n_raw / ne, "fingertip_dominant_delivery": n_fingertip_dominant / ne,
            "zero_body_contact_delivery": n_zero_body / ne, "body_assisted_delivery": n_body_assisted / ne,
            "body_driven_exploit_delivery": n_body_exploit / ne,
            "arm_body_rate": n_arm_body / ne, "arm_body_steps": body_steps_sum / ne,
            "arm_body_impulse": body_impulse_sum / ne}


def _both_contact_rate(env: PlanarGraspEnv, actor: Any, n_episodes: int, seed: int) -> float:
    """Fraction of steps in which BOTH fingertips touch the coin — the coordination metric the A/B tests.
    Thin wrapper over :func:`_coordination_metrics` (back-compat with prior experiment scripts)."""
    return _coordination_metrics(env, actor, n_episodes, seed)["both_contact"]


def measure_factory(*, coord: bool, difficulty: float, n_eval: int, treatment_hymeko: str = _COORD_HYMEKO,
                    ) -> "Any":
    """A Campaign ``measure(make_env, actor) -> {delivery, both_contact, coin_vel_to_zone}`` closed over the
    variant. Eval always uses the BASELINE env (coord=False) so delivery is scored on the true task, not the
    shaped training reward. ``coin_vel_to_zone`` is the target-directed coin velocity (CTDE contract cond. 5)."""
    def _measure(_make_env: Any, actor: Any) -> "dict[str, float]":
        delivery_env = make_env(coord=False, difficulty=difficulty)
        contact_env = make_env(coord=False, difficulty=difficulty)
        try:
            deliv = eval_delivery(delivery_env, actor, n_eval, _EVAL_SEED)
            coord = _coordination_metrics(contact_env, actor, min(n_eval, 12), _EVAL_SEED)
        finally:
            _close_env(delivery_env)
            _close_env(contact_env)
        return {"delivery": deliv, **coord}
    return _measure


@dataclass(frozen=True)
class AbConfig:
    """The A/B experiment configuration — one declaration site instead of a growing ``run(**kwargs)``
    signature (§6.5 #1/#6: new axes belong in config, not in new parameters).

    ``smoke=True`` shrinks every budget to a path check. ``uncertified_waiver`` is the explicit, logged
    escape hatch for deliberately studying a reward the oracle says does NOT deliver — never a default.
    """

    difficulty: float = 0.3
    smoke: bool = False
    n_demos: "int | None" = None            # None → 12 (smoke) / 200
    total_steps: "int | None" = None        # None → 3_000 (smoke) / 200_000
    bc_epochs: "int | None" = None          # None → 3 (smoke) / 200
    seeds: "tuple[int, ...] | None" = None  # None → (0,) (smoke) / (0, 1, 2)
    variants: "tuple[str, ...]" = ("baseline", "coord")
    treatment_hymeko: str = _COORD_HYMEKO
    treatment_name: str = "coord"
    uncertified_waiver: bool = False
    profile: "str | None" = None            # the declaring .hymeko, when built via from_hymeko (provenance)
    offpolicy: "dict[str, float] | None" = None  # off-policy knob overrides (the profile's @offpolicy term)
    bc_batch: int = 128                     # BC minibatch (512 + auto device = the measured GPU path)

    @classmethod
    def from_hymeko(cls, profile: "str | Path", *, smoke: bool = False) -> "AbConfig":
        """Build the campaign config from a declared ``experiment_spec`` profile (the canonical entry —
        the experiment is DATA, like the reward/strategy/scenario/controller; see meta_experiment.hymeko).
        ``smoke`` is the one runtime knob (a path check is a launch decision, not experiment identity).

        # Errors ``FileNotFoundError``; ``ValueError`` (no/malformed ``experiment_spec``, unknown field).
        """
        from hymeko_rl.env._profile import parse_fields, read_bundle
        fields: "dict[str, Any]" = {}
        refine: "dict[str, float]" = {}
        for _name, kind, body, _w in read_bundle(profile, "experiment_spec"):
            if kind == "offpolicy":               # free-form trainer overrides, validated by OffPolicyConfig
                refine.update({k: float(v) for k, v in parse_fields(body).items()
                               if isinstance(v, (int, float))})
            else:
                fields.update(parse_fields(body))
        variant = str(fields.pop("variant", "both"))
        known = {"difficulty", "total_steps", "n_demos", "bc_epochs", "seeds",
                 "treatment_hymeko", "treatment_name", "bc_batch"}
        unknown = sorted(fields.keys() - known)
        if unknown:
            raise ValueError(f"{profile}: unknown experiment field(s) {unknown}; expected {sorted(known)}")
        if "seeds" in fields:
            fields["seeds"] = tuple(int(s) for s in fields["seeds"])
        for k in ("total_steps", "n_demos", "bc_epochs", "bc_batch"):
            if k in fields:
                fields[k] = int(fields[k])
        return cls(variants=("baseline", "coord") if variant == "both" else (variant,),
                   smoke=smoke, profile=str(profile), offpolicy=refine or None, **fields)


def _certify_variants(cfg: AbConfig, variant_is_coord: dict[str, bool]) -> dict[str, bool]:
    """PRE-QUEUE GATE (2026-07-05, after a wasted overnight on the farming baseline): every variant's
    TRAINING reward must be oracle-certified as delivering BEFORE any training is queued. certify() costs
    ms; a wrong overnight costs hours. Machine-enforced precondition — not an operator habit.

    # Postconditions returns the per-variant certificates. # Errors ``ValueError`` if any variant's reward
      optimum does not deliver and ``cfg.uncertified_waiver`` is not set.
    """
    from hymeko_rl.eval.reward_oracle import certify
    certificates: dict[str, bool] = {}
    for name, coord in variant_is_coord.items():
        spec = make_env(coord=coord, difficulty=cfg.difficulty, treatment_hymeko=cfg.treatment_hymeko).reward_spec
        certificates[name] = bool(certify(spec).delivers)
        print(f"[oracle] variant {name!r}: training-reward optimum delivers = {certificates[name]}", flush=True)
    bad = [n for n, ok in certificates.items() if not ok]
    if bad and not cfg.uncertified_waiver:
        raise ValueError(f"training reward for variant(s) {bad} is NOT oracle-certified to deliver; "
                         f"fix the reward .hymeko or set AbConfig.uncertified_waiver to study it deliberately")
    return certificates


def resolve_budget(cfg: AbConfig) -> "tuple[tuple[int, ...], int, int, int, int, int]":
    """The effective ``(seeds, total_steps, n_demos, bc_epochs, eval_every, n_eval)`` for a config.

    ``smoke`` is a PATH CHECK: it **caps** every budget, including profile-declared ones — a fully
    specified profile must not turn ``--smoke`` into a full-scale launch (found the hard way,
    2026-07-05 03:51). # Postconditions under ``smoke``: 1 seed, ≤3k steps, ≤12 demos, ≤3 epochs.
    """
    seeds = cfg.seeds if cfg.seeds is not None else (0, 1, 2)
    total_steps = cfg.total_steps if cfg.total_steps is not None else 200_000
    n_demos = cfg.n_demos if cfg.n_demos is not None else 200
    bc_epochs = cfg.bc_epochs if cfg.bc_epochs is not None else 200
    if cfg.smoke:
        seeds, total_steps = seeds[:1], min(total_steps, 3_000)
        n_demos, bc_epochs = min(n_demos, 12), min(bc_epochs, 3)
    eval_every = 1_500 if cfg.smoke else max(25_000, total_steps // 8)
    return seeds, total_steps, n_demos, bc_epochs, eval_every, (3 if cfg.smoke else 50)


def run(cfg: AbConfig) -> "dict[str, Any]":
    smoke = cfg.smoke
    seeds, total_steps, n_demos, bc_epochs, eval_every, n_eval = resolve_budget(cfg)
    difficulty, treatment_hymeko = cfg.difficulty, cfg.treatment_hymeko

    _all = {"baseline": False, cfg.treatment_name: True}
    variants = tuple(cfg.treatment_name if v == "coord" else v for v in cfg.variants)
    certificates = _certify_variants(cfg, {n: _all[n] for n in variants})
    # The scripted demonstrator ignores the reward and the training seed-init, so its demos are identical
    # across seeds AND across the baseline/coord variants — collect once, reuse (a closure cache, not a
    # module global, §6.5 #11). Saves re-rolling ~16k samples per seed.
    _demo_cache: "dict[int, tuple[np.ndarray, np.ndarray]]" = {}

    def cached_demos(env: Any, n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
        if n not in _demo_cache:
            _demo_cache[n] = collect_galambos_demos(env, n, seed)
        return _demo_cache[n]

    summary: dict[str, Any] = {"difficulty": difficulty, "smoke": smoke,
                               "experiment_profile": cfg.profile,     # provenance: the declaring .hymeko
                               "reward_certificates": certificates,   # provenance: the pre-queue oracle verdicts
                               "budget": {"n_demos": n_demos, "total_steps": total_steps, "bc_epochs": bc_epochs,
                                          "seeds": list(seeds)}, "variants": {}}
    for name in variants:
        coord = _all[name]
        camp_cfg = CampaignConfig(
            name=f"galambos_coord_ab_{name}", select="delivery", seeds=seeds,
            total_steps=total_steps, eval_every=eval_every, n_demos=n_demos, bc_epochs=bc_epochs,
            n_eval=n_eval, offpolicy=cfg.offpolicy, bc_batch=cfg.bc_batch)

        def variant_env(coord: bool = coord) -> PlanarGraspEnv:
            return make_env(coord=coord, difficulty=difficulty, treatment_hymeko=treatment_hymeko)

        camp = Campaign(
            camp_cfg,
            make_env=variant_env,
            build=lambda env: build_collaborative_offpolicy(env, kind="sa_hsikan", hidden=64),
            measure=measure_factory(coord=coord, difficulty=difficulty, n_eval=n_eval,
                                    treatment_hymeko=treatment_hymeko),
            demos=cached_demos,
            gif=not smoke,
        )
        summary["variants"][name] = camp.run()
    return summary


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Run a DECLARED galambos A/B campaign (experiment-as-data).")
    ap.add_argument("profile", nargs="?", default="data/robotics/galambos_ab_deliver.hymeko",
                    help="the experiment_spec .hymeko declaring budget + arms (the experiment IS the file)")
    ap.add_argument("--smoke", action="store_true", help="1 seed × ~3k steps path check of the same profile")
    args = ap.parse_args(argv)
    summary = run(AbConfig.from_hymeko(args.profile, smoke=args.smoke))
    print("\n=== A/B SUMMARY ===")
    for name, res in summary["variants"].items():
        peak = res.get("peak_delivery_median", res.get("peak_median", "?"))
        print(f"  {name:9s}: peak delivery median = {peak}")
    print(json.dumps(summary, indent=2, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
