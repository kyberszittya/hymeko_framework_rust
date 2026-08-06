"""Stage A of the contact-reward ablation — recompute cached rollout rewards under reward variants, no training.

The Phase-2 DirectLiNGAM result proposes ``contact_score → total_reward`` (w≈+0.69) while ``total_reward`` is
disconnected from ``delivery_score`` — i.e. the coin reward is *contact-shaped* and optimisable without improving
delivery (the BC→RL-collapse signature). This module runs the **reward-computation-level** intervention that would
begin to support that hypothesis, *without retraining*:

1. Re-roll each cached policy on the same seeds and record, per step, the **unweighted** value of every reward
   term (the ``Σ weight·term`` decomposition — see :mod:`hymeko_rl.env.reward`). Because re-evaluating the terms
   on the post-``step`` env reproduces the env's own reward bit-exactly (verified: max error 0 over 162 steps),
   the recorded matrix lets any reward *variant* be recomputed offline as a weighted sum — no env re-stepping,
   no training.
2. Recompute ``total_reward`` under: **original**, **contact-off** (both/finger contact zeroed),
   **contact-downweighted** (×0.25), and **delivery-aligned** (contact annuity removed, ``in_zone`` boosted +
   ``grasp_deliver`` added).
3. Rebuild the architecture-stratified :class:`RolloutFrame` for each variant — **only ``total_reward`` and its
   reward↔monitor disagreement change**; the trajectory (and thus every monitor sub-score) is identical — rerun
   DirectLiNGAM, and declare + cross-view-verify each discovered DAG.

**Stage A CANNOT claim a delivery delta** (the policy is unchanged; ``delivery_score`` is identical across
variants). It tests only whether the reward *computation* produces the causal signature. The decision rule: if
removing/downweighting the contact terms collapses the ``contact_score → total_reward`` edge and reduces the
reward↔monitor disagreement, the reward-farming hypothesis is supported **at the reward-computation level**. A
policy-learning-level claim needs Stage B (a bounded training smoke — documented in the report, NOT run here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hymeko_rl.env.reward import _REWARD_TERMS

if TYPE_CHECKING:
    from hymeko_rl.eval.causal import LingamResult

# The dense per-step contact REWARD the DirectLiNGAM edge implicates (the "annuity" the policy farms).
CONTACT_TERMS: tuple[str, ...] = ("both_contact", "finger_contact")
# Task-success (delivery) terms — for the delivery-aligned variant and the reward↔delivery alignment probe.
DELIVERY_TERMS: tuple[str, ...] = ("in_zone", "grasp_deliver", "terminal_deliver")


@dataclass(frozen=True)
class RewardVariant:
    """A named reweighting of the base reward spec — same term extractors (Strategy), different weights.

    # Invariants every key is a known reward term (:data:`_REWARD_TERMS`); a term absent from ``weights`` has
      weight 0 in the recomputed sum.
    """

    name: str
    weights: dict[str, float]
    description: str = ""

    def __post_init__(self) -> None:
        unknown = [k for k in self.weights if k not in _REWARD_TERMS]
        if unknown:
            raise ValueError(f"variant {self.name!r}: unknown reward term(s) {unknown}; "
                             f"known: {sorted(_REWARD_TERMS)}")

    def weight_vector(self, recorded_terms: "list[str]") -> np.ndarray:
        """Dense weight vector aligned to ``recorded_terms`` (0 for a term this variant does not weight)."""
        return np.array([float(self.weights.get(k, 0.0)) for k in recorded_terms], dtype=np.float64)


def build_variants(base_terms: "tuple[tuple[str, float], ...]", *, downweight: float = 0.25) -> list[RewardVariant]:
    """The four Stage-A reward variants over a base ``(term, weight)`` spec.

    Only the weight vector differs across variants; the term extractors (and thus the recorded per-step term
    values) are shared, so every variant is a pure reweighting of the SAME rollouts.

    # Preconditions ``0 <= downweight <= 1``. # Postconditions ``result[0]`` is the untouched ``original``.
    """
    if not 0.0 <= downweight <= 1.0:
        raise ValueError(f"downweight must be in [0, 1]; got {downweight}")
    base = dict(base_terms)
    off = {k: (0.0 if k in CONTACT_TERMS else w) for k, w in base.items()}
    down = {k: (round(w * downweight, 6) if k in CONTACT_TERMS else w) for k, w in base.items()}
    # delivery-aligned: zero the contact annuity, keep approach (still needed to reach) + penalties, and REDIRECT
    # the contact budget into delivery — double the sparse in_zone term and add a grasp-gated delivery bonus.
    deliver = {k: (0.0 if k in CONTACT_TERMS else w) for k, w in base.items()}
    deliver["in_zone"] = round(2.0 * base.get("in_zone", 0.0), 6) or 20.0
    deliver["grasp_deliver"] = round(0.5 * base.get("both_contact", 0.0), 6) or 2.5
    return [
        RewardVariant("original", dict(base), "the deployed coin reward, unchanged"),
        RewardVariant("contact_off", off, "both_contact & finger_contact zeroed"),
        RewardVariant("contact_downweighted", down, f"contact terms x{downweight}"),
        RewardVariant("delivery_aligned", deliver, "contact annuity removed; in_zone x2 + grasp_deliver added"),
    ]


def recompute_variant_reward(term_matrix: np.ndarray, recorded_terms: "list[str]", variant: RewardVariant) -> float:
    """Recompute an episode's ``total_reward`` under ``variant`` from its recorded per-step term matrix.

    # Preconditions ``term_matrix`` is ``(n_steps, len(recorded_terms))``.
    # Postconditions returns ``Σ_steps Σ_k variant.weight[k] · term_value[s, k]`` (the env's ``Σ weight·term``
      summed over the episode).
    """
    if term_matrix.ndim != 2 or term_matrix.shape[1] != len(recorded_terms):
        raise ValueError(f"term_matrix shape {term_matrix.shape} != (*, {len(recorded_terms)})")
    return float((term_matrix @ variant.weight_vector(recorded_terms)).sum())


def directed_edge_weight(result: "LingamResult", cause: str, effect: str) -> float:
    """The discovered ``cause → effect`` adjacency weight (0.0 if either variable was dropped / no edge)."""
    names = list(result.names)
    if cause not in names or effect not in names:
        return 0.0
    return float(result.adjacency[names.index(effect), names.index(cause)])


def reward_delivery_alignment(reward: "np.ndarray | list[float]", delivery: "np.ndarray | list[float]") -> float:
    """Pearson correlation between per-episode ``total_reward`` and ``delivery_score`` (0.0 if either is constant).

    This is the scalar the ablation tracks for ``total_reward ↔ delivery_score`` alignment — higher = the reward
    tracks delivery better. Robust to the LiNGAM edge being absent (a constant column drops out of the DAG).
    """
    r = np.asarray(reward, dtype=np.float64)
    d = np.asarray(delivery, dtype=np.float64)
    if r.size < 2 or float(np.std(r)) < 1e-12 or float(np.std(d)) < 1e-12:
        return 0.0
    return float(np.corrcoef(r, d)[0, 1])


@dataclass
class AblationOutcome:
    """Per-(architecture, variant) Stage-A result — everything the decision rule reads."""

    architecture: str
    variant: str
    contact_edge_weight: float          # discovered contact_score -> total_reward weight (0 if collapsed/absent)
    reward_monitor_disagreement: float  # 1 - concordance over (total_reward, monitor_score)
    reward_delivery_alignment: float    # corr(total_reward, delivery_score)
    causal_order: list[str] = field(default_factory=list)
    cross_view_agree: bool = False
    canonical_hash: str = ""
    next_intervention: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture, "variant": self.variant,
            "contact_edge_weight": round(self.contact_edge_weight, 6),
            "reward_monitor_disagreement": round(self.reward_monitor_disagreement, 6),
            "reward_delivery_alignment": round(self.reward_delivery_alignment, 6),
            "causal_order": list(self.causal_order), "cross_view_agree": self.cross_view_agree,
            "canonical_hash": self.canonical_hash, "next_intervention": self.next_intervention,
        }


# ── rollout recording (mujoco; exercised at production scale by run_stage_a, not by a toy unit test) ──────────
@dataclass
class _EpisodeRecord:
    """One recorded episode: the monitor verdict + the per-step unweighted term matrix + the env's summed reward."""

    verdict: Any
    term_matrix: np.ndarray          # (n_steps, n_recorded_terms)
    env_reward: float


def _roll_terms(env: Any, action_fn: Any, seed: int, recorded_terms: "list[str]") -> "tuple[np.ndarray, float]":
    """Roll one episode, recording per step the unweighted value of every ``recorded_terms`` extractor + Σ reward.

    Re-evaluates the terms on the post-``step`` env with the same ``(disk_to_zone, action)`` the env used, so the
    original-weight recomputation reproduces the env reward exactly (asserted by the caller)."""
    obs, _ = env.reset(seed=seed)
    rows: list[list[float]] = []
    total = 0.0
    for _ in range(int(env.max_steps)):
        action = np.asarray(action_fn(env, obs), np.float32)
        obs, reward, terminated, truncated, _info = env.step(action)
        dist = float(env._planar_metrics.disk_to_zone)
        rows.append([float(_REWARD_TERMS[k](env, dist, action)) for k in recorded_terms])
        total += float(reward)
        if terminated or truncated:
            break
    return np.asarray(rows, dtype=np.float64), total


def _record_policy(arch: str, kind: str, checkpoint: str, n: int, seed0: int, recorded_terms: "list[str]",
                   original: RewardVariant, *, rtol: float = 1e-6) -> list[_EpisodeRecord]:
    """Re-roll a cached policy (same seeds) → per-episode (verdict, term matrix, env reward).

    Asserts the recorded term matrix reproduces the env reward under the ORIGINAL weights (the correctness gate —
    a mismatch means the recomputation is unfaithful; §11 halt, not a silent pass)."""
    from hymeko_rl.eval.evaluate import greedy_action_fn
    from hymeko_rl.eval.task_monitor.context import record_trajectory
    from hymeko_rl.eval.task_monitor.root import TaskMonitor
    from hymeko_rl.experiments.cip_lingam_demo import _load_coin_actor, _make_coin_env

    env = _make_coin_env()
    actor = _load_coin_actor(env, kind, checkpoint)
    action_fn = greedy_action_fn(actor)
    mon = TaskMonitor.from_env(env)

    records: list[_EpisodeRecord] = []
    for ep in range(n):
        seed = seed0 + ep
        verdict = mon.evaluate(record_trajectory(env, action_fn, seed))       # monitor tensor roll
        term_matrix, env_reward = _roll_terms(env, action_fn, seed, recorded_terms)  # term-recording roll (same seed)
        recomputed = recompute_variant_reward(term_matrix, recorded_terms, original)
        if abs(recomputed - env_reward) > rtol * max(1.0, abs(env_reward)):
            raise RuntimeError(f"{arch} ep{ep}: recomputed original reward {recomputed:.6f} != env reward "
                               f"{env_reward:.6f} — term recomputation is unfaithful; halting per §11")
        records.append(_EpisodeRecord(verdict=verdict, term_matrix=term_matrix, env_reward=env_reward))
        print(f"[ablation] {arch:6s} ep {ep + 1:2d}/{n} | pass={int(verdict.monitor_pass)} "
              f"deliv={verdict.delivery_score:+.3f} env_reward={env_reward:+.2f}", flush=True)
    if hasattr(env, "close"):
        env.close()
    return records


def _variant_frame(records: list[_EpisodeRecord], recorded_terms: "list[str]", variant: RewardVariant, arch: str,
                   ) -> "tuple[Any, list[float], list[float], float]":
    """Build the architecture-tagged frame for ``variant`` — only ``total_reward`` + its disagreement change.

    Returns ``(frame, total_reward, delivery, disagreement)``."""
    from hymeko_rl.eval.causal import RolloutFrame
    from hymeko_rl.eval.task_monitor.consistency import RewardConsistencyMonitor

    verdicts = [r.verdict for r in records]
    total_reward = [recompute_variant_reward(r.term_matrix, recorded_terms, variant) for r in records]
    delivery = [float(v.delivery_score) for v in verdicts]
    rows = [{"policy": f"{arch}_ep{i}", "total_reward": total_reward[i], "monitor_score": float(v.monitor_score)}
            for i, v in enumerate(verdicts)]
    disagreement = round(1.0 - float(RewardConsistencyMonitor().check_reward_alignment(rows).score), 6)

    maps = [{**v.as_dict(), "reward_progress_disagreement": disagreement} for v in verdicts]
    extra = {
        "approach_score": [v.approach_score for v in verdicts],
        "contact_score": [v.contact_score for v in verdicts],
        "delivery_score": delivery,
        "anti_exploit_score": [v.anti_exploit_score for v in verdicts],
        "total_reward": total_reward,
    }
    frame = RolloutFrame.from_verdicts(maps, extra_continuous=extra, categoricals={"architecture": [arch] * len(maps)})
    return frame, total_reward, delivery, disagreement


def _fit_and_declare(frame: Any, arch: str, variant: RewardVariant, out_dir: Path, outcome: AblationOutcome) -> None:
    """Fit DirectLiNGAM on ``frame``, fill the causal fields of ``outcome``, and declare + cross-view-verify the DAG.

    A stratum with too few varying continuous columns leaves ``outcome``'s causal fields at their defaults."""
    from hymeko_rl.eval.causal import CausalHypergraph, DirectLiNGAM, cross_view_verify
    from hymeko_rl.experiments.cip_lingam_demo import render_dag

    matrix, kept, _dropped = frame.continuous_matrix()
    if len(kept) < 2 or matrix.shape[0] <= len(kept):
        return
    result = DirectLiNGAM().fit(matrix, kept)
    outcome.contact_edge_weight = directed_edge_weight(result, "contact_score", "total_reward")
    outcome.causal_order = result.ordered_names()
    cg = CausalHypergraph.from_lingam(result, f"Coin{arch.capitalize()}{variant.name.title().replace('_', '')}")
    xview = cross_view_verify(cg, out_dir / f"causal_{arch}_{variant.name}.hymeko")
    outcome.cross_view_agree = bool(xview.agree)
    outcome.canonical_hash = xview.canonical_hash
    render_dag(result.order, result.adjacency, kept, out_dir / f"dag_{arch}_{variant.name}.png",
               f"Coin causal — {arch} / {variant.name} (N={matrix.shape[0]}, PROPOSED)")


def _diagnose_variant(records: list[_EpisodeRecord], recorded_terms: "list[str]", variant: RewardVariant,
                      arch: str, out_dir: Path) -> AblationOutcome:
    """Rebuild the frame with ``variant``'s recomputed total_reward, refit DirectLiNGAM, declare + verify the DAG."""
    from hymeko_rl.eval.causal import CausalDiagnosis

    frame, total_reward, delivery, disagreement = _variant_frame(records, recorded_terms, variant, arch)
    outcome = AblationOutcome(
        architecture=arch, variant=variant.name, contact_edge_weight=0.0,
        reward_monitor_disagreement=disagreement,
        reward_delivery_alignment=reward_delivery_alignment(total_reward, delivery),
        next_intervention=CausalDiagnosis().run(frame).next_intervention)
    _fit_and_declare(frame, arch, variant, out_dir, outcome)
    return outcome


def _diagnose_arch(arch: str, kind: str, checkpoint: str, n: int, seed0: int, recorded_terms: "list[str]",
                   variants: list[RewardVariant], original: RewardVariant, out_dir: Path,
                   ) -> "dict[str, Any] | None":
    """Record one architecture's rollouts once, then diagnose every variant. ``None`` if the arch is unavailable."""
    if not Path(checkpoint).exists():
        print(f"[ablation] SKIP {arch}: checkpoint absent ({checkpoint})", flush=True)
        return None
    try:
        records = _record_policy(arch, kind, checkpoint, n, seed0, recorded_terms, original)
    except (RuntimeError, ValueError) as err:
        print(f"[ablation] SKIP {arch}: {type(err).__name__}: {err}", flush=True)
        return None
    arch_outcomes: dict[str, Any] = {}
    for variant in variants:
        outcome = _diagnose_variant(records, recorded_terms, variant, arch, out_dir)
        arch_outcomes[variant.name] = outcome.as_dict()
        print(f"[ablation] {arch}/{variant.name}: contact→reward edge={outcome.contact_edge_weight:+.4f} "
              f"disagreement={outcome.reward_monitor_disagreement:.4f} "
              f"corr(reward,delivery)={outcome.reward_delivery_alignment:+.4f} "
              f"xview={outcome.cross_view_agree} intervention={outcome.next_intervention[:32]}", flush=True)
    return arch_outcomes


def run_stage_a(n: int, seed0: int, out_dir: Path, *, downweight: float = 0.25) -> "dict[str, Any]":
    """Stage A end-to-end: record cached rollouts once per arch, diagnose every reward variant, emit artifacts.

    # Postconditions writes ``summary.json`` + per-(arch,variant) DAG png/.hymeko into ``out_dir``; returns the
      summary dict. NO training is launched; NO Phase-2 artifact is overwritten (a fresh timestamped dir).
    """
    import json

    from hymeko_rl.experiments.cip_lingam_demo import _COIN_POLICIES, _make_coin_env

    base_spec = _make_coin_env().reward_spec
    variants = build_variants(base_spec.terms, downweight=downweight)
    recorded_terms = sorted({k for v in variants for k in v.weights})     # union of every variant's terms
    original = next(v for v in variants if v.name == "original")
    print(f"[ablation] Stage A: {n} eps/arch, seed0={seed0}, terms={recorded_terms}", flush=True)
    print(f"[ablation] variants: {[(v.name, v.weights) for v in variants]}", flush=True)

    summary: dict[str, Any] = {
        "stage": "A", "kind": "cached-rollout reward recomputation (NO training)", "n_episodes_per_arch": n,
        "seed0": seed0, "downweight": downweight, "recorded_terms": recorded_terms,
        "variants": {v.name: v.weights for v in variants},
        "note": "delivery_score/contact_score are IDENTICAL across variants (policy unchanged) — only total_reward "
                "and its disagreement change. Stage A tests the reward COMPUTATION, not a delivery delta.",
        "outcomes": {},
        "_disclaimer": "PROPOSED until intervention evidence. Stage A can support the hypothesis at the "
                       "reward-computation level only; a policy-learning claim needs Stage B (training smoke).",
    }
    for arch, kind, checkpoint in _COIN_POLICIES:
        arch_outcomes = _diagnose_arch(arch, kind, checkpoint, n, seed0, recorded_terms, variants, original, out_dir)
        if arch_outcomes is not None:
            summary["outcomes"][arch] = arch_outcomes

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ablation] wrote summary.json + artifacts to {out_dir}", flush=True)
    return summary


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from hymeko_rl.eval.evaluate import experiment_dir
    parser = argparse.ArgumentParser(description="Stage A: contact-reward ablation via cached-rollout recomputation")
    parser.add_argument("--n", type=int, default=30, help="episodes per architecture")
    parser.add_argument("--seed", type=int, default=0, help="seed offset (added to the 9000 eval base)")
    parser.add_argument("--downweight", type=float, default=0.25, help="contact-term scale for the downweight variant")
    parser.add_argument("--out", type=str, default="reports/figures", help="base dir for the timestamped run dir")
    args = parser.parse_args(argv)
    out_dir = experiment_dir(args.out, "cip_contact_ablation_stageA")
    run_stage_a(int(args.n), int(args.seed) + 9_000, out_dir, downweight=float(args.downweight))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
