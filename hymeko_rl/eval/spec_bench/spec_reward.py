"""Close the loop: an arbitrated HTL success spec becomes a per-step reward that *drives* a MetaWorld run.

The ``spec_bench`` gate produces an arbitrated HTL success formula ``phi*`` (e.g. coffee-push
``F(obj_to_target <= 0.071)``). Its quantitative semantics is the robustness ``rho`` — a signed geometric
margin — so the same formula yields *both* the dense per-step reward (``rho``) and the monitor verdict
(``sign rho``). This module is the seam between the *grading* pipeline (``spec_bench``) and the *execution*
pipeline (``exp_metaworld_spec_reward_ab`` / ``train_sac``):

* :func:`signals_from_metaworld_info` maps a MetaWorld ``info`` dict to the HTL signal names a spec reads
  (the same names ``metaworld_rollouts`` records, so a spec fit on saved traces is applied unchanged live);
* :class:`SpecRewardEnv` is a reward-override wrapper (mirrors ``MonitorAlignedEnv`` /
  ``HymekoRewardMetaWorld`` — §6.1: no re-implemented wrapper) whose step reward is the instantaneous
  robustness of ``phi*`` via the existing :class:`~hymeko_rl.control.htl_reward.HtlRewardSpec` adapter (no
  re-implemented logic engine, no re-implemented ``robustness_at``);
* :func:`spec_reward_separation` is the offline reward-quality metric — does the spec's per-episode return
  rank native success above failure, and by how much? This is the de-risk that must pass before any RL: a
  reward that does not separate success from failure cannot drive learning (CLAUDE.md §3 oracle-certify).

No training here; :class:`SpecRewardEnv` is a plain reward swap. The instantaneous ``rho`` is Markovian
(``G``/``F`` collapse to their leaf at a single event), so it is valid for an off-policy replay buffer; the
genuinely temporal reading of the same formula is the per-episode verdict (:meth:`SpecRewardEnv.verdict`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hymeko_rl.control.htl_reward import HtlRewardSpec
from hymeko_rl.eval.spec_bench.spec_bench import Rollout

# The canonical coffee-push specs (the discriminating A/B pair — see reports/2026-07-13-coffee-push-*).
ARBITRATED_COFFEE_SPEC = "F(obj_to_target <= 0.071)"                    # gate/formal, F1 1.0
RAW_COFFEE_SPEC = ("F(near_object >= 0.5 AND grasp_success >= 0.8 "
                   "AND in_place >= 0.9 AND obj_to_target <= 0.1)")     # weak-model raw, F1 0.0

# MetaWorld info key -> HTL signal name (the same aliasing metaworld_rollouts.py records; §6.1 single source).
_INFO_TO_SIGNAL: dict[str, str] = {
    "near_object": "near_object",
    "grasp_success": "grasp_success",
    "in_place_reward": "in_place",      # the recorded 'in_place' signal IS MetaWorld's in_place_reward
    "obj_to_target": "obj_to_target",
    "success": "success",
}


def signals_from_metaworld_info(info: "Mapping[str, Any]") -> "dict[str, float]":
    """Map a MetaWorld ``info`` dict to the HTL scalar signals a spec_bench spec reads.

    # Preconditions ``info`` is a MetaWorld V3 step-info dict (keys in :data:`_INFO_TO_SIGNAL` may be absent
      early; missing keys default to 0.0). # Postconditions every value is a finite float; the returned dict's
      keys are the HTL signal-name domain (``near_object``/``grasp_success``/``in_place``/``obj_to_target``/
      ``success``), matching the names the arbitrated spec was fit against."""
    return {sig: float(info.get(key, 0.0)) for key, sig in _INFO_TO_SIGNAL.items()}


def _signals_from_env(env: "SpecRewardEnv") -> "dict[str, float]":
    """HtlRewardSpec signal extractor for a :class:`SpecRewardEnv` — reads the wrapper's last step-info.

    Mirrors ``htl_reward.signals_from_planar(env)`` reading ``env._planar_metrics`` (same convention; the
    adapter's ``SignalFn`` takes the env and pulls the live metrics off it)."""
    return signals_from_metaworld_info(env.last_info)


class SpecRewardEnv:
    """Reward-override: the step reward is the instantaneous robustness of an HTL success spec over ``info``.

    Duck-typed (``reset``/``step``/spaces delegate) so any policy loop or ``train_sac`` drives it, exactly like
    :class:`~hymeko_rl.eval.cip.monitor_aligned_reward.MonitorAlignedEnv`. The MetaWorld env reward is preserved
    in ``info['env_reward']``; the spec reward in ``info['spec_reward']``; the boolean monitor verdict of the
    step in ``info['spec_satisfied']``.

    # Preconditions the wrapped env's step ``info`` exposes the MetaWorld component keys; ``formula`` parses.
    # Postconditions ``step`` returns a finite scalar equal to ``rho`` (or, if ``potential``, the potential-based
      shaping ``gamma*rho_t - rho_{t-1}`` which telescopes to ``gamma``-weighted ``rho_T - rho_0``).
    # Invariants no module-level state is mutated; signals travel through ``event.scalar_signals`` (§6.5 #11).
    """

    def __init__(self, env: Any, formula: str, *, potential: bool = False, gamma: float = 1.0) -> None:
        self.env = env
        self.formula = formula
        self._reward = HtlRewardSpec(formula, signals=_signals_from_env, verdict_formula=formula)
        self._potential = potential
        self._gamma = float(gamma)
        self.last_info: "dict[str, float]" = {}
        self._rho_prev: float = 0.0

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def _rho(self, action: Any) -> float:
        """Instantaneous robustness of the spec at the current ``last_info`` (delegates to HtlRewardSpec)."""
        return float(self._reward.evaluate(self, 0.0, action))

    def reset(self, **kw: Any) -> Any:
        obs, info = self.env.reset(**kw)
        self.last_info = dict(info) if isinstance(info, Mapping) else {}
        self._rho_prev = self._rho(None)     # rho at the reset state (potential baseline)
        return obs, info

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = dict(info)
        rho = self._rho(action)
        if self._potential:
            r = self._gamma * rho - self._rho_prev
            self._rho_prev = rho
        else:
            r = rho
        out_info = {**info, "env_reward": float(reward), "spec_reward": float(rho),
                    "spec_satisfied": bool(rho > 0.0)}
        return obs, float(r), terminated, truncated, out_info

    def verdict(self, *, horizon: int = 1024) -> Any:
        """A fresh HTL monitor for the temporal reading of the same formula (per-episode accountability, not the
        per-step reward). Feed it one :meth:`event` per step in non-decreasing time order."""
        return self._reward.episode_monitor(horizon=horizon)

    def event(self, t: float) -> Any:
        """A timestamped :class:`HypergraphEvent` at the current state, for feeding :meth:`verdict`."""
        return self._reward.event(self, t)


# ── offline reward-quality metric (the de-risk that gates RL) ────────────────────────────────────────────────
def _episode_spec_return(node: Any, rollout: Rollout) -> float:
    """Sum of the spec's instantaneous robustness over a rollout's trace (the per-episode spec-return)."""
    from hymeko_neuro.eval.htl import robustness_at
    return float(sum(robustness_at(node, ev) for ev in rollout.events()))


def _auc(pos: "Sequence[float]", neg: "Sequence[float]") -> float:
    """ROC-AUC = P(x_pos > x_neg) via the Mann–Whitney rank form; 0.5 on ties/empty, no sklearn dep.

    Positives are concatenated first, so ``ranks[:n_pos]`` are their (average, tie-corrected) ranks."""
    if not pos or not neg:
        return 0.5
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = np.argsort(allv, kind="stable")
    sorted_v = allv[order]
    ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0     # 1-based average rank for the tie block
        i = j + 1
    n_pos = len(pos)
    rank_pos_sum = float(ranks[:n_pos].sum())
    return (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * len(neg))


@dataclass(frozen=True)
class SpecRewardQuality:
    """How well a spec's per-episode return separates native success from failure (the RL-drivability proxy)."""

    formula: str
    n: int
    n_success: int
    mean_return_success: float
    mean_return_failure: float
    separation: float          # mean_success − mean_failure (reward-gradient signal; larger = more drivable)
    point_biserial: float      # corr(per-episode return, success label) in [-1, 1]
    auc: float                 # P(return_success > return_failure); 0.5 = no ranking, 1.0 = perfect

    def as_dict(self) -> "dict[str, Any]":
        return {"formula": self.formula, "n": self.n, "n_success": self.n_success,
                "mean_return_success": round(self.mean_return_success, 4),
                "mean_return_failure": round(self.mean_return_failure, 4),
                "separation": round(self.separation, 4), "point_biserial": round(self.point_biserial, 4),
                "auc": round(self.auc, 4)}


def spec_reward_separation(formula: str, rollouts: "Sequence[Rollout]") -> SpecRewardQuality:
    """Offline reward-quality of ``formula`` on labelled ``rollouts``: does its per-episode spec-return rank
    success above failure, and by how much?

    This is the offline oracle-certify (CLAUDE.md §3): a reward whose ``separation``/``auc`` is $\\approx$ chance
    cannot drive RL, regardless of a boolean "ranks success higher". # Preconditions ``formula`` parses; both
    classes present for a meaningful ``point_biserial``/``auc``. # Postconditions finite fields; ``auc`` in
    [0, 1]."""
    from hymeko_neuro.eval.htl import parse
    node = parse(formula)
    ret = np.asarray([_episode_spec_return(node, r) for r in rollouts], dtype=np.float64)
    labels = np.asarray([1.0 if r.success else 0.0 for r in rollouts], dtype=np.float64)
    pos = ret[labels > 0.5]
    neg = ret[labels <= 0.5]
    mean_s = float(pos.mean()) if pos.size else float("nan")
    mean_f = float(neg.mean()) if neg.size else float("nan")
    pb = 0.0 if float(np.std(ret)) < 1e-12 or float(np.std(labels)) < 1e-12 \
        else float(np.corrcoef(ret, labels)[0, 1])
    return SpecRewardQuality(
        formula=formula, n=len(rollouts), n_success=int(labels.sum()),
        mean_return_success=mean_s, mean_return_failure=mean_f,
        separation=(mean_s - mean_f) if (pos.size and neg.size) else float("nan"),
        point_biserial=pb, auc=_auc(list(pos), list(neg)))
