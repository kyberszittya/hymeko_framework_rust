"""HTL-robustness as the reward — the declarative temporal-geometric reward adapter.

The thesis (proof-of-concept on the galambos collaborative task): the quantitative semantics of an
STL/HTL predicate ``x >= θ`` is its robustness ``ρ = v(x) − θ`` — a **signed geometric margin**. So a
single declared formula yields *both* the dense training signal (``ρ``) *and* the monitor verdict
(``sign ρ``), collapsing reward shaping (`reward.py`) and runtime accountability (`hymeko_monitor` / HTL)
into one artifact. See ``docs/plans/2026-06-26-htl-reward-poc/``.

This module is a thin **adapter**: it reuses the existing non-core HTL evaluator under
``signedkan_wip/src/htl`` (``parse`` / ``robustness_at`` / ``HtlMonitor`` / ``HypergraphEvent``; operators
``AND/OR/NOT/G[a,b]/F[a,b]`` over ``ScalarPred``, robust-STL min/max semantics). It does **not**
re-implement the logic engine. The bridge to that package mirrors the established
``hymeko_ros2_demo/.../dashboard_node.py`` pattern (add ``signedkan_wip/src`` to ``sys.path``). The
module-level predicate *registry* in that package is **not** used (CLAUDE.md §6.5 #11): metrics are passed
through ``event.scalar_signals``, which ``signal_value`` resolves directly — the adapter mutates no global.

The per-step reward is the **instantaneous** robustness (``robustness_at``): temporal ``G``/``F`` collapse
to their leaf at a single event, so the reward is a **Markovian** min/max composition of geometric margins
— valid for an off-policy replay buffer. The genuinely temporal ``ρ`` (e.g. ``F[0,T](in_zone)`` over the
whole rollout) is non-Markovian (a reward machine) and is exposed only as the per-episode *verdict*
(:meth:`HtlRewardSpec.episode_monitor`), not as the reward (that is the FSM-structured-RL line,
``docs/plans/2026-06-23-fsm-structured-rl/``, deliberately out of scope here).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]   # hymeko_rl/control/htl_reward.py -> repo root
# Bridge to the non-core HTL evaluator (same pattern as the ros2 dashboard). Insert once, only if absent.
_HTL_SRC = _REPO / "signedkan_wip" / "src"
if str(_HTL_SRC) not in sys.path:
    sys.path.insert(0, str(_HTL_SRC))
try:
    from htl import HtlMonitor, HypergraphEvent, parse, robustness_at  # type: ignore[import-not-found]
except ImportError as exc:  # a clear failure at import time, not deep inside a step
    raise ImportError(
        f"HTL evaluator not importable from {_HTL_SRC} — expected signedkan_wip/src/htl. "
        f"Original error: {exc}") from exc

if TYPE_CHECKING:
    from htl import HtlNode

_DEFAULT_SPEC = _REPO / "data" / "robotics" / "galambos_spec.htl"
# The per-episode delivery verdict (temporal): did the coin ever reach the zone? Reported, not rewarded.
_DELIVERY_VERDICT = "F[0,200](in_zone > 0.5)"

SignalFn = Callable[[Any], "dict[str, float]"]


def _strip_comments(text: str) -> str:
    """Drop ``#`` comment lines and blanks, join the rest with spaces (the HTL tokenizer is
    whitespace-insensitive, so a multi-line formula collapses cleanly). # Postconditions non-empty."""
    body = " ".join(line.split("#", 1)[0].strip()
                     for line in text.splitlines()).strip()
    if not body:
        raise ValueError("HTL spec is empty after stripping comments")
    return body


def signals_from_planar(env: Any) -> dict[str, float]:
    """Map a :class:`~hymeko_rl.env.planar_grasp_env.PlanarGraspEnv`'s live metrics to the HTL scalar
    signals the spec reads — all in the metric units the margins are written against.

    # Preconditions ``env`` exposes ``_planar_metrics`` (a ``PlanarGraspMetrics``).
    # Postconditions every key is a finite float; binary signals are in ``{0.0, 1.0}``."""
    m = env._planar_metrics
    return {
        "disk_to_zone": float(m.disk_to_zone),
        "approach_l": float(m.left_tip_dist),
        "approach_r": float(m.right_tip_dist),
        "disk_speed": float(m.disk_speed),
        "in_zone": 1.0 if m.in_zone else 0.0,
        "both_contact": 1.0 if (m.left_contact and m.right_contact) else 0.0,
        "disk_oob": 1.0 if getattr(env, "_disk_out", False) else 0.0,
        "arm_self_contact": 1.0 if m.arm_self_contact else 0.0,
    }


class HtlRewardSpec:
    """A drop-in reward whose scalar value is an HTL formula's instantaneous robustness ``ρ``.

    Duck-types :meth:`hymeko_rl.env.reward.RewardSpec.evaluate` — ``evaluate(env, dist, action) -> float`` —
    so it slots into ``PlanarGraspEnv(reward_spec=…)`` (or ``env.reward_spec = …``) with no env change.

    # Preconditions the formula's signal names ⊆ the ``signals`` extractor's keys.
    # Invariants stateless across calls; no module-level state is mutated (the global HTL registry is
      bypassed — signals travel in ``event.scalar_signals``).
    """

    def __init__(self, formula: str | None = None, *, signals: SignalFn = signals_from_planar,
                 verdict_formula: str = _DELIVERY_VERDICT) -> None:
        text = formula if formula is not None else _DEFAULT_SPEC.read_text(encoding="utf-8")
        self.formula_text = _strip_comments(text)
        self._node: HtlNode = parse(self.formula_text)
        self._signals = signals
        self.verdict_formula = verdict_formula

    def evaluate(self, env: Any, dist: float, action: np.ndarray) -> float:
        """Instantaneous robustness ``ρ`` of the formula at the env's current state (the per-step reward).

        # Postconditions finite; ``> 0`` iff every conjunct's geometric margin is positive at this state."""
        event = HypergraphEvent(t=0.0, scalar_signals=self._signals(env))
        return float(robustness_at(self._node, event))

    def event(self, env: Any, t: float) -> "HypergraphEvent":
        """A timestamped :class:`HypergraphEvent` for feeding an :meth:`episode_monitor` over a rollout."""
        return HypergraphEvent(t=float(t), scalar_signals=self._signals(env))

    def episode_monitor(self, *, horizon: int = 1024) -> "HtlMonitor":
        """A fresh :class:`HtlMonitor` for the temporal delivery verdict (the accountability signal,
        scored per episode — NOT the per-step reward)."""
        return HtlMonitor(self.verdict_formula, horizon=horizon)
