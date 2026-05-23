"""Robot intervention policy module.

Evaluates each policy's condition predicate against the current
σ-trajectory + duration history. When a predicate fires, the
matching symbolic action is emitted.

Condition predicate mini-language (parsed by a small recursive
descent — kept restricted on purpose):

    sigma(<cycle_name>) < <number>
    sigma(<cycle_name>) <= <number>
    sigma(<cycle_name>) > <number>
    sigma(<cycle_name>) >= <number>
    <atom> and <atom>
    sustained(<cycle_name>, <int_frames>)

``sustained(c, k)`` evaluates true iff σ(c) has been negative for
the last `k` frames continuously. Combinations are
left-associative AND only — no OR / NOT in the v1 spec; the demo
policies don't need them.

Plan: docs/plans/2026-05-18-rapport-coherence-demo-nagoya/.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from .coalition import Coalition, Policy


# ─── Predicate parser ───────────────────────────────────────────────


_CMP_RE = re.compile(
    r"sigma\(\s*(?P<cyc>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*"
    r"(?P<op><=|>=|<|>)\s*(?P<num>-?\d+(?:\.\d+)?)"
)
_SUS_RE = re.compile(
    r"sustained\(\s*(?P<cyc>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(?P<k>\d+)\s*\)"
)


@dataclass
class PolicyEval:
    """A single time step's policy evaluation result."""
    fired: list[str] = field(default_factory=list)   # policy names that fired
    actions: list[str] = field(default_factory=list) # symbolic actions to dispatch


def _eval_atom(atom: str, sigma: dict[str, float],
               sigma_history: dict[str, deque[float]]) -> bool:
    atom = atom.strip()
    m = _CMP_RE.fullmatch(atom)
    if m:
        cyc = m.group("cyc")
        op = m.group("op")
        num = float(m.group("num"))
        s = sigma.get(cyc, 0.0)
        if op == "<":
            return s < num
        if op == "<=":
            return s <= num
        if op == ">":
            return s > num
        if op == ">=":
            return s >= num
        raise ValueError(f"unknown op {op!r}")
    m = _SUS_RE.fullmatch(atom)
    if m:
        cyc = m.group("cyc")
        k = int(m.group("k"))
        h = sigma_history.get(cyc)
        if h is None or len(h) < k:
            return False
        return all(v < 0 for v in list(h)[-k:])
    raise ValueError(f"unparseable predicate atom: {atom!r}")


def eval_condition(
    condition: str,
    sigma: dict[str, float],
    sigma_history: dict[str, deque[float]],
) -> bool:
    """Evaluate a left-associative AND-chain of atoms."""
    parts = [p.strip() for p in condition.split(" and ")]
    return all(_eval_atom(p, sigma, sigma_history) for p in parts)


# ─── Policy engine ──────────────────────────────────────────────────


class PolicyEngine:
    """Fires policy actions when their conditions evaluate to True.

    Maintains a per-cycle σ history (bounded deque) and a per-policy
    cooldown to avoid firing the same action every frame.

    Cooldown defaults to 20 frames — once a policy fires, it cannot
    re-fire for the next 20 frames. Tunable via ``cooldown_frames``.
    """

    def __init__(
        self,
        coalition: Coalition,
        history_depth: int = 50,
        cooldown_frames: int = 20,
    ) -> None:
        self.coalition = coalition
        self.history_depth = history_depth
        self.cooldown_frames = cooldown_frames
        self.history: dict[str, deque[float]] = {
            c.name: deque(maxlen=history_depth)
            for c in coalition.cycles.values()
        }
        self._last_fired: dict[str, int] = {}  # policy_name → last frame

    def update_history(self, sigma: dict[str, float]) -> None:
        for c_name, s in sigma.items():
            if c_name in self.history:
                self.history[c_name].append(s)

    def step(self, t: int, sigma: dict[str, float]) -> PolicyEval:
        """Update σ history and check every policy at this frame."""
        self.update_history(sigma)
        out = PolicyEval()
        for pol in self.coalition.policies.values():
            last = self._last_fired.get(pol.name, -10_000)
            if t - last < self.cooldown_frames:
                continue
            try:
                fired = eval_condition(pol.condition, sigma, self.history)
            except ValueError:
                continue
            if fired:
                self._last_fired[pol.name] = t
                out.fired.append(pol.name)
                out.actions.append(pol.action)
        return out

    def reset_cooldowns(self) -> None:
        self._last_fired.clear()
