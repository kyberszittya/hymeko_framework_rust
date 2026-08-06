"""The FANUC pick-place demonstrator as a HyMeKo **hybrid automaton** — the in-memory binding of
``pick_place_controller.hymeko``.

A hybrid system is a continuous flow punctuated by discrete mode switches. The discrete layer (modes + guarded
transitions) is read straight from the ``.hymeko`` by :class:`~hymeko_rl.control.controller_spec.ControllerSpec`
(reused, not re-implemented); this module binds the named **guards** to STATE PREDICATES on the pick-place env, so
the automaton can be *walked* over a trajectory (mode detection) and its Markov structure exposed.

Two views of the same automaton:
- :meth:`HybridAutomaton.walk` — the declarative FSM (``ControllerSpec.step``, phase memory): first-guard-wins,
  starting at ``reach``. The ground-truth mode sequence of a demonstrator rollout.
- :meth:`HybridAutomaton.classify` — the **Markov** mode from state ALONE (no phase memory), the constraint_mode
  pattern. Defined for every mode EXCEPT ``reach`` (whose clean transit needs a leading command — a purely
  state-derived reach sags into the table, measured). :attr:`MARKOV_MODES` names the state-classifiable modes;
  those are the ones a lock-step DAgger can relabel.

Design by contract: guards are pure functions of :class:`PickState`; a guard reads state, never mutates it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hymeko_rl.control.controller_spec import ControllerSpec

# The modes whose active-ness is a pure function of the observed state (constraint_mode-style) — hence
# DAgger-relabelable. `reach` is excluded: its clean transit is non-Markov (needs a leading command).
MARKOV_MODES: tuple[str, ...] = ("descend", "grasp", "lift", "transport", "place", "release", "retract")


@dataclass(frozen=True)
class PickState:
    """The pick-place state the guards read — extractable from a live env OR a recorded trajectory (so the same
    guards drive execution and detection). All lengths in metres; ``grasp_hold`` in steps.

    # Invariants a pure snapshot; guards never mutate it.
    """

    obj: np.ndarray            # object xyz
    tool: np.ndarray           # tool (wrist) xyz
    both_contact: bool         # both fingers touch the object
    grasp_hold: int            # consecutive steps both fingers have held
    target: np.ndarray         # target xy
    surf: float                # work-surface z (object rests at surf + box_zhalf)
    box_zhalf: float           # object half-height
    grasp_z: float             # tool z at the grasp pose (surf + box_half + _GRASP_TOOL_DZ)
    obj_vel: float             # object linear speed (for the at-rest / placed guard)

    @property
    def horiz_to_obj(self) -> float:
        return float(np.hypot(self.tool[0] - self.obj[0], self.tool[1] - self.obj[1]))

    @property
    def horiz_obj_to_target(self) -> float:
        return float(np.hypot(self.obj[0] - self.target[0], self.obj[1] - self.target[1]))

    @property
    def lifted(self) -> float:
        return float(max(0.0, self.obj[2] - (self.surf + self.box_zhalf)))

    @classmethod
    def from_env(cls, env: Any) -> "PickState":
        """# Preconditions ``env`` is a stepped/forwarded ``PickPlaceEnv``. # Postconditions a snapshot; no mutation."""
        # COPY every array — env.data.xpos rows are live views that mutate each step; a stored snapshot must not
        # alias the buffer (else every recorded state reads the final pose).
        obj = np.array(env._obj_xyz(), dtype=np.float64)
        left, right = env._both_contact()
        vadr = getattr(env, "_obj_vadr", None)
        vel = float(np.linalg.norm(env.data.qvel[vadr:vadr + 3])) if vadr is not None else 0.0
        return cls(
            obj=obj, tool=np.array(env.data.xpos[env._b_tool], dtype=np.float64),
            both_contact=bool(left and right), grasp_hold=int(env._grasp_hold),
            target=np.array(env.target_xy, dtype=np.float64), surf=float(env._surf),
            box_zhalf=float(env._box_zhalf),
            grasp_z=float(env._surf + env.box_half + env._GRASP_TOOL_DZ), obj_vel=vel)


# --- guard registry: guard-name -> predicate(state, params) -> bool (the `on <guard> to <phase>` events) ---
def _guards() -> "dict[str, Callable[[PickState, dict[str, float]], bool]]":
    def centered_over_object(s: PickState, p: "dict[str, float]") -> bool:
        return s.horiz_to_obj < p["over_horiz"]
    def tool_at_grasp(s: PickState, p: "dict[str, float]") -> bool:
        return s.tool[2] <= s.grasp_z + p["grasp_tol"]
    def grasp_settled(s: PickState, p: "dict[str, float]") -> bool:
        return s.both_contact and s.grasp_hold >= int(p["grasp_dwell"])
    def object_lost(s: PickState, _p: "dict[str, float]") -> bool:
        return not s.both_contact
    def object_lifted(s: PickState, _p: "dict[str, float]") -> bool:
        return s.lifted > 0.04
    def over_target(s: PickState, p: "dict[str, float]") -> bool:
        return s.horiz_obj_to_target < p["place_tol"]
    def object_on_surface(s: PickState, p: "dict[str, float]") -> bool:
        return s.obj[2] <= s.surf + s.box_zhalf + p["surface_tol"]
    def gripper_released(s: PickState, _p: "dict[str, float]") -> bool:
        return not s.both_contact
    return {"centered_over_object": centered_over_object, "tool_at_grasp": tool_at_grasp,
            "grasp_settled": grasp_settled, "object_lost": object_lost, "object_lifted": object_lifted,
            "over_target": over_target, "object_on_surface": object_on_surface,
            "gripper_released": gripper_released}


@dataclass(frozen=True)
class HybridAutomaton:
    """Pick-place hybrid automaton = the ``.hymeko`` FSM (:class:`ControllerSpec`) + bound state-predicate guards.

    # Preconditions built from ``pick_place_controller.hymeko`` (or a compatible controller profile).
    # Invariants ``guards`` covers every transition event named in ``spec``.
    """

    spec: ControllerSpec
    guards: "dict[str, Callable[[PickState, dict[str, float]], bool]]"

    @classmethod
    def from_hymeko(cls, profile: "str | Path" = "data/robotics/pick_place_controller.hymeko") -> "HybridAutomaton":
        spec = ControllerSpec.from_hymeko(profile)
        guards = _guards()
        missing = {e for ph in spec.phases for e, _d in ph.transitions} - set(guards)
        if missing:
            raise ValueError(f"{profile}: unbound guard event(s) {sorted(missing)}")
        return cls(spec=spec, guards=guards)

    def fires(self, event: str, state: PickState) -> bool:
        """Evaluate a named guard on ``state`` (# Errors ``KeyError`` if the event is unbound)."""
        return bool(self.guards[event](state, self.spec.params))   # normalise numpy bool → Python bool

    def walk(self, states: "list[PickState]") -> "list[str]":
        """The FSM mode at each step of a trajectory (``ControllerSpec.step``, phase memory; starts at
        ``spec.initial``). This is the ground-truth mode segmentation of a demonstrator rollout.

        # Postconditions ``len(out) == len(states)``; ``out[0] == spec.initial``.
        """
        phase = self.spec.initial
        out: "list[str]" = []
        for s in states:
            out.append(phase)
            phase, _ev = self.spec.step(phase, lambda e: self.fires(e, s))
        return out

    def classify(self, s: PickState) -> str:
        """The **Markov** mode from state alone (no phase memory) — defined for :data:`MARKOV_MODES`; ``reach`` is
        returned only as the residual (not state-classifiable). The constraint_mode-style decision tree.

        # Postconditions returns a mode name in ``spec`` phase set.
        """
        p = self.spec.params
        # RETRACT vs REACH are indistinguishable from instantaneous state EXCEPT that after a place the object sits
        # AT THE TARGET on the surface, released — so that (not the object's velocity) is what resolves it; requiring
        # at-rest would mis-classify the settle transient (object still moving) as `reach`.
        placed = (self.guards["over_target"](s, p) and self.guards["object_on_surface"](s, p)
                  and not s.both_contact)
        if placed:
            return "retract"
        if s.both_contact:
            if self.guards["over_target"](s, p):
                return "release" if self.guards["object_on_surface"](s, p) else "place"
            if self.guards["object_lifted"](s, p):
                return "transport"
            return "lift" if s.grasp_hold >= int(p["grasp_dwell"]) else "grasp"
        # not grasped: over the object + low → descending to grasp; else the non-Markov reach.
        if self.guards["centered_over_object"](s, p) and self.guards["tool_at_grasp"](s, p):
            return "descend"
        return "reach"

    def describe(self) -> "dict[str, Any]":
        """A JSON-serialisable dump of the automaton (modes, laws, guarded transitions, params, Markov split) — the
        HyMeKo hybrid-system description, for a report or `hymeko inspect`-style view."""
        return {
            "initial": self.spec.initial,
            "modes": [{"mode": ph.name, "law": ph.law,
                       "transitions": [{"on": e, "to": d} for e, d in ph.transitions],
                       "markov": ph.name in MARKOV_MODES} for ph in self.spec.phases],
            "params": dict(self.spec.params),
            "markov_modes": list(MARKOV_MODES),
            "non_markov_modes": [ph.name for ph in self.spec.phases if ph.name not in MARKOV_MODES]}
