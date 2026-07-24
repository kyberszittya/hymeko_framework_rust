"""§3 — first-class skill routing sourced from the `.hymeko`.

Each phase of ``coin_carry_option_v1.hymeko`` declares WHICH skill runs, whether it is a TRAINABLE upstream option or a
FROZEN downstream skill, its backend binding, and (for the frozen skill) the checkpoint identity/hash and the handoff
certificate. This module parses those attributes and GENERATES the framework `hymeko_rl.option_rl.hierarchy.SkillRoute` from
them — the runtime does not reconstruct the routing in Python. Validation is fail-closed: missing or contradictory
trained/frozen attributes raise, and a frozen skill can never be collected into an optimizer parameter set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hymeko_rl.coin_delivery.coin_carry_fsm import CARRY_OPTION_HYMEKO
from hymeko_rl.env._profile import read_bundle
from hymeko_rl.option_rl.hierarchy import SkillRoute

_ROLES = {"upstream", "downstream", "terminal"}
_TRAIN = {"trainable", "frozen", "none"}


def _attr(body: str, key: str) -> str | None:
    m = re.search(rf"\b{key}\s+([\w.]+)\s*;", body)
    return m.group(1) if m else None


@dataclass(frozen=True)
class SkillBinding:
    """One phase's skill routing, sourced from the `.hymeko` (not reconstructed in Python)."""

    phase: str
    skill_role: str
    training_state: str
    binding: str | None
    handoff_certificate: str | None
    checkpoint: str | None
    checkpoint_sha: str | None

    @property
    def trainable(self) -> bool:
        return self.training_state == "trainable"

    @property
    def frozen(self) -> bool:
        return self.training_state == "frozen"


@dataclass(frozen=True)
class CoinSkillRouting:
    bindings: tuple[SkillBinding, ...]

    def by_phase(self, phase: str) -> SkillBinding:
        return {b.phase: b for b in self.bindings}[phase]

    def skill_of(self, phase: str) -> str | None:
        """The backend binding a phase runs (its skill name), or None for a terminal marker."""
        return self.by_phase(phase).binding

    def upstream_phases(self) -> tuple[str, ...]:
        return tuple(b.phase for b in self.bindings if b.skill_role == "upstream")

    def frozen_phases(self) -> tuple[str, ...]:
        return tuple(b.phase for b in self.bindings if b.frozen)

    def trainable_bindings(self) -> set[str]:
        return {b.binding for b in self.bindings if b.trainable and b.binding}

    def frozen_bindings(self) -> set[str]:
        return {b.binding for b in self.bindings if b.frozen and b.binding}

    def downstream(self) -> SkillBinding:
        ds = [b for b in self.bindings if b.skill_role == "downstream"]
        return ds[0]

    def handoff_certificate(self) -> str:
        return self.downstream().handoff_certificate

    def to_skill_route(self, handed_off, downstream_skill) -> SkillRoute:
        """Generate the framework `SkillRoute` FROM the parsed `.hymeko` routing. Python supplies only the callables
        (``handed_off`` predicate on an option outcome, the frozen ``downstream_skill`` object); the route's identity,
        certificate, and upstream/downstream structure come from the description."""
        return SkillRoute(name=self.handoff_certificate(), handed_off=handed_off, downstream=downstream_skill,
                          upstream_owns_until_handoff=True)

    def manifest(self) -> dict:
        """Checkpoint hash/provenance of the frozen skill — carried into rollout/eval manifests."""
        d = self.downstream()
        return {"handoff_certificate": self.handoff_certificate(), "frozen_skill": d.binding,
                "frozen_checkpoint": d.checkpoint, "frozen_checkpoint_sha": d.checkpoint_sha,
                "trainable_skills": sorted(self.trainable_bindings())}


def validate_routing(bindings, path: str = "<memory>") -> CoinSkillRouting:
    """Fail-closed validation of a parsed skill routing.

    # Errors ``ValueError`` on: a missing role/training_state; unknown role/training_state; a role/training_state
      contradiction (upstream⇒trainable, downstream⇒frozen, terminal⇒none); a downstream skill without a
      handoff_certificate + checkpoint + sha; not exactly one downstream; no upstream."""
    for b in bindings:
        if b.skill_role is None or b.training_state is None:
            raise ValueError(f"{path}: phase {b.phase!r} missing skill_role/training_state (fail-closed)")
        if b.skill_role not in _ROLES or b.training_state not in _TRAIN:
            raise ValueError(f"{path}: phase {b.phase!r} bad skill_role {b.skill_role!r}/training_state {b.training_state!r}")
        bad = ((b.skill_role == "upstream" and b.training_state != "trainable")
               or (b.skill_role == "downstream" and b.training_state != "frozen")
               or (b.skill_role == "terminal" and b.training_state != "none"))
        if bad:
            raise ValueError(f"{path}: phase {b.phase!r} contradictory skill_role {b.skill_role!r} vs training_state {b.training_state!r}")
    ds = [b for b in bindings if b.skill_role == "downstream"]
    if len(ds) != 1:
        raise ValueError(f"{path}: expected exactly one downstream skill, found {[b.phase for b in ds]}")
    if not (ds[0].handoff_certificate and ds[0].checkpoint and ds[0].checkpoint_sha):
        raise ValueError(f"{path}: downstream skill {ds[0].phase!r} needs handoff_certificate + checkpoint + checkpoint_sha")
    if not any(b.skill_role == "upstream" for b in bindings):
        raise ValueError(f"{path}: no upstream (trainable) skill declared")
    return CoinSkillRouting(bindings=tuple(bindings))


def load_carry_skill_routing(path: str = CARRY_OPTION_HYMEKO) -> CoinSkillRouting:
    """Parse the per-phase skill routing from the carry-option `.hymeko` and validate it fail-closed (see
    :func:`validate_routing`)."""
    bindings = [SkillBinding(phase=name, skill_role=_attr(body, "skill_role"), training_state=_attr(body, "training_state"),
                             binding=_attr(body, "binding"), handoff_certificate=_attr(body, "handoff_certificate"),
                             checkpoint=_attr(body, "checkpoint"), checkpoint_sha=_attr(body, "checkpoint_sha"))
                for name, kind, body, _w in read_bundle(path, "controller_spec") if kind == "fsm_phase"]
    return validate_routing(bindings, path)


def skill_binding_trace(routing: CoinSkillRouting, phase_trace) -> tuple:
    """Map an executed phase trace (§1) to its per-step skill binding via the `.hymeko` routing (carry_option / settling_pi0
    / None for terminal markers) — the skill-binding trace is DERIVED from the description, not a parallel Python config."""
    known = {b.phase for b in routing.bindings}
    return tuple(routing.skill_of(p) if p in known else None for p in phase_trace)


def handoff_index(phase_trace) -> int | None:
    """First step at which control handed to the frozen downstream skill (the HANDOFF phase), or None if never."""
    return next((i for i, p in enumerate(phase_trace) if p == "HANDOFF"), None)


def optimizer_parameters(routing: CoinSkillRouting, module_by_binding: dict):
    """Collect optimizer parameters ONLY from the routing's TRAINABLE skills. # Errors ``ValueError`` if a FROZEN skill's
    module is offered for optimization (a frozen skill can never enter an optimizer parameter set)."""
    frozen = routing.frozen_bindings()
    params = []
    for name, module in module_by_binding.items():
        if name in frozen:
            raise ValueError(f"frozen skill {name!r} cannot be placed in an optimizer parameter set")
        if name in routing.trainable_bindings():
            params += list(module.parameters())
    return params
