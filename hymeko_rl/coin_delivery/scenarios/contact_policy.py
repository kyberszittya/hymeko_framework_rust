"""Contact-continuity policies for the coin-delivery scenarios.

A :class:`ContactPolicy` certifies ONLY the shape of the per-step fingertip-contact trace — whether the coin
was driven under a *continuous* grip (C0) or an *intermittent* contact→impulse→free→recontact pattern (C1). It
does **not** decide whether the coin reached the zone: that is the frozen delivery monitor's job, and the two
verdicts are reported separately (reward/monitor separation is binding). A body-shove that reaches the zone is
rejected by the *monitor*, never rescued by a contact policy.

The trace itself (:class:`ContactSample`) is produced by
:func:`hymeko_rl.coin_delivery.monitoring.contact_certification.contact_trace`; :meth:`ContactPolicy.certify`
is a pure function of that trace, testable in isolation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Contact(str, Enum):
    """Per-step fingertip↔coin contact state."""

    BOTH = "both"    # both fingertips touch the coin
    ONE = "one"      # exactly one fingertip touches the coin
    NONE = "none"    # neither fingertip touches the coin


@dataclass(frozen=True)
class ContactSample:
    """One step of the contact trace: which fingertips touched, whether an arm body touched, and the coin's
    toward-zone progress this step (``prev_dtz - dtz``; may be negative). ``state`` derives the tri-state."""

    t: int
    left: bool
    right: bool
    body: bool = False
    progress: float = 0.0

    @property
    def state(self) -> Contact:
        if self.left and self.right:
            return Contact.BOTH
        if self.left or self.right:
            return Contact.ONE
        return Contact.NONE


@dataclass(frozen=True)
class ContactCertification:
    """The contact-continuity verdict for one episode (independent of delivery success)."""

    policy_id: str
    certified: bool
    established: bool                 # the continuity requirement was met at least once
    first_contact_step: int          # step of first establishment, or -1
    max_loss_run: int                # longest run of contact-loss after establishment
    n_contact_phases: int            # number of distinct contact phases (C1 recontacts + 1)
    reason: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _establishment(trace: Sequence[ContactSample], require_both: bool) -> int:
    """Index of the first sample meeting the establishment requirement, or -1."""
    for i, s in enumerate(trace):
        met = (s.state is Contact.BOTH) if require_both else (s.state is not Contact.NONE)
        if met:
            return i
    return -1


def _loss_runs_after(trace: Sequence[ContactSample], start: int, require_both: bool) -> int:
    """Longest run of contact-loss (below the requirement) from ``start`` onward."""
    worst = run = 0
    for s in trace[start:]:
        met = (s.state is Contact.BOTH) if require_both else (s.state is not Contact.NONE)
        run = 0 if met else run + 1
        worst = max(worst, run)
    return worst


def _count_contact_phases(trace: Sequence[ContactSample]) -> int:
    """Number of maximal runs of (any) contact — one phase per uninterrupted touch."""
    phases = 0
    prev = False
    for s in trace:
        touching = s.state is not Contact.NONE
        if touching and not prev:
            phases += 1
        prev = touching
    return phases


class ContactPolicy(ABC):
    """A contact-continuity contract. Concrete policies are frozen (value-equal, serializable)."""

    KIND: str = "abstract"

    @property
    @abstractmethod
    def policy_id(self) -> str:
        """Stable id used in scenario ids and reports."""

    @abstractmethod
    def certify(self, trace: Sequence[ContactSample]) -> ContactCertification:
        """Judge whether ``trace`` satisfies this policy's continuity requirement.

        # Preconditions ``trace`` is a time-ordered sequence of :class:`ContactSample`.
        # Postconditions returns a :class:`ContactCertification`; never inspects delivery success."""

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-serializable form (carries ``kind`` + params for :func:`contact_policy_from_dict`)."""


@dataclass(frozen=True)
class C0Continuous(ContactPolicy):
    """C0 — CONTINUOUS grip. Both fingertips must contact the coin and stay in contact: after the first
    both-contact establishment, no run of contact-loss may exceed ``grace_steps``. Never establishing → not
    certified (continuity requires contact)."""

    KIND = "c0_continuous"
    grace_steps: int = 2
    require_both: bool = True

    @property
    def policy_id(self) -> str:
        return "c0"

    def certify(self, trace: Sequence[ContactSample]) -> ContactCertification:
        start = _establishment(trace, self.require_both)
        phases = _count_contact_phases(trace)
        if start < 0:
            return ContactCertification(self.policy_id, False, False, -1, len(trace), phases,
                                        "contact never established")
        worst = _loss_runs_after(trace, start, self.require_both)
        ok = worst <= self.grace_steps
        reason = "continuous grip within grace" if ok else f"contact lost for {worst} > {self.grace_steps} steps"
        return ContactCertification(self.policy_id, ok, True, start, worst, phases, reason)

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "grace_steps": self.grace_steps, "require_both": self.require_both}


@dataclass(frozen=True)
class C1Intermittent(ContactPolicy):
    """C1 — INTERMITTENT contact. Continuity is NOT required: the policy accepts
    CONTACT→IMPULSE→FREE→RECONTACT→CORRECTION→DELIVERY. It certifies as long as the coin was actually touched
    at least ``min_contact_phases`` time(s) (a pure free-motion / knock trace has no impulse and is not
    certified). It never judges delivery success — a body-shove is rejected by the delivery monitor, not here."""

    KIND = "c1_intermittent"
    min_contact_phases: int = 1

    @property
    def policy_id(self) -> str:
        return "c1"

    def certify(self, trace: Sequence[ContactSample]) -> ContactCertification:
        start = _establishment(trace, require_both=False)
        phases = _count_contact_phases(trace)
        ok = phases >= self.min_contact_phases
        reason = (f"intermittent contact ok ({phases} phase(s))" if ok
                  else f"no contact impulse ({phases} < {self.min_contact_phases} phases)")
        return ContactCertification(self.policy_id, ok, start >= 0, start,
                                    _loss_runs_after(trace, max(start, 0), require_both=True), phases, reason)

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "min_contact_phases": self.min_contact_phases}


_POLICY_KINDS: dict[str, type[ContactPolicy]] = {C0Continuous.KIND: C0Continuous, C1Intermittent.KIND: C1Intermittent}


def contact_policy_from_dict(d: dict) -> ContactPolicy:
    """Reconstruct a :class:`ContactPolicy` from :meth:`ContactPolicy.to_dict`.

    # Preconditions ``d['kind']`` is a registered policy kind. # Postconditions round-trips to an equal policy."""
    kind = d.get("kind")
    cls = _POLICY_KINDS.get(str(kind))
    if cls is None:
        raise ValueError(f"unknown contact-policy kind {kind!r}")
    params = {k: v for k, v in d.items() if k != "kind"}
    return cls(**params)
