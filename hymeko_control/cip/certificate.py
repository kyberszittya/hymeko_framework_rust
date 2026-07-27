"""CIP-0 --- external certificates.

A certificate is an EXTERNAL, reward-independent predicate over ``(state,
trace)``. Reward never appears in a certificate signature: success and safety
are judged by physics/geometry, not by whatever objective an RL run happens to
optimise. Certificates are composable value types (:func:`all_of`,
:func:`any_of`), and safety DOMINATES success -- a suite fails if any safety
certificate fails, regardless of the success certificates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .._frozen import freeze_mapping
from .option import ResponseTrace
from .structured_state import StructuredStateLike
from ..language.schema_v0 import CertificateKind

#: A predicate reads only state + trace. The absence of a reward parameter is
#: the machine-checkable guarantee of reward-independence.
PredicateFn = Callable[[StructuredStateLike, ResponseTrace], bool]


@runtime_checkable
class Predicate(Protocol):
    """A named, kinded certificate predicate over ``(state, trace)``."""

    name: str
    kind: CertificateKind

    def evaluate(self, state: StructuredStateLike, trace: ResponseTrace) -> bool: ...


@dataclass(frozen=True)
class Certificate:
    """A composable external certificate.

    # Invariants
    ``fn`` depends only on ``(state, trace)``; it must not consult any reward or
    mutate its arguments.
    """

    name: str
    kind: CertificateKind
    fn: PredicateFn

    def evaluate(self, state: StructuredStateLike, trace: ResponseTrace) -> bool:
        return bool(self.fn(state, trace))


def all_of(name: str, kind: CertificateKind, *certs: Certificate) -> Certificate:
    """Conjunction certificate: passes iff every ``cert`` passes."""

    def _fn(state: StructuredStateLike, trace: ResponseTrace) -> bool:
        return all(c.evaluate(state, trace) for c in certs)

    return Certificate(name, kind, _fn)


def any_of(name: str, kind: CertificateKind, *certs: Certificate) -> Certificate:
    """Disjunction certificate: passes iff any ``cert`` passes."""

    def _fn(state: StructuredStateLike, trace: ResponseTrace) -> bool:
        return any(c.evaluate(state, trace) for c in certs)

    return Certificate(name, kind, _fn)


@dataclass(frozen=True)
class CertificateResult:
    """The verdict of evaluating a certificate suite.

    ``passed`` is the overall verdict (safety-dominant). ``per_certificate``
    records each certificate's boolean outcome for auditing.
    """

    passed: bool
    success_passed: bool
    safety_passed: bool
    per_certificate: Mapping[str, bool] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_certificate", freeze_mapping(self.per_certificate))
        object.__setattr__(self, "details", freeze_mapping(self.details))


@dataclass(frozen=True)
class CertificateSuite:
    """A set of success + safety certificates evaluated together."""

    certificates: tuple[Certificate, ...]

    def evaluate(
        self, state: StructuredStateLike, trace: ResponseTrace
    ) -> CertificateResult:
        """Evaluate all certificates. Safety dominates success.

        # Postconditions
        ``passed`` iff every safety certificate passes AND every success
        certificate passes. Reward is not an input.
        """
        per: dict[str, bool] = {}
        success_ok = True
        safety_ok = True
        for cert in self.certificates:
            ok = cert.evaluate(state, trace)
            per[cert.name] = ok
            if cert.kind == CertificateKind.SAFETY:
                safety_ok = safety_ok and ok
            else:
                success_ok = success_ok and ok
        return CertificateResult(
            passed=safety_ok and success_ok,
            success_passed=success_ok,
            safety_passed=safety_ok,
            per_certificate=per,
        )
