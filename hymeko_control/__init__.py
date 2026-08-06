"""HyMeKo control language + CIP-0 runtime profile (v0).

This package is the SHARED CORE for every embodiment scenario (pick-and-place,
humanoid, AIBO). It is deliberately:

* **scenario-agnostic** -- it imports no ``scenarios.*`` and no coin / delivery /
  theta / galambos / pick-place / humanoid / aibo module;
* **torch-free** -- it imports no ``torch`` and no ``hymeko_rl.*`` runtime, so a
  scenario adapter never inherits a heavy dependency just to speak the contract;
* **stdlib-only** -- only the Python standard library.

Dependency direction is one-way::

    scenario adapter  ->  hymeko_control   (allowed)
    hymeko_control    ->  scenario adapter (FORBIDDEN)

The two layers are:

``hymeko_control.language``
    A declarative language (schema v0 + IR + validator) describing an
    embodiment's entities, morphology, ports, physics, hybrid control and task
    control.

``hymeko_control.cip``
    The CIP-0 runtime lifecycle expressed as typed value types + Protocols::

        OBSERVE -> IDENTIFY MODE -> FORM INTENT -> MEASURE AUTHORITY
        -> DECODE -> EXECUTE OPTION -> MEASURE RESPONSE -> CERTIFY -> TRANSITION
"""

from __future__ import annotations

PROFILE_VERSION = "hymeko-control-profile-v0"
CIP_VERSION = "CIP-0"

__all__ = ["PROFILE_VERSION", "CIP_VERSION"]
