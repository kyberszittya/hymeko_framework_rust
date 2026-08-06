"""Mechanism-classification adapter over the frozen mechanism classifier.

The frozen :class:`~hymeko_rl.train.coin_delivery_actor.Mechanism` taxonomy and its clean/unclean partition are
authoritative; this module only surfaces them for the runners (which read a :class:`DeliveryResult`, whose
``mechanism`` field is already the frozen classifier's output). No reclassification happens here.
"""
from __future__ import annotations

from hymeko_rl.train.coin_delivery_actor import Mechanism, DeliveryResult

# The frozen strict monitor's clean-mechanism set (mirrors _valid_delivery's `clean` predicate).
CLEAN_MECHANISMS: frozenset[Mechanism] = frozenset({
    Mechanism.SYM_PUSH, Mechanism.VPLOW, Mechanism.ASYM_PUSH, Mechanism.SETUP_PUSH, Mechanism.RECOVERY_PUSH,
})


def mechanism_of(result: DeliveryResult) -> Mechanism:
    """The frozen mechanism classification carried by ``result`` (as the :class:`Mechanism` enum)."""
    return Mechanism(result.mechanism)


def is_clean(mechanism: Mechanism) -> bool:
    """Whether ``mechanism`` is a clean delivery mechanism (not bulldoze / shove / free / stall / invalid)."""
    return mechanism in CLEAN_MECHANISMS
