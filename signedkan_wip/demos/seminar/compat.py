"""Checkpoint module-path compatibility for the seminar demos.

Committed HSiKAN checkpoints were pickled when ``SignedKAN*`` classes lived at
``signedkan_wip.src.signedkan``; the module has since moved to
``signedkan_wip.src.core.signedkan`` (``demo/checkpoint.py`` warns that renames
break the pickle). Rather than re-save every checkpoint, register the legacy
import path as an alias of the current module before ``torch.load`` runs — the
Adapter pattern (CLAUDE.md §7), kept in the non-core demo layer so the shared
loader is untouched (§1: workaround in non-core code preferred).

Idempotent: re-registering an already-present alias is a no-op.
"""
from __future__ import annotations

import importlib
import sys

# legacy pickled module path -> current module providing the same symbols.
_LEGACY_MODULE_ALIASES: dict[str, str] = {
    "signedkan_wip.src.signedkan": "signedkan_wip.src.core.signedkan",
}


def register_legacy_checkpoint_aliases() -> list[str]:
    """Install legacy module aliases needed to unpickle committed checkpoints.

    Postconditions: every key of ``_LEGACY_MODULE_ALIASES`` resolves in
    ``sys.modules`` to the live current module. Returns the aliases newly
    installed this call (empty if all were already present).
    """
    installed: list[str] = []
    for legacy, current in _LEGACY_MODULE_ALIASES.items():
        if legacy in sys.modules:
            continue
        sys.modules[legacy] = importlib.import_module(current)
        installed.append(legacy)
    return installed


__all__ = ["register_legacy_checkpoint_aliases"]
