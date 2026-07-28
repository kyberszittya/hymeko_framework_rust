"""Rollout provenance and success certificates, with a deterministic content hash that links outcomes to their rollout.

:class:`RolloutProvenance` bundles everything needed to reproduce and audit a single episode: the git SHA, the seed, the
initial-condition certificate, the coin/target poses, the mode trace, the transition count, and whether the measured
energy ledger was complete. :meth:`content_hash` is a stable SHA-256 over a canonical, float-rounded serialization — the
same rollout always hashes identically, and any change to a recorded field changes the hash. :class:`SuccessCertificate`
(e.g. strict-K6) carries that hash so a success claim is inseparably linked to the provenance that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from hymeko_rl.ir.hybrid_mode import ModeTrace
from hymeko_rl.ir.initial_condition import InitialConditionCertificate

_ROUND = 9  # decimal places for float canonicalization — matches the IR's 1e-9 tolerances


def _canon(pose: "NDArray[np.float64] | None") -> "list[float] | None":
    return None if pose is None else [round(float(x), _ROUND) for x in np.asarray(pose, dtype=np.float64).ravel()]


@dataclass(frozen=True)
class RolloutProvenance:
    """Reproducibility record for one episode. All poses are stored rounded for a stable hash."""

    git_sha: str
    seed: int
    ic_certificate: InitialConditionCertificate
    coin_pose: NDArray[np.float64]
    target_pose: Optional[NDArray[np.float64]]
    mode_trace: ModeTrace
    n_transitions: int
    energy_ledger_complete: bool

    def _payload(self) -> dict[str, object]:
        return {
            "git_sha": self.git_sha,
            "seed": int(self.seed),
            "ic_condition": self.ic_certificate.condition_name,
            "ic_valid": bool(self.ic_certificate.valid),
            "ic_violations": list(self.ic_certificate.violations),
            "coin_pose": _canon(self.coin_pose),
            "target_pose": _canon(self.target_pose),
            "mode_trace": [int(m) for m in self.mode_trace.modes],
            "n_transitions": int(self.n_transitions),
            "energy_ledger_complete": bool(self.energy_ledger_complete),
        }

    def content_hash(self) -> str:
        """Postcondition: a 64-char hex digest; deterministic in the recorded fields (float-rounded to 1e-9)."""
        blob = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SuccessCertificate:
    """An outcome certificate (e.g. strict K6) bound to the provenance hash of the rollout that produced it."""

    outcome_name: str
    success: bool
    metric_mm: float
    safe: bool
    provenance_hash: str

    def __post_init__(self) -> None:
        assert len(self.provenance_hash) == 64, "provenance_hash must be a SHA-256 hex digest linking to the rollout"
