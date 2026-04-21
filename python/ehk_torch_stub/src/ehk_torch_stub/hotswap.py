"""Weight transfer across hot-swapped model architectures.

Step 5 of the entropy hot-swap plan: given an *old* trained model and
a *new* model freshly constructed from a split architecture (either
hand-written or produced by `hymeko rewrite --emit-source`), carry
over every parameter tensor whose key **and** shape match between the
two `state_dict`s. Everything else remains at its fresh-initialisation
value.

This is deliberately architecture-agnostic — the function doesn't know
about clusters, HyMeKo, or PyTorch layer shapes beyond what `state_dict`
and `torch.Tensor.shape` expose. The matching is purely name-based, so
authors controlling the new model's attribute names dictate what
transfers: for the standard HyMeKo torch_dataflow emitter, layer
attributes are named `layer_<decl_name>`, so reusing the same decl
names in both old and new descriptions will match up automatically.

For orchestration with a [`SplitProposal`][ehk_torch_stub.proposal],
use [`reinfer_structure_and_rebuild`] which threads a model factory
plus the proposal through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

from .proposal import SplitProposal


@dataclass
class TransferReport:
    """Audit record of what moved / didn't during weight transfer."""
    transferred: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = \
        field(default_factory=list)
    fresh_in_new: list[str] = field(default_factory=list)
    dropped_from_old: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"transferred={len(self.transferred)} "
            f"shape_mismatch={len(self.shape_mismatch)} "
            f"fresh_in_new={len(self.fresh_in_new)} "
            f"dropped_from_old={len(self.dropped_from_old)}"
        )


def transfer_compatible_weights(
    old_model: nn.Module,
    new_model: nn.Module,
) -> TransferReport:
    """Copy entries from `old_model.state_dict()` into `new_model`
    where both the key and the tensor shape match. Mutates
    `new_model`'s parameters in place. Returns a [`TransferReport`]
    describing what moved, what was skipped for shape reasons, what
    stayed at fresh init, and what was discarded from the old model.

    Key matching is exact — no prefix rewrites, no heuristic remaps.
    Callers who need renaming should munge `old_model.state_dict()`
    before passing it in (or wrap `old_model` in a thin adapter that
    renames its parameters).
    """
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()

    report = TransferReport()
    patched: dict[str, torch.Tensor] = {}

    for key, new_tensor in new_sd.items():
        if key not in old_sd:
            report.fresh_in_new.append(key)
            continue
        old_tensor = old_sd[key]
        if tuple(old_tensor.shape) != tuple(new_tensor.shape):
            report.shape_mismatch.append(
                (key, tuple(old_tensor.shape), tuple(new_tensor.shape))
            )
            continue
        patched[key] = old_tensor.detach().clone()
        report.transferred.append(key)

    for key in old_sd:
        if key not in new_sd:
            report.dropped_from_old.append(key)

    # Apply the compatible subset in one call so non-strict loading
    # reports any residual mismatch we missed.
    if patched:
        incompatible = new_model.load_state_dict(patched, strict=False)
        # `incompatible.missing_keys` is expected (= fresh_in_new);
        # `incompatible.unexpected_keys` should be empty because we
        # only tried to load keys that already exist in new_sd.
        if incompatible.unexpected_keys:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Unexpected keys slipped through transfer filter: "
                f"{incompatible.unexpected_keys}"
            )

    return report


def reinfer_structure_and_rebuild(
    old_model: nn.Module,
    new_model_factory: Callable[[], nn.Module],
    *,
    proposal: Optional[SplitProposal] = None,
) -> tuple[nn.Module, TransferReport]:
    """Build a new model via `new_model_factory()`, transfer compatible
    weights from `old_model`, return `(new_model, report)`.

    `proposal` is optional — when supplied, the returned report is
    annotated with cluster metadata for downstream inspection, but the
    transfer logic itself is proposal-independent (purely name+shape
    based). This matches the spec's §5.4 signature while keeping the
    weight-transfer code agnostic to the entropy story above it.

    The old model remains valid and can be discarded by the caller.
    """
    new_model = new_model_factory()
    report = transfer_compatible_weights(old_model, new_model)
    if proposal is not None:
        # Attach the proposal for caller convenience — the report
        # dataclass carries it as a side attribute so telemetry can
        # correlate transfer outcomes against cluster membership.
        setattr(report, "proposal_scope", proposal.target_scope)
        setattr(report, "proposal_n_cross_edges", proposal.n_cross_edges)
    return new_model, report


__all__ = [
    "TransferReport",
    "transfer_compatible_weights",
    "reinfer_structure_and_rebuild",
]
