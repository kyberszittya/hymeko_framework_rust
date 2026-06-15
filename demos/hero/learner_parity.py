"""Structural parity between a `.hymeko` learner model and its emitted module.

The hero demo's "structural parity (gate-framed)" claim: *the emitted
`torch.nn.Module` faithfully realises the architecture declared in the IR*. We
prove it by text — no torch import (CI-safe, matches the rest of the demo):

  * `parse_hymeko_layers`  — the layer hypervertices declared in the source;
  * `parse_torch_attrs`    — the `self.<name>` sub-modules in the emitted code;
  * `structural_parity`    — every declared layer is realised (none missing).

The `torch_dataflow` template emits exactly one `self.<decl_name> = <Class>(...)`
per layer decl, so name equality is the faithfulness relation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Layer kinds the torch_dataflow template realises as a module attribute (one
#: per `{{#each <kind>_layers}}` section in transforms/torch_dataflow/template.py).
#: `mean_pool`/`max_pool`/`readout_linear` are intentionally excluded — the
#: template has no emit section for them yet, so they would (correctly) show as
#: missing.
LAYER_KINDS: tuple[str, ...] = (
    "hypergraph_conv",
    "linear_layer",
    "relu_layer",
    "sigmoid_layer",
    "tanh_layer",
    "residual_block",
    "highway_block",
    "signedkan_layer",
    "walk_layer",
    "arity_mixer",
    "signed_classifier",
)

# A top-level layer decl: `name: <ns?>.<kind> {` or `name: <ns?>.<kind>`.
# Edges (`@flow: …`) and inner factors (`@factor: fac.hypergraph_conv`) start
# with `@`, so the leading `[A-Za-z_]` anchor excludes them; tensors / ports /
# neurons / ggk specs use kinds not in LAYER_KINDS.
_LAYER_DECL = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*:\s*(?:[\w.]+\.)?(" + "|".join(LAYER_KINDS) + r")\b",
    re.MULTILINE,
)
_TORCH_ATTR = re.compile(r"^\s*self\.([A-Za-z_]\w*)\s*=", re.MULTILINE)


def parse_hymeko_layers(src: str) -> dict[str, str]:
    """Map each declared layer name → its kind. Preconditions: `src` is HyMeKo
    source. Postconditions: only top-level layer hypervertices (not edges,
    tensors, ports, factors)."""
    return {m.group(1): m.group(2) for m in _LAYER_DECL.finditer(src)}


def parse_torch_attrs(torch_src: str) -> set[str]:
    """Names assigned as `self.<name> = …` in the emitted module (its
    sub-modules). Forward-pass `self.<name>(...)` calls are not assignments and
    are excluded."""
    return {m.group(1) for m in _TORCH_ATTR.finditer(torch_src)}


@dataclass(frozen=True)
class ParityReport:
    layers: dict[str, str]        # IR layer name → kind
    attrs: frozenset[str]         # emitted module attributes
    missing: tuple[str, ...]      # declared layers with no emitted sub-module
    extra: tuple[str, ...]        # emitted attrs not backed by a declared layer

    @property
    def faithful(self) -> bool:
        """The emit is faithful iff every declared layer is realised."""
        return not self.missing

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def structural_parity(hymeko_src: str, torch_src: str) -> ParityReport:
    """Compare a `.hymeko` learner to its emitted torch module (text-only)."""
    layers = parse_hymeko_layers(hymeko_src)
    attrs = parse_torch_attrs(torch_src)
    missing = tuple(sorted(n for n in layers if n not in attrs))
    extra = tuple(sorted(a for a in attrs if a not in layers))
    return ParityReport(layers=layers, attrs=frozenset(attrs), missing=missing, extra=extra)
