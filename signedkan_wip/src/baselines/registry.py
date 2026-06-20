"""Strategy registry for signed-link-prediction baselines (Phase B audit).

The seven baselines (SGCN, SiGAT, SGT, SGCL, SiGformer, SE-SGformer, DADSGNN)
differ only along two orthogonal axes — *model architecture* and *training
hyperparameters*. Per CLAUDE.md §6.5 #1/#3/#9 that is a Strategy family behind
**one** dispatch point, not seven ``run_<method>.py`` scripts nor a
``match model_name`` ladder repeated at every call site.

Design
------
``SignedLinkBaseline`` (ABC) is the strategy. Each concrete strategy owns three
things and nothing else:

* ``build_model(meta, hp) -> nn.Module`` — construct the encoder+classifier.
* ``build_context(edges, signs, n_nodes, device) -> tuple`` — the *strict*,
  train-only message-passing structure the encoder consumes (sparse adjacency
  for SGCN, neighbour buckets for SiGAT, signed neighbour lists for SGT, ...).
* ``default_hparams() -> HParams`` — the method's training recipe.

The encoder contract is uniform: every model exposes ``encode_nodes(*ctx)``
returning per-node embeddings and ``edge_logits(z, edges_t)`` returning per-edge
logits, so the shared training loop (``run_baseline_audit``) never branches on
the model. New models inherit :class:`SignedLinkModule`, which supplies the
shared ``edge_logits`` / ``num_parameters`` so only ``encode_nodes`` and the
``classifier`` head differ.

Strict invariant (the audit's reason to exist): ``build_context`` is *only ever*
called with training edges/signs, so no test-edge sign can enter the encoder's
adjacency. The driver passes ``e_tr, s_tr`` exclusively; the loop asserts it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, ClassVar

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class GraphMeta:
    """Static graph shape handed to ``build_model``.

    Preconditions: ``n_nodes >= 1``; ``in_dim`` is the node-feature width (here
    always the learned-embedding width, the models are featureless).
    """

    n_nodes: int
    in_dim: int = 0

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {self.n_nodes}")


@dataclass(frozen=True)
class HParams:
    """Training recipe shared across baselines (overridable per strategy).

    Defaults match the SGCN EC recipe (``run_sgcn_baseline.run_one_sgcn``) so the
    existing audit number reproduces through the unified loop.
    """

    hidden: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_epochs: int = 120
    lr: float = 5e-3
    weight_decay: float = 1e-4
    early_stopping: bool = True
    val_every: int = 5
    patience: int | None = None  # stop after this many val checks w/o improvement
    class_weighted: bool = True
    aux_weight: float = 0.0  # weight on the model's optional aux_loss (SGCL)

    def merged(self, **overrides: Any) -> "HParams":
        """Return a copy with CLI overrides applied (None values ignored)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)


class SignedLinkModule(nn.Module, ABC):
    """Base class for the four reimplemented models.

    Subclasses set ``self.classifier`` (a head consuming ``[z_u ; z_v]``) and
    implement :meth:`encode_nodes`. ``edge_logits`` / ``num_parameters`` are
    shared so no per-model copy exists (the three legacy models predate this and
    carry their own identical copies — left untouched per the wrap-only plan).
    """

    classifier: nn.Module

    @abstractmethod
    def encode_nodes(self, *ctx: Any) -> torch.Tensor:
        """Per-node embeddings ``z`` of shape ``(n_nodes, d_out)``.

        Postcondition: ``z.shape[0] == n_nodes`` and ``z`` is finite.
        """

    def edge_logits(self, z: torch.Tensor, edges_t: torch.Tensor) -> torch.Tensor:
        """Per-edge logits for ``edges_t`` of shape ``(E, 2)`` (long)."""
        z_u = z[edges_t[:, 0]]
        z_v = z[edges_t[:, 1]]
        return self.classifier(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)

    def aux_loss(
        self, z: torch.Tensor, edges_t: torch.Tensor, signs_t: torch.Tensor
    ) -> torch.Tensor:
        """Optional auxiliary objective added to BCE (weighted by ``aux_weight``).

        Default: zero. SGCL overrides it with a sign-aware contrastive term.
        ``signs_t`` is ``(E_tr,)`` float in ``{+1,-1}``.
        """
        return z.new_zeros(())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SignedLinkBaseline(ABC):
    """A named baseline strategy: model factory + context builder + recipe."""

    name: ClassVar[str]

    @abstractmethod
    def build_model(self, meta: GraphMeta, hp: HParams) -> nn.Module:
        """Construct the model. Postcondition: returned module exposes
        ``encode_nodes(*ctx)`` and ``edge_logits(z, edges_t)``."""

    @abstractmethod
    def build_context(
        self,
        edges: np.ndarray,
        signs: np.ndarray,
        n_nodes: int,
        device: torch.device,
    ) -> tuple:
        """Strict (train-only) message-passing structure for the encoder.

        Preconditions: ``edges``/``signs`` are the *training* slice only;
        ``edges`` is ``(E_tr, 2)`` and ``signs`` is ``(E_tr,)`` in ``{+1,-1}``.
        Postcondition: a tuple splatted into ``model.encode_nodes(*ctx)``.
        """

    def default_hparams(self) -> HParams:
        return HParams()


# --- registry -------------------------------------------------------------

_REGISTRY: dict[str, SignedLinkBaseline] = {}
_STRATEGY_MODULES = (
    "sgcn", "sigat", "sgt",          # legacy wraps
    "sgcl", "sigformer", "sesgformer", "dadsgnn",  # Phase B reimplementations
    "cayley_rotor_baseline",         # inductive Cayley-rotor embedding (2026-06-16)
)
_loaded = False


def register(name: str) -> Callable[[type[SignedLinkBaseline]], type[SignedLinkBaseline]]:
    """Class decorator: instantiate and register a strategy under ``name``."""

    def _wrap(cls: type[SignedLinkBaseline]) -> type[SignedLinkBaseline]:
        cls.name = name
        if name in _REGISTRY:
            raise ValueError(f"baseline {name!r} already registered")
        _REGISTRY[name] = cls()
        return cls

    return _wrap


def _ensure_loaded() -> None:
    """Import strategy modules once so their ``@register`` side effects fire.

    Modules that are not yet implemented are skipped silently — the registry
    grows as Phase B lands each new method, and ``get_baseline`` reports the
    *available* set, not the planned one.
    """
    global _loaded
    if _loaded:
        return
    import importlib

    for mod in _STRATEGY_MODULES:
        try:
            importlib.import_module(f"signedkan_wip.src.baselines.{mod}")
        except ModuleNotFoundError:
            continue
    _loaded = True


def get_baseline(name: str) -> SignedLinkBaseline:
    """Look up a strategy by name.

    Raises ``KeyError`` with the valid set on an unknown name — a shallow,
    actionable failure at the boundary, not a deep panic (§6.5 #7).
    """
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown baseline {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


def list_baselines() -> list[str]:
    """Sorted names of all registered (implemented) strategies."""
    _ensure_loaded()
    return sorted(_REGISTRY)
