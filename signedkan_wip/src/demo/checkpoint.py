"""Checkpoint format for the demo.

A checkpoint is a ``torch.save``-able dict with these keys:

  - ``state_dict``    — model parameters.
  - ``model_class``   — qualified import path of the model class,
    e.g. ``"signedkan_wip.src.mixed_arity_signedkan.model.MixedAritySignedKAN"``.
  - ``cfg``           — the dataclass instance the model was built with
    (pickled by ``torch.save``).
  - ``meta``          — JSON-ish dict (see ``CheckpointMeta``).
  - ``inference_bundle`` (optional) — precomputed test-set inputs so
    the demo can run inference without re-enumerating cycles/walks.
    See ``InferenceBundle``.
  - ``format_version`` — int.

Caveats:
  - Model class is re-imported by string. Renames break the checkpoint.
  - ``cfg`` and ``inference_bundle`` are pickled. Brittle across class
    refactors but convenient for the demo.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


FORMAT_VERSION = 2


@dataclass
class InferenceBundle:
    """Precomputed test-set inputs for `MixedAritySignedKAN.encode_edges`.

    Stored alongside ``state_dict`` so the demo doesn't have to re-run
    the (potentially heavy) cycle / walk enumeration. ``per_arity_te``
    matches the call signature used in ``run_final_cell.py``.

    All tensors are CPU-resident; transfer happens at inference time.
    """

    # Same structure as run_final_cell's `per_arity_te` — list of
    # per-arity tuples. Treated as opaque by the demo; passed through
    # to the model's `encode_edges`.
    per_arity_te: Any
    query_edges: Any                     # (E, 2) long
    true_signs: Any                      # (E,)   int8


@dataclass
class CheckpointMeta:
    """Round-trippable metadata for a demo checkpoint."""

    dataset: str
    n_nodes: int
    tuple_specs: list = field(default_factory=list)
    seed: int = 0
    n_epochs: int = 0
    test_auc: float | None = None
    test_f1: float | None = None
    n_params: int | None = None
    train_args: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: Any,
    model_class: str,
    meta: CheckpointMeta,
    inference_bundle: InferenceBundle | None = None,
    classifier_module: nn.Module | None = None,
) -> Path:
    """Persist a trained model + cfg + metadata.

    Parameters
    ----------
    classifier_module
        For pipelines that wrap ``model.encode_edges`` with an external
        ``nn.Linear`` classifier (Bitcoin / Alpha / OTC paths in
        ``run_final_cell.py``), pass that module here. The demo's
        inference path will use it if present, else fall back to
        ``model.classifier``.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload: dict[str, Any] = {
        "state_dict": state_dict_cpu,
        "model_class": model_class,
        "cfg": cfg,
        "meta": asdict(meta),
        "format_version": FORMAT_VERSION,
    }
    if inference_bundle is not None:
        payload["inference_bundle"] = inference_bundle
    if classifier_module is not None:
        # Pickle the whole module so the demo doesn't need to know its
        # construction kwargs. Move to CPU first.
        payload["classifier_module"] = classifier_module.to("cpu")
    torch.save(payload, path)
    return path


def _import_model_class(qualified: str) -> type:
    if "." not in qualified:
        raise ValueError(f"model_class must be qualified, got {qualified!r}")
    mod_path, cls_name = qualified.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> tuple[nn.Module, Any, CheckpointMeta, InferenceBundle | None, nn.Module | None]:
    """Load a checkpoint and reconstruct the model.

    Returns ``(model, cfg, meta, inference_bundle_or_None, classifier_or_None)``.
    Model and classifier (if present) are in eval mode and on
    ``map_location``.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {"state_dict", "model_class", "cfg", "meta"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(
            f"{path} is not a demo checkpoint (missing keys: {missing})"
        )
    model_cls = _import_model_class(payload["model_class"])
    cfg = payload["cfg"]
    model = model_cls(cfg).to(map_location)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = CheckpointMeta(**payload["meta"])
    bundle = payload.get("inference_bundle")
    classifier = payload.get("classifier_module")
    if classifier is not None:
        classifier = classifier.to(map_location).eval()
    return model, cfg, meta, bundle, classifier
