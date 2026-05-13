"""Catalogue of canonical "best" checkpoints, plus a loader.

The catalogue is a YAML file (default ``signedkan_wip/src/demo/models.yaml``)
listing entries that look like:

.. code-block:: yaml

    - id: hsikan_bitcoin_alpha_optuna
      framework: HSiKAN
      dataset: bitcoin_alpha
      label: HSiKAN — Bitcoin Alpha (Optuna-best, AUC 0.9959)
      path: checkpoints/hsikan/bitcoin_alpha_optuna_best.pt
      metrics: { test_auc: 0.9959, n_seeds: 10, n_params: 30487, ... }
      train_cmd: |
        ... shell command ...

Entries with ``path`` that doesn't exist on disk are reported as
``available=False``. The GUI lists those greyed-out with the ``train_cmd``
so the user can reproduce them.

Override the registry path with ``HYMEKO_MODEL_REGISTRY=/path/to/file.yaml``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "models.yaml"


@dataclass
class ModelEntry:
    """One catalogued model. See module docstring for the YAML schema."""

    id: str
    framework: str
    dataset: str
    label: str
    path: Path
    notes: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    report: str | None = None
    train_cmd: str = ""

    @property
    def available(self) -> bool:
        return self.path.is_file()

    @property
    def auc(self) -> float | None:
        v = self.metrics.get("test_auc")
        return float(v) if v is not None else None

    @property
    def n_params(self) -> int | None:
        v = self.metrics.get("n_params")
        return int(v) if v is not None else None

    @property
    def n_seeds(self) -> int | None:
        v = self.metrics.get("n_seeds")
        return int(v) if v is not None else None


def _resolve_path(raw: str) -> Path:
    """Absolute paths kept as-is; otherwise resolve relative to REPO_ROOT."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_registry(path: str | Path | None = None) -> list[ModelEntry]:
    """Load the catalogue.

    Precedence: explicit ``path`` arg > ``HYMEKO_MODEL_REGISTRY`` env var
    > the packaged default.

    Returns an empty list (with a printed warning) if the file is missing
    rather than raising — the GUI degrades gracefully to upload-only mode.
    """
    if path is None:
        env = os.environ.get("HYMEKO_MODEL_REGISTRY")
        path = Path(env) if env else DEFAULT_REGISTRY
    path = Path(path).expanduser()
    if not path.is_file():
        print(f"[demo.registry] WARNING: no registry at {path}")
        return []
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    raw_models = doc.get("models", [])
    out: list[ModelEntry] = []
    for raw in raw_models:
        try:
            entry = ModelEntry(
                id=str(raw["id"]),
                framework=str(raw.get("framework", "unknown")),
                dataset=str(raw.get("dataset", "unknown")),
                label=str(raw.get("label", raw["id"])),
                path=_resolve_path(raw["path"]),
                notes=str(raw.get("notes", "")).strip(),
                metrics=dict(raw.get("metrics", {})),
                report=raw.get("report"),
                train_cmd=str(raw.get("train_cmd", "")).strip(),
            )
        except KeyError as e:
            print(f"[demo.registry] skipping malformed entry "
                  f"(missing {e.args[0]!r}): {raw!r}")
            continue
        out.append(entry)
    return out


def dropdown_choices(entries: list[ModelEntry]) -> list[tuple[str, str]]:
    """``(label, id)`` pairs for a Gradio Dropdown.

    Available models come first (alphabetised by label); unavailable ones
    follow with a ``[NOT TRAINED]`` prefix.
    """
    avail = sorted([e for e in entries if e.available], key=lambda e: e.label)
    miss = sorted([e for e in entries if not e.available], key=lambda e: e.label)
    pairs: list[tuple[str, str]] = [(e.label, e.id) for e in avail]
    pairs += [(f"[NOT TRAINED] {e.label}", e.id) for e in miss]
    return pairs


def find_by_id(entries: list[ModelEntry], entry_id: str) -> ModelEntry | None:
    for e in entries:
        if e.id == entry_id:
            return e
    return None


__all__ = [
    "ModelEntry",
    "REPO_ROOT",
    "DEFAULT_REGISTRY",
    "load_registry",
    "dropdown_choices",
    "find_by_id",
]
