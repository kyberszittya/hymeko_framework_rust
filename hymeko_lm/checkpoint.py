"""Checkpoint save/load for the FSR LM (config + weights round-trip).

Stores the config as plain values (enum ``.value`` strings) so a checkpoint is self-describing and a
model can be rebuilt without the caller knowing its hyperparameters (§4: long runs must checkpoint).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from hymeko_lm.config import Activation, FSRConfig, GateMode, ResidualMode
from hymeko_lm.model import FSRLanguageModel

_ENUM_FIELDS = {"activation": Activation, "gate_mode": GateMode, "residual_mode": ResidualMode}


def config_to_dict(cfg: FSRConfig) -> dict[str, Any]:
    """FSRConfig -> plain dict (enum members -> their string values)."""
    d = asdict(cfg)
    for field in _ENUM_FIELDS:
        d[field] = d[field].value
    return d


def config_from_dict(d: dict[str, Any]) -> FSRConfig:
    """Inverse of :func:`config_to_dict`."""
    d = dict(d)
    for field, enum in _ENUM_FIELDS.items():
        d[field] = enum(d[field])
    return FSRConfig(**d)


def save_checkpoint(path: str | Path, model: FSRLanguageModel, step: int,
                    extra: dict[str, Any] | None = None) -> None:
    """Write ``{config, state_dict, step, extra}`` to ``path``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config_to_dict(model.cfg), "state_dict": model.state_dict(),
                "step": step, "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu"
                    ) -> tuple[FSRLanguageModel, int, dict[str, Any]]:
    """Rebuild ``(model, step, extra)`` from a checkpoint.

    # Postconditions the returned model reproduces the saved logits exactly.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = FSRLanguageModel(config_from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model, int(ckpt["step"]), dict(ckpt["extra"])
