"""Time-series forecasting benchmark for HSIKAN.

See ``docs/plans/2026-05-21-hsikan-timeseries-control/`` for the design.
"""

from .datasets import (
    DATASETS,
    TSConfig,
    load,
    lorenz_x,
    mackey_glass,
    noisy_sine,
    sine,
    windowed,
)
from .models import (
    GRUForecaster,
    HSIKANSeqForecaster,
    LinearAR,
    MLP,
    MODELS,
)

__all__ = [
    "DATASETS", "TSConfig", "load", "windowed",
    "sine", "noisy_sine", "mackey_glass", "lorenz_x",
    "MODELS", "LinearAR", "MLP", "GRUForecaster", "HSIKANSeqForecaster",
]
