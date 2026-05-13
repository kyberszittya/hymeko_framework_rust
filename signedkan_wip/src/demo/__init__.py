"""Interactive demo for trained HSiKAN / MixedArity / HSiKAN-family models.

Layers:

  - ``checkpoint`` — round-trippable checkpoint format (state_dict +
    config + dataset name + split rng).
  - ``inference`` — load a checkpoint, reconstruct the model, run on a
    dataset, return metrics + per-edge predictions.
  - ``plotting`` — pure-matplotlib helpers for ROC curves, per-arity α
    bar charts, and signed-graph subgraph rendering.
  - ``gui`` — Gradio frontend.

Run the GUI with::

    pip install gradio
    PYTHONPATH=. python -m signedkan_wip.src.demo.gui

Train a checkpoint with::

    python -m signedkan_wip.src.run_final_cell \\
        --dataset bitcoin_alpha --hidden 8 --n-epochs 80 --seed 0 \\
        --save-checkpoint ./alpha_optuna_best.pt
"""
from .checkpoint import save_checkpoint, load_checkpoint, CheckpointMeta
from .inference import (
    ModelBundle, PredictionResult, load_bundle, predict_test_edges,
)

__all__ = [
    "CheckpointMeta",
    "ModelBundle",
    "PredictionResult",
    "load_bundle",
    "load_checkpoint",
    "predict_test_edges",
    "save_checkpoint",
]
