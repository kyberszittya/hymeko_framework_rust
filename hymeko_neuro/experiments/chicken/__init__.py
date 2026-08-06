"""Chicken-aggression detection pipeline (Éva Rampasek collaboration).

Three building blocks:

  * :mod:`simulator`  — synthetic flock simulator with peck events and
    ground-truth aggressor labels, used while waiting for real video.
  * :mod:`interactions`  — trajectories → signed interaction graph
    (peck = -1, peaceful proximity = +1).  Same converter feeds real
    pose CSV / DeepLabCut / SLEAP outputs and synthetic trajectories.
  * :mod:`aggressor` — per-vertex aggressor classifier on top of HSiKAN
    cycle embeddings (a thin head, reuses the existing SignedKAN
    encoder + M_vt for vertex-level pooling).

The synthetic pipeline lets us iterate on the model architecture
before real data lands.  The real-data plug-in point is the
``Trajectories`` dataclass in :mod:`interactions`; see ``README.md``
for the expected file format.
"""
from .simulator import ChickenFlockSim, simulate_flock
from .interactions import (
    Trajectories,
    InteractionEvent,
    trajectories_to_signed_graph,
    detect_peck_events,
)
from .unsupervised import (
    negative_out_degree_score,
    cartwright_harary_score,
    hsikan_self_supervised_score,
)

__all__ = [
    "ChickenFlockSim",
    "simulate_flock",
    "Trajectories",
    "InteractionEvent",
    "trajectories_to_signed_graph",
    "detect_peck_events",
    "negative_out_degree_score",
    "cartwright_harary_score",
    "hsikan_self_supervised_score",
]
