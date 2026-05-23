"""Map HSIKAN architecture-P-graph unit names → ``run_compare.run_one``
kwargs.

The sweep file ``data/hsikan/sweep_msg.hymeko`` describes the HSIKAN
architecture search as a P-graph with three orthogonal axes:

  * cycle setup: ``cycle_topk_m{4,16,64}``  → ``m_cycles`` (cycle-pool size)
  * hidden width: ``model_h{8,16,32}``      → ``hidden`` (HighwaySignedKAN width)
  * training length: ``train_{short,long}`` → ``n_epochs``

Phase 7 (2026-05-19) shipped this module alongside the parallel Gömb
mapping (`gomb_pgraph_mapping.py`) so the HSIKAN driver can consume
the ABB-selected unit set the same way the Gömb driver does.

Extend :data:`HSIKAN_UNIT_TO_KNOBS` when adding units to a custom
HSIKAN P-graph sweep file.
"""
from __future__ import annotations

from typing import Any


HSIKAN_UNIT_TO_KNOBS: dict[str, dict[str, Any]] = {
    # Cycle setup axis — picks the cycle-pool size (m_cycles).
    "cycle_topk_m4":  {"m_cycles":  4},
    "cycle_topk_m16": {"m_cycles": 16},
    "cycle_topk_m64": {"m_cycles": 64},
    # Hidden-width axis — picks the HighwaySignedKAN inner width.
    "model_h8":  {"hidden":  8},
    "model_h16": {"hidden": 16},
    "model_h32": {"hidden": 32},
    # Training-length axis — picks the n_epochs schedule.
    "train_short": {"n_epochs": 10},
    "train_long":  {"n_epochs": 60},
    # ---- Phase 6 by-product variant kept here for the divergence test
    # so the mapping also resolves selections drawn from
    # `sweep_msg_byproduct.hymeko`. Same knobs as above; no new
    # symbols are introduced in that fixture.
    # ---- Phase 16 (2026-05-20): depth axis. Maps to n_layers in
    # the StackedSignedKAN / MultiLayerSignedKAN config. Empirical
    # finding on Bitcoin Alpha: L=1 wins (deeper hurts) — see
    # `reports/2026-05-20-stackable-hsikan-resnet-phase16.md`.
    "depth_l1": {"n_layers": 1},
    "depth_l2": {"n_layers": 2},
    "depth_l4": {"n_layers": 4},
    "depth_l8": {"n_layers": 8},
}


def merge_structure_knobs(unit_names: list[str]) -> dict[str, Any]:
    """Merge knob dicts for all units in one SSG / ABB solution
    structure.

    Later keys win on collision (prefer the last unit in
    ``unit_names``). Raises ``KeyError`` on any unrecognised unit name
    rather than silently defaulting — this is the same contract the
    Gömb mapping uses.
    """
    merged: dict[str, Any] = {}
    for raw in unit_names:
        u = raw.strip()
        patch = HSIKAN_UNIT_TO_KNOBS.get(u)
        if patch is None:
            raise KeyError(
                f"unknown HSIKAN P-graph unit {u!r}; add it to "
                f"HSIKAN_UNIT_TO_KNOBS in hsikan_pgraph_mapping.py"
            )
        merged.update(patch)
    return merged


def run_one_kwargs(
    *,
    dataset: str,
    seed: int,
    structure: dict[str, Any],
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the kwargs dict for
    ``signedkan_wip.experiments.runs.run_compare.run_one``.

    ``base`` may override defaults at the call site (e.g. ``lr``,
    ``model_name``). ``structure`` is the merged P-graph-derived dict
    (use :func:`merge_structure_knobs`).
    """
    base = dict(base or {})
    hidden = int(structure.get("hidden", base.get("hidden", 16)))
    n_epochs = int(structure.get("n_epochs", base.get("n_epochs", 60)))
    m_cycles = int(structure.get("m_cycles", base.get("m_cycles", 16)))
    n_layers = int(structure.get("n_layers", base.get("n_layers", 1)))
    return {
        # `run_compare.run_one` accepts "signedkan" /
        # "signedkan_entropy" / "vanillakan". Default to the bare
        # SignedKAN; callers wanting the highway variant should
        # subclass / wrap.
        "model_name": base.get("model_name", "signedkan"),
        "dataset": dataset,
        "hidden": hidden,
        "seed": seed,
        "n_epochs": n_epochs,
        "lr": float(base.get("lr", 5e-2)),
        # HSIKAN's cycle-pool size is controlled by env (m_cycles is
        # not a run_one kwarg today). We surface it as both a kwarg
        # *and* an env var so callers can either patch or set.
        "m_cycles": m_cycles,
        # Phase 16: depth axis for ResNet-style stackable HSIKAN.
        # n_layers > 1 routes run_compare to MultiLayerSignedKAN.
        "n_layers": n_layers,
    }


__all__ = [
    "HSIKAN_UNIT_TO_KNOBS",
    "merge_structure_knobs",
    "run_one_kwargs",
]
