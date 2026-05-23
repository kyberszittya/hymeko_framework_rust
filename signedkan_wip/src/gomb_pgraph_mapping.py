"""Map P-graph operating-unit names → ``run_gomb_smoke`` kwargs.

The toy graph ``data/hsikan/sweep_msg_gomb.hymeko`` uses ``gomb_fast`` /
``gomb_slow`` / ``gomb_fit``.  Extend :data:`GOMB_UNIT_TO_KNOBS` when you add
units to a custom P-graph sweep file.
"""
from __future__ import annotations

from typing import Any

GOMB_UNIT_TO_KNOBS: dict[str, dict[str, Any]] = {
    # Cheap pool + short implicit budget (n_epochs comes from CLI unless set).
    "gomb_fast": {
        "topk": 12,
        "cycle_abb_mode": "none",
        "n_epochs": 2,
    },
    # Larger pool + ABB on cycles + a few more epochs.
    "gomb_slow": {
        "topk": 48,
        "cycle_abb_mode": "start_local",
        "n_epochs": 4,
    },
    # Structural “train to product” step — does not override numerical knobs.
    "gomb_fit": {},
}


def merge_structure_knobs(unit_names: list[str]) -> dict[str, Any]:
    """Merge knob dicts for all units in one SSG / ABB solution structure.

    Later keys win on collision (prefer the last unit in ``unit_names``).
    """
    merged: dict[str, Any] = {}
    for raw in unit_names:
        u = raw.strip()
        patch = GOMB_UNIT_TO_KNOBS.get(u)
        if patch is None:
            raise KeyError(
                f"unknown Gömb P-graph unit {u!r}; add it to GOMB_UNIT_TO_KNOBS "
                f"in gomb_pgraph_mapping.py"
            )
        merged.update(patch)
    return merged


def smoke_argv_from_knobs(
    *,
    dataset: str,
    device: str,
    edge_split: str,
    seed: int,
    base: dict[str, Any],
    structure: dict[str, Any],
) -> list[str]:
    """Build ``python -m signedkan_wip.experiments.runs.run_gomb_smoke`` argv."""
    topk = int(structure.get("topk", base.get("gomb_topk", 48)))
    abb = str(structure.get("cycle_abb_mode", base.get("gomb_cycle_abb_mode", "none")))
    n_epochs = int(structure.get("n_epochs", base.get("n_epochs", 4)))
    d_embed = int(base.get("gomb_d_embed", 16))
    d_outer = int(base.get("gomb_d_outer", 8))
    m_outer = int(base.get("gomb_M_outer", 2))
    d_middle = int(base.get("gomb_d_middle", 16))
    d_core = int(base.get("gomb_d_core", 16))
    k = int(base.get("gomb_k", 3))
    abb_gate = float(base.get("gomb_cycle_abb_fullness_gate", 0.25))

    py = base.get("_python_executable")
    if not isinstance(py, str) or not py:
        import sys

        py = sys.executable

    cmd: list[str] = [
        py,
        "-m",
        "signedkan_wip.experiments.runs.run_gomb_smoke",
        "--dataset",
        dataset,
        "--edge-split",
        edge_split,
        "--device",
        device,
        "--seed",
        str(seed),
        "--n-epochs",
        str(n_epochs),
        "--topk",
        str(topk),
        "--d-embed",
        str(d_embed),
        "--d-outer",
        str(d_outer),
        "--M-outer",
        str(m_outer),
        "--d-middle",
        str(d_middle),
        "--d-core",
        str(d_core),
        "--k",
        str(k),
        "--cycle-abb-fullness-gate",
        str(abb_gate),
    ]
    if abb and abb != "none":
        cmd.extend(["--cycle-abb-mode", abb])
    return cmd
