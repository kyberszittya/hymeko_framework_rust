"""MSG / ABB / SSG for HSIKAN-Gömb architecture search.

Implements three algorithms in the P-graph tradition, applied
to a discrete neural-architecture axis-product:

- :func:`msg_enumerate` — Maximum Structure Generation
  (Cartesian product over chosen axes).
- :func:`abb_prune` — Algorithmic Bounding Branch (predict-and-
  reject by GPU memory, wall, and parameter budgets BEFORE
  launching the cell).
- :func:`ssg_pareto` — Subset Structure Generation, Pareto
  filter along (wall, −lift) axes.

The bound predictors are coarse heuristics calibrated against
the 2026-05-20 / 2026-05-21 observed peak-memory + wall data
points. They aren't a substitute for runtime OOM-handling —
ABB just gets cheap wins on candidates we already know will
fail, sparing 7+ s of cycle-enum / GPU-init per such cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Iterable, Optional, Sequence


# --- Calibration constants -------------------------------------------
#
# Calibrated against 2026-05-20/21 observed peak-memory + wall-time
# data. Conservative: 30% safety margin baked in. Numbers in GiB
# unless noted.

# Per-dataset base memory + per-cell base wall (no outer HSIKAN).
_DATASET_BASELINE = {
    # (peak_mem_gib_no_hsikan, base_wall_s_60ep, n_cycles_typical)
    "bitcoin_alpha":  (0.7,   3.0,    13_000),
    "bitcoin_otc":    (0.8,   3.5,    16_000),
    "slashdot":       (1.8,  10.0,   133_000),
    "epinions":       (2.5,  30.0,   200_000),
}

# Outer-HSIKAN per-layer memory cost (GiB), highway vs cr_highway,
# without grad checkpoint.
_PER_LAYER_MEM_GIB = {
    "highway":     0.45,   # observed: d=4 BA ≈ 1.8 GiB above baseline
    "cr_highway":  0.65,   # CR-perturbation adds ~40% per layer
    "residual":    0.40,
    "none":        0.35,
    "auto":        0.45,
}

# Per-arity-input "data layer" overhead per cell (GiB), independent
# of depth. Used when M_vt is not cached (legacy path).
_DATA_LAYER_OVERHEAD_GIB = 0.15

# Grad-checkpoint relief multiplier on per-layer memory.
_GRAD_CKPT_RELIEF = 0.40   # ~60% reduction observed

# Wall multipliers (relative to baseline).
_WALL_PER_LAYER_MULT = 0.18   # ~18% per added layer
_WALL_CKPT_MULT     = 1.35    # ~35% slower with grad ckpt


@dataclass(frozen=True)
class ArchCandidate:
    """One discrete point in the architectural search space."""
    dataset: str
    outer_hsikan_n_layers: int
    inner_skip: str = "highway"
    use_arc_weights: bool = False
    grad_checkpoint: bool = False
    middle_n_layers: int = 1
    seed: int = 0
    n_epochs: int = 60

    @property
    def name(self) -> str:
        ds = {"bitcoin_alpha": "ba", "bitcoin_otc": "otc",
              "slashdot": "sd", "epinions": "epi"}.get(
                  self.dataset, self.dataset)
        skip = "cr" if self.inner_skip == "cr_highway" else "hw"
        return (f"{ds}_d{self.outer_hsikan_n_layers}_{skip}"
                f"{'_aw' if self.use_arc_weights else ''}"
                f"{'_ckpt' if self.grad_checkpoint else ''}"
                f"_s{self.seed}")

    def predicted_peak_mem_gib(self) -> float:
        """Heuristic: base + L * per_layer_skip - ckpt_relief."""
        baseline, _, _ = _DATASET_BASELINE.get(
            self.dataset, (1.0, 5.0, 10_000),
        )
        per_layer = _PER_LAYER_MEM_GIB.get(self.inner_skip, 0.45)
        L = self.outer_hsikan_n_layers
        # Outer HSIKAN cost
        hsikan_cost = per_layer * L
        if self.grad_checkpoint:
            hsikan_cost *= _GRAD_CKPT_RELIEF
        # Arc weights adds modest overhead (per-edge CR coefs).
        if self.use_arc_weights:
            hsikan_cost *= 1.10
        # Middle stacking adds analogous cost.
        if self.middle_n_layers > 1:
            hsikan_cost += per_layer * (self.middle_n_layers - 1)
        return baseline + hsikan_cost + _DATA_LAYER_OVERHEAD_GIB

    def predicted_wall_s(self) -> float:
        """Heuristic: base × (1 + L·wall_mult) × ckpt_mult."""
        _, base_wall, _ = _DATASET_BASELINE.get(
            self.dataset, (1.0, 5.0, 10_000),
        )
        L = self.outer_hsikan_n_layers
        wall = base_wall * (1.0 + _WALL_PER_LAYER_MULT * L)
        if self.grad_checkpoint:
            wall *= _WALL_CKPT_MULT
        if self.use_arc_weights:
            wall *= 1.20
        if self.middle_n_layers > 1:
            wall *= (1.0 + 0.15 * (self.middle_n_layers - 1))
        return wall

    def predicted_param_count(self) -> int:
        """Rough param-count estimate (for sanity bounds)."""
        # Per-dataset n_nodes (approximate).
        n_nodes = {
            "bitcoin_alpha": 3783, "bitcoin_otc": 5881,
            "slashdot": 82140, "epinions": 131828,
        }.get(self.dataset, 10_000)
        d_embed = {
            "bitcoin_alpha": 32, "bitcoin_otc": 32,
            "slashdot": 16, "epinions": 16,
        }.get(self.dataset, 16)
        # Two embeddings (base + outer HSIKAN's internal).
        embed = 2 * n_nodes * d_embed
        # Per HSIKAN layer: roughly d^2 × G × const, with G=5 grid.
        layer = d_embed * d_embed * 5 * 6
        L = self.outer_hsikan_n_layers
        # Outer + middle.
        return int(embed + L * layer + self.middle_n_layers * layer)

    def to_cli_args(self) -> list[str]:
        """CLI flags for ``run_gomb_smoke``."""
        args = [
            "--dataset", self.dataset,
            "--seed", str(self.seed),
            "--n-epochs", str(self.n_epochs),
            "--model", "outer_hsikan_gomb",
            "--outer-hsikan-n-layers", str(self.outer_hsikan_n_layers),
            "--outer-hsikan-inner-skip", self.inner_skip,
            "--outer-hsikan-jk-mode", "last",
        ]
        if self.use_arc_weights:
            # The outer HSIKAN doesn't currently expose
            # ``use-arc-weights`` directly — the run_final_cell
            # plumbing does (used by HSIKAN-only runs). For Gömb
            # the arc weights enter through ``inner_skip=cr_highway``
            # at the layer level instead. ``use_arc_weights`` here is
            # a forward-looking knob: when False (default) cr_highway
            # uses the gate without arc-weight modulation.
            pass
        if self.grad_checkpoint:
            args.append("--outer-hsikan-grad-checkpoint")
        if self.middle_n_layers > 1:
            args.extend([
                "--middle-n-layers", str(self.middle_n_layers),
            ])
        return args


def msg_enumerate(
    axes: dict[str, Sequence],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    n_epochs: int = 60,
) -> list[ArchCandidate]:
    """Maximum Structure Generation — Cartesian product over axes.

    ``axes`` is a dict mapping field name to a list of values.
    Recognised fields: ``dataset``, ``outer_hsikan_n_layers``,
    ``inner_skip``, ``use_arc_weights``, ``grad_checkpoint``,
    ``middle_n_layers``. Missing fields take their
    :class:`ArchCandidate` defaults.
    """
    keys = list(axes.keys())
    vals = [list(axes[k]) for k in keys]
    out: list[ArchCandidate] = []
    for combo in product(*vals):
        kwargs = dict(zip(keys, combo))
        for s in seeds:
            kwargs["seed"] = s
            kwargs["n_epochs"] = n_epochs
            out.append(ArchCandidate(**kwargs))
    return out


def abb_prune(
    candidates: Iterable[ArchCandidate],
    *,
    mem_cap_gib: float = 7.0,
    wall_cap_s: float = 90.0,
    param_cap: int = 10_000_000,
) -> tuple[list[ArchCandidate], list[tuple[ArchCandidate, str]]]:
    """Algorithmic Bounding Branch — prune by predicted bounds.

    Returns ``(survivors, pruned)`` where ``pruned`` is a list of
    ``(candidate, reason)`` tuples.
    """
    survivors: list[ArchCandidate] = []
    pruned: list[tuple[ArchCandidate, str]] = []
    for c in candidates:
        m = c.predicted_peak_mem_gib()
        w = c.predicted_wall_s()
        p = c.predicted_param_count()
        if m > mem_cap_gib:
            pruned.append((c, f"mem {m:.2f} > {mem_cap_gib} GiB"))
            continue
        if w > wall_cap_s:
            pruned.append((c, f"wall {w:.1f} > {wall_cap_s} s"))
            continue
        if p > param_cap:
            pruned.append((c, f"params {p:,} > {param_cap:,}"))
            continue
        survivors.append(c)
    return survivors, pruned


def ssg_pareto(
    candidates: Iterable[ArchCandidate],
    *,
    objectives: Sequence[str] = ("predicted_wall_s",),
) -> list[ArchCandidate]:
    """Subset Structure Generation — Pareto-dominant subset.

    Default keeps only the wall-minimal candidates per
    (dataset, inner_skip, seed) bucket; if multiple objectives
    are given, drops candidates dominated along all of them.

    For now this implements the simplest version: within each
    (dataset, seed) group, keep the candidate with the smallest
    ``predicted_wall_s`` per ``outer_hsikan_n_layers``. (I.e.,
    if a non-ckpt and a ckpt version of the same depth both
    survived ABB, prefer the non-ckpt one — same numerical result,
    less recompute overhead.)
    """
    by_bucket: dict[tuple, list[ArchCandidate]] = {}
    for c in candidates:
        key = (c.dataset, c.seed, c.outer_hsikan_n_layers,
                c.inner_skip, c.use_arc_weights, c.middle_n_layers)
        by_bucket.setdefault(key, []).append(c)
    out: list[ArchCandidate] = []
    for cs in by_bucket.values():
        # Pick the wall-minimal one.
        best = min(cs, key=lambda c: c.predicted_wall_s())
        out.append(best)
    return out
