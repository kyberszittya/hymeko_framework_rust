"""Continuous-weight Bitcoin and Reddit dataset loaders.

Companion to ``signedkan_wip/src/datasets.py``. Where the binary
loader thresholds the raw rating to a sign in $\\{-1, +1\\}$ at
entry, this loader preserves the full magnitude:

  Bitcoin Alpha / OTC: rating $r \\in [-10, +10]$ → $w = r / 10 \\in [-1, +1]$
  Reddit Hyperlinks:   LINK_SENTIMENT $\\in \\{-1, +1\\}$ — no
                       magnitude in the source data, so the
                       continuous loader is a no-op for Reddit.

Plan: ``docs/plans/2026-05-17-general-weighted-hyperedges/plan.tex``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .legacy import DATA_DIR, FORMATS, URLS, download


@dataclass
class WeightedSignedGraph:
    """A signed graph whose edges carry continuous per-edge weights
    in $[-1, +1]$ (or potentially $\\mathbb{R}$ for other domains).

    ``weights`` replaces ``signs`` from the binary
    :class:`~signedkan_wip.src.datasets.SignedGraph`.
    """

    edges: np.ndarray   # (E, 2) src, dst
    weights: np.ndarray  # (E,) float in [-1, +1]
    n_nodes: int

    def stats(self) -> dict:
        n_pos = int((self.weights > 0).sum())
        n_neg = int((self.weights < 0).sum())
        return {
            "n_nodes": self.n_nodes,
            "n_edges": int(self.edges.shape[0]),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_frac": n_pos / max(1, n_pos + n_neg),
            "weight_mean": float(self.weights.mean()),
            "weight_std": float(self.weights.std()),
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
        }

    @property
    def signs(self) -> np.ndarray:
        """Backward-compat: derived binary signs $\\mathrm{sign}(w)$.

        For interop with the existing binary pipeline (training
        scripts that expect a ``.signs`` attribute). The continuous
        weights remain available via the ``.weights`` attribute.
        """
        return np.sign(self.weights).astype(np.int8)


def load_continuous(name: str) -> WeightedSignedGraph:
    """Load a signed dataset, preserving continuous-rating magnitude.

    Supported datasets:
      - ``bitcoin_alpha``: rating $\\in [-10, +10]$, normalised to $/10$.
      - ``bitcoin_otc``: same.
      - ``slashdot`` / ``epinions`` / ``wiki_*`` / ``reddit_*``: the
        underlying source data only carries $\\pm 1$ binary signs, so
        the continuous load returns the binary signs as floats.
    """
    fmt = FORMATS[name]
    raw_path = download(name)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    nodes: set[int] = set()
    name_to_id: dict[str, int] = {}  # for reddit

    def _node_id_for(s: str) -> int:
        v = name_to_id.get(s)
        if v is None:
            v = len(name_to_id)
            name_to_id[s] = v
        return v

    with raw_path.open() as f:
        if fmt == "bitcoin":
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 3:
                    continue
                try:
                    s = int(row[0])
                    t = int(row[1])
                    r = float(row[2])
                except ValueError:
                    continue
                if r == 0.0:
                    continue
                w = r / 10.0  # normalise [-10, +10] → [-1, +1]
                edges.append((s, t))
                weights.append(w)
                nodes.add(s); nodes.add(t)
        elif fmt == "snap_signed":
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or row[0].startswith("#") or row[0].startswith("%"):
                    continue
                if len(row) < 3:
                    continue
                try:
                    s = int(row[0])
                    t = int(row[1])
                    rf = float(row[2])
                except ValueError:
                    continue
                if rf == 0.0:
                    continue
                edges.append((s, t))
                weights.append(1.0 if rf > 0 else -1.0)
                nodes.add(s); nodes.add(t)
        elif fmt == "reddit_hyperlinks":
            for raw_line in f:
                if not raw_line.strip() or raw_line.startswith("SOURCE_SUBREDDIT"):
                    continue
                fields = raw_line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                try:
                    sentiment = int(fields[4])
                except ValueError:
                    continue
                if sentiment == 0:
                    continue
                u = _node_id_for(fields[0])
                v = _node_id_for(fields[1])
                edges.append((u, v))
                weights.append(float(sentiment))  # already in {-1, +1}
                nodes.add(u); nodes.add(v)
        else:
            raise NotImplementedError(
                f"continuous loader not implemented for format {fmt!r} (dataset {name!r})"
            )
    node_list = sorted(nodes)
    remap = {n: i for i, n in enumerate(node_list)}
    edges_arr = np.array([(remap[s], remap[t]) for s, t in edges], dtype=np.int64)
    weights_arr = np.array(weights, dtype=np.float32)
    return WeightedSignedGraph(
        edges=edges_arr, weights=weights_arr, n_nodes=len(node_list),
    )
