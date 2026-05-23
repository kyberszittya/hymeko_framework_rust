"""Typed loader for `hymeko rewrite --json` output.

The Rust CLI emits a JSON document describing a k=2 split of some
hypervertex scope — two cluster vertex lists, a per-edge cluster
assignment, an inertia score, and a cross-edge count. Python-side
consumers (weight transfer, hot-swap orchestration) parse this into a
small dataclass-based record so cluster membership can be looked up
by decl name rather than by reading the raw JSON shape.

Schema (as emitted by `hymeko rewrite --json`):

    {
      "target_scope": "<scope_name>",
      "cluster_a": ["<decl_name>", ...],
      "cluster_b": ["<decl_name>", ...],
      "edge_assignments": [
        {"edge": "<edge_name>", "cluster": "A" | "B" | "cross"},
        ...
      ],
      "n_cross_edges": <int>,
      "inertia": <float>
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Union

ClusterTag = Literal["A", "B", "cross"]


@dataclass(frozen=True)
class SplitProposal:
    """A single k=2 split proposal, matched to the Rust SplitProposal."""
    target_scope: str
    cluster_a: tuple[str, ...]
    cluster_b: tuple[str, ...]
    edge_assignments: tuple[tuple[str, ClusterTag], ...]
    n_cross_edges: int
    inertia: float

    def vertex_cluster(self, name: str) -> ClusterTag | None:
        """Which cluster does this vertex decl name belong to? `None` if
        the name isn't a vertex in either cluster."""
        if name in self.cluster_a:
            return "A"
        if name in self.cluster_b:
            return "B"
        return None

    def edge_cluster(self, name: str) -> ClusterTag | None:
        """Cluster assignment for an edge decl name, or `None` if the
        name isn't an assigned edge."""
        for edge, cluster in self.edge_assignments:
            if edge == name:
                return cluster
        return None

    @property
    def cluster_a_edges(self) -> tuple[str, ...]:
        return tuple(e for e, c in self.edge_assignments if c == "A")

    @property
    def cluster_b_edges(self) -> tuple[str, ...]:
        return tuple(e for e, c in self.edge_assignments if c == "B")

    @property
    def cross_edges(self) -> tuple[str, ...]:
        return tuple(e for e, c in self.edge_assignments if c == "cross")


def load_proposal(src: Union[str, Path, Mapping, bytes]) -> SplitProposal:
    """Load a split proposal from:
      - a path (str / Path) pointing at the JSON file
      - a raw JSON string (must start with `{`)
      - a bytes object
      - an already-parsed dict
    """
    data = _coerce_to_dict(src)
    _validate_schema(data)
    return SplitProposal(
        target_scope=str(data["target_scope"]),
        cluster_a=tuple(str(x) for x in data["cluster_a"]),
        cluster_b=tuple(str(x) for x in data["cluster_b"]),
        edge_assignments=tuple(
            (str(e["edge"]), _normalise_cluster(str(e["cluster"])))
            for e in data["edge_assignments"]
        ),
        n_cross_edges=int(data["n_cross_edges"]),
        inertia=float(data["inertia"]),
    )


def _coerce_to_dict(src: Union[str, Path, Mapping, bytes]) -> Mapping:
    if isinstance(src, Mapping):
        return src
    if isinstance(src, bytes):
        return json.loads(src.decode("utf-8"))
    if isinstance(src, Path):
        return json.loads(src.read_text(encoding="utf-8"))
    # str — distinguish path-like from literal JSON by the first char
    text = src.strip()
    if text.startswith("{"):
        return json.loads(text)
    return json.loads(Path(text).read_text(encoding="utf-8"))


def _validate_schema(data: Mapping) -> None:
    required = ("target_scope", "cluster_a", "cluster_b",
                "edge_assignments", "n_cross_edges", "inertia")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Proposal JSON missing keys: {missing}")


def _normalise_cluster(tag: str) -> ClusterTag:
    lowered = tag.lower()
    if lowered in ("a",):
        return "A"
    if lowered in ("b",):
        return "B"
    if lowered in ("cross", "x"):
        return "cross"
    raise ValueError(f"Unknown cluster tag: {tag!r}")


__all__ = ["SplitProposal", "load_proposal", "ClusterTag"]
