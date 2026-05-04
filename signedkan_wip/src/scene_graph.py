"""Scene graph adapter for HSiKAN.

A scene graph encodes objects (vertices) and visual relations
(edges or hyperedges). HSiKAN consumes signed graphs with binary
edges + per-edge ±1 sign, so this module:

  1. Defines a generic ``SceneGraph`` data structure that supports
     arity-≥2 hyperedges (for relations like "A between B and C").
  2. Provides ``to_signed_graph()`` for the binary-relation case
     (loses arity-≥3 info but compatible with current HSiKAN).
  3. Provides a stub for the **Berge cycle** extension that would
     handle arity-≥3 relations natively (Rust enumerator extension
     scoped in the OVERNIGHT_PLAN_2026_05_03.md).

Sign assignment for visual relations
-------------------------------------
The ±1 binary on each relation captures a task-relevant dichotomy.
Common choices:

    sign = +1   spatial-positive ("above", "on", "supports")
    sign = -1   spatial-negative ("below", "under", "blocks")

    sign = +1   affirmative semantic ("holding", "wearing", "using")
    sign = -1   negating semantic ("avoiding", "ignoring")

    sign = +1   contains/whole-of (mereological inclusion)
    sign = -1   adjacent/touches (boundary contact)

The adapter takes a user-provided ``relation_to_sign`` mapping and
applies it.

Usage
-----

>>> sg = SceneGraph()
>>> sg.add_object("table", "furniture")
>>> sg.add_object("apple", "food")
>>> sg.add_object("plate", "tableware")
>>> sg.add_relation(("apple", "plate"), "on")     # binary
>>> sg.add_relation(("plate", "table"), "on")     # binary
>>> sg.add_relation(("apple", "plate", "table"), "stacked")  # ternary
>>> g, obj_names = sg.to_signed_graph(
...     relation_to_sign={"on": +1, "under": -1},
...     binary_only=True,
... )
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .datasets import SignedGraph


@dataclass
class SceneObject:
    name: str
    category: str = ""
    bbox: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class SceneRelation:
    """Generic relation. ``vertices`` is a tuple of object names of
    arity ≥ 2 — supports binary edges (k=2) and higher-order
    hyperedges (k=3 e.g. "A between B and C")."""
    vertices: tuple[str, ...]
    relation_type: str
    confidence: float = 1.0
    direction: tuple[int, ...] | None = None   # cycle-order, optional


class SceneGraph:
    """Container for objects and (typed, possibly higher-arity) relations.

    Internally stores objects as a dict keyed by name; relations as a
    list. Arity-2 relations can be exported to ``SignedGraph`` directly;
    arity-≥3 relations require Berge-cycle support (TODO — the Rust
    enumerator extension in OVERNIGHT_PLAN_2026_05_03.md).
    """
    def __init__(self) -> None:
        self.objects: dict[str, SceneObject] = {}
        self.relations: list[SceneRelation] = []

    def add_object(self, name: str, category: str = "",
                    bbox: tuple[float, float, float, float] | None = None,
                    **attrs) -> None:
        self.objects[name] = SceneObject(
            name=name, category=category, bbox=bbox, attrs=dict(attrs),
        )

    def add_relation(self, vertices, relation_type: str,
                       confidence: float = 1.0,
                       direction=None) -> None:
        v = tuple(vertices) if not isinstance(vertices, tuple) else vertices
        if len(v) < 2:
            raise ValueError(
                f"relation requires ≥2 vertices, got {len(v)}"
            )
        for name in v:
            if name not in self.objects:
                raise KeyError(f"vertex {name!r} not in objects "
                                f"(call add_object first)")
        self.relations.append(SceneRelation(
            vertices=v, relation_type=relation_type,
            confidence=confidence, direction=direction,
        ))

    def stats(self) -> dict:
        n_by_arity: dict[int, int] = defaultdict(int)
        n_by_type: dict[str, int] = defaultdict(int)
        for r in self.relations:
            n_by_arity[len(r.vertices)] += 1
            n_by_type[r.relation_type] += 1
        return {
            "n_objects": len(self.objects),
            "n_relations": len(self.relations),
            "n_relations_by_arity": dict(n_by_arity),
            "n_relations_by_type": dict(n_by_type),
        }

    def to_signed_graph(
        self,
        relation_to_sign: Mapping[str, int],
        *,
        binary_only: bool = True,
        unknown_sign: int | None = None,
    ) -> tuple[SignedGraph, list[str]]:
        """Convert to ``SignedGraph`` (HSiKAN consumable).

        ``relation_to_sign`` maps each relation type to ±1.
        ``binary_only=True`` (default): drop arity-≥3 relations (with a
        warning printed). When False: raise — the caller should switch
        to the Berge-cycle path (not yet implemented).

        ``unknown_sign``: sign to use for relation types not in
        ``relation_to_sign``. ``None`` → drop those edges silently.
        """
        obj_names = list(self.objects.keys())
        name_to_idx = {n: i for i, n in enumerate(obj_names)}
        edges, signs = [], []
        n_dropped_arity = 0
        n_dropped_unknown = 0
        for r in self.relations:
            if len(r.vertices) > 2:
                if not binary_only:
                    raise NotImplementedError(
                        f"arity-{len(r.vertices)} relation requires "
                        f"Berge-cycle extension; pass binary_only=True "
                        f"to drop these."
                    )
                n_dropped_arity += 1
                continue
            sign = relation_to_sign.get(r.relation_type, unknown_sign)
            if sign is None:
                n_dropped_unknown += 1
                continue
            u, v = r.vertices
            edges.append((name_to_idx[u], name_to_idx[v]))
            signs.append(int(sign))
        if n_dropped_arity:
            print(f"[SceneGraph.to_signed_graph] dropped "
                  f"{n_dropped_arity} arity-≥3 relations "
                  f"(use Berge extension to keep them)")
        if n_dropped_unknown:
            print(f"[SceneGraph.to_signed_graph] dropped "
                  f"{n_dropped_unknown} unknown-sign relations")
        edges_arr = (np.array(edges, dtype=np.int64)
                     if edges else np.zeros((0, 2), dtype=np.int64))
        signs_arr = (np.array(signs, dtype=np.int8)
                     if signs else np.zeros((0,), dtype=np.int8))
        g = SignedGraph(edges=edges_arr, signs=signs_arr,
                          n_nodes=len(obj_names))
        return g, obj_names

    # --- Future: Berge-cycle support ---

    def get_hyperedges(self) -> list[tuple[tuple[int, ...], int, str]]:
        """For the Berge-cycle path (not yet wired into HSiKAN's Rust
        enumerator). Returns list of (vertex_indices, sign, type) for
        all relations.

        When the Rust ``enumerate_berge_cycles_rs`` lands, this is the
        format it'll consume.
        """
        name_to_idx = {n: i for i, n in enumerate(self.objects)}
        out = []
        for r in self.relations:
            v_idx = tuple(name_to_idx[name] for name in r.vertices)
            out.append((v_idx, 0, r.relation_type))   # sign=0 placeholder
        return out


# --- Demo: small synthetic scene graph ---

def demo_kitchen_scene() -> SceneGraph:
    """Hand-built kitchen scene: table + plate + apple + cup + chair.
    Mix of binary (on/under/next-to) and ternary (between) relations.
    Useful as a smoke fixture for the adapter."""
    sg = SceneGraph()
    for obj, cat in [("table", "furniture"), ("chair", "furniture"),
                       ("plate", "tableware"), ("cup", "tableware"),
                       ("apple", "food")]:
        sg.add_object(obj, category=cat)
    # Binary relations
    sg.add_relation(("plate", "table"), "on")
    sg.add_relation(("apple", "plate"), "on")
    sg.add_relation(("cup", "table"), "on")
    sg.add_relation(("chair", "table"), "next_to")
    sg.add_relation(("cup", "plate"), "next_to")
    # Ternary
    sg.add_relation(("apple", "plate", "cup"), "between")
    return sg


if __name__ == "__main__":
    sg = demo_kitchen_scene()
    print(f"Demo kitchen scene: {sg.stats()}")
    g, names = sg.to_signed_graph(
        relation_to_sign={"on": +1, "next_to": +1, "under": -1},
        binary_only=True,
    )
    print(f"\nAs SignedGraph: {g.stats()}")
    print(f"Vertex order: {names}")
    print(f"Edges: {g.edges.tolist()}")
    print(f"Signs: {g.signs.tolist()}")
