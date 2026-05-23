"""Star-expand a HyMeKo description into a signed graph that HSiKAN
can consume (Thesis-2 reduction: hypergraph → bipartite graph).

Pipeline:

  1. Parse the HyMeKo source via the existing `hymeko inspect` CLI.
     Each `kind=Edge` decl with arc lines is a hypergraph hyperedge
     with `(+ src, ~ port|layer, - dst)` incidence.

  2. Star expansion: each hyperedge `e` becomes a centroid vertex
     `c_e` plus `arity(e)` regular edges between c_e and the
     hyperedge's incident original vertices.  Centroid IDs are
     allocated in `[n_orig_vertices, n_orig_vertices + n_hyperedges)`.

  3. Sign assignment for each star-expansion edge:
        +  →  +1   (source / forward dataflow)
        -  →  -1   (sink / backward dataflow)
        ~  →   0   (port / through; mapped to +1 by default since
                    HSiKAN is binary {-1, +1}; configurable via
                    `--port-sign`)

The output is the (edges_u, edges_v, signs) tuple that
``hymeko.enumerate_k_cycles_rs`` and the SignedKAN bench harness
already consume — no new code on the data path.

Usage:
    python3 scripts/hymeko_to_signed_graph.py data/nn/mnist_resmlp_3.hymeko
    python3 scripts/hymeko_to_signed_graph.py data/nn/*.hymeko --enumerate
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


REPO = Path(__file__).resolve().parent.parent
HYMEKO_BIN = REPO / "target" / "release" / "hymeko"


@dataclass
class StarExpansion:
    n_orig_vertices: int
    n_hyperedges:    int
    edges_u:  np.ndarray
    edges_v:  np.ndarray
    signs:    np.ndarray
    vertex_names: list[str] = field(default_factory=list)
    centroid_names: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.n_orig_vertices + self.n_hyperedges

    def stats(self) -> dict:
        return {
            "n_orig":    self.n_orig_vertices,
            "n_centroids": self.n_hyperedges,
            "n_total":   self.n_nodes,
            "n_edges":   int(self.edges_u.shape[0]),
            "n_pos":     int((self.signs == +1).sum()),
            "n_neg":     int((self.signs == -1).sum()),
            "max_arity": (self._arity_per_edge().max()
                            if self.n_hyperedges else 0),
            "mean_arity": (float(self._arity_per_edge().mean())
                             if self.n_hyperedges else 0.0),
        }

    def _arity_per_edge(self) -> np.ndarray:
        """Per-hyperedge arity from the centroid degrees."""
        counts = np.zeros(self.n_hyperedges, dtype=int)
        for v in self.edges_v:
            cidx = int(v) - self.n_orig_vertices
            if 0 <= cidx < self.n_hyperedges:
                counts[cidx] += 1
        return counts


# ─── parsing ─────────────────────────────────────────────────────────


# Arc lines from `hymeko inspect`:
#   - arc#0: +foo.bar, 0foo.layer, -foo.baz
#                       ^   ^ port (~) renders as the literal "0" in the inspect output.
_ARC_RE = re.compile(
    r"-\s+arc#\d+:\s*(.+)$"
)
_INCIDENCE_RE = re.compile(
    r"([+\-0])([\w_.]+)"
)


def _run_inspect(path: Path) -> str:
    if not HYMEKO_BIN.exists():
        raise FileNotFoundError(f"hymeko CLI not built: {HYMEKO_BIN}")
    res = subprocess.run(
        [str(HYMEKO_BIN), "inspect", str(path)],
        cwd=REPO, capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"hymeko inspect failed:\n{res.stderr}")
    return res.stdout


def _parse_inspect_arcs(inspect_text: str) -> list[list[tuple[str, str]]]:
    """Return list of hyperedges, each as a list of (sign, target) pairs.

    Sign chars: '+', '-', '0' (port).
    """
    arcs_per_edge: list[list[tuple[str, str]]] = []
    current_edge: list[tuple[str, str]] | None = None
    for line in inspect_text.splitlines():
        # Look for the `kind=Edge` declarations first (they're indented
        # decls with hash and the word `Edge`).  These open a new arc
        # bucket; we accumulate any following `- arc#N:` lines until
        # the next decl line (any non-arc-and-non-blank line).
        if "kind=Edge" in line and "did=" in line and "Edge\thash=" not in line:
            current_edge = []
            arcs_per_edge.append(current_edge)
            continue
        m = _ARC_RE.search(line)
        if m and current_edge is not None:
            payload = m.group(1)
            for sgn, tgt in _INCIDENCE_RE.findall(payload):
                current_edge.append((sgn, tgt.strip(",")))
        elif line.strip() and "did=" in line and current_edge is not None:
            # Different decl — close the current arc bucket.
            current_edge = None
    # Some edges have no arcs (meta-types); drop them.
    return [e for e in arcs_per_edge if e]


def star_expand(path: Path, port_sign: int = +1) -> StarExpansion:
    """Star-expand the HyMeKo file at `path`.  `port_sign` selects
    how the `~` port incidence is mapped to {-1, +1} (HSiKAN is
    binary)."""
    text = _run_inspect(path)
    arcs_per_edge = _parse_inspect_arcs(text)

    vertex_id: dict[str, int] = {}
    def vid(name: str) -> int:
        if name not in vertex_id:
            vertex_id[name] = len(vertex_id)
        return vertex_id[name]

    eu: list[int] = []
    ev: list[int] = []
    sg: list[int] = []
    centroid_names: list[str] = []

    for ei, arcs in enumerate(arcs_per_edge):
        # All non-port endpoints get vertex IDs first; centroid is
        # allocated last so we can keep `[0, n_orig_vertices)` clean.
        for sgn, tgt in arcs:
            vid(tgt)
        centroid_names.append(f"_he_{ei}")

    n_orig = len(vertex_id)
    n_he = len(arcs_per_edge)

    for ei, arcs in enumerate(arcs_per_edge):
        c_id = n_orig + ei
        for sgn, tgt in arcs:
            v = vertex_id[tgt]
            s = (+1 if sgn == "+" else
                 -1 if sgn == "-" else port_sign)
            eu.append(v); ev.append(c_id); sg.append(s)

    return StarExpansion(
        n_orig_vertices=n_orig,
        n_hyperedges=n_he,
        edges_u=np.asarray(eu, dtype=np.uint32),
        edges_v=np.asarray(ev, dtype=np.uint32),
        signs=np.asarray(sg, dtype=np.int8),
        vertex_names=[n for n, _ in sorted(vertex_id.items(),
                                              key=lambda kv: kv[1])],
        centroid_names=centroid_names,
    )


# ─── enumerate cycles in the star expansion ──────────────────────────


def enumerate_cycles(se: StarExpansion, ks: Iterable[int] = (4, 6, 8),
                     max_cycles: int | None = 10000):
    """Run `hymeko.enumerate_k_cycles_rs` on the star-expansion edges
    for each k.  Cycles in star expansions alternate vertex↔centroid,
    so a "k_he hyperedges sharing a vertex" pattern shows up as a
    `2 * k_he`-cycle.  Even k's only.
    """
    import hymeko
    out = {}
    for k in ks:
        if k % 2 != 0:
            # Bipartite — no odd cycles possible.
            out[k] = (np.zeros((0, k), dtype=np.uint32), 0)
            continue
        cycles = hymeko.enumerate_k_cycles_rs(
            se.edges_u.tolist(),
            se.edges_v.tolist(),
            int(se.n_nodes),
            int(k),
            max_cycles,
            0,    # seed
        )
        out[k] = (cycles, int(cycles.shape[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--port-sign", type=int, default=+1, choices=[-1, +1],
                    help="Sign assigned to ~ (port) incidences.")
    ap.add_argument("--enumerate", action="store_true",
                    help="Also enumerate k-cycles in the expansion.")
    ap.add_argument("--ks", nargs="+", type=int,
                    default=[4, 6, 8],
                    help="Cycle lengths to enumerate (even only).")
    args = ap.parse_args()

    print(f"{'file':>40s}  {'n_orig':>7s}  {'n_he':>5s}  "
          f"{'n_edges':>7s}  {'+ / - / 0':>10s}  "
          f"{'mean_ar':>7s}  {'max_ar':>6s}")
    print("─" * 100)
    for p in args.paths:
        try:
            se = star_expand(p, port_sign=args.port_sign)
        except Exception as e:
            print(f"{p.name:>40s}  ERROR: {e}")
            continue
        s = se.stats()
        print(f"{p.name:>40s}  {s['n_orig']:>7d}  "
              f"{s['n_centroids']:>5d}  {s['n_edges']:>7d}  "
              f"{s['n_pos']:>3d} / {s['n_neg']:>3d} / {s['n_edges']-s['n_pos']-s['n_neg']:>3d}  "
              f"{s['mean_arity']:>7.2f}  {s['max_arity']:>6d}")
        if args.enumerate:
            cyc = enumerate_cycles(se, ks=args.ks)
            for k, (_, n) in cyc.items():
                print(f"   k={k}: {n} cycles")


if __name__ == "__main__":
    main()
