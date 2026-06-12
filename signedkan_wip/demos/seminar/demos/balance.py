"""Demo 1 / plan-item 5 — affective structural balance (the opener, slide 12).

Loads a signed (affective) graph, enumerates its short cycles via the Rust
``enumerate_top_k_cycles_rs`` path, and reports **structural balance**: the
fraction of cycles whose sign-product is positive. Frustration = 1 − balance;
the most-frustrated cycles (negative sign-product) are surfaced and drawn.

Honest framing (printed): balance/frustration here is the **cycle sign-product
statistic**, not a learned quantity — it sets up the ``link`` demo, where the
same cycles drive prediction.

Design note (CLAUDE.md §6.5 #2): cycle *enumeration* is the Rust algorithm; the
per-cycle sign-product is a trivial reduction over the enumerated pool, kept in
Python. The Rust ``fraction_negative`` scorer only ranks which cycles survive
the ``k_keep`` cap — balance is classified from the sign-product directly, never
from the score (a 2-negative-edge cycle is *balanced* despite a high
fraction_negative).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..base import DemoContext, DemoResult

if TYPE_CHECKING:
    from signedkan_wip.src.datasets import SignedGraph

# Display ranking scorer (most-frustrated first under a binding cap). Balance is
# NOT read from this score — see module docstring.
SURFACE_SCORER = "fraction_negative"
MAX_SURFACED = 8


@dataclass
class BalanceStats:
    """Outcome of a balance evaluation over a cycle pool.

    Invariants: ``0 <= balance <= 1`` when ``n_cycles > 0`` (else ``balance`` is
    ``None``); ``frustrated`` rows are a subset of the enumerated pool, ordered
    most-frustrated first; ``cap_hit`` is True iff the pool size equalled the cap
    (the balance is then a lower bound over the kept pool, not exhaustive).
    """

    k: int
    n_cycles: int
    n_frustrated: int
    balance: float | None
    frustrated: list[list[int]]
    cap_hit: bool


def cycle_sign_products(
    edges: np.ndarray, signs: np.ndarray, cycles: np.ndarray,
) -> np.ndarray:
    """Sign-product (±1) of each cycle from the graph's edge signs.

    Preconditions: ``edges`` is (E, 2); ``signs`` aligns with ``edges``, values
    in {-1, +1}; every consecutive vertex pair in each cycle (with wraparound)
    is an edge of the graph.
    Postconditions: returns an int8 array, one entry per cycle row; a cycle is
    *frustrated/unbalanced* iff its entry is −1.
    Raises ``KeyError`` if a cycle uses a non-edge (caller passed a pool that
    does not match ``edges`` — a bug, not a recoverable state).
    """
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(edges.tolist(), signs.tolist()):
        sign_of.setdefault((min(u, v), max(u, v)), int(s))
    out = np.ones(len(cycles), dtype=np.int8)
    for i, cyc in enumerate(cycles.tolist()):
        prod = 1
        for j in range(len(cyc)):
            a, b = cyc[j], cyc[(j + 1) % len(cyc)]
            prod *= sign_of[(min(a, b), max(a, b))]
        out[i] = prod
    return out


def balance_statistics(
    graph: "SignedGraph", *, k: int, k_keep: int,
) -> BalanceStats:
    """Enumerate ``k``-cycles and classify each as balanced / frustrated.

    Preconditions: ``hymeko`` importable; ``k >= 3``; ``k_keep >= 1``.
    Postconditions: a :class:`BalanceStats`; ``cap_hit`` flags a binding cap.
    """
    # hymeko is a compiled PyO3 module with no py.typed marker (no stubs).
    import hymeko  # type: ignore[import-untyped]

    e = np.asarray(graph.edges)
    u = e[:, 0].astype(np.uint32)
    v = e[:, 1].astype(np.uint32)
    s = np.asarray(graph.signs).astype(np.int8)
    cycles, scores = hymeko.enumerate_top_k_cycles_rs(
        u, v, s, int(graph.n_nodes), int(k), int(k_keep), SURFACE_SCORER,
        "none", "none",
    )
    cycles = np.asarray(cycles)
    n = len(cycles)
    if n == 0:
        return BalanceStats(k=k, n_cycles=0, n_frustrated=0, balance=None,
                            frustrated=[], cap_hit=False)
    products = cycle_sign_products(e, s, cycles)
    frustrated_mask = products < 0
    n_frustrated = int(frustrated_mask.sum())
    balance = 1.0 - n_frustrated / n
    # Surface the most-frustrated cycles (highest surface-score first).
    order = np.argsort(-np.asarray(scores))
    frustrated = [cycles[i].tolist() for i in order if frustrated_mask[i]]
    return BalanceStats(
        k=k, n_cycles=n, n_frustrated=n_frustrated, balance=balance,
        frustrated=frustrated[:MAX_SURFACED], cap_hit=(n == k_keep),
    )


# ─── Small affective fixtures (hand-built, deterministic) ───────────────


def build_affective_graph(variant: str) -> tuple["SignedGraph", frozenset[int] | None]:
    """Return ``(graph, planted_frustrated_triad | None)``.

    ``camps``   — two 3-cliques, + within / − across: Harary-balanced (→ 1.0).
    ``planted`` — ``camps`` with one within-camp edge flipped: triad {0,1,2}
                  becomes frustrated (→ balance 0.5, that triad surfaced).
    ``karate``  — Zachary's club, faction-signed (real affective split).
    """
    from signedkan_wip.src.datasets import SignedGraph, karate_faction_signed

    if variant == "karate":
        return karate_faction_signed(), None
    # camps: A={0,1,2} all +, B={3,4,5} all +, cross {0-3,1-4,2-5} all −.
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]
    signs = [1, 1, 1, 1, 1, 1, -1, -1, -1]
    planted: frozenset[int] | None = None
    if variant == "planted":
        signs[0] = -1  # flip edge 0-1 → triad {0,1,2} frustrated
        planted = frozenset({0, 1, 2})
    g = SignedGraph(
        edges=np.array(edges, dtype=np.int64),
        signs=np.array(signs, dtype=np.int8),
        n_nodes=6,
    )
    return g, planted


class BalanceDemo:
    name = "balance"
    help = ("Affective structural balance: enumerate cycles, report balance / "
            "frustration, surface and draw the frustrated cycles.")

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--graph", default="planted",
            choices=["planted", "camps", "karate"],
            help="affective fixture: planted (frustrated triad), camps "
                 "(balanced), or karate (real faction split).",
        )
        parser.add_argument("--k", type=int, default=3,
                            help="cycle length to enumerate (>=3).")
        parser.add_argument("--k-keep", type=int, default=100_000,
                            help="top-K cycle cap (CLAUDE.md §2 contract). A "
                                 "binding cap makes balance a lower bound.")
        parser.add_argument("--no-figures", action="store_true",
                            help="skip the frustrated-cycle PNG.")

    def run(self, args: argparse.Namespace, ctx: DemoContext) -> DemoResult:
        os.environ.setdefault("MPLBACKEND", "Agg")
        if args.k < 3:
            raise ValueError(f"--k must be >= 3 (a cycle needs 3 edges); got {args.k}")
        graph, planted = build_affective_graph(args.graph)
        stats = balance_statistics(graph, k=args.k, k_keep=args.k_keep)

        result = DemoResult(name=self.name)
        src = f"{args.graph} fixture · enumerate_top_k_cycles_rs(k={args.k})"
        self._populate_metrics(result, args, graph, stats, src)
        self._add_notes(result, args, stats, planted)

        if not args.no_figures and stats.n_cycles:
            result.artifacts.append(
                self._render(graph, stats, args.graph, ctx.out_dir))
        return result

    @staticmethod
    def _populate_metrics(
        result: DemoResult, args: argparse.Namespace, graph: "SignedGraph",
        stats: BalanceStats, src: str,
    ) -> None:
        signs = np.asarray(graph.signs)
        result.metrics["graph"] = args.graph
        result.metrics["n_nodes"] = int(graph.n_nodes)
        result.metrics["n_edges"] = int(len(signs))
        result.metrics["pct_negative"] = round(float((signs < 0).mean()) * 100, 1)
        result.metrics["k"] = stats.k
        result.metrics["n_cycles"] = stats.n_cycles
        result.metrics["n_frustrated"] = stats.n_frustrated
        bal = "n/a (no cycles)" if stats.balance is None else round(stats.balance, 4)
        result.metrics["balance"] = bal
        result.metrics["frustration"] = (
            "n/a" if stats.balance is None else round(1.0 - stats.balance, 4))
        for key in ("balance", "frustration", "n_frustrated", "n_cycles"):
            result.provenance[key] = src

    @staticmethod
    def _add_notes(
        result: DemoResult, args: argparse.Namespace, stats: BalanceStats,
        planted: frozenset[int] | None,
    ) -> None:
        result.notes.append(
            "balance/frustration is the cycle sign-product statistic, not a "
            "learned quantity — it sets up the link demo (same cycles predict).")
        if stats.frustrated:
            shown = ", ".join("{" + ",".join(map(str, c)) + "}"
                              for c in stats.frustrated)
            result.notes.append(f"most-frustrated cycles: {shown}")
        if planted is not None:
            surfaced = any(frozenset(c) == planted for c in stats.frustrated)
            result.metrics["planted_cycle_surfaced"] = surfaced
            result.notes.append(
                f"planted frustrated triad {set(planted)} "
                f"{'surfaced' if surfaced else 'NOT surfaced'}.")
        if stats.cap_hit:
            result.notes.append(
                f"WARNING: k_keep cap ({args.k_keep}) was binding — balance is a "
                f"lower bound over the kept (most-frustrated) pool, not exhaustive.")

    @staticmethod
    def _render(
        graph: "SignedGraph", stats: BalanceStats, name: str, out_dir: Path,
    ) -> Path:
        from signedkan_wip.src.demo.plotting import frustration_figure
        import matplotlib.pyplot as plt

        fig = frustration_figure(
            np.asarray(graph.edges), np.asarray(graph.signs),
            stats.frustrated, int(graph.n_nodes), balance=stats.balance,
            title=f"Affective balance — {name}",
        )
        path = out_dir / f"frustration_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return path


__all__ = [
    "BalanceDemo", "BalanceStats", "balance_statistics",
    "cycle_sign_products", "build_affective_graph",
]
