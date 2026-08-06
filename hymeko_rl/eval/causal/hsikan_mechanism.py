"""HSiKAN as a nonlinear mechanism model over a DirectLiNGAM / LiNGAM-SH causal hypergraph.

The CIP pipeline discovers a causal signed hypergraph over MetaWorld monitor variables (``near_fraction``,
``grasp_fraction``, ``obj_to_target_delta``, ``action_noise`` → ``total_reward`` …). DirectLiNGAM PROPOSES that
structure and ``mechanism_factorization`` fits **linear** loadings over it. This module lets HSiKAN be the
**nonlinear** mechanism model over the same structure — message-passing along the causal edges — so we can ask
whether the reward is a *structural, nonlinear* function of the monitors that the linear loadings and a flat MLP
miss, with a directed incidence-scramble as the decider (doctrine: LiNGAM proposes, HSiKAN models, the scramble
decides).

## Translating LiNGAM into a signed hypergraph

DirectLiNGAM returns a linear structural equation model ``x = B x + e`` (``B[effect, cause]`` the causal
coefficient, ``e`` non-Gaussian). The **faithful** translation into the framework's signed-hypergraph operator is
the sign split

    B = A⁺ − A⁻ ,   A⁺ = max(B, 0)  (excitatory causes),   A⁻ = max(−B, 0)  (inhibitory causes),

which is *exactly* HyMeKo's signed incidence ``B = A⁺ − A⁻`` and *exactly* HSiKAN's receiver-from-sender
convention (``A[dst, src]`` with ``dst = effect``). This preserves both **sign and magnitude** — unlike the
topological bridge below (:func:`causal_hg_to_structure`), which routes through ``HypergraphState.dense_signed_adj``
and so collapses weights to row-normalised arc counts.

**Operator compatibility (a restricted-layer embedding — NOT a containment claim).** A HSiKAN/SignedKAN layer is
``h' = φ(W_self x + W₊ Â⁺x + W₋ Â⁻x)`` (+ highway skip); with ``φ = id``, ``W_self = 0``, ``W₊ = W₋ = I`` and the
*raw* (un-normalised) split ``Â± = A±`` it evaluates to ``h' = A⁺x − A⁻x = B x`` — the **one-step linear** LiNGAM
prediction. So the one-step linear LiNGAM operator *embeds* as a restricted HSiKAN layer, and HSiKAN can then model
**nonlinear** (KAN spline ``φ``) and **multi-hop** (stacked layers) functions over the *same* proposed operator.
This is a principled **bridge** — LiNGAM proposes a signed causal operator, HSiKAN models over it — and is deliberately
NOT a claim that HSiKAN contains the full LiNGAM SEM solution, that it beats LiNGAM, or that any structural advantage
is thereby established (that is the harness's job, and it is gated on the directed scramble).

Two translation levels, both provided — and they are NOT equivalent:

* **weighted / faithful** — :func:`lingam_to_signed_adjacency` (``B → (A⁺, A⁻)``, reconstructs ``B`` exactly): feed
  the raw matrices straight to :class:`~hymeko_neuro.core.SignedKANBackbone`. This is the faithful operator bridge —
  it carries sign **and** magnitude.
* **topological / hypergraph** — :func:`causal_hg_to_structure` (mechanisms → star-expanded ``HypergraphState``):
  the sign-preserving hyperedge structure with mechanism hubs, useful for HyMeKo representation and cross-view
  verification. It routes through ``dense_signed_adj`` (row-normalised arc counts), so it does **not** preserve the
  full weighted LiNGAM operator unless weights/signs are explicitly carried through.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.eval.causal.hymeko_emit import CausalHypergraph
from hymeko_rl.eval.causal.lingam import LingamResult


def signed_adjacency_split(b: np.ndarray, *, min_abs: float = 0.0, row_normalize: bool = False,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Split a causal coefficient matrix ``B`` into the signed-hypergraph operator ``(A⁺, A⁻)`` with ``A⁺ − A⁻ = B``.

    ``A⁺ = max(B, 0)`` (excitatory), ``A⁻ = max(−B, 0)`` (inhibitory); entries with ``|B| <= min_abs`` are pruned to
    zero (LiNGAM's tiny-coefficient noise). With ``row_normalize`` each receiver's incoming weights are divided by
    their absolute-degree (HSiKAN's ``dense_signed_adj`` convention) — otherwise the split is *raw* and reconstructs
    ``B`` exactly.

    # Preconditions ``b`` is a square 2-D array; ``min_abs >= 0``.
    # Postconditions ``A⁺, A⁻ >= 0``; ``A⁺ * A⁻ == 0`` (disjoint supports); if not normalised and ``min_abs == 0``
      then ``A⁺ − A⁻ == B``.
    """
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError(f"B must be square 2-D, got shape {b.shape}")
    if min_abs < 0:
        raise ValueError("min_abs must be >= 0")
    keep = np.abs(b) > min_abs
    bp = np.where(keep, b, 0.0)
    a_pos = np.maximum(bp, 0.0)
    a_neg = np.maximum(-bp, 0.0)
    if row_normalize:
        deg = (a_pos + a_neg).sum(axis=1, keepdims=True)
        deg = np.where(deg > 0.0, deg, 1.0)
        a_pos, a_neg = a_pos / deg, a_neg / deg
    return a_pos.astype(np.float64), a_neg.astype(np.float64)


def lingam_to_signed_adjacency(result: LingamResult, *, min_abs: float = 0.0, row_normalize: bool = False,
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Translate a :class:`LingamResult` into the signed-hypergraph operator ``(A⁺, A⁻)`` over its variables.

    Thin wrapper over :func:`signed_adjacency_split` on ``result.adjacency`` (``B[effect, cause]``). The matrices are
    indexed by ``result.names`` order and can be handed directly to :class:`SignedKANBackbone` as the causal prior.

    # Postconditions same shape as ``result.adjacency``; ``A⁺ − A⁻`` reconstructs the (pruned) LiNGAM ``B``.
    """
    return signed_adjacency_split(np.asarray(result.adjacency, dtype=np.float64),
                                  min_abs=min_abs, row_normalize=row_normalize)


@dataclass(frozen=True)
class CausalStructure:
    """A causal hypergraph rendered for HSiKAN: the signed graph + the name↔index map + the sink variable.

    ``hg`` message-passes cause→effect (each effect aggregates from its causes, HSiKAN's ``A[dst, src]`` convention).
    ``variable_index`` locates the observed monitor variables among the (variables + mechanism-hub) vertices;
    ``hub_index`` locates the latent mechanism hubs (which carry no observed feature — zero-padded).
    """

    hg: HypergraphState
    variable_index: dict[str, int]
    hub_index: dict[str, int]
    sink: str

    @property
    def n_vertices(self) -> int:
        return self.hg.n_vertices

    @property
    def variable_names(self) -> list[str]:
        return list(self.variable_index)


def causal_hg_to_structure(chg: CausalHypergraph, *, sink: str) -> CausalStructure:
    """Bridge a :class:`CausalHypergraph` to a HSiKAN :class:`CausalStructure` via its star projection.

    # Preconditions ``sink`` is one of ``chg.variables``; the star projection references only known nodes.
    # Postconditions ``hg.n_vertices == len(variables) + len(hubs)``; every star-incidence arc is present with its
      sign; ``variable_index``/``hub_index`` partition the vertex set.
    """
    if sink not in chg.variables:
        raise ValueError(f"sink {sink!r} is not a causal variable: {chg.variables}")
    star = chg.star_projection()
    nodes = list(star.variables) + list(star.hubs)
    idx = {name: i for i, name in enumerate(nodes)}
    edges = [(idx[src], idx[dst]) for src, dst, _s in star.incidence]
    signs = [int(s) for _src, _dst, s in star.incidence]
    hg = HypergraphState(
        vertex_labels=tuple(nodes),
        edges=np.asarray(edges, np.int64) if edges else np.zeros((0, 2), np.int64),
        signs=np.asarray(signs, np.int64) if signs else np.zeros((0,), np.int64),
        topo_hash=f"causal:{chg.name}")
    return CausalStructure(
        hg=hg,
        variable_index={v: idx[v] for v in star.variables},
        hub_index={h: idx[h] for h in star.hubs},
        sink=sink)


def node_features_from_frame(structure: CausalStructure, columns: dict[str, np.ndarray], *,
                             mask_sink: bool = True) -> np.ndarray:
    """Lay per-episode variable columns onto the vertex feature matrix ``(n_samples, n_vertices, 1)``.

    Observed variables occupy their vertices; mechanism hubs (latent) are zero. With ``mask_sink`` the sink
    variable's own feature is zeroed so the model must *predict* it from its causes, not read it as an input.

    # Preconditions every ``structure.variable_names`` (except possibly the sink) is a key in ``columns`` with a
      common length ``n_samples``. # Postconditions shape ``(n_samples, n_vertices, 1)``.
    """
    present = [v for v in structure.variable_names if v in columns]
    if not present:
        raise ValueError("no causal variables found among the provided columns")
    n = len(columns[present[0]])
    x = np.zeros((n, structure.n_vertices, 1), dtype=np.float32)
    for v, i in structure.variable_index.items():
        if v == structure.sink and mask_sink:
            continue
        if v in columns:
            col = np.asarray(columns[v], dtype=np.float32)
            if col.shape[0] != n:
                raise ValueError(f"column {v!r} length {col.shape[0]} != {n}")
            x[:, i, 0] = col
    return x
