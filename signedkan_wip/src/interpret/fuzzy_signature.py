"""Fuzzy signature view of HSIKAN — 2026-05-20.

HSIKAN factors each per-edge prediction into a sum of per-cycle
contributions; each cycle has an inherent sign-product (its
"vote") and a membership weight in the query's neighbourhood
(its "firing strength"). Reading these together makes the
model's reasoning chain directly observable.

For a query edge $e_q = (u, v)$, the fuzzy signature is the
set of triples

    \\{(c, σ_c, α_c, h_c, k_c) : M_e[e_q, c] ≠ 0\\}

where σ_c ∈ {+1, -1} is the cycle's σ-product, α_c is its
attention-weighted (or uniform) membership, h_c is its
per-cycle embedding (post-JK), and k_c is the arity slot.

The extractor uses a tiny side-channel
(``model._signature_capture``) populated by
:mod:`signedkan_wip.src.mixed_arity_signedkan.encoding_full`
to record everything in one forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class CycleContribution:
    """One cycle's contribution to a query edge's prediction.

    Reads as a fuzzy rule firing:

    - ``sigma_prod`` is the rule conclusion (the Cartwright-Harary
      balance vote: +1 for a balanced cycle, -1 for an unbalanced
      one). Computed as the product of ``edge_signs``.
    - ``membership`` is the firing strength α_c ∈ [0, 1].
    - ``embedding`` is the rule's vector contribution h_c.

    Note on ``sigma_assignment`` vs ``edge_signs``:
    ``sigma_assignment`` is HSIKAN's per-vertex σ_i ∈ {+1, -1}
    (each vertex's parity of incident negative edges within the
    cycle). It's what the model's per-vertex KAN spline uses.
    By construction (every negative edge flips parity at two
    vertices), ``Π σ_i ≡ +1`` always — so per-vertex sigma is
    NOT a valid cycle-level vote. ``edge_signs`` is the right
    primitive for that.
    """
    arity: int
    arity_kind: str               # 'cycle' | 'walk' | 'unknown'
    cycle_idx: int                # position within the arity slot
    vertices: tuple[int, ...]
    sigma_assignment: tuple[int, ...]  # per-vertex σ_i (structural)
    edge_signs: tuple[int, ...]   # per-edge signs (interpretive vote)
    sigma_prod: int               # ±1, = Π edge_signs (the vote)
    balanced: bool                # σ_prod == +1
    membership: float             # α_c — fuzzy firing strength
    embedding: np.ndarray         # (d_jk,) per-cycle embedding
    arc_weights: tuple[float, ...] = ()  # per-edge continuous weights
                                         # ∈ [−1, +1], same length as
                                         # edge_signs. Empty when the
                                         # source graph is unweighted
                                         # (legacy binary signed-graph
                                         # datasets) or the caller
                                         # didn't pass arity_arc_weights.


@dataclass
class FuzzySignature:
    """All cycle contributions to a single query edge's prediction.

    Walks like a fuzzy rule-base: each :class:`CycleContribution`
    is one rule, the model's final logit is their weighted
    aggregate. Stacking contributions by (arity, σ) gives the
    "how does each arity vote?" summary view.
    """
    query_edge: tuple[int, int]
    query_idx: int
    contributions: list[CycleContribution]
    arity_alpha: np.ndarray                # learned αₖ over arities
    arity_kinds: list[str]
    logit: Optional[float] = None
    prob_positive: Optional[float] = None

    # Convenience aggregates -------------------------------------------------
    def vote_by_arity(self) -> dict[str, dict[int, float]]:
        """Sum of ``membership`` per (arity_kind+arity, σ) bucket."""
        out: dict[str, dict[int, float]] = {}
        for c in self.contributions:
            tag = f"{c.arity_kind}{c.arity}"
            bkt = out.setdefault(tag, {-1: 0.0, +1: 0.0})
            bkt[c.sigma_prod] += c.membership
        return out

    def net_vote(self) -> float:
        """Signed sum: $\\sum_c σ_c · α_c$. The fuzzy aggregate vote."""
        return float(sum(c.sigma_prod * c.membership
                          for c in self.contributions))

    def total_membership(self) -> float:
        return float(sum(c.membership for c in self.contributions))

    def mean_abs_arc_weight(self) -> float:
        """Mean of the absolute arc weights across all edges in all
        contributing cycles. A summary scalar — high values indicate
        the prediction was driven by strong-magnitude evidence,
        low values indicate weak-magnitude evidence even when many
        cycles fired."""
        ws: list[float] = []
        for c in self.contributions:
            ws.extend(abs(w) for w in c.arc_weights)
        return float(np.mean(ws)) if ws else 0.0


# ─── Extractor ────────────────────────────────────────────────────────


def _sign_product(row) -> int:
    """Product of a sign row, returns ±1 (treats 0 as +1)."""
    s = int(np.prod(np.asarray(row, dtype=np.int64)))
    return 1 if s > 0 else -1


def _infer_arity_kind(arity: int, kind_hint: Optional[str]) -> str:
    """Without explicit hints, we can't tell cycles from walks at
    arity 3+ (the kind comes from how cycles were enumerated, not
    from the tuples themselves). Return ``'unknown'`` unless told.
    The kind for arity 2 is always 'cycle' (k=2 cycles are edge
    pairs)."""
    if kind_hint is not None:
        return kind_hint
    if arity == 2:
        return "cycle"
    return "unknown"


def _resolve_model(model) -> "torch.nn.Module":
    """If the model is a Phase 21 side-stacked wrapper, return its
    first branch (the interpretable proxy — branches are independent
    and the first is a faithful representative for the signature
    view)."""
    if hasattr(model, "branches") and hasattr(model, "fusion"):
        return model.branches[0]
    return model


@torch.no_grad()
def extract_signature(
    model,
    per_arity_inputs: Sequence[tuple[torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]],
    query_edges: torch.Tensor,
    query_idx: int,
    arity_kinds: Optional[Sequence[str]] = None,
    arity_edge_signs: Optional[Sequence[np.ndarray]] = None,
    arity_arc_weights: Optional[Sequence[np.ndarray]] = None,
) -> FuzzySignature:
    """Extract a :class:`FuzzySignature` for ``query_edges[query_idx]``.

    Parameters
    ----------
    model
        :class:`MixedAritySignedKAN` or :class:`SideMixedAritySignedKAN`.
    per_arity_inputs
        The same `per_arity_inputs` list passed to
        :meth:`encode_edges`.
    query_edges
        ``(E, 2)`` long tensor of query edges.
    query_idx
        Row of ``query_edges`` to extract the signature for.
    arity_kinds
        Optional per-arity kind labels (e.g.,
        ``['cycle', 'cycle', 'walk', 'walk', 'walk']`` for
        ``c3, c5, w2, w3, w4``). When None, kinds are inferred as
        'cycle' for arity 2 and 'unknown' otherwise.
    arity_edge_signs
        Optional list of per-cycle edge-sign arrays (one
        ``(T_k, n_edges_per_cycle)`` ``np.ndarray`` per arity).
        When provided, ``sigma_prod`` is computed as the product
        of edge signs (the Cartwright-Harary balance vote — the
        right primitive). When None, we fall back to the product
        of per-vertex σ_i which is **structurally always +1**
        (every negative edge flips parity at two vertices); a
        warning is emitted in that case.
    arity_arc_weights
        Optional list of per-cycle continuous arc-weight arrays
        (same shape as ``arity_edge_signs``). When provided, each
        :class:`CycleContribution` carries its per-edge arc weights;
        :func:`plot_signature` adds a third panel showing the
        magnitude distribution per (arity, σ) bucket. Built by the
        ``arc_weights`` helpers in
        :mod:`signedkan_wip.src.core.arc_weights`.
    """
    inner = _resolve_model(model)
    if not hasattr(inner, "encode_edges"):
        raise TypeError(
            f"model must expose encode_edges; got {type(model).__name__}"
        )

    # Wire the capture side channel.
    capture: dict[str, list] = {}
    setattr(inner, "_signature_capture", capture)
    was_training = inner.training
    inner.eval()
    try:
        edge_emb = inner.encode_edges(
            list(per_arity_inputs), query_edges=query_edges,
            collect_attn_entropy=False,
        )
        logits = inner.classifier(edge_emb).squeeze(-1)
        # Match the final prediction probability through sigmoid.
        prob = torch.sigmoid(logits[query_idx]).item()
        logit = logits[query_idx].item()
    finally:
        try:
            delattr(inner, "_signature_capture")
        except AttributeError:
            pass
        if was_training:
            inner.train()

    if not capture:
        # encoding_full wasn't invoked (cycle_batch_size may have
        # routed to encoding_batched). The batched path is a
        # follow-up; raise a clear error rather than returning
        # silently-empty data.
        raise RuntimeError(
            "Signature capture is empty. The model likely used "
            "the batched encode path (cycle_batch_size set). "
            "Try cfg.base.cycle_batch_size = None for inspection."
        )

    n_arities = len(inner.cfg.arities)
    if len(capture["arity_h_final"]) != n_arities:
        raise RuntimeError(
            f"Captured {len(capture['arity_h_final'])} per-arity "
            f"slots but model has {n_arities} arities; inspector "
            f"can't align."
        )

    arity_alpha_t = inner.alpha().detach().cpu().numpy()
    arity_kinds_resolved = [
        _infer_arity_kind(int(k),
                            arity_kinds[ai] if arity_kinds is not None
                            and ai < len(arity_kinds) else None)
        for ai, k in enumerate(inner.cfg.arities)
    ]

    q = int(query_idx)
    contributions: list[CycleContribution] = []
    for ai, (triad_v, triad_sigma, _M_vt, M_e) in enumerate(
            per_arity_inputs):
        h_final = capture["arity_h_final"][ai]
        idx_t = capture["m_e_indices"][ai]
        val_t = capture["m_e_values"][ai]
        attn_t = capture["attn_vals"][ai]
        arity = int(inner.cfg.arities[ai])

        # Filter to (row == q).
        rows = idx_t[0].cpu().numpy()
        cols = idx_t[1].cpu().numpy()
        mask = rows == q
        if not np.any(mask):
            continue
        local_cols = cols[mask]
        # Resolve the per-cycle membership.
        if attn_t is not None:
            # attn_vals is parallel to the original (rows, cols) order.
            attn_np = attn_t.cpu().numpy()
            local_membership = attn_np[mask]
        else:
            # Uniform pool: M_e[q, c] is the membership directly.
            local_membership = val_t.cpu().numpy()[mask]

        triad_v_np = triad_v.cpu().numpy()
        triad_sigma_np = triad_sigma.cpu().numpy()
        h_final_np = h_final.cpu().numpy()
        es_arr = (arity_edge_signs[ai]
                   if arity_edge_signs is not None
                   and ai < len(arity_edge_signs)
                   else None)
        aw_arr = (arity_arc_weights[ai]
                   if arity_arc_weights is not None
                   and ai < len(arity_arc_weights)
                   else None)
        for cidx, alpha_c in zip(local_cols, local_membership):
            sigma_row = triad_sigma_np[cidx]
            verts = triad_v_np[cidx]
            if es_arr is not None:
                edge_row = tuple(int(s) for s in es_arr[cidx])
                vote = _sign_product(edge_row)
            else:
                edge_row = ()
                vote = _sign_product(sigma_row)  # always +1 by construction
            if aw_arr is not None:
                arc_row = tuple(float(w) for w in aw_arr[cidx])
            else:
                arc_row = ()
            contributions.append(CycleContribution(
                arity=arity,
                arity_kind=arity_kinds_resolved[ai],
                cycle_idx=int(cidx),
                vertices=tuple(int(v) for v in verts),
                sigma_assignment=tuple(int(s) for s in sigma_row),
                edge_signs=edge_row,
                sigma_prod=vote,
                balanced=(vote == 1),
                membership=float(alpha_c),
                embedding=h_final_np[cidx].copy(),
                arc_weights=arc_row,
            ))

    if arity_edge_signs is None:
        import warnings
        warnings.warn(
            "extract_signature called without arity_edge_signs; "
            "sigma_prod falls back to the per-vertex σ product, "
            "which is structurally +1 for every cycle and therefore "
            "uninformative as a 'vote'. Pass arity_edge_signs for "
            "the Cartwright-Harary balance vote.",
            stacklevel=2,
        )

    q_pair = tuple(int(x) for x in query_edges[q].cpu().numpy())
    return FuzzySignature(
        query_edge=q_pair,
        query_idx=q,
        contributions=contributions,
        arity_alpha=arity_alpha_t,
        arity_kinds=arity_kinds_resolved,
        logit=float(logit),
        prob_positive=float(prob),
    )


# ─── Plot ─────────────────────────────────────────────────────────────


def plot_signature(sig: FuzzySignature, axes=None):
    """Render the fuzzy signature as a multi-panel matplotlib figure.

    Top: per-arity stacked bar over $\\sigma \\in \\{+1, -1\\}$
    showing the total firing strength of each arity-kind bucket.

    Middle: scatter of individual cycles — x = ``membership``,
    y = ``\\|embedding\\|``, colour = sign (blue=+, red=-), size
    scales with arity.

    Bottom (only when any ``CycleContribution.arc_weights`` is
    non-empty): per-σ histogram of absolute arc weights — exposes
    whether the firing cycles carry strong vs weak edge-magnitude
    evidence.
    """
    import matplotlib.pyplot as plt

    has_arc = any(len(c.arc_weights) > 0 for c in sig.contributions)
    n_panels = 3 if has_arc else 2
    if axes is None:
        if has_arc:
            fig, (ax_top, ax_bot, ax_arc) = plt.subplots(
                3, 1, figsize=(8, 9),
                gridspec_kw={"height_ratios": [1, 1.4, 0.9]},
            )
        else:
            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(8, 7),
                gridspec_kw={"height_ratios": [1, 1.4]},
            )
            ax_arc = None
    else:
        if has_arc and len(axes) >= 3:
            ax_top, ax_bot, ax_arc = axes[:3]
        else:
            ax_top, ax_bot = axes[:2]
            ax_arc = None
        fig = ax_top.figure

    # --- Top: stacked bar over (arity_kind+arity) × σ -----------------
    votes = sig.vote_by_arity()
    keys = sorted(votes.keys())
    pos = [votes[k][+1] for k in keys]
    neg = [votes[k][-1] for k in keys]
    x = np.arange(len(keys))
    ax_top.bar(x, pos, color="#2b6cb0", label=r"$\sigma = +1$")
    ax_top.bar(x, [-n for n in neg], color="#c53030",
                label=r"$\sigma = -1$")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(keys, rotation=0)
    ax_top.set_ylabel(r"sum of memberships $\sum \alpha_c$")
    ax_top.axhline(0, color="black", linewidth=0.6)
    ax_top.legend(loc="best", frameon=False, fontsize=9)
    net = sig.net_vote()
    ax_top.set_title(
        f"FuzzySignature on edge {sig.query_edge}  "
        f"(net σ·α = {net:+.3f}, "
        f"p(+) = {sig.prob_positive:.3f})"
    )

    # --- Bottom: scatter of individual cycles -------------------------
    if sig.contributions:
        mb = np.array([c.membership for c in sig.contributions])
        nm = np.array([np.linalg.norm(c.embedding)
                        for c in sig.contributions])
        sg = np.array([c.sigma_prod for c in sig.contributions])
        ar = np.array([c.arity for c in sig.contributions])
        colors = np.where(sg > 0, "#2b6cb0", "#c53030")
        # Sizes by arity: arity 2 -> 30, 3 -> 60, 4 -> 90, etc.
        sizes = 30 + 30 * (ar - 2)
        ax_bot.scatter(mb, nm, c=colors, s=sizes, alpha=0.6,
                        edgecolors="black", linewidths=0.4)
        ax_bot.set_xlabel(r"membership $\alpha_c$")
        ax_bot.set_ylabel(r"embedding norm $\|h_c\|$")
        ax_bot.set_title(
            f"individual cycles ({len(sig.contributions)} total; "
            f"size ∝ arity, color = σ)"
        )
        ax_bot.grid(alpha=0.3)
    else:
        ax_bot.text(0.5, 0.5,
                     "(no cycles touched this query edge)",
                     ha="center", va="center",
                     transform=ax_bot.transAxes,
                     fontsize=11, color="gray")
        ax_bot.set_axis_off()

    # --- Arc-weight panel (optional) ----------------------------------
    if ax_arc is not None:
        pos_ws: list[float] = []
        neg_ws: list[float] = []
        for c in sig.contributions:
            for w in c.arc_weights:
                bucket = pos_ws if c.sigma_prod > 0 else neg_ws
                bucket.append(abs(float(w)))
        bins = np.linspace(0.0, 1.0, 21)
        if pos_ws:
            ax_arc.hist(pos_ws, bins=bins, alpha=0.6,
                          color="#2b6cb0", label=r"$\sigma = +1$")
        if neg_ws:
            ax_arc.hist(neg_ws, bins=bins, alpha=0.6,
                          color="#c53030", label=r"$\sigma = -1$")
        ax_arc.set_xlim(0.0, 1.0)
        ax_arc.set_xlabel(r"$|w_e|$ (arc-weight magnitude)")
        ax_arc.set_ylabel("count")
        mean_abs = sig.mean_abs_arc_weight()
        ax_arc.set_title(
            f"arc-weight magnitudes per σ  "
            f"(mean $|w|$ = {mean_abs:.3f})"
        )
        ax_arc.legend(loc="best", frameon=False, fontsize=9)
        ax_arc.grid(alpha=0.3)

    fig.tight_layout()
    if ax_arc is not None:
        return (ax_top, ax_bot, ax_arc)
    return (ax_top, ax_bot)
