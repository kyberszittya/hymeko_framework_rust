"""Gömb fuzzy signature — multi-shell interpretability view.

Mirror of the HSIKAN ``fuzzy_signature`` view, but with one extra
dimension: per-shell breakdown of each cycle's contribution
through the V1 (Outer FIR) → V2 (Middle HSIKAN) → V4 (Inner CPML)
cortical cascade.

For a query edge $e_q = (u, v)$:

    \\{(c, σ_c, |h_c^{outer}|, |h_c^{middle}|, ...) :
       \\text{cycle } c \\text{ touches } e_q\\}

The per-shell magnitudes expose which cortical layer drove a
prediction, and the cross-shell propagation pattern shows how
a cycle's vote gets transformed up the hierarchy.

Capture is via the same attribute-driven side channel used for
HSIKAN's signature: set ``shell._signature_capture = {}`` before
forward, walk the dict afterward.

The Inner CPML core uses capsule routing without a clean per-cycle
intermediate; this MVP captures only the outer and middle shells.
A future extension will reach into the CPML pre-pool state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class GombCycleContribution:
    """One cycle's contribution to a Gömb prediction, broken down
    by cortical shell.

    ``per_shell_magnitude`` is the L2 norm of the cycle's feature
    vector at each shell — a scalar quantifier of "how much this
    cycle contributed at this shell". The full per-shell embedding
    is kept in ``per_shell_embedding`` for richer downstream
    analysis.
    """
    cycle_idx: int
    vertices: tuple[int, ...]
    sigma_assignment: tuple[int, ...]
    edge_signs: tuple[int, ...]
    sigma_prod: int               # ±1, the Cartwright-Harary vote
    balanced: bool
    arc_weights: tuple[float, ...]
    per_shell_magnitude: dict[str, float]
    per_shell_embedding: dict[str, np.ndarray]


@dataclass
class GombFuzzySignature:
    """All cycles' contributions to a single query edge's Gömb
    prediction, with per-shell breakdown."""
    query_edge: tuple[int, int]
    query_idx: int
    contributions: list[GombCycleContribution]
    cycle_arity: int
    shells: tuple[str, ...]
    logit: Optional[float] = None
    prob_positive: Optional[float] = None

    def shell_dominance(self) -> dict[str, float]:
        """Mean per-cycle magnitude per shell — exposes which
        cortical layer is doing the heavy lifting for this query."""
        out: dict[str, float] = {}
        for s in self.shells:
            mags = [c.per_shell_magnitude.get(s, 0.0)
                    for c in self.contributions]
            out[s] = float(np.mean(mags)) if mags else 0.0
        return out

    def cross_shell_consistency(self) -> float:
        """Rank correlation of cycle ordering between the outer
        and middle shells. 1.0 = same cycles dominate at both
        shells (the cortical hierarchy is just rescaling); 0.0 =
        independent (each shell re-prioritises the cycles)."""
        if len(self.shells) < 2 or len(self.contributions) < 3:
            return 0.0
        s0, s1 = self.shells[0], self.shells[1]
        a = np.array([c.per_shell_magnitude.get(s0, 0.0)
                       for c in self.contributions])
        b = np.array([c.per_shell_magnitude.get(s1, 0.0)
                       for c in self.contributions])
        # Pearson correlation of magnitudes (rank-correlation would be
        # cleaner but adds a scipy dep we don't otherwise need).
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def net_vote(self) -> float:
        """$\\sum_c σ_c · |h_c^{middle}|$ — the magnitude-weighted
        signed aggregate at the middle (HSIKAN) shell."""
        s_mid = "middle" if "middle" in self.shells else self.shells[-1]
        return float(sum(
            c.sigma_prod * c.per_shell_magnitude.get(s_mid, 0.0)
            for c in self.contributions
        ))


# ─── Extractor ────────────────────────────────────────────────────────


def _sign_product(row) -> int:
    s = int(np.prod(np.asarray(row, dtype=np.int64)))
    return 1 if s > 0 else -1


@torch.no_grad()
def extract_gomb_signature(
    model,
    cycles: torch.Tensor,
    signs: torch.Tensor,
    tier_of: torch.Tensor,
    edges_to_score: torch.Tensor,
    query_idx: int,
    arc_weights: Optional[np.ndarray] = None,
    edge_signs: Optional[np.ndarray] = None,
) -> GombFuzzySignature:
    """Extract a :class:`GombFuzzySignature` for
    ``edges_to_score[query_idx]``.

    Capture is via ``shell._signature_capture`` side channels on
    ``model.outer`` and ``model.middle``. The inner CPML shell is
    not captured in this MVP (no clean per-cycle intermediate);
    its contribution shows up implicitly through the final logit.

    Parameters
    ----------
    model
        :class:`HymeKoGomb` or one of its ablation siblings that
        exposes ``.outer`` and ``.middle`` submodules.
    cycles
        ``(M_c, k)`` long tensor of cycle vertex indices.
    signs
        ``(M_c, k)`` per-cycle edge / vertex signs.
    tier_of
        ``(N,)`` long tensor of per-vertex tier IDs (used by the
        inner CPML core).
    edges_to_score
        ``(E, 2)`` long tensor of query edges.
    query_idx
        Row of ``edges_to_score`` to extract for.
    arc_weights
        Optional ``(M_c, k)`` per-edge arc-weight array (in the
        same edge-ordering as ``signs``). When provided, each
        :class:`GombCycleContribution` carries its arc weights.
    edge_signs
        Optional ``(M_c, k)`` per-edge sign array used to compute
        the Cartwright-Harary balance vote. When None, falls back
        to ``signs`` (which is per-position in Gömb conventions and
        gives the same product up to construction details).
    """
    outer = getattr(model, "outer", None)
    middle = getattr(model, "middle", None)
    if outer is None and middle is None:
        raise ValueError(
            "model has neither .outer nor .middle — is this a Gömb "
            f"variant? type={type(model).__name__}"
        )
    capture_outer: dict = {}
    capture_middle: dict = {}
    if outer is not None:
        setattr(outer, "_signature_capture", capture_outer)
    if middle is not None:
        setattr(middle, "_signature_capture", capture_middle)
    was_training = model.training
    model.eval()
    try:
        scores = model.forward(
            cycles, signs, tier_of, edges_to_score,
        ).detach()
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        prob = torch.sigmoid(scores[query_idx]).item()
        logit = scores[query_idx].item()
    finally:
        if outer is not None and hasattr(outer, "_signature_capture"):
            delattr(outer, "_signature_capture")
        if middle is not None and hasattr(middle, "_signature_capture"):
            delattr(middle, "_signature_capture")
        if was_training:
            model.train()

    # Filter to cycles incident to the query edge.
    q = int(query_idx)
    q_u, q_v = (int(x) for x in edges_to_score[q].cpu().numpy())
    cycles_np = cycles.cpu().numpy()
    signs_np = signs.cpu().numpy()
    M_c, k = cycles_np.shape

    # A cycle "touches" the query edge when both endpoints appear
    # in the cycle. Cheap O(M_c · k) loop.
    incident_idx: list[int] = []
    for ci in range(M_c):
        verts = set(int(v) for v in cycles_np[ci])
        if q_u in verts and q_v in verts:
            incident_idx.append(ci)

    outer_feats = capture_outer.get("outer_per_cycle")
    middle_feats = capture_middle.get("middle_per_cycle")
    if outer_feats is None and middle_feats is None:
        raise RuntimeError(
            "Both outer and middle signature captures are empty — "
            "did the forward execute the standard cascade path?"
        )

    shells: list[str] = []
    if outer_feats is not None:
        shells.append("outer")
    if middle_feats is not None:
        shells.append("middle")

    contributions: list[GombCycleContribution] = []
    for ci in incident_idx:
        per_shell_mag: dict[str, float] = {}
        per_shell_emb: dict[str, np.ndarray] = {}
        if outer_feats is not None:
            v = outer_feats[ci].cpu().numpy()
            per_shell_mag["outer"] = float(np.linalg.norm(v))
            per_shell_emb["outer"] = v.copy()
        if middle_feats is not None:
            v = middle_feats[ci].cpu().numpy()
            per_shell_mag["middle"] = float(np.linalg.norm(v))
            per_shell_emb["middle"] = v.copy()

        sigma_row = signs_np[ci]
        if edge_signs is not None:
            es_row = tuple(int(s) for s in edge_signs[ci])
            vote = _sign_product(es_row)
        else:
            es_row = tuple(int(s) for s in sigma_row)
            vote = _sign_product(sigma_row)
        aw_row = (tuple(float(w) for w in arc_weights[ci])
                   if arc_weights is not None else ())

        contributions.append(GombCycleContribution(
            cycle_idx=ci,
            vertices=tuple(int(v) for v in cycles_np[ci]),
            sigma_assignment=tuple(int(s) for s in sigma_row),
            edge_signs=es_row,
            sigma_prod=vote,
            balanced=(vote == 1),
            arc_weights=aw_row,
            per_shell_magnitude=per_shell_mag,
            per_shell_embedding=per_shell_emb,
        ))

    return GombFuzzySignature(
        query_edge=(q_u, q_v),
        query_idx=q,
        contributions=contributions,
        cycle_arity=int(k),
        shells=tuple(shells),
        logit=float(logit),
        prob_positive=float(prob),
    )


# ─── Plot ─────────────────────────────────────────────────────────────


def plot_gomb_signature(sig: GombFuzzySignature, axes=None):
    """Render the Gömb fuzzy signature as a 4-row matplotlib figure.

    Row 1: stacked bar over $\\sigma \\in \\{+1, -1\\}$ — sum of
    per-cycle magnitudes (mean across shells) per σ. ``How does
    the cascade as a whole vote?``

    Row 2: per-shell mean magnitude bar — which cortical layer
    contributed the most.

    Row 3: per-cycle propagation lines — for the top-K firing
    cycles (by mean magnitude across shells), connected dots
    showing magnitude at each shell, coloured by σ.

    Row 4 (only when arc_weights present): per-σ histogram of
    arc-weight magnitudes.
    """
    import matplotlib.pyplot as plt

    has_arc = any(len(c.arc_weights) > 0 for c in sig.contributions)
    n_rows = 4 if has_arc else 3
    if axes is None:
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(8, 2 * n_rows + 1),
            gridspec_kw={"height_ratios": ([1.0, 0.8, 1.4, 0.9]
                                              if has_arc
                                              else [1.0, 0.8, 1.4])},
        )
    else:
        fig = axes[0].figure

    if has_arc:
        ax_vote, ax_shell, ax_prop, ax_arc = axes
    else:
        ax_vote, ax_shell, ax_prop = axes
        ax_arc = None

    # --- Row 1: σ-vote bar (mean across shells) -----------------------
    pos = sum(np.mean(list(c.per_shell_magnitude.values()))
               for c in sig.contributions if c.sigma_prod > 0)
    neg = sum(np.mean(list(c.per_shell_magnitude.values()))
               for c in sig.contributions if c.sigma_prod < 0)
    ax_vote.bar([0], [pos], color="#2b6cb0", label=r"$\sigma = +1$")
    ax_vote.bar([1], [neg], color="#c53030", label=r"$\sigma = -1$")
    ax_vote.set_xticks([0, 1])
    ax_vote.set_xticklabels([r"$\sigma = +1$ (balanced)",
                              r"$\sigma = -1$ (unbalanced)"])
    ax_vote.set_ylabel("Σ mean magnitude")
    net = sig.net_vote()
    ax_vote.set_title(
        f"GombFuzzySignature on edge {sig.query_edge}  "
        f"(net σ·|h_mid| = {net:+.3f}, "
        f"p(+) = {sig.prob_positive:.3f})"
    )

    # --- Row 2: per-shell dominance -----------------------------------
    dom = sig.shell_dominance()
    keys = list(dom.keys())
    vals = [dom[k] for k in keys]
    palette = {"outer": "#dd6b20", "middle": "#5a67d8",
               "inner": "#319795"}
    colors = [palette.get(k, "#777") for k in keys]
    ax_shell.bar(np.arange(len(keys)), vals, color=colors)
    ax_shell.set_xticks(np.arange(len(keys)))
    ax_shell.set_xticklabels(keys)
    ax_shell.set_ylabel("mean |h|")
    consistency = sig.cross_shell_consistency()
    ax_shell.set_title(
        f"shell dominance  (cross-shell consistency r = {consistency:+.2f})"
    )
    ax_shell.grid(alpha=0.3, axis="y")

    # --- Row 3: per-cycle propagation across shells -------------------
    n_top = min(40, len(sig.contributions))
    if n_top == 0:
        ax_prop.text(0.5, 0.5,
                      "(no cycles touched this query)",
                      ha="center", va="center",
                      transform=ax_prop.transAxes,
                      fontsize=11, color="gray")
        ax_prop.set_axis_off()
    else:
        sorted_contribs = sorted(
            sig.contributions,
            key=lambda c: np.mean(list(c.per_shell_magnitude.values())),
            reverse=True,
        )[:n_top]
        x_positions = np.arange(len(sig.shells))
        for c in sorted_contribs:
            ys = [c.per_shell_magnitude.get(s, 0.0)
                   for s in sig.shells]
            colour = "#2b6cb0" if c.sigma_prod > 0 else "#c53030"
            ax_prop.plot(x_positions, ys, color=colour,
                          alpha=0.4, linewidth=0.9,
                          marker="o", markersize=3)
        ax_prop.set_xticks(x_positions)
        ax_prop.set_xticklabels(list(sig.shells))
        ax_prop.set_ylabel(r"$|h_c|$")
        ax_prop.set_title(
            f"per-cycle propagation across shells  "
            f"(top {n_top} firing cycles; blue = balanced, red = unbalanced)"
        )
        ax_prop.grid(alpha=0.3)

    # --- Row 4: arc-weight panel (optional) ---------------------------
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
        ax_arc.set_xlabel(r"$|w_e|$")
        ax_arc.set_ylabel("count")
        ax_arc.set_title("arc-weight magnitudes per σ")
        ax_arc.legend(loc="best", frameon=False, fontsize=9)
        ax_arc.grid(alpha=0.3)

    fig.tight_layout()
    return axes
