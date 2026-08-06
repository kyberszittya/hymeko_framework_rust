"""Render star- vs clique-expansion heatmaps for the seminar "tensor view" slide.

Deterministic (seeded) 50-node / 50-edge / 0.30-incidence-density *signed*
hypergraph — the same example slide 12 cites — materialised two ways:

  * star expansion   = signed vertex×hyperedge incidence B ∈ {-1,0,+1}
                       (the bipartite Levi/Berge form; sparse, O(|E|·d̄));
  * clique expansion = vertex×vertex co-membership adjacency A
                       (each hyperedge becomes a clique; dense, O(|E|·d̄²)).

Outputs two PNGs into docs/seminar/figures/ suitable as a slide background, and
prints each materialisation's NNZ (the efficiency argument). Pure stdlib +
numpy + matplotlib (already in the env — no new dependency).

Run:  python docs/seminar/make_tensor_heatmaps.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
N_NODES = 50
N_EDGES = 50
DENSITY = 0.30
SEED = 0


def build_signed_hypergraph() -> tuple[np.ndarray, np.ndarray]:
    """Return (incidence B, signs S): B[v,e]=1 iff v∈e; S[v,e]∈{-1,+1}."""
    rng = np.random.default_rng(SEED)
    incidence = (rng.random((N_NODES, N_EDGES)) < DENSITY).astype(np.int8)
    # Guarantee every hyperedge has arity >= 2 (no empty/degenerate columns).
    for e in range(N_EDGES):
        if incidence[:, e].sum() < 2:
            incidence[rng.choice(N_NODES, size=2, replace=False), e] = 1
    signs = rng.choice([-1, 1], size=(N_NODES, N_EDGES)).astype(np.int8)
    return incidence, signs


def star_matrix(incidence: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Signed vertex×hyperedge incidence (the star / Levi expansion)."""
    return (incidence * signs).astype(np.int8)


def clique_matrix(incidence: np.ndarray) -> np.ndarray:
    """Vertex×vertex co-membership adjacency (the clique expansion)."""
    adj = incidence @ incidence.T          # shared-hyperedge counts
    np.fill_diagonal(adj, 0)
    return adj.astype(np.int32)


def _save_heatmap(mat: np.ndarray, path: Path, cmap: str, title: str) -> int:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    vmax = float(np.abs(mat).max()) or 1.0
    ax.imshow(mat, cmap=cmap, vmin=-vmax if mat.min() < 0 else 0, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return int(np.count_nonzero(mat))


def save_aggregated(star, clique, star_nnz, clique_nnz) -> Path:  # type: ignore[no-untyped-def]
    """One aggregated tensor-view figure: star (sparse) vs clique (dense),
    with the matrix DIMENSIONS labelled and the honest takeaway —
    star uses fewer NNZ but augments the vertex set with |E| hub dimensions;
    the sparse structure is preferred."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200)
    vmaxL = float(np.abs(star).max()) or 1.0
    axL.imshow(star, cmap="coolwarm", vmin=-vmaxL, vmax=vmaxL, interpolation="nearest", aspect="equal")
    axL.set_title("Star expansion — signed incidence", fontsize=12, fontweight="bold")
    axL.set_xlabel(f"hyperedges  |E| = {N_EDGES}", fontsize=10)
    axL.set_ylabel(f"vertices  |V| = {N_NODES}", fontsize=10)
    axL.text(0.5, -0.20, f"sparse:  O(|E|·d̄)   ·   NNZ = {star_nnz} (illustrative)",
             transform=axL.transAxes, ha="center", fontsize=10, color="#1b6ca8", fontweight="bold")
    axR.imshow(clique, cmap="magma", vmin=0, vmax=float(clique.max()) or 1.0, interpolation="nearest", aspect="equal")
    axR.set_title("Clique expansion — co-membership", fontsize=12, fontweight="bold")
    axR.set_xlabel(f"vertices  |V| = {N_NODES}", fontsize=10)
    axR.set_ylabel(f"vertices  |V| = {N_NODES}", fontsize=10)
    axR.text(0.5, -0.20, f"dense:  O(|E|·d̄²)   ·   NNZ = {clique_nnz} (illustrative)",
             transform=axR.transAxes, ha="center", fontsize=10, color="#8a2d6b", fontweight="bold")
    for ax in (axL, axR):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Two materialisations of the same hypergraph", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015,
             "Star uses far fewer non-zeros; it pays one extra hub “dimension” per hyperedge, "
             "but the sparse structure is preferred over the dense clique blow-up.",
             ha="center", fontsize=10.5, style="italic")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = OUT / "tensor_view.png"
    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return path


def save_panels(star, clique) -> Path:  # type: ignore[no-untyped-def]
    """Clean two-panel figure (matrices + titles + axis labels only) used on the
    "Star vs clique" slide. No supertitle, no illustrative-NNZ captions, no baked
    takeaway — the slide itself carries those, so they cannot drift out of sync."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    vmaxL = float(np.abs(star).max()) or 1.0
    axL.imshow(star, cmap="coolwarm", vmin=-vmaxL, vmax=vmaxL, interpolation="nearest", aspect="equal")
    axL.set_title("Star expansion — signed incidence", fontsize=13, fontweight="bold", color="#0E6B4F")
    axL.set_xlabel(f"hyperedges  |E| = {N_EDGES}", fontsize=11)
    axL.set_ylabel(f"vertices  |V| = {N_NODES}", fontsize=11)
    axR.imshow(clique, cmap="magma", vmin=0, vmax=float(clique.max()) or 1.0, interpolation="nearest", aspect="equal")
    axR.set_title("Clique expansion — co-membership", fontsize=13, fontweight="bold", color="#2147A8")
    axR.set_xlabel(f"vertices  |V| = {N_NODES}", fontsize=11)
    axR.set_ylabel(f"vertices  |V| = {N_NODES}", fontsize=11)
    for ax in (axL, axR):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=1.0)
    path = OUT / "tensor_panels.png"
    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    incidence, signs = build_signed_hypergraph()
    star = star_matrix(incidence, signs)
    clique = clique_matrix(incidence)

    star_nnz = _save_heatmap(
        star, OUT / "star_expansion.png", "coolwarm",
        f"Star expansion — signed incidence ({N_NODES}×{N_EDGES})",
    )
    clique_nnz = _save_heatmap(
        clique, OUT / "clique_expansion.png", "magma",
        f"Clique expansion — co-membership ({N_NODES}×{N_NODES})",
    )
    agg = save_aggregated(star, clique, star_nnz, clique_nnz)
    panels = save_panels(star, clique)

    # Faded, square background variant of the aggregated view — composited onto
    # white so it reads as a visible (but text-safe) slide background. FADE is the
    # visibility knob: 0 = invisible (pure white), 1 = full opacity.
    FADE = 0.5
    rgba = plt.imread(str(agg))                  # H×W×4 in [0,1]
    rgb, alpha = rgba[..., :3], rgba[..., 3:4]
    on_white = rgb * alpha + (1.0 - alpha)       # flatten transparency onto white
    faded = np.clip((1.0 - FADE) + on_white * FADE, 0.0, 1.0)
    bg = OUT / "tensor_view_bg.png"
    plt.imsave(str(bg), faded)

    print(f"star  expansion: {star_nnz} NNZ  -> {OUT / 'star_expansion.png'}")
    print(f"clique expansion: {clique_nnz} NNZ -> {OUT / 'clique_expansion.png'}")
    print(f"ratio clique/star = {clique_nnz / max(star_nnz, 1):.1f}x")
    print(f"aggregated tensor view -> {agg}")
    print(f"clean tensor panels (slide 12) -> {panels}")
    print(f"aggregated tensor background (FADE={FADE}) -> {bg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
