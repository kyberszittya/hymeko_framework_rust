"""Leakage-free per-node structural features for the inductive rotor line.

The rotor signed-link line was shown (2026-06-17 → 2026-06-18) to be bottlenecked
by a *degree-only* node input: four readout/propagation levers landed flat, while
adding signed-cycle participation lifted the deduped AUROC (+0.008 alpha / +0.019
otc, gated). So the lever is the **input**, and this module is its extensible home.

A :class:`FeatureExtractor` registry (Strategy, §6.5 #1/#9 — no per-variant
hardcoding): :func:`build_node_features` concatenates a *spec* of named extractors
into one ``(n_nodes, D)`` matrix, with ``D`` derived from the spec. Every extractor
is a pure function of the **train-only** ``(edges, signs, n_nodes)`` — inductive and
leakage-free (the shuffle gate must still fall to chance per config).

Extractors:

* ``degree`` (6) — the original signed-degree profile (default; reproduces the
  legacy ``_structural_features`` bit-for-bit).
* ``cycle_k3`` (3), ``cycle_k34`` (6) — per-node balanced/unbalanced signed
  ``k``-cycle participation + balanced fraction (``_cycle_features``; capped
  enumeration → the per-node count is a *sample* under the cap).
* ``walk_k3`` (9) — **exact** per-node signed ``A^k`` reach profile (``k=1..3``):
  signed reach ``A^k·1``, total reach ``|A|^k·1``, and their ratio. ``O(K·nnz)``
  sparse mat-vecs, no cap/sampling bias — the multi-scale generalisation of signed
  degree (``k=1``).
* ``ratios`` (2) — signed and unsigned local clustering (triangle density), via
  ``diag(A^3)`` computed as ``(A ∘ A²)·1`` without forming ``A^3``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]  # scipy ships no py.typed stubs

# ─── degree profile (the legacy default; canonical home) ────────────────────
STRUCT_DIM = 6


def _structural_features(
    edges: np.ndarray, signs: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Per-node inductive features from the (train-only) signed degree profile.

    Returns ``(n_nodes, STRUCT_DIM)`` float32: log-degree, log-pos-degree,
    log-neg-degree, positive fraction, negative fraction, and signed balance
    ``(pos-neg)/deg``. Functions of local structure, not node identity — no
    memorisation channel.
    """
    deg = np.zeros(n_nodes, dtype=np.float64)
    pos = np.zeros(n_nodes, dtype=np.float64)
    neg = np.zeros(n_nodes, dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        deg[u] += 1.0
        deg[v] += 1.0
        if s > 0:
            pos[u] += 1.0
            pos[v] += 1.0
        else:
            neg[u] += 1.0
            neg[v] += 1.0
    eps = 1e-6
    feats = np.stack([
        np.log1p(deg), np.log1p(pos), np.log1p(neg),
        pos / (deg + eps), neg / (deg + eps), (pos - neg) / (deg + eps),
    ], axis=1)
    return feats.astype(np.float32)


# ─── signed k-cycle participation (capped enumeration; canonical home) ──────
CYC_KS = (3,)
CAP_PER_K = 2_000_000  # subsample guard for dense graphs (logged if hit)
CYC_DIM = STRUCT_DIM + 3 * len(CYC_KS)


def _cycle_features(
    edges: np.ndarray, signs: np.ndarray, n_nodes: int,
    ks: Sequence[int] = CYC_KS, cap: int | None = CAP_PER_K,
) -> np.ndarray:
    """Per-node signed-cycle participation: for each arity ``k``, the (log) count
    of *balanced* and *unbalanced* k-cycles the node sits in, plus the balanced
    fraction. Balanced = even number of negative edges (Davis). Reuses
    ``core.n_tuples.construct_k_arrays``; train-only (inductive).

    Returns ``(n_nodes, 3 * len(ks))`` float32. Under ``cap`` the per-node counts
    are a *sample* (enumeration is bounded), unlike the exact walk profile.
    """
    from signedkan_wip.src.core.n_tuples import construct_k_arrays
    from signedkan_wip.src.datasets.legacy import SignedGraph

    g = SignedGraph(edges=edges.astype(np.int64),
                    signs=signs.astype(np.int64), n_nodes=n_nodes)
    cols = []
    for k in ks:
        v, _sigma, edge_signs = construct_k_arrays(g, k, max_cycles=cap)
        bal = np.zeros(n_nodes, dtype=np.float64)
        unb = np.zeros(n_nodes, dtype=np.float64)
        v = np.asarray(v)
        if v.size:
            es = np.asarray(edge_signs)
            is_bal = ((es == -1).sum(axis=1) % 2 == 0)
            for j in range(v.shape[1]):
                np.add.at(bal, v[:, j][is_bal], 1.0)
                np.add.at(unb, v[:, j][~is_bal], 1.0)
        tot = bal + unb
        cols += [np.log1p(bal), np.log1p(unb), bal / (tot + 1e-6)]
    return np.stack(cols, axis=1).astype(np.float32)


# ─── signed adjacency helpers (shared by walk + ratio extractors) ───────────
def _signed_adjacency(
    edges: np.ndarray, signs: np.ndarray, n_nodes: int
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Symmetric signed ``A`` (entries ±1) and unsigned ``|A|`` as CSR.

    Preconditions: ``edges`` is ``(E,2)``; ``signs`` in ``{+1,-1}``.
    Postconditions: both ``(n,n)``, symmetric, zero diagonal; built from the given
    (train-only) edges, so they carry no test-edge structure.
    """
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    s = np.where(signs > 0, 1.0, -1.0)
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    sgn = np.concatenate([s, s])
    a_signed = sp.csr_matrix((sgn, (rows, cols)), shape=(n_nodes, n_nodes))
    a_abs = sp.csr_matrix((np.abs(sgn), (rows, cols)), shape=(n_nodes, n_nodes))
    # Collapse any parallel entries to a single ±1 / 1 (sign(sum) keeps it in {±1}).
    a_signed.sum_duplicates()
    a_abs.sum_duplicates()
    a_signed.data = np.sign(a_signed.data)
    a_abs.data = np.minimum(a_abs.data, 1.0)
    return a_signed, a_abs


WALK_K = 3
WALK_DIM = 3 * WALK_K


def _walk_profile_features(
    edges: np.ndarray, signs: np.ndarray, n_nodes: int, K: int = WALK_K,
) -> np.ndarray:
    """Exact per-node signed ``A^k`` reach profile, ``k = 1..K``.

    For each ``k``: signed reach ``s_k = A^k·1`` (net balanced−unbalanced walk
    weight reaching the node), total reach ``t_k = |A|^k·1``, and the ratio
    ``s_k/t_k ∈ [-1,1]`` (a multi-scale local-balance descriptor). ``k=1`` is signed
    degree; higher ``k`` is multi-hop signed structure that degree alone cannot see.

    Preconditions: ``K >= 1``. Postconditions: ``(n_nodes, 3*K)`` float32; **exact**
    (no enumeration / cap), ``O(K·nnz)`` sparse mat-vecs; train-only (inductive).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1; got {K}")
    a_signed, a_abs = _signed_adjacency(edges, signs, n_nodes)
    sv = np.ones(n_nodes, dtype=np.float64)
    tv = np.ones(n_nodes, dtype=np.float64)
    eps = 1e-6
    cols = []
    for _ in range(K):
        sv = a_signed @ sv            # signed k-reach  A^k · 1
        tv = a_abs @ tv               # total k-reach  |A|^k · 1
        cols += [np.sign(sv) * np.log1p(np.abs(sv)),
                 np.log1p(tv),
                 sv / (tv + eps)]
    return np.stack(cols, axis=1).astype(np.float32)


RATIO_DIM = 2


def _ratio_features(
    edges: np.ndarray, signs: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Per-node signed + unsigned local clustering (triangle density).

    ``diag(A^3)_i = 2·(signed) triangles through i`` is computed as the row-sum of
    ``A ∘ A²`` (no ``A^3`` materialised). Returns ``(n_nodes, 2)`` float32:
    unsigned clustering ``tri_i / C(deg_i,2) ∈ [0,1]`` and signed clustering
    ``(balanced−unbalanced)_i / C(deg_i,2) ∈ [-1,1]``. Train-only (inductive).
    """
    a_signed, a_abs = _signed_adjacency(edges, signs, n_nodes)
    deg = np.asarray(a_abs.sum(axis=1)).ravel()
    pairs = deg * (deg - 1.0) / 2.0                       # C(deg, 2)
    # diag(M^3)/2 = (M ∘ (M@M)) · 1 / 2, per node.
    tri_abs = np.asarray(a_abs.multiply(a_abs @ a_abs).sum(axis=1)).ravel() / 2.0
    tri_sgn = np.asarray(a_signed.multiply(a_signed @ a_signed).sum(axis=1)).ravel() / 2.0
    eps = 1e-6
    return np.stack([tri_abs / (pairs + eps),
                     tri_sgn / (pairs + eps)], axis=1).astype(np.float32)


# ─── registry ───────────────────────────────────────────────────────────────
FeatureFn = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class FeatureExtractor:
    """A named, fixed-width, leakage-free per-node feature extractor."""
    name: str
    dim: int
    fn: FeatureFn

    def __call__(self, edges: np.ndarray, signs: np.ndarray,
                 n_nodes: int) -> np.ndarray:
        out = self.fn(edges, signs, n_nodes)
        assert out.shape == (n_nodes, self.dim), \
            f"{self.name}: expected ({n_nodes}, {self.dim}); got {out.shape}"
        return out


_REGISTRY: dict[str, FeatureExtractor] = {}


def _register(name: str, dim: int, fn: FeatureFn) -> None:
    _REGISTRY[name] = FeatureExtractor(name, dim, fn)


_register("degree", STRUCT_DIM, _structural_features)
_register("cycle_k3", 3, lambda e, s, n: _cycle_features(e, s, n, ks=(3,)))
_register("cycle_k34", 6, lambda e, s, n: _cycle_features(e, s, n, ks=(3, 4)))
_register("walk_k3", WALK_DIM, _walk_profile_features)
_register("ratios", RATIO_DIM, _ratio_features)

DEFAULT_SPEC: tuple[str, ...] = ("degree",)


def available_features() -> list[str]:
    """Sorted names of the registered extractors."""
    return sorted(_REGISTRY)


def feature_dim(spec: Sequence[str]) -> int:
    """Total width of the concatenated feature vector for ``spec``."""
    return sum(_get(name).dim for name in spec)


def _get(name: str) -> FeatureExtractor:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown feature {name!r}; available: {available_features()}"
        ) from None


def build_node_features(
    spec: Sequence[str], edges: np.ndarray, signs: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Concatenate the ``spec`` extractors into one ``(n_nodes, feature_dim(spec))``.

    Preconditions: ``spec`` is non-empty; every name is registered
    (``available_features``); ``edges``/``signs`` are train-only.
    Postconditions: float32 ``(n_nodes, feature_dim(spec))``; a pure function of the
    inputs (inductive, leakage-free). ``("degree",)`` reproduces
    ``_structural_features`` exactly.
    """
    if not spec:
        raise ValueError("feature spec must name at least one extractor")
    cols = [_get(name)(edges, signs, n_nodes) for name in spec]
    return np.concatenate(cols, axis=1).astype(np.float32)
