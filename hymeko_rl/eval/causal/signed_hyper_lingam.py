"""SignedHyperLiNGAM — native signed-hyperedge causal discovery over the same SSG the spec arbiter uses.

DirectLiNGAM recovers a *pairwise* linear SEM ``x = Bx + e``; we then post-hoc split ``B = A⁺ − A⁻`` and group
pairwise edges into mechanism hyperedges. This module makes the causal primitive **native**: for each effect (in
DirectLiNGAM's non-Gaussian causal order — reused, not re-implemented), select its **signed-hyperedge tail** — the
subset of prior variables that *jointly* produce it — by reducing tail-selection to a ``hymeko_pgraph`` SSG
solution-structure search (the *same* reduction ``pgraph_refine`` uses for conjunct-pruning), scored by an
**interaction-aware** fit. Each tail member is then signed (``A⁺`` excitatory / ``A⁻`` inhibitory), and the result
is emitted directly as a :class:`~hymeko_rl.eval.causal.hymeko_emit.CausalHypergraph` (HyMeKo IR) — no matrix to
translate.

**Why it exists (measured, honest).** On *additive* SEMs (linear or per-edge-monotone-nonlinear) DirectLiNGAM
already recovers the support perfectly (F1 ≈ 1.0) — SignedHyperLiNGAM **ties**, no headroom. The win is on
**joint-interaction** mechanisms (e.g. ``x4 = x0·x1 + x2``): a parent's *marginal* effect vanishes
(``corr(x4,x0) ≈ 0`` because ``x1`` is zero-mean), so pairwise DirectLiNGAM *structurally* misses ``x0,x1``
(recall ≈ 0.33), while SignedHyperLiNGAM scores the *subset jointly* and recovers the hyperedge ``{x0,x1}→x4``
(recall 1.0). This is exactly ``signed hypergraphs capture joint causal structure pairwise misses`` — on causal
discovery. Doctrine unchanged: **it PROPOSES structure; ablations decide.**
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from hymeko_rl.eval.causal.lingam import DirectLiNGAM, LingamConfig, _standardize
from hymeko_rl.eval.spec_bench.pgraph_refine import predicates_to_pgraph_hymeko, solve_ssg

# Per-column feature basis (identity + monotone nonlinearities, the harness's per-edge family). The identity term
# carries the sign and the linear part; the monotone terms let a purely-nonlinear additive edge (e.g. signed-square)
# be fully fit — so tail-scoring does not drop it, and its member weight is not zeroed by a linear-only statistic.
_BASIS: "tuple[Callable[[np.ndarray], np.ndarray], ...]" = (
    lambda z: z, np.tanh, lambda z: np.sign(z) * z * z, lambda z: z / (1.0 + np.abs(z)))
_N_BASIS = len(_BASIS)


@dataclass(frozen=True)
class SHLConfig:
    """Configuration for :class:`SignedHyperLiNGAM`.

    ``max_predecessors`` caps the per-target candidate pool (most-recent predecessors in causal order) so the SSG
    subset search stays bounded — it does NOT filter by marginal dependence (that would drop joint parents);
    ``parsimony`` is the per-tail-member penalty on the interaction-R² fit score (drops distractors);
    ``joint_sign_eps`` — a tail member whose *linear* coefficient is below this is reported as ``sign = 0`` (a
    pure-joint member, no clean A⁺/A⁻); ``weight_floor`` drops members with negligible joint-aware weight."""

    lingam: LingamConfig = field(default_factory=LingamConfig)
    max_predecessors: int = 8
    parsimony: float = 0.03
    joint_sign_eps: float = 0.05
    weight_floor: float = 0.05


@dataclass(frozen=True)
class SignedHyperResult:
    """A recovered signed-hyperedge causal model. **Proposed structure, not proof.**

    ``order`` — variable indices, most exogenous → most endogenous (DirectLiNGAM's order). ``mechanisms`` — per
    effect index, the list of ``(cause_index, sign, weight, is_joint)`` tail members (``sign ∈ {+1,-1,0}``; ``0`` =
    pure-joint). ``adjacency`` — a signed ``B``-form (``B[effect, cause] = sign·weight``) for
    :func:`recovery_metrics` comparability with DirectLiNGAM. ``names`` — column names."""

    order: list[int]
    mechanisms: "dict[int, list[tuple[int, int, float, bool]]]"
    adjacency: np.ndarray
    names: list[str]

    def hyperedges(self) -> "list[tuple[tuple[str, ...], str, tuple[int, ...]]]":
        """Native hyperedges: ``(tail_names, head_name, tail_signs)`` — the multi-tail mechanisms."""
        out: "list[tuple[tuple[str, ...], str, tuple[int, ...]]]" = []
        for eff, tail in self.mechanisms.items():
            if tail:
                out.append((tuple(self.names[c] for c, _s, _w, _j in tail), self.names[eff],
                            tuple(s for _c, s, _w, _j in tail)))
        return out

    def to_causal_hypergraph(self, name: str = "signed_hyper_lingam") -> object:
        """Emit a :class:`CausalHypergraph` (mechanism form) — one signed hyperedge per multi-tail mechanism."""
        from hymeko_rl.eval.causal.mechanism_proposal import MechanismProposal, proposals_to_causal_hypergraph
        proposals = [
            MechanismProposal(
                name=f"mech_{self.names[eff]}", tail=tuple(self.names[c] for c, _s, _w, _j in tail),
                head=(self.names[eff],), strength=float(np.mean([abs(w) for _c, _s, w, _j in tail])),
                sign=1 if sum(s for _c, s, _w, _j in tail) >= 0 else -1, confidence=1.0,
                source="signed_hyper_lingam", evidence={"joint": [c for c, _s, _w, j in tail if j]})
            for eff, tail in self.mechanisms.items() if tail]
        return proposals_to_causal_hypergraph(self.names, proposals, name=name)


class SignedHyperLiNGAM:
    """Native signed-hyperedge causal discovery: DirectLiNGAM ordering + SSG tail-selection + per-member signs.

    # Preconditions ``fit`` receives ``X`` shape ``(n, d)`` with ``n > d``, finite, ≥2 non-constant columns.
    # Postconditions each mechanism's tail is a subset of the effect's causal-order predecessors (acyclic);
      ``adjacency`` is a signed ``B``-form strictly triangular in ``order``.
    # Invariants holds no mutable state across ``fit`` calls; config is immutable (§6.5 #11)."""

    def __init__(self, config: "SHLConfig | None" = None) -> None:
        self.config = config or SHLConfig()

    def fit(self, x: np.ndarray, names: "list[str] | None" = None) -> SignedHyperResult:
        x = np.asarray(x, dtype=np.float64)
        self._validate(x)
        d = x.shape[1]
        names = list(names) if names is not None else [f"x{i}" for i in range(d)]
        order = DirectLiNGAM(self.config.lingam).fit(x, names).order
        xs = np.column_stack([_standardize(x[:, i]) for i in range(d)])
        mechanisms, adjacency = self._discover(xs, order, d)
        return SignedHyperResult(order=order, mechanisms=mechanisms, adjacency=adjacency, names=names)

    # -- helpers --------------------------------------------------------------------------------------------------
    @staticmethod
    def _validate(x: np.ndarray) -> None:
        if x.ndim != 2 or x.shape[0] <= x.shape[1]:
            raise ValueError(f"SignedHyperLiNGAM.fit: need X (n, d) with n > d, got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("SignedHyperLiNGAM.fit: X contains non-finite values")

    def _discover(self, xs: np.ndarray, order: list[int], d: int,
                  ) -> "tuple[dict[int, list[tuple[int, int, float, bool]]], np.ndarray]":
        """Per effect (in causal order): select+fit its signed-hyperedge tail; fill the signed ``B``-form."""
        adjacency = np.zeros((d, d), dtype=np.float64)
        mechanisms: "dict[int, list[tuple[int, int, float, bool]]]" = {}
        for pos, target in enumerate(order):
            members = self._mechanism_for_target(xs, target, order[:pos])
            if members:
                mechanisms[target] = members
                for c, s, w, _j in members:
                    adjacency[target, c] = (s if s != 0 else 1) * w
        return mechanisms, adjacency
    def _mechanism_for_target(self, xs: np.ndarray, target: int, preds: list[int],
                              ) -> "list[tuple[int, int, float, bool]]":
        """Select the target's signed-hyperedge tail from its predecessors and fit the members (empty if none)."""
        cands = self._candidate_predecessors(preds)
        if not cands:
            return []
        return self._fit_members(xs, target, self._select_tail(xs, target, cands))

    def _fit_members(self, xs: np.ndarray, target: int, tail: list[int]) -> "list[tuple[int, int, float, bool]]":
        """Fit the selected tail's interaction OLS and read each member's weight/sign from the coefficients.

        A member's weight is ``|linear coef| + Σ|interaction coefs it participates in|`` — so a **joint** member
        (near-zero *linear* coef but large *interaction* coef, e.g. ``x0`` in ``x0·x1``) keeps a large weight and
        survives the floor, where a marginal statistic would zero it. Sign is the linear-coef sign, or ``0`` (a
        pure-joint member) when the linear part is below ``joint_sign_eps``."""
        cols = list(tail)
        if not cols:
            return []
        design, pairs = self._design(xs, cols)
        coef, *_ = np.linalg.lstsq(design, xs[:, target], rcond=None)
        inter_c = coef[_N_BASIS * len(cols): _N_BASIS * len(cols) + len(pairs)]
        members: "list[tuple[int, int, float, bool]]" = []
        for k, c in enumerate(cols):
            mono = coef[k * _N_BASIS: (k + 1) * _N_BASIS]                 # this col's basis coefs (mono[0] = linear)
            id_coef = float(mono[0])
            w_mono = float(np.abs(mono).sum())                           # additive-monotone contribution
            w_int = sum(abs(float(inter_c[p])) for p, (a, b) in enumerate(pairs) if k in (a, b))
            weight = w_mono + w_int
            if weight < self.config.weight_floor:
                continue
            is_joint = w_int > w_mono                                    # dominated by interaction ⇒ pure-joint
            sign = 0 if (is_joint or abs(id_coef) < self.config.joint_sign_eps) else (1 if id_coef > 0 else -1)
            members.append((c, sign, weight, is_joint))
        return members

    def _candidate_predecessors(self, preds: list[int]) -> list[int]:
        """The candidate pool = all causal-order predecessors, capped to the most-recent ``max_predecessors``.

        Crucially it does **not** filter by *marginal* dependence: a joint parent's marginal effect vanishes
        (``corr(x4,x0)≈0`` for ``x4=x0·x1``), so a marginal filter would drop exactly the parents we need. The SSG
        subset search + interaction-aware R² recover the joint tail from the full pool; the cap only bounds the
        ``2^m`` search for large ``d`` (the dropped predecessors, if any, are the most causally distant)."""
        return list(preds) if len(preds) <= self.config.max_predecessors else list(preds[-self.config.max_predecessors:])

    @staticmethod
    def _design(xs: np.ndarray, cols: Sequence[int]) -> "tuple[np.ndarray, list[tuple[int, int]]]":
        """Design matrix ``[per-col monotone basis | pairwise products | intercept]`` for a candidate tail.

        Columns are laid out col-major: col ``k``'s ``_N_BASIS`` basis features occupy ``[k·_N_BASIS, …)``, then the
        pairwise products (interaction), then a constant — so a member's contribution is attributable by slice."""
        col_feats = [f(xs[:, c]) for c in cols for f in _BASIS]
        pairs = list(itertools.combinations(range(len(cols)), 2))
        products = [xs[:, cols[a]] * xs[:, cols[b]] for a, b in pairs]
        return np.column_stack([*col_feats, *products, np.ones(xs.shape[0])]), pairs

    def _interaction_r2(self, xs: np.ndarray, target: int, cols: Sequence[int]) -> float:
        """Penalised R² of an OLS over the per-column monotone basis + pairwise products predicting the target."""
        if not cols:
            return -1e9
        design, _pairs = self._design(xs, cols)
        coef, *_ = np.linalg.lstsq(design, xs[:, target], rcond=None)
        resid = xs[:, target] - design @ coef
        r2 = 1.0 - float(resid @ resid) / float(xs[:, target] @ xs[:, target] + 1e-12)
        return r2 - self.config.parsimony * len(cols)

    def _select_tail(self, xs: np.ndarray, target: int, preds: list[int]) -> list[int]:
        """Pick the joint tail: the SSG subset of ``preds`` maximising the interaction-aware penalised R²."""
        subsets = self._ssg_subsets(preds)
        best = max(subsets, key=lambda c: self._interaction_r2(xs, target, c))
        return list(best)

    def _ssg_subsets(self, preds: list[int]) -> "list[list[int]]":
        """Feasible tail subsets via ``hymeko_pgraph`` SSG (the arbiter's reduction); exhaustive fallback."""
        pgraph_preds = [(f"cause_{c} >= 0", f"cause_{c}") for c in preds]
        structures = solve_ssg(predicates_to_pgraph_hymeko(pgraph_preds))
        if structures is None:
            return [list(c) for r in range(1, len(preds) + 1) for c in itertools.combinations(preds, r)]
        return [[preds[int(u[1:])] for u in s if u[1:].isdigit()] for s in structures] or [preds]


def _independent_pair(x: np.ndarray, parents: list[int], max_corr: float) -> "tuple[int, int] | None":
    """The most-independent parent pair (min |corr|), or ``None`` if even the best exceeds ``max_corr``.

    A clean joint-product demonstration needs (near-)uncorrelated parents: if the pair is correlated the product
    ``z_a·z_c`` picks up a *marginal* component (``corr(x_a, z_a z_c) ∝ ρ``), which pairwise LiNGAM would then
    detect — defeating the ``pairwise misses joint`` property. So we only joint-ify an independent pair."""
    best: "tuple[float, tuple[int, int]] | None" = None
    for a, c in itertools.combinations(parents, 2):
        rho = abs(float(np.corrcoef(x[:, a], x[:, c])[0, 1]))
        if best is None or rho < best[0]:
            best = (rho, (a, c))
    return best[1] if best is not None and best[0] <= max_corr else None


def sample_interaction_sem(b: np.ndarray, n_samples: int, *, seed: int, joint_frac: float = 0.6,
                           noise_scale: float = 0.4, max_pair_corr: float = 0.2) -> np.ndarray:
    """Sample ``(n, d)`` from a SEM over support ``b`` where eligible effects use a **joint product** mechanism over
    a near-independent parent pair (non-additive — marginal effects vanish) plus additive remaining parents.

    ``b[effect, cause]`` gives the support and per-edge weight/sign; an effect is joint iff it has an independent
    parent pair (``|corr| <= max_pair_corr``) and ``rng < joint_frac``. The standardised product is unit-scale (a
    raw product of small root-noise variables would be swamped by ``eps``). Non-Gaussian (uniform) noise per LiNGAM.

    # Preconditions ``b`` square, strictly lower-triangular (acyclic). # Postconditions generating edges ⊆ nonzeros
      of ``b``; joint effects have $\\approx 0$ marginal correlation with each product-pair parent."""
    d = b.shape[0]
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, d), dtype=np.float64)
    eps = rng.uniform(-1.0, 1.0, size=(n_samples, d)) * noise_scale
    for i in range(d):
        parents = [j for j in range(i) if b[i, j] != 0.0]
        acc = eps[:, i].copy()
        joint_pair = (_independent_pair(x, parents, max_pair_corr)
                      if len(parents) >= 2 and rng.random() < joint_frac else None)
        if joint_pair is not None:
            a, c = joint_pair
            acc += float(abs(b[i, a]) + abs(b[i, c])) * _standardize(x[:, a]) * _standardize(x[:, c])
        for j in parents:
            if joint_pair is not None and j in joint_pair:
                continue
            acc += b[i, j] * x[:, j]                                             # additive remainder
        x[:, i] = acc
    return x
