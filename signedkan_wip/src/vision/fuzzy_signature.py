"""FuzzySignatureLayer / Classifier — Atanassov-pair redesign with
C+B fallbacks (revision 3).

Plan:          ``docs/plans/2026-05-30-fuzzy-signature-layer/plan.tex``
Background:    ``docs/plans/2026-05-30-fuzzy-signature-layer/background.tex``
               (20 pp foundations — primitives, Atanassov IFS, Kóczy
               signatures, defuzzification, linguistic hedges, TSK
               derivation, OWA, open questions).
Report rev 2:  ``reports/2026-05-30-fuzzy-signature-atanassov-redesign.md``
               (failure diagnosis: contraction-map collapse).

Revision 1 (abandoned) used a single CR membership per channel and
collapsed to ~0.1135 across all 9 (t-norm, t-conorm) combinations on the
production-scale smoke. The user rejected the obvious patch (Linear+σ
channel mixer) on the grounds that HSiKAN already has the plasticity;
the right move is to reuse it in fuzzy-systems language.

Revision 2 (Atanassov pair, polarity-mean dropped, learnable τ,
t-conorm e→v) failed the pre-registered 0.5 smoke gate at
test_accuracy=0.114 because the layer is a contraction map: K=25
t-norm-min collapses fuzzy variance, 9-edge probsum saturates to ~1.0,
and the 0.5-averaging residual compresses the rest. [0,1]-preservation
held; non-contraction did not.

Revision 3 (this file) adds two orthogonal axes from the report's
ordered fallbacks — without removing revision 2's defaults from the
search space:

  * ``cr_input_scale`` ∈ {"unit_to_grid", "raw"}, default
    "unit_to_grid": rescale fuzzy input x∈[0,1] to 6x-3 ∈ [-3,3] before
    the CR spline so the full 8-control-point grid is used. Revision 2
    used only the upper ~1.5 of 7 segments.
  * ``residual_kind`` ∈ {"avg", "max", "probsum"}, default "max":
    replace 0.5(x + h_v) with a non-contracting fuzzy operator. "max"
    is max-conorm; "probsum" is probabilistic sum; "avg" preserves
    revision 2's behaviour. All three keep [0,1] (Theorem 7.1 extends
    trivially).

Revision 2's design is otherwise preserved:

  * Drop the polarity-mean reference (HSiKAN's μ_e). Centre the gate at
    ½, the natural midpoint of the fuzzy unit interval.
  * Two CR branches per channel: μ⁺ and μ⁻, the Atanassov intuitionistic
    fuzzy pair.
  * Learnable per-channel hedge τ (softplus-positive) controlling the
    sharpness of the μ⁺/μ⁻ mixer. τ→∞: crisp commit; τ→0: uniform mix.
  * T-norm at v→e (fuzzy AND) — already in tree from revision 1.
  * T-conorm at e→v (fuzzy OR) — replaces revision 1's einsum-sum
    scatter, closing the fuzzy loop to a layer that preserves [0,1]
    inside.
  * Rule strength W_e: one tied scalar per layer, clamped to [0,1] via
    σ(W_e)·2 ∧ 1.

The original revision-2 behaviour is exactly
``cr_input_scale="raw", residual_kind="avg"`` and remains a runnable
point in the search space for ablations.

Each learnable parameter is in bijection with a named fuzzy-systems
primitive — see plan.tex §"Component-to-fuzzy-logic mapping" and
background.tex §7 for the full table. Nothing is added that isn't a
classical fuzzy operator; nothing is removed that HSiKAN had as
plasticity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hsikan_vision import CRActivation, build_rf_edge_members


# ─── Receptive-field index helpers ─────────────────────────────────


def build_rf_vertex_edges(
    H: int, W: int, kernel: int, stride: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """For each vertex, return the list of edges (RFs) it belongs to,
    padded to a fixed ``max_E_per_v`` width.

    Returns:
        v_edges:      (V, max_E_per_v) long; edge indices per vertex.
                      Padded entries reference edge 0 (safe; masked).
        v_edges_mask: (V, max_E_per_v) float; 1.0 for real entries,
                      0.0 for padding.
        max_E_per_v:  int, max number of edges any vertex belongs to.

    Used by the e→v t-conorm: gather edge memberships per vertex, mask
    padding to the t-conorm's identity element (0), then pool.
    """
    rows = list(range(0, H - kernel + 1, stride))
    cols = list(range(0, W - kernel + 1, stride))
    V = H * W
    per_v: list[list[int]] = [[] for _ in range(V)]
    e_idx = 0
    for r in rows:
        for c in cols:
            for dr in range(kernel):
                for dc in range(kernel):
                    v = (r + dr) * W + (c + dc)
                    per_v[v].append(e_idx)
            e_idx += 1
    max_E_per_v = max((len(e) for e in per_v), default=0) or 1
    v_edges = torch.zeros(V, max_E_per_v, dtype=torch.long)
    v_edges_mask = torch.zeros(V, max_E_per_v, dtype=torch.float32)
    for v, edges in enumerate(per_v):
        for j, e in enumerate(edges):
            v_edges[v, j] = e
            v_edges_mask[v, j] = 1.0
    return v_edges, v_edges_mask, max_E_per_v


# ─── T-norms and T-conorms (the fuzzy aggregators) ─────────────────


_EPS = 1e-7


def t_norm(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Apply a t-norm over the second-to-last dim. ``x`` must be in
    [0, 1]. Shape (..., K, d) → (..., d).

    Boundary conditions (tested in test_fuzzy_signature_layer.py):
      T(1, ..., 1) = 1   (identity element 1)
      T(0, ..., a) = 0   (annihilator 0)
      monotonic in each argument
      ordering: Łuk ≤ product ≤ Gödel (Prop. 2.2 in background.tex)
    """
    if kind == "min":
        return x.amin(dim=-2)
    if kind == "product":
        # exp(Σ log) avoids underflow for large K (see background.tex §11.2).
        return torch.exp(torch.log(x.clamp(min=_EPS)).sum(dim=-2))
    if kind == "lukasiewicz":
        K = x.shape[-2]
        return (x.sum(dim=-2) - (K - 1)).clamp(min=0.0)
    raise ValueError(
        f"unknown t-norm kind {kind!r}; expected min|product|lukasiewicz"
    )


def t_conorm(x: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
    """Apply a t-conorm over the second-to-last dim, respecting
    ``mask``. ``x`` must be in [0, 1]; ``mask`` is 1 for real entries,
    0 for padding.

    Padding handling (per t-conorm's identity element, S(0, a) = a):
      Padded ``x`` values must be 0 — then S(0, a, b, ...) = S(a, b, ...).
      The mask multiplication enforces this.

    Boundary conditions (tested):
      S(0, ..., 0) = 0  (identity element 0)
      S(1, ..., a) = 1  (annihilator 1)
      monotonic, ordering: max ≤ probsum ≤ Łuk (Prop. 2.3 in background.tex)
    """
    x_masked = x * mask.unsqueeze(-1)
    if kind == "max":
        return x_masked.amax(dim=-2)
    if kind == "probsum":
        # S(a, b) = a + b − ab; n-ary: 1 − Π (1 − a_i).
        # Pad-safe: x_masked=0 at padding → (1 − 0) = 1, no effect.
        return 1.0 - torch.exp(torch.log((1.0 - x_masked).clamp(min=_EPS)).sum(dim=-2))
    if kind == "lukasiewicz":
        return x_masked.sum(dim=-2).clamp(max=1.0)
    raise ValueError(
        f"unknown t-conorm kind {kind!r}; expected max|probsum|lukasiewicz"
    )


# ─── FuzzySignatureLayer (Atanassov-pair redesign) ─────────────────


class FuzzySignatureLayer(nn.Module):
    """One layer of a learnable Kóczy fuzzy signature (revision 3).

    Forward (Definition 7.1 in background.tex, generalised to the
    cr_input_scale and residual_kind axes):

        x̂_v    = 6·x_v - 3  if cr_input_scale="unit_to_grid"  else  x_v
        μ⁺_v   = σ(CR⁺(x̂_v))                           # Atanassov μ⁺
        μ⁻_v   = σ(CR⁻(x̂_v))                           # Atanassov μ⁻
        g_v    = σ(softplus(τ) ⊙ (x_v − ½))             # learnable hedge
        μ_v    = g_v · μ⁺_v + (1 − g_v) · μ⁻_v          # IFS mix
        h_e    = T_norm({μ_v : v ∈ e})                  # v → e, fuzzy AND
        h̃_e   = min(σ(W_e)·2, 1) · h_e                 # rule strength
        h_v    = T_conorm({h̃_e : e ∋ v}, mask_v)       # e → v, fuzzy OR
        out_v  = R(x_v, h_v)                            # residual op

    where R is one of:
        avg:     R(a, b) = ½ (a + b)                    # revision 2 default
        max:     R(a, b) = max(a, b)  (t-conorm max)    # non-contracting
        probsum: R(a, b) = a + b − a·b  (t-conorm prob) # saturates to 1

    Theorem 7.1 (background.tex): inputs in [0,1] → outputs in [0,1],
    composable to arbitrary depth without normalisation. The theorem
    extends trivially to all three residual operators (each is a
    [0,1]-preserving binary fuzzy operator).

    Parameters per layer (for fixed d, m, W_e tied):
      2·d·m  (CR⁺ + CR⁻ control points) + d (τ) + 1 (W_e tied).
    """

    def __init__(self, d: int, H: int, W: int, kernel: int, stride: int,
                 *, t_norm_kind: str = "min",
                 t_conorm_kind: str = "max",
                 m: int = 8, init_scale: float = 0.05,
                 learnable_tau: bool = True, tau_init: float = 4.0,
                 tied_we: bool = True,
                 cr_input_scale: str = "unit_to_grid",
                 residual_kind: str = "lerp",
                 init_kind: str = "ramp",
                 ramp_strength: float = 1.5,
                 lerp_alpha_init: float = 0.05,
                 gate_center_learnable: bool = True):
        """Revision-4 axes (user direction 2026-05-30):

        ``residual_kind`` ∈ {"avg", "max", "probsum", "lerp"}:
          Revision-3 added "max" but the 9-cell smoke matrix showed
          every option saturates at random baseline because the gate
          and Atanassov mix conspire to a fixed point. "lerp" is the
          new highway-gate-style residual: out = (1-α)·x + α·h_v with
          α = σ(α_raw) learnable per channel. At init α ≈ 0.05 (skip-
          dominant), so the layer is near-identity and gradient flow
          is clean. Gradient grows α only where h_v carries useful
          signal — same pattern that worked on Slashdot edge_cr SOTA.

        ``gate_center_learnable`` ∈ {True, False}: revision-4 axis.
          When True, the gate centre c is a learnable per-channel
          parameter (initialised at ½). Breaks the symmetry-induced
          fixed point diagnosed in the 9-cell smoke. When False
          preserves rev-3 (c fixed at ½).

        ``init_kind`` ∈ {"random", "ramp", "asymmetric_ramp"}:
          Revision-4 adds "asymmetric_ramp": μ⁺ is the increasing ramp
          (as in "ramp"), but μ⁻ is a Gaussian bump centred at low x
          (NOT the symmetric decreasing ramp). Breaks the μ⁻(x) ≈
          μ⁺(1-x) symmetry that the 9-cell smoke implicated. The Gaussian
          bump is approximated by setting μ⁻'s CP to a discrete bump
          profile at the low end of the grid.

        ``lerp_alpha_init``: residual mixing init when residual_kind=
          "lerp". Default 0.05 → α_raw ≈ σ⁻¹(0.05) ≈ -2.94; at init
          out ≈ 0.95·x + 0.05·h_v (skip-dominant).
        """
        super().__init__()
        if cr_input_scale not in ("unit_to_grid", "raw"):
            raise ValueError(
                f"cr_input_scale must be 'unit_to_grid' or 'raw'; "
                f"got {cr_input_scale!r}"
            )
        if residual_kind not in (
            "avg", "max", "probsum", "lerp", "additive_centered",
        ):
            raise ValueError(
                "residual_kind must be 'avg', 'max', 'probsum', 'lerp', or "
                f"'additive_centered'; got {residual_kind!r}"
            )
        if init_kind not in ("random", "ramp", "asymmetric_ramp"):
            raise ValueError(
                f"init_kind must be 'random', 'ramp', or 'asymmetric_ramp'; "
                f"got {init_kind!r}"
            )
        self.d = d
        self.H, self.W = H, W
        self.kernel, self.stride = kernel, stride
        self.t_norm_kind = t_norm_kind
        self.t_conorm_kind = t_conorm_kind
        self.cr_input_scale = cr_input_scale
        self.residual_kind = residual_kind
        self.init_kind = init_kind
        self.ramp_strength = ramp_strength
        self.gate_center_learnable = gate_center_learnable
        self.lerp_alpha_init = lerp_alpha_init
        em, n_e = build_rf_edge_members(H, W, kernel, stride)
        v_edges, v_mask, _ = build_rf_vertex_edges(H, W, kernel, stride)
        self.register_buffer("edge_members", em)
        self.register_buffer("vertex_edges", v_edges)
        self.register_buffer("vertex_edges_mask", v_mask)
        self.n_edges = n_e
        # Atanassov μ⁺ and μ⁻ — two independent CRActivations with
        # clamp_to_01=True (σ on top of the spline). Independent random
        # seeds at init ensure they don't collapse to the same function.
        self.mu_plus = CRActivation(
            channels=d, n_branches=1, m=m, init_scale=init_scale,
            clamp_to_01=True,
        )
        self.mu_minus = CRActivation(
            channels=d, n_branches=1, m=m, init_scale=init_scale,
            clamp_to_01=True,
        )
        if init_kind == "ramp":
            # μ⁺: monotone increasing in x; μ⁻: monotone decreasing.
            # Grid = linspace(-3, 3, m) is CRActivation's input range.
            grid = torch.linspace(-3.0, 3.0, m)
            with torch.no_grad():
                self.mu_plus.cpts.data = (
                    grid.view(1, 1, -1).expand(1, d, -1).clone()
                    * ramp_strength
                )
                self.mu_minus.cpts.data = (
                    -grid.view(1, 1, -1).expand(1, d, -1).clone()
                    * ramp_strength
                )
        elif init_kind == "asymmetric_ramp":
            # Revision-4: break the μ⁻(x) ≈ μ⁺(1-x) symmetry.
            # μ⁺ = monotone ramp (as in "ramp").
            # μ⁻ = Gaussian-like bump centred at low x (so it's NOT the
            # mirror of μ⁺). Use a triangular bump on the CP grid
            # centred at the leftmost-third of the grid (peaks at
            # idx ≈ m/3), decaying linearly to 0 outside.
            grid = torch.linspace(-3.0, 3.0, m)
            bump_peak = m // 3
            bump_width = max(1, m // 3)
            bump = torch.zeros(m)
            for i in range(m):
                dist = abs(i - bump_peak)
                bump[i] = max(0.0, 1.0 - dist / bump_width) * 3.0
            # Scale so σ(bump) is ~0.95 at peak, ~0.5 at decay edges.
            with torch.no_grad():
                self.mu_plus.cpts.data = (
                    grid.view(1, 1, -1).expand(1, d, -1).clone()
                    * ramp_strength
                )
                self.mu_minus.cpts.data = (
                    bump.view(1, 1, -1).expand(1, d, -1).clone()
                    * ramp_strength
                )
        # Learnable hedge dilation τ (one scalar per channel). Stored as
        # raw ℓ with τ = softplus(ℓ) — positivity by construction.
        # Init: softplus⁻¹(tau_init) so τ starts at tau_init exactly.
        if learnable_tau:
            tau_raw_init = torch.log(torch.expm1(torch.tensor(tau_init)))
            self.tau_raw = nn.Parameter(torch.full((d,), tau_raw_init.item()))
        else:
            tau_raw_init = torch.log(torch.expm1(torch.tensor(tau_init)))
            self.register_buffer("tau_raw", torch.full((d,), tau_raw_init.item()))
        self.learnable_tau = learnable_tau
        # Rule strength W_e (TSK rule salience). Tied: one scalar per
        # layer. Stored raw; effective W_e ∈ [0,1] via σ(W_e_raw)·2 ∧ 1.
        # Init: σ⁻¹(0.5) = 0 → effective W_e = 1.0 at init (full strength).
        self.tied_we = tied_we
        if tied_we:
            self.W_e_raw = nn.Parameter(torch.zeros(1))
        else:
            self.W_e_raw = nn.Parameter(torch.zeros(n_e))
        # Revision-4: learnable gate centre c per channel.
        # When True, c is a parameter (init ½). When False, fixed buffer ½.
        if gate_center_learnable:
            self.c_gate = nn.Parameter(torch.full((d,), 0.5))
        else:
            self.register_buffer("c_gate", torch.full((d,), 0.5))
        # Revision-4: learnable α for lerp residual.
        # α = σ(α_raw); init α ≈ lerp_alpha_init (default 0.05 → skip-dominant).
        # Only allocated when residual_kind="lerp" (other modes use no α).
        if residual_kind == "lerp":
            # α_raw_init = σ⁻¹(lerp_alpha_init) = logit(lerp_alpha_init)
            p = max(min(lerp_alpha_init, 1.0 - 1e-6), 1e-6)
            alpha_raw_init = torch.logit(torch.tensor(p))
            self.alpha_raw = nn.Parameter(
                torch.full((d,), alpha_raw_init.item())
            )
        else:
            self.register_parameter("alpha_raw", None)

    @property
    def tau(self) -> torch.Tensor:
        """Effective τ = softplus(τ_raw) ∈ (0, ∞)."""
        return F.softplus(self.tau_raw)

    @property
    def W_e_eff(self) -> torch.Tensor:
        """Effective rule strength ∈ [0, 1]: min(σ(W_e_raw)·2, 1).
        At W_e_raw = 0 → σ(0)·2 = 1.0 (full strength).
        At W_e_raw → −∞ → 0 (rule muted).
        At W_e_raw → +∞ → 1 (saturated full strength)."""
        return torch.clamp(torch.sigmoid(self.W_e_raw) * 2.0, max=1.0)

    @property
    def alpha_eff(self) -> torch.Tensor | None:
        """Effective lerp mixing weight α ∈ (0, 1) when residual_kind=
        'lerp'; None otherwise. Exposed for telemetry."""
        if self.alpha_raw is None:
            return None
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, d) in [0, 1]. Returns (B, V, d) in [0, 1]."""
        # 1. Optional input rescale: x∈[0,1] → x̂∈[-3,3] so the full CR
        # spline domain is used (revision-3 fix C). "raw" preserves rev-2.
        if self.cr_input_scale == "unit_to_grid":
            x_hat = 6.0 * x - 3.0                  # (B, V, d) in [-3, 3]
        else:
            x_hat = x
        # 2. Atanassov pair: μ⁺_v, μ⁻_v ∈ [0, 1]^(B,V,d).
        mu_p = self.mu_plus(x_hat, branch_idx=0)   # (B, V, d) in [0,1]
        mu_n = self.mu_minus(x_hat, branch_idx=0)  # (B, V, d) in [0,1]
        # 3. Learnable hedge gate at x (NOT x_hat — the centre lives in
        # fuzzy coordinates, regardless of CR's input range). τ and c
        # broadcast over (B, V, d). c is learnable per-channel (rev 4)
        # when gate_center_learnable=True; otherwise fixed at ½.
        tau = self.tau                             # (d,)
        g = torch.sigmoid(tau * (x - self.c_gate)) # (B, V, d) in [0,1]
        # 4. Atanassov mix μ_v = g · μ⁺ + (1−g) · μ⁻ ∈ [0,1].
        mu = g * mu_p + (1.0 - g) * mu_n           # (B, V, d) in [0,1]
        # 5. v → e gather + t-norm (fuzzy AND).
        x_per_e = mu[:, self.edge_members, :]      # (B, E, K, d)
        h_e = t_norm(x_per_e, self.t_norm_kind)    # (B, E, d) in [0,1]
        # 6. Rule strength W_e (broadcasts over B, d).
        w_eff = self.W_e_eff                        # (1,) or (E,)
        if w_eff.shape[0] == 1:
            h_e = h_e * w_eff                       # scalar broadcast
        else:
            h_e = h_e * w_eff.view(-1, 1)           # (E,) → broadcast
        # 7. e → v gather + t-conorm (fuzzy OR).
        h_per_v = h_e[:, self.vertex_edges, :]     # (B, V, max_E_per_v, d)
        h_v = t_conorm(h_per_v, self.vertex_edges_mask, self.t_conorm_kind)
        # 8. Residual (revision-4: lerp; revision-5: additive_centered).
        # avg/max/probsum/lerp preserve [0,1]; additive_centered does NOT
        # (signal goes unbounded — HSiKAN pattern per user direction
        # 2026-05-30 after 18-cell smoke matrix exhausted bounded-fuzzy
        # space).
        if self.residual_kind == "avg":
            return 0.5 * (x + h_v)
        if self.residual_kind == "max":
            return torch.maximum(x, h_v)
        if self.residual_kind == "probsum":
            return x + h_v - x * h_v
        if self.residual_kind == "additive_centered":
            # HSiKAN-style additive residual. h_v is in [0,1], so
            # (h_v − 0.5) ∈ [−0.5, 0.5] (centered). x grows linearly
            # with depth (signal is unbounded internally). The fuzzy
            # semantics live in μ⁺/μ⁻/t-norm/t-conorm; x carries the
            # unbounded representation. Theorem 7.1 no longer applies
            # for x — it still holds for μ, h_e, h_v internally.
            return x + (h_v - 0.5)
        # lerp: out = (1 − α) · x + α · h_v with α = σ(α_raw) ∈ [0,1].
        # At init α ≈ 0.05 → out ≈ 0.95·x + 0.05·h_v (skip-dominant,
        # near-identity). Highway-gate pattern; non-contracting since
        # α=0 reproduces x exactly. α grows under gradient only where
        # h_v carries signal.
        alpha = torch.sigmoid(self.alpha_raw)       # (d,) in (0, 1)
        return (1.0 - alpha) * x + alpha * h_v


# ─── Multi-arity layer (αₖ mixer; HSiKAN-style) ────────────────────


class MultiArityFuzzySignatureLayer(nn.Module):
    """Wraps one ``FuzzySignatureLayer`` per receptive-field arity
    and mixes them with a learnable softmax ``αₖ``.

    Architectural lever taken from HSiKAN-vision (which gets 0.97
    MNIST at h=16/L=8/14.5k params via multi-arity). The αₖ mixer
    lets the model attend to multiple RF scales simultaneously —
    e.g. a fine kernel=3 capture (local edges) and a coarse kernel=5
    capture (digit-scale features). At init αₖ is uniform; gradient
    descent learns the per-scale importance.

    Parameter cost vs single-arity FuzzySignatureLayer:
      base: one layer per arity (≈ 2dm + 3d + 1 each at rev-5 defaults).
      plus: ``len(arities)`` αₖ logits per layer.

    Output dimensionality is preserved: each per-arity layer outputs
    (B, V, d) and the αₖ-weighted sum is also (B, V, d). All arities
    share the same V grid (28×28 vertices); they differ only in K
    (RF size) and which edges are constructed.
    """

    def __init__(self, d: int, H: int, W: int,
                 arities: list[tuple[int, int]],
                 *, multi_arity_mixer: str = "softmax",
                 **layer_kwargs):
        """``multi_arity_mixer`` ∈ {"softmax", "t_norm", "t_conorm",
        "owa"}.

        - ``"softmax"`` (default, preserves validated 0.5354 MNIST
          result): αₖ = softmax(logits); out = Σ αₖ · branch_k(x).
          A constrained OWA — convex combination across arities.
        - ``"t_norm"``: out = element-wise min over branches. Fuzzy
          AND across scales — "ALL scales must agree". No learnable
          per-arity weights. Safe on unbounded carrier (additive_centered).
        - ``"t_conorm"``: out = element-wise max over branches. Fuzzy
          OR across scales — "ANY scale fires". The fuzzy-consistent
          choice for the FuzzySignature stack (matches the e→v
          t-conorm at the layer interior). No learnable per-arity
          weights. Safe on unbounded carrier.
        - ``"owa"`` (Yager 1988): sort branches per-element, weight
          by learnable OWA weights (softmax-normalised), sum. Subsumes
          min / max / mean continuously; ``len(arities)`` learnable
          params.

        See ``docs/plans/2026-05-30-fuzzy-signature-layer/background.tex``
        §10 for the OWA framing of the softmax αₖ. The fuzzy-mixer
        axis was added 2026-06-01 to close the "softmax is the last
        non-fuzzy primitive" gap in the fuzzy stack.
        """
        super().__init__()
        if not arities:
            raise ValueError("arities must have at least one (kernel, stride)")
        if multi_arity_mixer not in ("softmax", "t_norm", "t_conorm", "owa"):
            raise ValueError(
                f"multi_arity_mixer must be 'softmax', 't_norm', "
                f"'t_conorm', or 'owa'; got {multi_arity_mixer!r}"
            )
        self.arities = list(arities)
        self.multi_arity_mixer = multi_arity_mixer
        self.layers = nn.ModuleList([
            FuzzySignatureLayer(
                d=d, H=H, W=W, kernel=k, stride=s, **layer_kwargs,
            )
            for (k, s) in arities
        ])
        # Learnable weights: only softmax and owa need them.
        if multi_arity_mixer in ("softmax", "owa"):
            self.alpha_logits = nn.Parameter(torch.zeros(len(arities)))
        else:
            # t_norm and t_conorm are parameter-free aggregators.
            self.register_parameter("alpha_logits", None)

    def alpha_weights(self) -> torch.Tensor:
        """Effective αₖ ∈ [0,1] with Σαₖ = 1 (softmax / owa).
        For t_norm / t_conorm returns a uniform 1/K vector (informational
        only — the mixer doesn't use weighted sums)."""
        if self.alpha_logits is not None:
            return F.softmax(self.alpha_logits, dim=0)
        K = len(self.arities)
        return torch.full((K,), 1.0 / K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [L(x) for L in self.layers]    # K tensors (B, V, d)
        if self.multi_arity_mixer == "softmax":
            alphas = F.softmax(self.alpha_logits, dim=0)
            out = 0.0
            for alpha, o in zip(alphas, branch_outs):
                out = out + alpha * o
            return out
        # Stack branches along a new leading axis for parameter-free
        # aggregators.
        stacked = torch.stack(branch_outs, dim=0)     # (K, B, V, d)
        if self.multi_arity_mixer == "t_norm":
            # element-wise min across arities — fuzzy AND
            return stacked.amin(dim=0)
        if self.multi_arity_mixer == "t_conorm":
            # element-wise max across arities — fuzzy OR
            return stacked.amax(dim=0)
        # OWA: sort along K dim, weight by learnable softmax weights.
        # `descending=True` so weight 0 attends to the *largest* branch
        # (matches Yager's classical OWA convention).
        sorted_vals, _ = stacked.sort(dim=0, descending=True)
        weights = F.softmax(self.alpha_logits, dim=0)
        # weights: (K,) → (K, 1, 1, 1) for broadcasting
        return (weights.view(-1, *([1] * (sorted_vals.dim() - 1)))
                * sorted_vals).sum(dim=0)


# ─── FuzzySignatureClassifier ──────────────────────────────────────


class FuzzySignatureClassifier(nn.Module):
    """Vision classifier:
        image → fuzzification → L × FuzzySignatureLayer
              → Sugeno pool → TSK output rule → logits.

    Mirrors HSiKANVisionClassifier's outer interface so
    ``vision_bench_cell`` dispatch hands it (B, 1, H, W) and gets
    (B, n_classes) logits.
    """

    def __init__(self, H: int, W: int, n_classes: int,
                 *, d: int = 16, n_layers: int = 8,
                 arities: list[tuple[int, int]] | None = None,
                 t_norm_kind: str = "min",
                 t_conorm_kind: str = "max",
                 m: int = 8,
                 learnable_tau: bool = True, tau_init: float = 4.0,
                 tied_we: bool = True,
                 cr_input_scale: str = "unit_to_grid",
                 residual_kind: str = "lerp",
                 init_kind: str = "ramp",
                 ramp_strength: float = 1.5,
                 lerp_alpha_init: float = 0.05,
                 gate_center_learnable: bool = True,
                 fuzzification_kind: str = "sigmoid",
                 multi_arity_mixer: str = "softmax"):
        """Default architecture: d=16, L=8, single arity (5,2) — the
        depth-winner from 2026-05-30. ``arities`` is a list of (kernel,
        stride); only the FIRST pair is used (one membership function
        per layer; multi-arity OWA mixing across scales is a downstream
        extension — see background.tex §10).

        ``cr_input_scale`` and ``residual_kind`` are the revision-3
        axes (see module docstring); defaults reflect Fallback C and
        Fallback B-option-1 from the failure report.

        ``fuzzification_kind`` ∈ {"sigmoid", "linear"} — revision-5
        axis: "sigmoid" applies σ after the embed Linear (input is
        fuzzified to [0,1], classical TSK rule). "linear" omits the
        σ — the embed output is unbounded, matching the HSiKAN pattern
        where internal signals are NOT confined to [0,1]. Use
        "linear" with residual_kind="additive_centered" for the
        HSiKAN-style relaxation."""
        super().__init__()
        if fuzzification_kind not in ("sigmoid", "linear"):
            raise ValueError(
                "fuzzification_kind must be 'sigmoid' or 'linear'; "
                f"got {fuzzification_kind!r}"
            )
        if arities is None:
            arities = [(5, 2)]
        # TSK fuzzification: crisp pixel → d fuzzy memberships in [0,1].
        self.embed = nn.Linear(1, d)
        self.fuzzification_kind = fuzzification_kind
        # Per-layer construction: single-arity → FuzzySignatureLayer;
        # multi-arity → MultiArityFuzzySignatureLayer (αₖ mixer).
        layer_kwargs = dict(
            t_norm_kind=t_norm_kind, t_conorm_kind=t_conorm_kind,
            m=m, learnable_tau=learnable_tau, tau_init=tau_init,
            tied_we=tied_we,
            cr_input_scale=cr_input_scale, residual_kind=residual_kind,
            init_kind=init_kind, ramp_strength=ramp_strength,
            lerp_alpha_init=lerp_alpha_init,
            gate_center_learnable=gate_center_learnable,
        )
        if len(arities) == 1:
            kernel, stride = arities[0]
            self.layers = nn.ModuleList([
                FuzzySignatureLayer(
                    d=d, H=H, W=W, kernel=kernel, stride=stride,
                    **layer_kwargs,
                )
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MultiArityFuzzySignatureLayer(
                    d=d, H=H, W=W, arities=arities,
                    multi_arity_mixer=multi_arity_mixer,
                    **layer_kwargs,
                )
                for _ in range(n_layers)
            ])
        self.arities = arities
        self.multi_arity_mixer = multi_arity_mixer
        # TSK output rule (defuzzification): from d fuzzy channels → logits.
        self.head = nn.Linear(d, n_classes)
        self.d = d
        self.n_layers = n_layers
        self.t_norm_kind = t_norm_kind
        self.t_conorm_kind = t_conorm_kind
        self.learnable_tau = learnable_tau
        self.tied_we = tied_we
        self.cr_input_scale = cr_input_scale
        self.residual_kind = residual_kind
        self.init_kind = init_kind
        self.ramp_strength = ramp_strength
        self.lerp_alpha_init = lerp_alpha_init
        self.gate_center_learnable = gate_center_learnable

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        # x_img: (B, 1, H, W) in [0, 1] (already fuzzified by ToTensor()).
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)                # (B, V, 1)
        v = self.embed(v)                        # (B, V, d) unbounded
        if self.fuzzification_kind == "sigmoid":
            v = torch.sigmoid(v)                 # → [0,1] (TSK fuzzification)
        # If "linear", v stays unbounded — HSiKAN-style internals.
        for L in self.layers:
            v = L(v)
        z = v.mean(dim=1)                        # (B, d) Sugeno-like pool
        return self.head(z)                      # (B, n_classes)


__all__ = [
    "FuzzySignatureLayer",
    "MultiArityFuzzySignatureLayer",
    "FuzzySignatureClassifier",
    "build_rf_vertex_edges",
    "t_norm",
    "t_conorm",
    "expected_param_count",
]


# ─── Closed-form parameter count helper (regression check) ──────────


def expected_param_count(d: int, n_layers: int, n_classes: int = 10,
                          m: int = 8, tied_we: bool = True,
                          gate_center_learnable: bool = True,
                          residual_kind: str = "lerp") -> int:
    """Closed-form predicted parameter count for the Atanassov-pair
    layer (revision 4). Matches plan.tex revision 4.

        N(d, L) = 2d  (embed)
                + L · (2dm + d + 1_we + 1_c + 1_α)
                + (d+1) · n_classes

    where:
      1_we counts the tied W_e scalar per layer (if untied: n_edges).
      1_c  counts d for the learnable gate centre (0 if not learnable).
      1_α  counts d for the learnable lerp α (0 if residual_kind ≠ "lerp").
    """
    embed = 2 * d                          # Linear(1→d) + bias
    we_per = 1 if tied_we else 0          # caller's responsibility if untied
    c_per = d if gate_center_learnable else 0
    a_per = d if residual_kind == "lerp" else 0
    per_layer = 2 * d * m + d + we_per + c_per + a_per
    layers = n_layers * per_layer
    head = (d + 1) * n_classes
    return embed + layers + head
