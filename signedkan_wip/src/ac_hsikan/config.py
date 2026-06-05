"""AC-HSiKAN configuration dataclass.

Single frozen dataclass; passed explicitly down the call chain
(no env-var reads at depth — CLAUDE.md §6.5 #11).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AcHsikanConfig:
    """Configuration for an AC-HSiKAN block + classifier head."""

    # ── Token / position geometry ─────────────────────────────────
    d_model: int = 64
    n_positions: int = 128       # max sequence length (graph node count)

    # ── Sign head ────────────────────────────────────────────────
    sign_head_kind: str = "bilinear" # "bilinear" | "quaternion"
                                     # quaternion: Hamilton-product real-part
                                     # scoring (q_a·k_a − q_b·k_b − q_c·k_c
                                     # − q_d·k_d) like the production graph
                                     # HSiKAN attention (HSIKAN_ATTENTION_M_E)
    sign_head_hidden: int = 16       # bilinear path: hidden rank
                                     # quaternion path: must be divisible by 4
    sign_temperature: float = 1.0    # tanh scale; smaller = sharper sign
    sign_ste: bool = False           # straight-through estimator at inference

    # ── Cycle pool ───────────────────────────────────────────────
    arities: tuple[int, ...] = (2, 3, 4, 5)
    top_k_per_position: int = 16     # K cycles per position per arity
    n_jumps_per_anchor: int = 0      # >0: include this many remote (non-local)
                                     # positions in each anchor's candidate set.
                                     # Deterministic per layer (seeded at init).
    walk_kind: str = "star"          # "star" | "chain" | "cycle"
                                     # star  : sign = prod_k sign(anchor, n_k)
                                     # chain : sign = prod_k sign(n_{k-1}, n_k)
                                     #         starting from sign(anchor, n_0)
                                     # cycle : chain + closing edge sign(n_last, anchor)
    jumps_seed: int = 1729           # rng seed for the fixed jump indices

    # ── αₖ mixture ───────────────────────────────────────────────
    alpha_init: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    alpha_temperature: float = 1.0   # softmax temperature on αₖ

    # ── Stack ────────────────────────────────────────────────────
    n_layers: int = 2
    n_classes: int = 2
    dropout: float = 0.1

    # ── v1.1: expressivity boosters (2026-06-04, post IMDB clean loss) ──
    # Each is opt-in (defaults preserve v1 behaviour for the parity
    # task numbers in reports/2026-06-04-ac-hsikan-synthetic-parity.md).
    use_value_projection: bool = False    # add W_v: d->d before pooling
    ffn_hidden: int = 0                   # > 0 enables post-attention FFN
    use_magnitude_weight: bool = False    # softmax(|score|) weighting
                                          # alongside sign(score)

    # ── v1.4: speed via Clifford-FIR pre-context + sparse sign head ─────
    # Both opt-in; together they take the layer from O(L²·d) to
    # O(L · (K_fir + K_total) · d) -- linear in L. See
    # signedkan_wip/src/sequence/clifford_fir.py for the underlying
    # multivector FIR; the Gömb architecture's primary speed primitive.
    use_clifford_fir_context: bool = False  # pre-process x via CliffordFIR
    clifford_fir_K: int = 8                 # FIR filter length (taps)
    sparse_sign_head: bool = False          # only compute signs for
                                            # (anchor, candidate) pairs
                                            # instead of full (L, L)

    # ── v1.5: building-block speedups (2026-06-04 evening) ──────────────
    # Each is a component-level knob picked up by
    # signedkan_wip.src.ac_hsikan.components.*.build_*(cfg) factories.
    n_sign_heads: int = 1                   # #1: multi-head sparse sign
                                            #     attention (parallel via
                                            #     batched einsum). Requires
                                            #     sparse_sign_head=True and
                                            #     sign_head_kind="quaternion".
    use_fused_walk: bool = False            # #2: fused chain/cycle walk
                                            #     sign-product (log+exp+sign
                                            #     instead of per-edge prod).
    ffn_bottleneck_ratio: int = 0           # #3: > 0 selects BottleneckFFN
                                            #     with ff_hidden = d_model //
                                            #     ratio (e.g. 1 = identity-ish,
                                            #     2 = half-width). 0 falls back
                                            #     to ffn_hidden or IdentityFFN.

    # ── v1.6: dual-path attention (2026-06-05) ─────────────────────────
    # Adds a FusedPoolScatter primitive in parallel with the walk-op
    # sign-product path. Biological / mathematical interpretation:
    #   pool-scatter  = synaptic transmission (asymmetric, per-pair
    #                    vector-CR signed-V aggregation, Hamilton field)
    #   walk-op       = rhythm + phase coherence (sign-product over
    #                    cycle, α_k regime compass)
    # The two outputs are mixed via a learnable sigmoid gate; the
    # regime compass survives because α_k still parameterises walk_op.
    use_pool_scatter: bool = False          # opt-in dual-path
    pool_scatter_h: int = 8                 # h dim of the vector-CR
                                            # head (must equal 4 * n_quat
                                            # for v1; multi-block in v2)
    pool_scatter_n_quat: int = 2            # quaternion blocks at h /4
    pool_scatter_G: int = 8                 # CR grid points per channel
    # 2D CR-patch surface (per-channel bilinear over (R, V_c)).
    # Replaces the per-channel 1D CR + multiplicative `S * V_c` gate
    # with a learned non-separable surface activation:
    #   pool_h += CR_2D(R_pool, V_c, coef_pos/neg_2d[h, G, G])
    # The V_c information is baked into the surface (no separate
    # multiply), making the (R, V) interaction a learned bivariate
    # non-linearity rather than the fixed multiplicative form.
    # Memory: 2 × h × G² = 1024 params (vs 1D: 128). Compute: 16
    # control-point reads per CR (vs 4). Uses the PyTorch reference
    # path -- Triton kernel extension is future work.
    use_2d_cr: bool = False
    pool_scatter_temperature: float = 1.0
    pool_scatter_gate_init: float = 0.0     # initial logit value -- 0 -> 0.5
                                            # gate; both paths start equal
    # Entropy-feedback Hamilton rotor on the scatter direction:
    #   scatter_h ← M ⊗ scatter_h,   M = exp(β · H · n)
    # where H is the per-layer scalar entropy of (alpha, path_gate),
    # n is a learnable unit imaginary quaternion axis (per block),
    # and β is a learnable scalar. β=0 reduces to identity.
    use_pool_scatter_rotor: bool = False    # opt-in entropy-rotor modulation

    # Dynamic top-K candidate selection (the AC-HSiKAN analogue of ABB
    # top-K-per-vertex on simple HSiKAN). When True, at every forward
    # the dense (L, L) sign-magnitude matrix is computed once via the
    # sign-head, then per-anchor top-K positions (by batch-averaged
    # |score|) replace the static `_local_indices_cache`. The pool-
    # scatter primitive sees a normal (L, K_total) index tensor;
    # only the *content* changes per step. Requires the dense sign-
    # head path (`sparse_sign_head=False`).
    use_dynamic_topk: bool = False
    # Mixed-selection budget: K_static positional + (K-K_static) dynamic.
    # 0 = pure dynamic top-K (current behaviour when use_dynamic_topk).
    # > 0 = hybrid: first K_static slots from the static positional
    # neighbours (guaranteed local coverage), remaining (K-K_static)
    # slots from the magnitude-top-K of the *non-static* positions
    # (adaptive long-range signal). Combines the proven local prior
    # with content-adaptive selection.
    dynamic_topk_static_k: int = 0
    # Long-cycle inductive prior. The selection score is multiplied by
    # ``(|i - j| / (L - 1)) ** long_cycle_prior_alpha`` so that
    # positionally distant candidates are preferred -- these form
    # longer walks/cycles through the walk-op composition.
    # α = 0: pure magnitude (no distance bias).
    # α = 1: linear distance weighting.
    # α > 1: stronger long-cycle preference.
    # Replaces the simple-HSiKAN DAG-axiom pruning with a sequence-
    # native inductive bias.
    long_cycle_prior_alpha: float = 0.0

    # ── Depth-stability options (essential at n_layers >= 4) ─────
    # Pre-norm vs post-norm. Default = post-norm (current behaviour).
    # Pre-norm puts the LayerNorm BEFORE the layer body, with the
    # residual identity bypassing the norm. Modern depth-trainable
    # transformer practice; required for deeper stacks (>= 4 layers).
    use_pre_norm: bool = False
    # LayerScale (Touvron et al, 2021, CaiT). Multiplies the layer's
    # output residual branch by a per-channel learnable scalar
    # initialised to ``layer_scale_init`` (default 1e-4). Strongly
    # stabilises depth training; pairs well with pre-norm.
    use_layer_scale: bool = False
    layer_scale_init: float = 1e-4

    # ── Classification head ──────────────────────────────────────
    pool: str = "mean"               # mean | cls | max

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1; got {self.d_model}")
        if self.n_positions < 2:
            raise ValueError(f"n_positions must be >= 2; got {self.n_positions}")
        if len(self.arities) != len(self.alpha_init):
            raise ValueError(
                f"alpha_init must align with arities: "
                f"got {len(self.alpha_init)} alpha vs {len(self.arities)} arities"
            )
        if self.top_k_per_position < 1:
            raise ValueError(
                f"top_k_per_position must be >= 1; got {self.top_k_per_position}"
            )
        if self.pool not in {"mean", "cls", "max"}:
            raise ValueError(f"pool must be mean|cls|max; got {self.pool!r}")
        if self.sign_head_kind not in {"bilinear", "quaternion"}:
            raise ValueError(
                f"sign_head_kind must be 'bilinear' or 'quaternion'; "
                f"got {self.sign_head_kind!r}"
            )
        if self.sign_head_kind == "quaternion" and self.sign_head_hidden % 4 != 0:
            raise ValueError(
                f"sign_head_kind='quaternion' requires sign_head_hidden % 4 == 0;"
                f" got {self.sign_head_hidden}"
            )
        if self.walk_kind not in {"star", "chain", "cycle"}:
            raise ValueError(
                f"walk_kind must be 'star'|'chain'|'cycle'; "
                f"got {self.walk_kind!r}"
            )
        if self.n_jumps_per_anchor < 0:
            raise ValueError(
                f"n_jumps_per_anchor must be >= 0; "
                f"got {self.n_jumps_per_anchor}"
            )
        if self.use_clifford_fir_context:
            if self.d_model % 4 != 0:
                raise ValueError(
                    f"use_clifford_fir_context requires d_model % 4 == 0; "
                    f"got d_model={self.d_model}"
                )
            if self.clifford_fir_K < 1:
                raise ValueError(
                    f"clifford_fir_K must be >= 1; "
                    f"got {self.clifford_fir_K}"
                )
        # v1.5 cross-checks
        if self.n_sign_heads < 1:
            raise ValueError(
                f"n_sign_heads must be >= 1; got {self.n_sign_heads}"
            )
        if self.n_sign_heads > 1:
            if not self.sparse_sign_head:
                raise ValueError(
                    "n_sign_heads > 1 requires sparse_sign_head=True"
                )
            if self.sign_head_hidden % self.n_sign_heads != 0:
                raise ValueError(
                    f"sign_head_hidden ({self.sign_head_hidden}) "
                    f"must be divisible by n_sign_heads ({self.n_sign_heads})"
                )
            per_head = self.sign_head_hidden // self.n_sign_heads
            if (self.sign_head_kind == "quaternion"
                    and per_head % 4 != 0):
                raise ValueError(
                    f"quaternion + multi-head: hidden / n_heads "
                    f"({per_head}) must be % 4 == 0"
                )
        if self.ffn_bottleneck_ratio < 0:
            raise ValueError(
                f"ffn_bottleneck_ratio must be >= 0; "
                f"got {self.ffn_bottleneck_ratio}"
            )
        if self.use_pool_scatter:
            if self.pool_scatter_h != 4 * self.pool_scatter_n_quat:
                raise ValueError(
                    f"pool_scatter_h ({self.pool_scatter_h}) must equal "
                    f"4 * pool_scatter_n_quat ({4 * self.pool_scatter_n_quat})"
                )
            if self.pool_scatter_G < 4:
                raise ValueError(
                    f"pool_scatter_G must be >= 4 for valid CR basis; "
                    f"got {self.pool_scatter_G}"
                )
