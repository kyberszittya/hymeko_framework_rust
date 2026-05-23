"""Core HSiKAN / SignedKAN modules.

Moved here from ``signedkan_wip/src/`` flat layout on 2026-05-19 as
part of Slice F of the directory-reorganisation
(``docs/plans/2026-05-19-signedkan-wip-organize/``). 23 algorithmic
modules that constitute the HSiKAN inner architecture, the
Catmull-Rom and BSpline activation primitives, the cycle/walk
hyperedge construction, the highway/attention/bilinear/multi-layer
SignedKAN heads, the various regulariser families, the training
loop, the Triton-kernel sensitivity probe, the iterative pruner,
and the symbolic-distillation helpers.

The public surface is re-exported flat through this ``__init__`` so
external code that uses ``from signedkan_wip.src.core import X``
keeps working unchanged. Pre-Slice-F code that wrote
``from signedkan_wip.src.<name> import X`` has been migrated to use
``from signedkan_wip.src.core.<name> import X`` (or via the package
namespace).

Object-oriented note: each submodule typically exposes one or two
public classes (config dataclass + main class) plus a few free
helpers. The pattern is uniform; the per-submodule ``__init__``
does no work beyond re-export.
"""

# ── Activations + spline primitives ─────────────────────────────────────────
from .splines import (
    BSplineActivation,
    BatchedBSplineActivation,
    BatchedCatmullRomActivation,
    BatchedKochanekBartelsActivation,
    CatmullRomActivation,
    DiagonalBatchedBSplineActivation,
    DiagonalBatchedCatmullRomActivation,
    DiagonalBatchedKochanekBartelsActivation,
    KochanekBartelsActivation,
    cox_de_boor,
    main,
    make_uniform_knots,
)

# ── Hyperedge + cycle + walk construction ─────────────────────────────────────────
from .hyperedges import SignedTriad, construct, main, stats
from .n_tuples import (
    SignedNTuple,
    construct_2,
    construct_k,
    enumerate_k_cycles,
    stats,
)
from .walks import construct_walks

# ── Spectral / Laplacian init ─────────────────────────────────────────
from .signed_laplacian import (
    make_spectral_init,
    signed_normalised_laplacian,
    top_k_eigenvectors,
)
from .spectral_init import compute_spectral_init, signed_laplacian

# ── Core SignedKAN model + layers ─────────────────────────────────────────
from .signedkan import (
    MultiLayerSignedKAN,
    MultiLayerSignedKANConfig,
    SignedKAN,
    SignedKANConfig,
    SignedKANLayer,
    build_vertex_triad_incidence,
    main,
)
from .highway_signedkan import HighwaySignedKAN, HighwaySignedKANConfig
from .bilinear_head import BilinearHead, LowRankBilinearHead

# ── Attention / triad / scene ─────────────────────────────────────────
from .attention import (
    AttentionConfig,
    SignedTriadAttention,
    attention_entropy_loss,
    build_attention_pairs,
)
from .triad_loss import TriadLoss, TriadLossConfig, build_triad_pairs
from .scene_graph import SceneGraph, SceneObject, SceneRelation, demo_kitchen_scene

# ── CPML routing / capsule / tier ─────────────────────────────────────────
from .cpml import (
    CPML,
    CPMLConfig,
    CapsuleHypergraphRouter,
    ClifFIRTierAggregator,
    SignedKANTierAggregator,
    TierAggregator,
    TierSpec,
    restrict_cycles_to_tier,
)

# ── Learnable M_e + sigma masking ─────────────────────────────────────────
from .learnable_m_e import LearnableMe
from .sigma_masking import (
    eval_with_sigma_masking,
    patch_per_arity_for_query,
    patch_sigma_for_query,
)

# ── Regularisers ─────────────────────────────────────────
from .cross_branch_reg import CrossBranchRegConfig, CrossBranchRegulariser
from .entropy_reg import (
    CoefEntropyRegulariser,
    EntropyRegConfig,
    EntropyRegulariser,
    SplineSmoothRegulariser,
)
from .participation_reg import (
    HyperedgeDensityRegulariser,
    ParticipationRegulariser,
    triad_degree,
    triad_density,
)
from .n_tuple_loss import (
    NTupleBalanceLoss,
    NTupleBalanceLossConfig,
    build_ntuple_balance_tensors,
)

# ── Pruning + distillation ─────────────────────────────────────────
from .iter_prune import PruneMask, count_active_splines
from .prune_distill import (
    SymbolicFit,
    distill_activation,
    evaluate_symbolic,
    fit_summary,
    fit_symbolic,
    measure_activity,
    prune_inactive,
    sample_spline_activation,
)

# ── Training loop ─────────────────────────────────────────
from .train import TrainConfig, build_edge_to_triads, evaluate, main, train

# ── Profiling / sensitivity ─────────────────────────────────────────
from .triton_kernels_sensitivity import (
    main,
    measure_memory,
    sweep_G,
    sweep_T,
    sweep_block_D,
    sweep_block_T,
    sweep_d,
    sweep_k,
)

__all__ = [
    "AttentionConfig",
    "BSplineActivation",
    "BatchedBSplineActivation",
    "BatchedCatmullRomActivation",
    "BatchedKochanekBartelsActivation",
    "BilinearHead",
    "CPML",
    "CPMLConfig",
    "CapsuleHypergraphRouter",
    "CatmullRomActivation",
    "ClifFIRTierAggregator",
    "CoefEntropyRegulariser",
    "CrossBranchRegConfig",
    "CrossBranchRegulariser",
    "DiagonalBatchedBSplineActivation",
    "DiagonalBatchedCatmullRomActivation",
    "DiagonalBatchedKochanekBartelsActivation",
    "EntropyRegConfig",
    "EntropyRegulariser",
    "HighwaySignedKAN",
    "HighwaySignedKANConfig",
    "HyperedgeDensityRegulariser",
    "KochanekBartelsActivation",
    "LearnableMe",
    "LowRankBilinearHead",
    "MultiLayerSignedKAN",
    "MultiLayerSignedKANConfig",
    "NTupleBalanceLoss",
    "NTupleBalanceLossConfig",
    "ParticipationRegulariser",
    "PruneMask",
    "SceneGraph",
    "SceneObject",
    "SceneRelation",
    "SignedKAN",
    "SignedKANConfig",
    "SignedKANLayer",
    "SignedKANTierAggregator",
    "SignedNTuple",
    "SignedTriad",
    "SignedTriadAttention",
    "SplineSmoothRegulariser",
    "SymbolicFit",
    "TierAggregator",
    "TierSpec",
    "TrainConfig",
    "TriadLoss",
    "TriadLossConfig",
    "attention_entropy_loss",
    "build_attention_pairs",
    "build_edge_to_triads",
    "build_ntuple_balance_tensors",
    "build_triad_pairs",
    "build_vertex_triad_incidence",
    "compute_spectral_init",
    "construct",
    "construct_2",
    "construct_k",
    "construct_walks",
    "count_active_splines",
    "cox_de_boor",
    "demo_kitchen_scene",
    "distill_activation",
    "enumerate_k_cycles",
    "eval_with_sigma_masking",
    "evaluate",
    "evaluate_symbolic",
    "fit_summary",
    "fit_symbolic",
    "main",
    "make_spectral_init",
    "make_uniform_knots",
    "measure_activity",
    "measure_memory",
    "patch_per_arity_for_query",
    "patch_sigma_for_query",
    "prune_inactive",
    "restrict_cycles_to_tier",
    "sample_spline_activation",
    "signed_laplacian",
    "signed_normalised_laplacian",
    "stats",
    "sweep_G",
    "sweep_T",
    "sweep_block_D",
    "sweep_block_T",
    "sweep_d",
    "sweep_k",
    "top_k_eigenvectors",
    "train",
    "triad_degree",
    "triad_density",
]