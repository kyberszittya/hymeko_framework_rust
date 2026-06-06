"""AC-HSiKAN composable building blocks.

Strategy / Adapter pattern decomposition of the AC-HSiKAN attention
block. Each axis is a small ABC plus a few concrete implementations
that the layer can compose:

    SignAttention     -- compute per-pair signs (dense or sparse;
                          bilinear, quaternion, or multi-head quaternion).
    WalkOp            -- combine (anchor->candidate) signs into the
                          per-arity cycle sign-product (star, chain,
                          cycle; with an optional fused implementation).
    FFNBlock          -- post-attention feed-forward; standard 4x
                          expansion or bottleneck for speed.
    ContextEncoder    -- pre-attention contextualisation; identity,
                          Clifford-FIR (causal multivector convolution).

Build the right concrete class from an AcHsikanConfig via the
factories ``build_sign_attention(cfg)`` etc.
"""
from .attention import (
    SignAttention, BilinearSparseAttention, QuaternionSparseAttention,
    MultiHeadQuaternionSparseAttention, DenseAttentionAdapter,
    build_sign_attention,
)
from .context import (
    ContextEncoder, IdentityContext, CliffordFIRContext,
    build_context_encoder,
)
from .ffn import (
    FFNBlock, IdentityFFN, StandardFFN, BottleneckFFN, build_ffn_block,
)
from .walk_op import (
    WalkOp, StarWalkOp, ChainWalkOp, CycleWalkOp, FusedChainWalkOp,
    FusedCycleWalkOp, build_walk_op,
)

__all__ = [
    # Sign attention
    "SignAttention", "BilinearSparseAttention", "QuaternionSparseAttention",
    "MultiHeadQuaternionSparseAttention", "DenseAttentionAdapter",
    "build_sign_attention",
    # Context
    "ContextEncoder", "IdentityContext", "CliffordFIRContext",
    "build_context_encoder",
    # FFN
    "FFNBlock", "IdentityFFN", "StandardFFN", "BottleneckFFN",
    "build_ffn_block",
    # Walk op
    "WalkOp", "StarWalkOp", "ChainWalkOp", "CycleWalkOp",
    "FusedChainWalkOp", "FusedCycleWalkOp", "build_walk_op",
]
