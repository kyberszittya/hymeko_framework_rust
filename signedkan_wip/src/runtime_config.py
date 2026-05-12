"""Frozen runtime configuration parsed once at process startup.

CLAUDE.md §6.5 anti-pattern #11: env-var reads scattered through deep call
chains are forbidden. This module is the canonical place to parse them.

Usage:
    from signedkan_wip.src.runtime_config import HSiKANConfig, HyMeKoConfig

    cfg = HSiKANConfig.from_env()
    # ... pass cfg explicitly down the call chain — never imported at depth
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return bool(int(v))


# ─── HSiKAN — top-K cycle enumeration knobs ─────────────────────────


@dataclass(frozen=True)
class TopKConfig:
    """Top-K cycle-enumeration knobs (env: HSIKAN_TOPK_*).

    `mode` selects which legacy enumerator path is taken:
        ""                       → no top-K (fall through)
        "global"                 → top-K-global, regular scorer
        "global_bb"              → top-K-global + ABB
        "entropy"                → top-K-global + entropy/hybrid heuristic
        "per_vertex"             → per-vertex top-m
        "per_vertex_adaptive"    → per-vertex with degree-adaptive m_v
        "per_vertex_tiered"      → per-vertex with tiered (CPG) m_v
    """
    mode:          str   = ""
    k_keep:        int   = 16
    scorer:        str   = "fraction_negative"
    pruner:        str   = "none"

    # Entropy / hybrid (only used when mode == "entropy")
    heuristic:     str   = "entropy"
    hybrid_alpha:  float = 0.0
    hybrid_signal: str   = "fraction_negative"

    # Per-vertex ABB toggles
    use_per_vertex_abb:           bool  = False
    per_vertex_abb_mode:          str   = "start"  # "start" | "global"
    per_vertex_abb_fullness_gate: float = 1.0

    # Vertex pre-filter
    vertex_filter:            str = "none"
    vertex_filter_min_degree: int = 2

    # Tiered m_v (CPG)
    tiers_spec: str = "100.0:128"  # "<pct>:<cap>,<pct>:<cap>,..."

    # Degree-adaptive m_v
    adaptive_m_min: int   = 1
    adaptive_m_max: int   = 0     # 0 → use k_keep
    adaptive_c:     float = 0.0

    def fingerprint(self) -> dict[str, str]:
        """Cache-key fingerprint covering every field that affects cycle
        enumeration output. The dict shape matches the legacy
        `cycle_cache._topk_fingerprint()` for backward-compat with on-disk
        cache keys."""
        return {
            "HSIKAN_TOPK_MODE":                       self.mode,
            "HSIKAN_TOPK_K":                          str(self.k_keep),
            "HSIKAN_TOPK_SCORER":                     self.scorer,
            "HSIKAN_TOPK_PRUNER":                     self.pruner,
            "HSIKAN_TOPK_M_V_C":                      str(self.adaptive_c),
            "HSIKAN_TOPK_M_V_MIN":                    str(self.adaptive_m_min),
            "HSIKAN_TOPK_M_V_MAX":                    str(self.adaptive_m_max),
            "HSIKAN_TOPK_HEURISTIC":                  self.heuristic,
            "HSIKAN_TOPK_HYBRID_ALPHA":               str(self.hybrid_alpha),
            "HSIKAN_TOPK_SIGNAL":                     self.hybrid_signal,
            "HSIKAN_TOPK_TIERS":                      self.tiers_spec,
            "HSIKAN_USE_PER_VERTEX_ABB":              "1" if self.use_per_vertex_abb else "0",
            "HSIKAN_USE_PER_VERTEX_ABB_MODE":         self.per_vertex_abb_mode,
            "HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE":    str(self.per_vertex_abb_fullness_gate),
            "HSIKAN_VERTEX_FILTER":                   self.vertex_filter,
            "HSIKAN_VERTEX_FILTER_MIN_DEGREE":        str(self.vertex_filter_min_degree),
        }


# ─── Cache / cycle-cache knobs ──────────────────────────────────────


@dataclass(frozen=True)
class CycleCacheConfig:
    """Knobs for `cycle_cache.py` (env: HYMEKO_CYCLE_*)."""
    enabled:       bool = False
    enum_seed:     int  = 0
    cache_format:  str  = "npz"


# ─── Training / model toggles (env: HSIKAN_*) ───────────────────────


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop + architecture-toggle knobs (env: HSIKAN_*)."""
    arities:           tuple[int, ...]   = (3,)
    max_k2:            int               = 1_000_000
    max_k3:            int               = 30_000
    max_k4:            int               = 200_000
    mixed_tuples:      str               = ""        # e.g. "c3,c4,w2,w3"
    walk_lens:         tuple[int, ...]   = ()
    cycle_batch:       int | None        = None
    entropy_lambda:    float             = 0.0
    gumbel_hard:       bool              = False
    gumbel_tau:        float             = 1.0
    per_edge_gate:     bool              = False
    attention_kind:    str               = "none"
    direct_messaging:  bool              = False
    spline_kind:       str               = "catmull_rom"
    strict_protocol:   bool              = False
    sparse_attn_k:     int               = 0     # top-K sparse attention


@dataclass(frozen=True)
class KernelConfig:
    """Triton / KAN kernel toggles (env: HSIKAN_*)."""
    triton_kernel:    bool   = False
    triton_backward:  bool   = True
    chunk_t:          int    = 0
    kb_preset:        str    = ""
    kb_init_tcb:      str    = ""


@dataclass(frozen=True)
class CompileConfig:
    """torch.compile toggles (env: HSIKAN_TORCH_COMPILE / HSIKAN_COMPILE_MODE)."""
    enabled:  bool  = False
    mode:     str   = "reduce-overhead"


# ─── Aggregated runtime config ──────────────────────────────────────


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level frozen config. Parse once per logical entry-point and
    pass explicitly. **Do not import `_RUNTIME` — call `get_runtime()`,
    which re-parses env each call so orchestrators that mutate
    `os.environ` between phases see fresh values.**"""
    topk:         TopKConfig       = field(default_factory=TopKConfig)
    training:     TrainingConfig   = field(default_factory=TrainingConfig)
    kernel:       KernelConfig     = field(default_factory=KernelConfig)
    compile:      CompileConfig    = field(default_factory=CompileConfig)
    cycle_cache:  CycleCacheConfig = field(default_factory=CycleCacheConfig)

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Parse the entire HSiKAN/HyMeKo env-var surface in one place."""
        topk = _parse_topk()
        training = _parse_training()
        kernel = _parse_kernel()
        compile_cfg = _parse_compile()
        cycle_cache = _parse_cycle_cache()
        return cls(
            topk=topk, training=training, kernel=kernel,
            compile=compile_cfg, cycle_cache=cycle_cache,
        )

    def with_topk(self, **overrides) -> "RuntimeConfig":
        return replace(self, topk=replace(self.topk, **overrides))


# ─── Per-section parsers (each ≤ 30 LOC, single concern) ────────────


def _parse_topk() -> TopKConfig:
    return TopKConfig(
        mode=_env_str("HSIKAN_TOPK_MODE", ""),
        k_keep=_env_int("HSIKAN_TOPK_K", 16),
        scorer=_env_str("HSIKAN_TOPK_SCORER", "fraction_negative"),
        pruner=_env_str("HSIKAN_TOPK_PRUNER", "none"),
        heuristic=_env_str("HSIKAN_TOPK_HEURISTIC", "entropy"),
        hybrid_alpha=_env_float("HSIKAN_TOPK_HYBRID_ALPHA", 0.0),
        hybrid_signal=_env_str("HSIKAN_TOPK_SIGNAL", "fraction_negative"),
        use_per_vertex_abb=_env_bool("HSIKAN_USE_PER_VERTEX_ABB", False),
        per_vertex_abb_mode=_env_str("HSIKAN_USE_PER_VERTEX_ABB_MODE", "start"),
        per_vertex_abb_fullness_gate=_env_float(
            "HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE", 1.0),
        vertex_filter=_env_str("HSIKAN_VERTEX_FILTER", "none"),
        vertex_filter_min_degree=_env_int("HSIKAN_VERTEX_FILTER_MIN_DEGREE", 2),
        tiers_spec=_env_str("HSIKAN_TOPK_TIERS", "100.0:128"),
        adaptive_m_min=_env_int("HSIKAN_TOPK_M_V_MIN", 1),
        adaptive_m_max=_env_int(
            "HSIKAN_TOPK_M_V_MAX", _env_int("HSIKAN_TOPK_K", 16)),
        adaptive_c=_env_float("HSIKAN_TOPK_M_V_C", 0.0),
    )


def _parse_training() -> TrainingConfig:
    arities_str = _env_str("HSIKAN_ARITIES", "3")
    arities = tuple(int(x) for x in arities_str.split(",") if x.strip())
    walk_lens_str = _env_str("HSIKAN_WALK_LENS", "")
    walk_lens = tuple(int(x) for x in walk_lens_str.split(",") if x.strip())
    cycle_batch_str = os.environ.get("HSIKAN_CYCLE_BATCH")
    cycle_batch = int(cycle_batch_str) if cycle_batch_str else None
    return TrainingConfig(
        arities=arities,
        max_k2=_env_int("HSIKAN_MAX_K2", 1_000_000),
        max_k3=_env_int("HSIKAN_MAX_K3", 30_000),
        max_k4=_env_int("HSIKAN_MAX_K4", 200_000),
        mixed_tuples=_env_str("HSIKAN_MIXED_TUPLES", ""),
        walk_lens=walk_lens,
        cycle_batch=cycle_batch,
        entropy_lambda=_env_float("HSIKAN_ENTROPY_LAMBDA", 0.0),
        gumbel_hard=_env_bool("HSIKAN_GUMBEL_HARD", False),
        gumbel_tau=_env_float("HSIKAN_GUMBEL_TAU", 1.0),
        per_edge_gate=_env_bool("HSIKAN_PER_EDGE_GATE", False),
        attention_kind=_env_str("HSIKAN_ATTENTION_M_E", "none").lower(),
        direct_messaging=_env_bool("HSIKAN_DIRECT_MESSAGING", False),
        spline_kind=_env_str("HSIKAN_SPLINE_KIND", "catmull_rom"),
        strict_protocol=_env_bool("HSIKAN_STRICT_PROTOCOL", False),
        sparse_attn_k=_env_int("HSIKAN_SPARSE_ATTN_K", 0),
    )


def _parse_kernel() -> KernelConfig:
    return KernelConfig(
        triton_kernel=_env_bool("HSIKAN_TRITON_KERNEL", False),
        triton_backward=_env_bool("HSIKAN_TRITON_BACKWARD", True),
        chunk_t=_env_int("HSIKAN_CHUNK_T", 0),
        kb_preset=_env_str("HSIKAN_KB_PRESET", ""),
        kb_init_tcb=_env_str("HSIKAN_KB_INIT_TCB", ""),
    )


def _parse_compile() -> CompileConfig:
    return CompileConfig(
        enabled=_env_str("HSIKAN_TORCH_COMPILE", "0") == "1",
        mode=_env_str("HSIKAN_COMPILE_MODE", "reduce-overhead"),
    )


def _parse_cycle_cache() -> CycleCacheConfig:
    return CycleCacheConfig(
        enabled=_env_bool("HYMEKO_CYCLE_CACHE", False),
        enum_seed=_env_int("HYMEKO_CYCLE_ENUM_SEED", 0),
        cache_format=_env_str("HYMEKO_CACHE_FORMAT", "npz"),
    )


# ─── Accessor ───────────────────────────────────────────────────────


def get_runtime() -> RuntimeConfig:
    """Return a fresh `RuntimeConfig.from_env()` snapshot.

    Re-parses every call. Cheap (≈ 40 string lookups + dataclass build).
    Orchestrator code that mutates `os.environ` between training phases
    will see the new values on the next call.

    Prefer passing the returned config explicitly via parameters; this
    helper is a bridge while AP-11 migration is in progress.
    """
    return RuntimeConfig.from_env()


def parse_tiers_spec(spec: str) -> list[tuple[float, int]]:
    """Parse '<pct>:<cap>,<pct>:<cap>,...' into a sorted list."""
    tiers: list[tuple[float, int]] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        pct_str, cap_str = piece.split(":")
        tiers.append((float(pct_str), int(cap_str)))
    return tiers
