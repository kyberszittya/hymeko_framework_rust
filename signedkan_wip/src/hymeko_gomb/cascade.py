"""Full HymeKo-Gömb cascade composer + ablation wrappers + mixed-arity.

Three model classes that consume the shells defined in `shells.py`:
  - `HymeKoGomb`     : the full three-shell cascade (the plan's
                      mainline architecture)
  - `GombNoOuter` / `GombNoMiddle` / `GombNoInner`  : one-shell-dropped
                      ablation models — separate model classes, not
                      forward-time flags, per plan §Sequencing
  - `MixedArityGomb` : one full stack per cycle arity in `cycle_ks`
                      with learned αₖ softmax fusion of edge logits
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shells import InnerCPMLCore, MiddleHSiKAN, OuterFIRShell


@dataclass
class GombConfig:
    """HymeKo-Gömb configuration."""
    n_nodes: int = 0
    d_embed: int = 32
    d_outer: int = 16
    M_outer: int = 8
    d_middle: int = 32
    d_core: int = 32
    n_tiers: int = 3
    cycle_k: int = 3
    middle_grid: int = 5
    d_predictor_hidden: int = 32


# ─── Full three-shell cascade ───────────────────────────────────────


class HymeKoGomb(nn.Module):
    """The three-shell cascade.

    Forward:
        x_embed → outer_shell → middle_shell → inner_core → edge_predictor

    Forward signature:
        cycles         : (M_c, k) long
        signs          : (M_c, k)
        tier_of        : (N,) long
        edges_to_score : (E, 2) long
        return         : (E,) edge sign logits
    """

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)

        self.outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        middle_in = cfg.d_embed + outer_out
        self.middle = MiddleHSiKAN(
            n_nodes=cfg.n_nodes, d_in=middle_in, d_layer=cfg.d_middle,
            cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
        )
        core_in = cfg.d_embed + outer_out + cfg.d_middle
        self.core = InnerCPMLCore(
            d_in=core_in, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
        )

    def forward(
        self,
        cycles: torch.Tensor,
        signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.node_embed.weight
        x_outer = self.outer(x_embed, cycles, signs)
        x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
        x_middle = self.middle(x_for_middle, cycles, signs)
        x_for_core = torch.cat([x_embed, x_outer, x_middle], dim=-1)
        scores, _ = self.core(x_for_core, cycles, signs, tier_of, edges_to_score)
        return scores

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Ablation wrappers ──────────────────────────────────────────────


class GombNoOuter(nn.Module):
    """Cascade without the Outer FIR volume."""

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.middle = MiddleHSiKAN(
            n_nodes=cfg.n_nodes, d_in=cfg.d_embed, d_layer=cfg.d_middle,
            cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
        )
        self.core = InnerCPMLCore(
            d_in=cfg.d_embed + cfg.d_middle, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
        )

    def forward(self, cycles, signs, tier_of, edges_to_score):
        x_embed = self.node_embed.weight
        x_middle = self.middle(x_embed, cycles, signs)
        x_for_core = torch.cat([x_embed, x_middle], dim=-1)
        scores, _ = self.core(x_for_core, cycles, signs, tier_of, edges_to_score)
        return scores

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GombNoMiddle(nn.Module):
    """Cascade without the Middle HSiKAN shell."""

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        self.core = InnerCPMLCore(
            d_in=cfg.d_embed + outer_out, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
        )

    def forward(self, cycles, signs, tier_of, edges_to_score):
        x_embed = self.node_embed.weight
        x_outer = self.outer(x_embed, cycles, signs)
        x_for_core = torch.cat([x_embed, x_outer], dim=-1)
        scores, _ = self.core(x_for_core, cycles, signs, tier_of, edges_to_score)
        return scores

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GombNoInner(nn.Module):
    """Cascade without the Inner CPML core; plain MLP edge head."""

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        self.middle = MiddleHSiKAN(
            n_nodes=cfg.n_nodes, d_in=cfg.d_embed + outer_out,
            d_layer=cfg.d_middle,
            cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
        )
        final_dim = cfg.d_embed + outer_out + cfg.d_middle
        self.head = nn.Sequential(
            nn.Linear(2 * final_dim, cfg.d_predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_predictor_hidden, 1),
        )

    def forward(self, cycles, signs, tier_of, edges_to_score):
        del tier_of   # no CPML tier dispatch in this ablation
        x_embed = self.node_embed.weight
        x_outer = self.outer(x_embed, cycles, signs)
        x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
        x_middle = self.middle(x_for_middle, cycles, signs)
        x_final = torch.cat([x_embed, x_outer, x_middle], dim=-1)
        u = x_final[edges_to_score[:, 0]]
        v = x_final[edges_to_score[:, 1]]
        return self.head(torch.cat([u, v], dim=-1)).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Mixed-arity (k=3+k=4, k=4+k=5, …) ──────────────────────────────


class MixedArityGomb(nn.Module):
    """One full (outer, middle, inner) stack per cycle arity in `cycle_ks`,
    with learned αₖ softmax fusion of edge logits.

    Memory `project_phase9_k45_sweet_spot_2026_05_02`: k=4+k=5 mixed
    with learned αₖ Pareto-dominates single-arity k=3 on every signed
    dataset.

    Forward signature:
        cycles_by_k    : {k -> (M_c_k, k) long}
        signs_by_k     : {k -> (M_c_k, k) float}
        tier_of        : (N,) long
        edges_to_score : (E, 2) long
        return         : (E,) edge sign logits
    """

    def __init__(self, cfg: GombConfig, cycle_ks: tuple[int, ...] = (3, 4)):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        if len(cycle_ks) < 1:
            raise ValueError("cycle_ks must contain at least one arity")
        self.cfg = cfg
        self.cycle_ks = tuple(cycle_ks)

        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)

        outer_out = cfg.M_outer * cfg.d_outer
        middle_in = cfg.d_embed + outer_out
        core_in = cfg.d_embed + outer_out + cfg.d_middle

        self.outers = nn.ModuleDict()
        self.middles = nn.ModuleDict()
        self.cores = nn.ModuleDict()
        for k in self.cycle_ks:
            sk = str(k)
            self.outers[sk] = OuterFIRShell(
                d_in=cfg.d_embed, d_layer=cfg.d_outer,
                M=cfg.M_outer, cycle_k=k,
            )
            self.middles[sk] = MiddleHSiKAN(
                n_nodes=cfg.n_nodes, d_in=middle_in, d_layer=cfg.d_middle,
                cycle_k=k, grid=cfg.middle_grid,
            )
            self.cores[sk] = InnerCPMLCore(
                d_in=core_in, d_layer=cfg.d_core,
                n_tiers=cfg.n_tiers, cycle_k=k,
            )

        self.alpha_logits = nn.Parameter(torch.zeros(len(self.cycle_ks)))

    def forward(
        self,
        cycles_by_k: dict[int, torch.Tensor],
        signs_by_k: dict[int, torch.Tensor],
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.node_embed.weight
        alpha = F.softmax(self.alpha_logits, dim=0)
        out = None
        for i, k in enumerate(self.cycle_ks):
            sk = str(k)
            cycles_k = cycles_by_k[k]
            signs_k = signs_by_k[k]
            x_outer = self.outers[sk](x_embed, cycles_k, signs_k)
            x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
            x_middle = self.middles[sk](x_for_middle, cycles_k, signs_k)
            x_for_core = torch.cat([x_embed, x_outer, x_middle], dim=-1)
            scores_k, _ = self.cores[sk](
                x_for_core, cycles_k, signs_k, tier_of, edges_to_score,
            )
            contribution = alpha[i] * scores_k
            out = contribution if out is None else out + contribution
        assert out is not None
        return out

    def alpha(self) -> torch.Tensor:
        """Current αₖ (softmaxed) — diagnostic for per-arity weight tracking."""
        return F.softmax(self.alpha_logits, dim=0).detach()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "GombConfig",
    "HymeKoGomb",
    "GombNoOuter", "GombNoMiddle", "GombNoInner",
    "MixedArityGomb",
]
