"""Feature Pyramid Network (FPN) for HyMeYOLO Stage C.

A 2-level FPN that takes intermediate ResNet-tiny features at /4
(16x16) and /8 (8x8), produces sampleable feature maps at both
scales for multi-scale bilinear sampling at the query corners.

Architecture (matches docs/plans/2026-05-16-hymeyolo-stage-c-fpn/plan.tex):

    Backbone /4 (c_in_4, 16x16) ──── lateral_p4 (1x1) ─┐
                                                       ├─→ smooth_p4 (3x3) ─→ P4 (c_out, 16x16)
    Backbone /8 (c_in_8,  8x8) ──── lateral_p8 (1x1) ──┴────────────────────→ P8 (c_out,  8x8)
                                            │
                                            └─ upsample 2x (nearest) ──→ feeds the add above

Parameter budget at c_in_4=32, c_in_8=d_hidden=32, c_out=32:
    lateral_p4: 32*32 + 32 = 1056
    lateral_p8: 32*32 + 32 = 1056
    smooth_p4 (conv + BN): 32*32*9 + 64 = 9280
    Total: ~11.4k

Plan: docs/plans/2026-05-16-hymeyolo-stage-c-fpn/.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN2Level(nn.Module):
    """2-level FPN: lateral 1x1 + top-down upsample + 3x3 smooth.

    Input: (P_in_4, P_in_8) backbone features at /4 and /8.
    Output: (P_4, P_8) FPN features at /4 and /8, both at c_out channels.
    """

    def __init__(self, c_in_4: int, c_in_8: int, c_out: int) -> None:
        super().__init__()
        self.lateral_p4 = nn.Conv2d(c_in_4, c_out, kernel_size=1, bias=True)
        self.lateral_p8 = nn.Conv2d(c_in_8, c_out, kernel_size=1, bias=True)
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=False),
        )
        self.c_out = c_out

    def forward(
        self,
        p_in_4: torch.Tensor,
        p_in_8: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_8 = self.lateral_p8(p_in_8)                   # (B, c_out, H/8, W/8)
        p_4_lat = self.lateral_p4(p_in_4)               # (B, c_out, H/4, W/4)
        # Top-down: upsample P8 to /4 and add to lateral.
        up = F.interpolate(p_8, scale_factor=2.0, mode="nearest")
        p_4 = p_4_lat + up
        p_4 = self.smooth_p4(p_4)
        return p_4, p_8
