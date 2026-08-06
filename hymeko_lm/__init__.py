"""hymeko_lm — Gömb / HSiKAN / Fiber-Spike-Rotor language model (Phase 0 skeleton).

A composition crate over the existing HyMeKo primitives (no new algorithm code):
the sequence mixer replaces softmax self-attention with a causal, signed,
rotor-transported, spike-gated walk-holonomy over token positions; the channel
mixer is the HSiKAN CR-Chebyshev cell (``hymeko_neuro.core.make_activation('cr_cheby')``);
the residual stream lives on the unit hypersphere (Gömb).

Plan: ``docs/plans/2026-06-29-gomb-hsikan-fsr-llm/``. Phase 0 = forward/backward
smoke on a lag-copy toy; the go/no-go A/B vs a matched transformer is Phase 1.
"""
from __future__ import annotations

from hymeko_lm.block import FSRBlock
from hymeko_lm.channel_mixer import HSiKANChannelMixer
from hymeko_lm.config import Activation, FSRConfig, GateMode
from hymeko_lm.model import FSRLanguageModel
from hymeko_lm.sequence_mixer import FiberSpikeRotorMixer
from hymeko_lm.sphere import SphereEmbedding, l2_normalize, spherical_residual

__all__ = [
    "Activation",
    "FSRConfig",
    "GateMode",
    "FSRBlock",
    "FSRLanguageModel",
    "FiberSpikeRotorMixer",
    "HSiKANChannelMixer",
    "SphereEmbedding",
    "l2_normalize",
    "spherical_residual",
]
