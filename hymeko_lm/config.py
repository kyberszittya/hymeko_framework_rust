"""Typed configuration for the FSR language model (enums, not strings — §6.5#7)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Activation(str, Enum):
    """Edge-activation kind for the HSiKAN channel mixer (maps to ``signed_kan.make_activation``)."""

    CR = "cr"
    CR_CHEBY = "cr_cheby"
    BSPLINE = "bspline"


class GateMode(str, Enum):
    """Spike-gate selection. ``SOFTMAX`` = sharp winner-take-most over the past (needed for
    content-addressed recall); ``SIGMOID`` = soft independent gates (the diffuse ablation)."""

    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


class ResidualMode(str, Enum):
    """Residual-stream geometry. ``SPHERE`` = Gömb: retract to S^{d-1} each sublayer (normalised-
    transformer). ``PRENORM`` = standard additive residual with pre-LayerNorm (the proven control;
    isolates whether the sphere constraint helps or hurts)."""

    SPHERE = "sphere"
    PRENORM = "prenorm"


@dataclass(frozen=True)
class FSRConfig:
    """Hyperparameters of the Gömb/HSiKAN/Fiber-Spike-Rotor LM.

    The hidden width is ``d_model = 3 * n_blocks`` because the rotor (quaternion,
    Cl(0,2)+) rotates 3-vectors, so channels are grouped into 3-blocks.

    # Preconditions all sizes ``>= 1``; ``2 <= n_layers``; ``grid >= 8`` for ``cr_cheby``.
    """

    vocab_size: int
    n_blocks: int = 16
    n_layers: int = 4
    max_seq_len: int = 64
    gate_rank: int = 16
    channel_mult: int = 2
    activation: Activation = Activation.CR_CHEBY
    gate_mode: GateMode = GateMode.SOFTMAX
    residual_mode: ResidualMode = ResidualMode.SPHERE
    spike_k: int | None = None        # None = dense O(T^2); int = hard top-k spike gate, O(T*k)
    grid: int = 12

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.n_blocks, self.n_layers, self.max_seq_len,
                self.gate_rank, self.channel_mult) < 1:
            raise ValueError("all FSRConfig sizes must be >= 1")
        if self.activation is Activation.CR_CHEBY and self.grid < 8:
            raise ValueError(f"cr_cheby needs grid >= 8; got {self.grid}")
        if self.spike_k is not None and self.spike_k < 1:
            raise ValueError(f"spike_k must be >= 1 or None; got {self.spike_k}")

    @property
    def d_model(self) -> int:
        """Hidden width = 3 · n_blocks (one quaternion per 3-block)."""
        return 3 * self.n_blocks
