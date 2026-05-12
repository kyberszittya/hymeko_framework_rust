"""Runtime gate for the Triton fused backward kernels (CLAUDE.md §6.5 #4 split)."""
from __future__ import annotations

from ..runtime_config import get_runtime


def _triton_backward_enabled() -> bool:
    """Whether the Triton-fused backward kernels are active.

    Default True; set ``HSIKAN_TRITON_BACKWARD=0`` to fall back to the
    PyTorch-reference recomputation (slower, useful for debugging)."""
    return get_runtime().kernel.triton_backward
