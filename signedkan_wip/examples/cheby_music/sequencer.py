"""The sequencer: one ChebyshevCRActivation channel = one melodic voice."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from signed_kan import ChebyshevCRActivation

from signedkan_wip.examples.cheby_music.scale import Scale


@dataclass(frozen=True)
class Note:
    """One sounding note. ``start``/``duration`` are in sequencer *steps*."""
    channel: int
    start: int
    duration: int
    pitch: int
    velocity: int


class ChebyshevSequencer:
    """Turn a ``ChebyshevCRActivation`` into a multi-voice score.

    Each channel's band-limited curve, sampled over ``n_steps`` points of
    ``t ∈ [-1, 1]``, is one voice's melodic contour. Consecutive equal pitches
    are merged into one sustained note (so a slow contour holds rather than
    re-triggers).

    Preconditions: ``n_channels >= 1``, ``n_steps >= 2``, ``degree`` (Chebyshev
    order) ``2 <= degree <= grid``.
    Postconditions: ``compose()`` returns notes with ``channel ∈ [0, n_channels)``,
    pitches on ``scale``, and per-channel coverage of all ``n_steps`` steps.
    """

    def __init__(self, n_channels: int = 5, n_steps: int = 32, *,
                 degree: int = 6, grid: int = 16, scale: Scale | None = None,
                 velocity: int = 80, seed: int = 0) -> None:
        if n_channels < 1 or n_steps < 2:
            raise ValueError("need n_channels >= 1 and n_steps >= 2")
        torch.manual_seed(seed)
        self.cell = ChebyshevCRActivation(n_channels, grid=grid, k=degree)
        self.n_channels = n_channels
        self.n_steps = n_steps
        self.scale = scale or Scale()
        self.velocity = velocity

    def contours(self) -> torch.Tensor:
        """``(n_steps, n_channels)`` — column ``c`` is channel ``c``'s curve over
        the time grid, per-channel normalised to span ``[-1, 1]``."""
        t = torch.linspace(-1.0, 1.0, self.n_steps)
        x = t.unsqueeze(-1).expand(self.n_steps, self.n_channels)
        with torch.no_grad():
            y = self.cell(x)                                  # (n_steps, n_channels)
        y = y - y.mean(dim=0, keepdim=True)
        peak = y.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
        return y / peak

    def compose(self) -> list[Note]:
        y = self.contours()
        notes: list[Note] = []
        for c in range(self.n_channels):
            pitches = [self.scale.quantize(v.item()) for v in y[:, c]]
            start = 0
            for step in range(1, self.n_steps + 1):
                if step == self.n_steps or pitches[step] != pitches[start]:
                    notes.append(Note(
                        channel=c, start=start, duration=step - start,
                        pitch=pitches[start], velocity=self.velocity,
                    ))
                    start = step
        return notes
