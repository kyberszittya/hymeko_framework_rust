"""Musical scale: continuous value in [-1, 1] → quantised MIDI pitch."""
from __future__ import annotations

# Semitone offsets from the root, per scale name.
SCALES: dict[str, tuple[int, ...]] = {
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
}


class Scale:
    """Maps a value in ``[-1, 1]`` to a MIDI pitch on a fixed scale.

    Preconditions: ``root`` a valid MIDI note (0–127); ``octaves >= 1``;
    ``name`` in ``SCALES``.
    Postconditions: ``quantize`` returns a pitch in ``[root, 127]`` drawn from
    the scale's pitch table, monotone non-decreasing in its input.
    """

    def __init__(self, root: int = 57, name: str = "minor_pentatonic",
                 octaves: int = 3) -> None:
        if name not in SCALES:
            raise ValueError(f"unknown scale {name!r}; choose from {sorted(SCALES)}")
        if octaves < 1:
            raise ValueError(f"octaves must be >= 1; got {octaves}")
        intervals = SCALES[name]
        self.pitches = [
            root + 12 * o + i for o in range(octaves) for i in intervals
            if root + 12 * o + i <= 127
        ]
        self.name = name

    def quantize(self, x: float) -> int:
        """Value in [-1, 1] → a scale pitch (clamped, then index-mapped)."""
        x = max(-1.0, min(1.0, float(x)))
        idx = round((x + 1.0) * 0.5 * (len(self.pitches) - 1))
        return self.pitches[idx]
