"""Electronic / industrial beat patterns + bass / arp riffs (16th-note grid).

Rhythm source for the wavetable synth: named 1-bar (16-step) drum patterns in the
four-on-the-floor / techno / industrial / breakbeat idioms, plus simple bass and
arpeggio riffs over a scale. These produce ``Note``s; the synth renders pitched
notes through the Chebyshev-CR / wavelet wavetables and drum notes as percussion.
"""
from __future__ import annotations

from signedkan_wip.examples.cheby_music.beat import (
    CLAP,
    HAT_CLOSED,
    KICK,
    LOW_TOM,
    OPEN_HAT,
    SNARE,
)
from signedkan_wip.examples.cheby_music.scale import Scale
from signedkan_wip.examples.cheby_music.sequencer import Note

BAR = 16  # steps per bar (16th notes)

# name → {drum pitch: [step indices within the bar]}
DRUM_PATTERNS: dict[str, dict[int, list[int]]] = {
    "four_on_floor": {
        KICK: [0, 4, 8, 12],
        CLAP: [4, 12],
        HAT_CLOSED: [2, 6, 10, 14],
    },
    "techno": {
        KICK: [0, 4, 8, 12],
        CLAP: [4, 12],
        OPEN_HAT: [2, 6, 10, 14],
        HAT_CLOSED: [0, 1, 3, 5, 7, 9, 11, 13, 15],
    },
    "industrial": {
        KICK: [0, 3, 6, 8, 11, 14],
        SNARE: [4, 12],
        LOW_TOM: [7, 15],
        HAT_CLOSED: list(range(0, 16, 2)),
        OPEN_HAT: [10],
    },
    "breakbeat": {
        KICK: [0, 10],
        SNARE: [4, 7, 12],
        HAT_CLOSED: list(range(0, 16, 2)),
    },
}


def drum_pattern(name: str, n_bars: int, channel: int, *, velocity: int = 100) -> list[Note]:
    """Expand a named drum pattern over ``n_bars`` into drum ``Note``s.

    Preconditions: ``name`` in ``DRUM_PATTERNS``; ``n_bars >= 1``.
    Postconditions: every note on ``channel`` with a drum pitch, ``duration == 1``.
    """
    if name not in DRUM_PATTERNS:
        raise ValueError(f"unknown pattern {name!r}; choose {sorted(DRUM_PATTERNS)}")
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1; got {n_bars}")
    pat = DRUM_PATTERNS[name]
    notes: list[Note] = []
    for bar in range(n_bars):
        base = bar * BAR
        for pitch, steps in pat.items():
            vel = velocity if pitch in (KICK, SNARE, CLAP) else velocity - 25
            notes.extend(Note(channel, base + s, 1, pitch, max(30, vel)) for s in steps)
    return notes


# Bass riffs as scale-degree indices over a bar (None = rest). Low octave.
_BASS_RIFFS: dict[str, list[int | None]] = {
    "rolling": [0, None, 0, 0, None, 0, 3, None, 0, None, 0, 2, None, 0, 5, None],
    "offbeat": [0, None, None, 0, None, None, 0, None, 5, None, None, 3, None, None, 7, None],
    "driving": [0, 0, 7, 0, 0, 0, 5, 0, 0, 0, 7, 0, 3, 0, 5, 0],
}


def bassline(root: int, scale_name: str, n_bars: int, channel: int, *,
             riff: str = "rolling", velocity: int = 105) -> list[Note]:
    """A bass riff over ``n_bars`` on the wavetable channel ``channel``.

    Preconditions: ``riff`` in ``_BASS_RIFFS``; ``n_bars >= 1``.
    """
    if riff not in _BASS_RIFFS:
        raise ValueError(f"unknown riff {riff!r}; choose {sorted(_BASS_RIFFS)}")
    scale = Scale(root=root, name=scale_name, octaves=2)
    seq = _BASS_RIFFS[riff]
    notes: list[Note] = []
    for bar in range(n_bars):
        for s, deg in enumerate(seq):
            if deg is None:
                continue
            pitch = scale.pitches[min(deg, len(scale.pitches) - 1)]
            notes.append(Note(channel, bar * BAR + s, 1, pitch, velocity))
    return notes


def arp(root: int, scale_name: str, n_bars: int, channel: int, *,
        degrees: tuple[int, ...] = (0, 2, 4, 7), step: int = 2,
        velocity: int = 85) -> list[Note]:
    """An arpeggio cycling ``degrees`` (scale indices) every ``step`` steps,
    one octave above the bass."""
    scale = Scale(root=root + 12, name=scale_name, octaves=2)
    notes: list[Note] = []
    k = 0
    for pos in range(0, n_bars * BAR, step):
        deg = degrees[k % len(degrees)]
        pitch = scale.pitches[min(deg, len(scale.pitches) - 1)]
        notes.append(Note(channel, pos, step, pitch, velocity))
        k += 1
    return notes
