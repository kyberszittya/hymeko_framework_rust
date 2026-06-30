"""Background beat: a percussion pattern on a dedicated drum channel.

Drum hits are modelled as ``Note``s whose ``pitch`` is a General-MIDI drum note
(kick / snare / hat). They live on their own channel; ``midi`` routes that
channel to GM percussion (MIDI channel 10), and ``synth`` renders kick/snare/hat
by ear instead of a pitched oscillator.
"""
from __future__ import annotations

import numpy as np

from signedkan_wip.examples.cheby_music.sequencer import Note

# General-MIDI percussion note numbers.
KICK = 36
SNARE = 38
CLAP = 39
LOW_TOM = 45
HAT_CLOSED = 42
OPEN_HAT = 46
CRASH = 49
DRUM_PITCHES = (KICK, SNARE, CLAP, LOW_TOM, HAT_CLOSED, OPEN_HAT, CRASH)


def make_beat(n_steps: int, channel: int, *, bar: int = 8, hat_every: int = 1,
              velocity: int = 96, seed: int = 0) -> list[Note]:
    """A simple 4/4 backing pattern over ``n_steps`` (steps = eighth notes).

    Kick on the down-beats (positions 0, 4 of each ``bar``), snare on the
    back-beats (2, 6), hi-hat every ``hat_every`` steps with light seed-based
    velocity variation (a touch of human feel).

    Preconditions: ``n_steps >= 1``, ``bar >= 1``, ``hat_every >= 1``.
    Postconditions: every returned ``Note`` is on ``channel`` with a drum pitch
    in ``DRUM_PITCHES`` and ``duration == 1``.
    """
    if n_steps < 1 or bar < 1 or hat_every < 1:
        raise ValueError("need n_steps, bar, hat_every all >= 1")
    rng = np.random.default_rng(seed)
    kick_pos = {0, bar // 2}
    snare_pos = {bar // 4, bar // 4 + bar // 2}
    notes: list[Note] = []
    for s in range(n_steps):
        pos = s % bar
        if pos in kick_pos:
            notes.append(Note(channel, s, 1, KICK, velocity))
        if pos in snare_pos:
            notes.append(Note(channel, s, 1, SNARE, velocity))
        if s % hat_every == 0:
            jitter = int(rng.integers(-12, 13))
            notes.append(Note(channel, s, 1, HAT_CLOSED,
                              max(30, velocity - 30 + jitter)))
    return notes
