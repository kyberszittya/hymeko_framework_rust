"""Minimal Standard MIDI File (format 1) writer — no third-party dependency."""
from __future__ import annotations

import struct
from pathlib import Path

from hymeko_neuro.examples.cheby_music.sequencer import Note

# General-MIDI program per voice index (piano, marimba, vibes, nylon guitar,
# strings, flute, choir, bass) — distinct timbres so the voices separate.
GM_PROGRAMS = (0, 12, 11, 24, 48, 73, 52, 33)


def vlq(value: int) -> bytes:
    """MIDI variable-length quantity encoding of a non-negative int."""
    if value < 0:
        raise ValueError(f"vlq value must be >= 0; got {value}")
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack(">I", len(body)) + body


def notes_to_midi(notes: list[Note], n_channels: int, *, ticks_per_step: int = 240,
                  ticks_per_quarter: int = 480, bpm: int = 110,
                  drum_channel: int | None = None) -> bytes:
    """Render notes to Standard MIDI File (format 1) bytes — one track per voice.

    ``drum_channel`` (if given) is routed to GM percussion (MIDI channel 10,
    index 9) with no melodic program change — its note numbers are drum sounds.

    Preconditions: every ``Note.channel < n_channels``; positive rates.
    Postconditions: returns a valid SMF byte string beginning with ``MThd``.
    """
    header = struct.pack(">HHH", 1, n_channels + 1, ticks_per_quarter)
    tracks = [_chunk(b"MThd", header)]

    us_per_quarter = round(60_000_000 / bpm)
    tempo = vlq(0) + bytes([0xFF, 0x51, 0x03]) + us_per_quarter.to_bytes(3, "big")
    end = vlq(0) + bytes([0xFF, 0x2F, 0x00])
    tracks.append(_chunk(b"MTrk", tempo + end))

    by_channel: dict[int, list[Note]] = {c: [] for c in range(n_channels)}
    for n in notes:
        by_channel[n.channel].append(n)

    for c in range(n_channels):
        is_drum = c == drum_channel
        ch = 9 if is_drum else c % 16          # GM percussion lives on channel 10
        # Drums need no program change (the kit is fixed on channel 10).
        events: list[tuple[int, bytes]] = (
            [] if is_drum else
            [(0, bytes([0xC0 | ch, GM_PROGRAMS[c % len(GM_PROGRAMS)]]))]
        )
        for n in by_channel[c]:
            on = n.start * ticks_per_step
            off = (n.start + n.duration) * ticks_per_step
            events.append((on, bytes([0x90 | ch, n.pitch, n.velocity])))
            events.append((off, bytes([0x80 | ch, n.pitch, 0])))
        events.sort(key=lambda e: e[0])
        body = b""
        prev = 0
        for tick, msg in events:
            body += vlq(tick - prev) + msg
            prev = tick
        body += vlq(0) + bytes([0xFF, 0x2F, 0x00])
        tracks.append(_chunk(b"MTrk", body))

    return b"".join(tracks)


def write_midi(notes: list[Note], n_channels: int, path: str | Path, **kw) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(notes_to_midi(notes, n_channels, **kw))
    return dest
