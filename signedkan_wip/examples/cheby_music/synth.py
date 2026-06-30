"""Software synth → WAV: a configurable signal chain over the sequencer notes.

Stages (all numpy + stdlib ``wave``; no third-party audio dependency):

    per-note oscillator (harmonic timbre)  ─┐
        + LFO  (vibrato = pitch mod, tremolo = amp mod; sine or Chebyshev shape)
        + noise (per-note white-noise blend)
                                             ├─ per-channel mix gains (+ pan)
    pool channel buses (SUM / MEAN / MAX / SOFT)
        → distortion (tanh waveshaper, `drive`)
        → peak-normalise → 16-bit PCM (mono, or stereo when panned)

Everything is driven by a ``SynthConfig`` (a config object, not a kwargs pile).
Reuses ``ChebyshevCRActivation`` for the optional band-limited LFO shape.
"""
from __future__ import annotations

import enum
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from signed_kan import ChebyshevCRActivation

from signedkan_wip.examples.cheby_music.beat import CLAP, KICK, LOW_TOM, OPEN_HAT, SNARE
from signedkan_wip.examples.cheby_music.oscillator import Wavetable, chebyshev_bank
from signedkan_wip.examples.cheby_music.sequencer import Note


class Pooling(enum.Enum):
    """How the per-channel buses pool into the master bus.

    ``SUM`` — classic additive mix. ``MEAN`` — headroom-safe average.
    ``MAX`` — per-sample largest-magnitude voice (sharp, sparse). ``SOFT`` —
    ``tanh(Σ)``, a soft-saturating pool that glues voices with mild drive.
    """

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    SOFT = "soft"


@dataclass
class LFO:
    """Low-frequency modulator applied globally over the timeline.

    ``rate_hz`` 0 disables it. ``vibrato`` is pitch depth in semitones;
    ``tremolo`` is amplitude depth in ``[0, 1]``. ``shape`` is ``"sine"`` or
    ``"cheby"`` (one band-limited Chebyshev cycle per period).
    """

    rate_hz: float = 0.0
    vibrato: float = 0.0
    tremolo: float = 0.0
    shape: str = "sine"


@dataclass
class Filter:
    """Resonant state-variable low-pass with a per-note downward cutoff sweep —
    the classic acid/industrial filter. ``cutoff`` 0 disables it.

    Per-note cutoff ``= cutoff + env_amount · exp(-t · env_decay)`` Hz; ``q`` is
    resonance (higher = more peak; ~0.7 clean, 4–12 squelchy).
    """

    cutoff: float = 0.0
    resonance: float = 0.7
    env_amount: float = 0.0
    env_decay: float = 8.0


# Vowel formant frequencies (Hz) + relative gains [F1, F2, F3] — the resonances
# that give a vocal "aaah/eee" character; sweeping the vowel = a talking/scream.
FORMANTS: dict[str, tuple[tuple[float, float], ...]] = {
    "a": ((800, 1.0), (1150, 0.6), (2900, 0.3)),
    "e": ((400, 1.0), (2000, 0.5), (2800, 0.3)),
    "i": ((300, 1.0), (2300, 0.5), (3000, 0.3)),
    "o": ((450, 1.0), (800, 0.6), (2830, 0.2)),
    "u": ((325, 1.0), (700, 0.5), (2530, 0.2)),
}


@dataclass
class Formant:
    """Vowel formant filter — parallel band-passes at the vowel's resonances,
    blended with the dry signal by ``mix``. The vocal/scream effect. ``mix`` 0
    disables it; ``q`` is the resonance sharpness."""

    vowel: str = "a"
    mix: float = 0.0
    q: float = 9.0


@dataclass
class SynthConfig:
    sample_rate: int = 44100
    bpm: int = 110
    ticks_per_step: int = 240
    ticks_per_quarter: int = 480
    lfo: LFO = field(default_factory=LFO)
    pooling: Pooling = Pooling.SUM
    drive: float = 0.0                          # distortion amount, 0 = clean
    noise: float = 0.0                          # per-note white-noise blend [0,1]
    gains: tuple[float, ...] | None = None      # per-channel mix gain
    pans: tuple[float, ...] | None = None       # per-channel pan [-1,1] (→ stereo)
    drum_channel: int | None = None             # this channel renders as percussion
    filter: Filter | None = None                # resonant filter + envelope (pitched voices)
    formant: Formant | None = None              # vowel formant filter (scream/vocal)
    ring_mod: float = 0.0                        # ring-modulation blend [0,1]
    ring_ratio: float = 1.5                      # ring-mod carrier = ratio · note freq
    sidechain: float = 0.0                       # duck pitched bus by the kick [0,1]
    seed: int = 0                               # noise / LFO-shape determinism

    @property
    def step_seconds(self) -> float:
        return (self.ticks_per_step / self.ticks_per_quarter) * (60.0 / self.bpm)


def midi_to_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _envelope(n: int, sr: int) -> np.ndarray:
    """Click-free attack/release ramp envelope over ``n`` samples."""
    env = np.ones(n, dtype=np.float32)
    a = min(int(0.010 * sr), n // 2)
    r = min(int(0.060 * sr), n // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _svf_lowpass(x: np.ndarray, cutoff: np.ndarray, q: float, sr: int) -> np.ndarray:
    """Chamberlin state-variable low-pass with per-sample cutoff (Hz) and
    resonance ``q``. Recursive (per-sample) — used on short per-note buffers."""
    n = len(x)
    fc = np.clip(np.broadcast_to(cutoff, (n,)), 20.0, sr * 0.45)
    f = (2.0 * np.sin(np.pi * fc / sr)).astype(np.float32)
    damp = float(np.clip(1.0 / max(q, 0.5), 0.0, 2.0))
    low = band = 0.0
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        low += f[i] * band
        high = x[i] - low - damp * band
        band += f[i] * high
        out[i] = low
    return out


def _svf_bandpass(x: np.ndarray, fc: float, q: float, sr: int) -> np.ndarray:
    """Chamberlin SVF band-pass output at fixed cutoff ``fc`` — one resonant peak."""
    n = len(x)
    f = float(2.0 * np.sin(np.pi * min(fc, sr * 0.45) / sr))
    damp = float(np.clip(1.0 / max(q, 0.5), 0.0, 2.0))
    low = band = 0.0
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        low += f * band
        high = x[i] - low - damp * band
        band += f * high
        out[i] = band
    return out


def _formant_filter(x: np.ndarray, formant: Formant, sr: int) -> np.ndarray:
    """Blend the dry signal with a sum of band-passes at the vowel's formants."""
    peaks = FORMANTS.get(formant.vowel, FORMANTS["a"])
    voiced = np.zeros_like(x)
    norm = 0.0
    for freq, gain in peaks:
        voiced += gain * _svf_bandpass(x, freq, formant.q, sr)
        norm += gain
    voiced /= max(norm, 1e-6)
    return ((1.0 - formant.mix) * x + formant.mix * voiced).astype(np.float32)


def _sidechain_env(total: int, kick_starts: list[int], depth: float, sr: int) -> np.ndarray:
    """Ducking gain in [1-depth, 1]: dips at each kick, recovers over ~150 ms."""
    env = np.ones(total, dtype=np.float32)
    r = max(1, int(0.15 * sr))
    ramp = (1.0 - depth) + depth * np.linspace(0.0, 1.0, r, dtype=np.float32)
    for k in kick_starts:
        seg = env[k:k + r]
        np.minimum(seg, ramp[:len(seg)], out=seg)
    return env


class Synth:
    """Render a note list to audio through the configured signal chain.

    ``bank`` is the wavetable per pitched channel (the *oscillator timbres*) —
    Chebyshev-CR and/or wavelet tables. Defaults to an 8-channel Chebyshev-CR
    bank if not supplied. Drum-channel notes bypass the bank (rendered as
    percussion).
    """

    def __init__(self, config: SynthConfig | None = None,
                 bank: list[Wavetable] | None = None) -> None:
        self.cfg = config or SynthConfig()
        self.bank = bank if bank is not None else chebyshev_bank(8, seed=self.cfg.seed)

    # -- modulator -------------------------------------------------------- #

    def _lfo(self, total: int) -> np.ndarray:
        """Global LFO signal in [-1, 1] over ``total`` samples (zeros if off)."""
        lfo = self.cfg.lfo
        if lfo.rate_hz <= 0.0 or (lfo.vibrato == 0.0 and lfo.tremolo == 0.0):
            return np.zeros(total, dtype=np.float32)
        t = np.arange(total, dtype=np.float32) / self.cfg.sample_rate
        phase = (lfo.rate_hz * t) % 1.0
        if lfo.shape == "cheby":
            torch.manual_seed(self.cfg.seed)
            cell = ChebyshevCRActivation(1, grid=16, k=7)
            x = torch.as_tensor(2.0 * phase - 1.0, dtype=torch.float32).unsqueeze(-1)
            with torch.no_grad():
                y = cell(x).squeeze(-1).numpy()
            return (y / max(1e-6, np.abs(y).max())).astype(np.float32)
        return np.sin(2.0 * np.pi * phase).astype(np.float32)

    # -- per-note voice --------------------------------------------------- #

    def _render_note(self, n: Note, bus: np.ndarray, lfo: np.ndarray,
                     rng: np.random.Generator) -> None:
        sr = self.cfg.sample_rate
        start = int(n.start * self.cfg.step_seconds * sr)
        n_samp = max(1, int(n.duration * self.cfg.step_seconds * sr))
        seg_lfo = lfo[start:start + n_samp]
        n_use = len(seg_lfo)
        if n_use == 0:
            return
        freq = midi_to_hz(n.pitch)
        # Vibrato via FM: per-sample instantaneous frequency.
        if self.cfg.lfo.vibrato:
            inst_f = freq * 2.0 ** (self.cfg.lfo.vibrato / 12.0 * seg_lfo)
        else:
            inst_f = freq
        # The oscillator IS the channel's wavetable (Chebyshev-CR or wavelet).
        wt = self.bank[n.channel % len(self.bank)]
        wave_buf = wt.render(inst_f, n_use, sr).astype(np.float32)
        t = np.arange(n_use, dtype=np.float32) / sr
        if self.cfg.ring_mod:                              # metallic ring modulation
            carrier = np.sin(2.0 * np.pi * freq * self.cfg.ring_ratio * t)
            wave_buf = ((1.0 - self.cfg.ring_mod) * wave_buf
                        + self.cfg.ring_mod * wave_buf * carrier)
        if self.cfg.filter and self.cfg.filter.cutoff > 0.0:   # resonant filter sweep
            flt = self.cfg.filter
            co = flt.cutoff + flt.env_amount * np.exp(-t * flt.env_decay)
            wave_buf = _svf_lowpass(wave_buf, co, flt.resonance, sr)
        if self.cfg.formant and self.cfg.formant.mix > 0.0:    # vowel formant (scream)
            wave_buf = _formant_filter(wave_buf, self.cfg.formant, sr)
        if self.cfg.noise:
            wave_buf = ((1.0 - self.cfg.noise) * wave_buf
                        + self.cfg.noise * rng.standard_normal(n_use).astype(np.float32))
        env = _envelope(n_use, sr)
        if self.cfg.lfo.tremolo:
            env = env * (1.0 - self.cfg.lfo.tremolo * (0.5 - 0.5 * seg_lfo))
        bus[start:start + n_use] += wave_buf * env * (n.velocity / 127.0)

    def _render_drum(self, n: Note, bus: np.ndarray,
                     rng: np.random.Generator) -> None:
        """Synthesize a percussion hit by drum note: kick (pitch-dropping sine),
        snare (tone + noise burst), hat (short filtered noise)."""
        sr = self.cfg.sample_rate
        start = int(n.start * self.cfg.step_seconds * sr)
        vel = n.velocity / 127.0
        if n.pitch == KICK:
            t = np.arange(int(0.18 * sr), dtype=np.float32) / sr
            f = 110.0 * np.exp(-t * 30.0) + 40.0          # pitch drop
            sig = np.sin(2.0 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 16.0)
        elif n.pitch in (SNARE, CLAP):
            decay = 22.0 if n.pitch == SNARE else 30.0
            t = np.arange(int(0.16 * sr), dtype=np.float32) / sr
            tone = 0.5 * np.sin(2.0 * np.pi * 180.0 * t) if n.pitch == SNARE else 0.0
            noise = rng.standard_normal(len(t)).astype(np.float32)
            sig = (tone + noise) * np.exp(-t * decay)
        elif n.pitch == LOW_TOM:
            t = np.arange(int(0.20 * sr), dtype=np.float32) / sr
            f = 160.0 * np.exp(-t * 8.0) + 90.0
            sig = np.sin(2.0 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 12.0)
        else:  # HAT_CLOSED / OPEN_HAT (filtered noise; open = longer)
            decay = 35.0 if n.pitch == OPEN_HAT else 80.0
            dur = 0.12 if n.pitch == OPEN_HAT else 0.05
            t = np.arange(int(dur * sr), dtype=np.float32) / sr
            noise = np.diff(rng.standard_normal(len(t) + 1).astype(np.float32))
            sig = noise * np.exp(-t * decay)
        sig = (sig * vel).astype(np.float32)
        seg = bus[start:start + len(sig)]
        seg += sig[:len(seg)]

    # -- pooling / distortion / normalise --------------------------------- #

    def _pool(self, buses: list[np.ndarray]) -> np.ndarray:
        stack = np.stack(buses, axis=0)
        mode = self.cfg.pooling
        if mode is Pooling.SUM:
            return stack.sum(axis=0)
        if mode is Pooling.MEAN:
            return stack.mean(axis=0)
        if mode is Pooling.SOFT:
            return np.tanh(stack.sum(axis=0)).astype(np.float32)
        idx = np.argmax(np.abs(stack), axis=0)                 # MAX (by magnitude)
        return np.take_along_axis(stack, idx[None], axis=0)[0]

    def _distort(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.drive <= 0.0:
            return x
        g = 1.0 + 6.0 * self.cfg.drive
        return (np.tanh(g * x) / np.tanh(g)).astype(np.float32)

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        peak = float(np.abs(x).max())
        return (x / peak * 0.89) if peak > 0 else x

    # -- top level -------------------------------------------------------- #

    def render(self, notes: list[Note], n_channels: int) -> np.ndarray:
        """Render to a float32 array: ``(N,)`` mono or ``(N, 2)`` stereo (when
        ``pans`` is set). Preconditions: ``notes`` non-empty."""
        if not notes:
            raise ValueError("no notes to render")
        sr = self.cfg.sample_rate
        end_step = max(n.start + n.duration for n in notes)
        total = int((end_step * self.cfg.step_seconds + 0.2) * sr)
        lfo = self._lfo(total)
        rng = np.random.default_rng(self.cfg.seed)
        buses = [np.zeros(total, dtype=np.float32) for _ in range(n_channels)]
        for n in notes:
            if n.channel == self.cfg.drum_channel:
                self._render_drum(n, buses[n.channel], rng)
            else:
                self._render_note(n, buses[n.channel], lfo, rng)

        gains = self.cfg.gains or (1.0,) * n_channels
        buses = [b * float(gains[c % len(gains)]) for c, b in enumerate(buses)]

        # Sidechain: duck the pitched (non-drum) buses by the kick envelope.
        if self.cfg.sidechain > 0.0 and self.cfg.drum_channel is not None:
            kicks = [int(n.start * self.cfg.step_seconds * sr)
                     for n in notes
                     if n.channel == self.cfg.drum_channel and n.pitch == KICK]
            if kicks:
                env = _sidechain_env(total, kicks, self.cfg.sidechain, sr)
                for c in range(n_channels):
                    if c != self.cfg.drum_channel:
                        buses[c] = buses[c] * env

        if self.cfg.pans is not None:
            # Equal-power pan each voice, then pool per side.
            pans = self.cfg.pans
            ang = [(float(pans[c % len(pans)]) + 1.0) * 0.25 * np.pi
                   for c in range(n_channels)]
            left = self._pool([b * float(np.cos(a)) for b, a in zip(buses, ang)])
            right = self._pool([b * float(np.sin(a)) for b, a in zip(buses, ang)])
            stereo = np.stack([self._distort(left), self._distort(right)], axis=-1)
            return self._normalise(stereo)

        return self._normalise(self._distort(self._pool(buses)))

    def write_wav(self, notes: list[Note], n_channels: int, path: str | Path) -> Path:
        audio = self.render(notes, n_channels)
        pcm = (audio * 32767.0).astype("<i2")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dest), "wb") as w:
            w.setnchannels(2 if audio.ndim == 2 else 1)
            w.setsampwidth(2)
            w.setframerate(self.cfg.sample_rate)
            w.writeframes(pcm.tobytes())
        return dest


def render_wav(notes: list[Note], path: str | Path, *, n_channels: int | None = None,
               config: SynthConfig | None = None, bpm: int = 110,
               bank: list[Wavetable] | None = None) -> Path:
    """Convenience wrapper: render ``notes`` to a WAV at ``path``.

    ``n_channels`` defaults to ``max(channel)+1``. ``config`` overrides all synth
    settings; ``bank`` overrides the wavetable timbres (Chebyshev-CR / wavelet).
    """
    if not notes:
        raise ValueError("no notes to render")
    if n_channels is None:
        n_channels = max(n.channel for n in notes) + 1
    cfg = config or SynthConfig(bpm=bpm)
    return Synth(cfg, bank=bank).write_wav(notes, n_channels, path)
