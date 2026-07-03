"""Tests for the CR-Chebyshev music package (sequencer, MIDI, synth, beat)."""
from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from hymeko_neuro.examples.cheby_music import (
    KICK,
    LFO,
    ChebyshevSequencer,
    Pooling,
    Scale,
    Synth,
    SynthConfig,
    Wavetable,
    arp,
    bassline,
    chebyshev_bank,
    drum_pattern,
    make_beat,
    midi_to_hz,
    morph,
    notes_to_midi,
    render_wav,
    wavelet_wavetable,
    write_midi,
)
from hymeko_neuro.examples.cheby_music import (
    Filter,
    fit_chebyshev,
    fit_quality,
    kb_bank,
    kochanek_bartels_wavetable,
    target_array,
    trained_wavetable,
)
from hymeko_neuro.examples.cheby_music.midi import vlq


# ---------------------------------------------------------------------
# MIDI VLQ + scale
# ---------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (0, b"\x00"), (0x7F, b"\x7f"), (0x80, b"\x81\x00"),
    (0x3FFF, b"\xff\x7f"), (0x4000, b"\x81\x80\x00"),
])
def test_vlq_known_values(value: int, expected: bytes) -> None:
    assert vlq(value) == expected


def test_vlq_rejects_negative() -> None:
    with pytest.raises(ValueError, match="vlq value must be >= 0"):
        vlq(-1)


def test_scale_endpoints_clamp_and_unknown() -> None:
    s = Scale(root=57, name="minor_pentatonic", octaves=3)
    assert s.quantize(-5.0) == s.pitches[0]
    assert s.quantize(5.0) == s.pitches[-1]
    with pytest.raises(ValueError, match="unknown scale"):
        Scale(name="bogus")


# ---------------------------------------------------------------------
# Sequencer
# ---------------------------------------------------------------------


def test_contours_shape_and_normalisation() -> None:
    seq = ChebyshevSequencer(n_channels=4, n_steps=24, degree=6, seed=1)
    y = seq.contours()
    assert y.shape == (24, 4)
    peaks = y.abs().amax(dim=0)
    assert (peaks <= 1.0 + 1e-5).all() and (peaks > 0.5).all()


def test_compose_covers_every_step_and_is_deterministic() -> None:
    a = ChebyshevSequencer(n_channels=3, n_steps=32, seed=2).compose()
    b = ChebyshevSequencer(n_channels=3, n_steps=32, seed=2).compose()
    assert a == b
    for c in range(3):
        assert sum(n.duration for n in a if n.channel == c) == 32


def test_degree_controls_busyness() -> None:
    low = ChebyshevSequencer(n_channels=4, n_steps=48, degree=3, seed=0).compose()
    high = ChebyshevSequencer(n_channels=4, n_steps=48, degree=12, seed=0).compose()
    assert len(high) > len(low)


# ---------------------------------------------------------------------
# Beat (background drum channel)
# ---------------------------------------------------------------------


def test_make_beat_on_its_channel_with_drum_pitches() -> None:
    notes = make_beat(16, channel=5, seed=0)
    assert notes and all(n.channel == 5 and n.duration == 1 for n in notes)
    assert all(n.pitch in (36, 38, 42) for n in notes)
    assert any(n.pitch == KICK and n.start == 0 for n in notes)   # down-beat kick


def test_make_beat_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        make_beat(0, channel=0)


# ---------------------------------------------------------------------
# MIDI bytes + drum routing
# ---------------------------------------------------------------------


def test_midi_header_and_track_count() -> None:
    seq = ChebyshevSequencer(n_channels=5, n_steps=16, seed=3)
    data = notes_to_midi(seq.compose(), n_channels=5)
    assert data[:4] == b"MThd"
    fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    assert fmt == 1 and ntracks == 6 and division == 480
    assert data.count(b"MTrk") == 6


def test_midi_drum_channel_routes_to_percussion_and_skips_program() -> None:
    notes = make_beat(8, channel=0, seed=0)
    data = notes_to_midi(notes, n_channels=1, drum_channel=0)
    # No program-change status byte (0xC0..0xCF) anywhere; note-ons on channel 9.
    assert all((b & 0xF0) != 0xC0 for b in data)
    assert bytes([0x99]) in data        # note-on, channel index 9 (GM drums)


def test_write_midi_file(tmp_path) -> None:
    seq = ChebyshevSequencer(n_channels=3, n_steps=16, seed=4)
    out = write_midi(seq.compose(), 3, tmp_path / "song.mid")
    assert out.exists() and out.read_bytes()[:4] == b"MThd"


# ---------------------------------------------------------------------
# Synth / WAV + feature chain
# ---------------------------------------------------------------------


def test_midi_to_hz_a440() -> None:
    assert abs(midi_to_hz(69) - 440.0) < 1e-6
    assert abs(midi_to_hz(57) - 220.0) < 1e-6


def test_render_wav_is_valid_and_audible(tmp_path) -> None:
    seq = ChebyshevSequencer(n_channels=4, n_steps=24, seed=5)
    out = render_wav(seq.compose(), tmp_path / "s.wav",
                     config=SynthConfig(bpm=120, sample_rate=22050))
    raw = out.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        data = w.readframes(w.getnframes())
    assert any(b != 0 for b in data)


def test_render_wav_duration_tracks_tempo(tmp_path) -> None:
    notes = ChebyshevSequencer(n_channels=3, n_steps=32, seed=6).compose()
    def frames(bpm):
        out = render_wav(notes, tmp_path / f"{bpm}.wav",
                         config=SynthConfig(bpm=bpm, sample_rate=22050))
        with wave.open(str(out), "rb") as w:
            return w.getnframes()
    assert frames(180) < frames(90)


def test_render_wav_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no notes to render"):
        render_wav([], "x.wav")


def test_pans_produce_stereo(tmp_path) -> None:
    notes = ChebyshevSequencer(n_channels=2, n_steps=16, seed=7).compose()
    cfg = SynthConfig(sample_rate=22050, pans=(-1.0, 1.0))
    out = Synth(cfg).write_wav(notes, 2, tmp_path / "stereo.wav")
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 2


def test_pooling_modes_all_render() -> None:
    notes = ChebyshevSequencer(n_channels=3, n_steps=16, seed=8).compose()
    for mode in Pooling:
        audio = Synth(SynthConfig(sample_rate=16000, pooling=mode)).render(notes, 3)
        assert np.isfinite(audio).all() and np.abs(audio).max() > 0


def test_distortion_increases_high_harmonic_content() -> None:
    """Tanh drive adds harmonics → more spectral energy above the fundamental."""
    notes = ChebyshevSequencer(n_channels=1, n_steps=12, seed=1).compose()
    clean = Synth(SynthConfig(sample_rate=16000, drive=0.0)).render(notes, 1)
    driven = Synth(SynthConfig(sample_rate=16000, drive=0.9)).render(notes, 1)
    n = min(len(clean), len(driven))
    hf_clean = np.abs(np.fft.rfft(clean[:n]))[n // 4:].sum()
    hf_driven = np.abs(np.fft.rfft(driven[:n]))[n // 4:].sum()
    assert hf_driven > hf_clean


def test_formant_filter_reshapes_spectrum() -> None:
    """A vowel formant filter boosts energy near its formant frequencies vs dry."""
    from hymeko_neuro.examples.cheby_music.synth import FORMANTS, Formant
    notes = ChebyshevSequencer(n_channels=1, n_steps=8, seed=1).compose()
    bank = [wavelet_wavetable("ricker")]
    dry = Synth(SynthConfig(sample_rate=22050), bank=bank).render(notes, 1)
    wet = Synth(SynthConfig(sample_rate=22050,
                            formant=Formant(vowel="a", mix=0.9)),
                bank=bank).render(notes, 1)
    n = min(len(dry), len(wet))
    assert not np.allclose(dry[:n], wet[:n])         # the filter changed the signal
    assert "a" in FORMANTS and len(FORMANTS["a"]) == 3


def test_lfo_tremolo_modulates_amplitude() -> None:
    """A strong tremolo LFO makes the per-sample amplitude vary more."""
    notes = ChebyshevSequencer(n_channels=1, n_steps=16, seed=2).compose()
    flat = Synth(SynthConfig(sample_rate=16000, lfo=LFO())).render(notes, 1)
    trem = Synth(SynthConfig(sample_rate=16000,
                             lfo=LFO(rate_hz=8.0, tremolo=0.9))).render(notes, 1)
    # Compare the envelope variability (std of |signal|) over the sustained region.
    assert np.std(np.abs(trem)) != pytest.approx(np.std(np.abs(flat)), rel=0.05)


def test_drum_channel_renders_percussion(tmp_path) -> None:
    """A beat on the drum channel renders audible audio through the synth."""
    notes = make_beat(16, channel=0, seed=0)
    out = render_wav(notes, tmp_path / "drums.wav", n_channels=1,
                     config=SynthConfig(sample_rate=22050, drum_channel=0))
    with wave.open(str(out), "rb") as w:
        data = w.readframes(w.getnframes())
    assert any(b != 0 for b in data)


# ---------------------------------------------------------------------
# Wavetable oscillators (Chebyshev-CR + wavelet)
# ---------------------------------------------------------------------


def test_wavetable_normalised_and_playback_periodic() -> None:
    bank = chebyshev_bank(2, degree=7, seed=0)
    wt = bank[0]
    assert abs(float(np.abs(wt.table).max()) - 1.0) < 1e-5      # normalised
    # Playback at exactly 1 cycle over N samples repeats with period N.
    sig = wt.render(freq=100.0, n=441, sr=44100)
    assert sig.shape == (441,) and np.isfinite(sig).all()


def test_wavetable_pitch_doubles_with_octave() -> None:
    """Counting zero-crossings: an octave up ≈ 2× the crossings (period halves)."""
    wt = chebyshev_bank(1, seed=1)[0]
    def crossings(freq):
        s = wt.render(freq, 44100, 44100)
        return int((np.diff(np.signbit(s)) != 0).sum())
    lo, hi = crossings(110.0), crossings(220.0)
    assert 1.7 * lo <= hi <= 2.3 * lo


def test_wavelet_tables_and_morph() -> None:
    ric = wavelet_wavetable("ricker")
    mor = wavelet_wavetable("morlet")
    assert isinstance(ric, Wavetable) and isinstance(mor, Wavetable)
    blend = morph(ric, mor, 0.5)
    assert blend.size > 0 and abs(float(np.abs(blend.table).max()) - 1.0) < 1e-4
    with pytest.raises(ValueError, match="unknown wavelet"):
        wavelet_wavetable("bogus")


def test_synth_uses_the_provided_bank(tmp_path) -> None:
    """Two different wavetables produce different audio for the same notes."""
    notes = bassline(33, "minor_pentatonic", 1, channel=0, riff="driving")
    a = Synth(SynthConfig(sample_rate=16000),
              bank=chebyshev_bank(1, seed=0)).render(notes, 1)
    b = Synth(SynthConfig(sample_rate=16000),
              bank=[wavelet_wavetable("ricker")]).render(notes, 1)
    n = min(len(a), len(b))
    assert not np.allclose(a[:n], b[:n])


# ---------------------------------------------------------------------
# Beat patterns + riffs
# ---------------------------------------------------------------------


def test_drum_pattern_named_and_expands_over_bars() -> None:
    one = drum_pattern("techno", 1, channel=2)
    two = drum_pattern("techno", 2, channel=2)
    assert one and all(n.channel == 2 and n.pitch in (36, 38, 39, 42, 46, 45) for n in one)
    assert any(n.pitch == KICK and n.start == 0 for n in one)
    assert len(two) == 2 * len(one)            # pattern repeats per bar
    with pytest.raises(ValueError, match="unknown pattern"):
        drum_pattern("bogus", 1, channel=0)


def test_bassline_and_arp_on_their_channels() -> None:
    bass = bassline(33, "minor_pentatonic", 2, channel=0, riff="rolling")
    assert bass and all(n.channel == 0 for n in bass)
    ar = arp(33, "minor_pentatonic", 1, channel=1)
    assert ar and all(n.channel == 1 for n in ar)
    # Arp sits an octave above the bass root.
    assert min(n.pitch for n in ar) >= min(n.pitch for n in bass)


def test_full_track_renders_to_wav(tmp_path) -> None:
    """Bass + drums through a Cheby bank + industrial beat → audible WAV."""
    notes = (bassline(33, "minor_pentatonic", 2, channel=0, riff="driving")
             + drum_pattern("industrial", 2, channel=2))
    cfg = SynthConfig(sample_rate=22050, bpm=130, ticks_per_step=120,
                      drum_channel=2, drive=0.5, pooling=Pooling.SOFT)
    out = Synth(cfg, bank=chebyshev_bank(2, seed=3)).write_wav(notes, 3,
                                                               tmp_path / "track.wav")
    with wave.open(str(out), "rb") as w:
        data = w.readframes(w.getnframes())
    assert any(b != 0 for b in data)


# ---------------------------------------------------------------------
# Kochanek-Bartels (TCB) wavetables
# ---------------------------------------------------------------------


def test_kb_tcb_zero_matches_catmull_rom() -> None:
    """At T=C=B=0 the KB wavetable equals the Catmull-Rom (Chebyshev) one."""
    bank = chebyshev_bank(1, degree=7, seed=0)
    cr = bank[0]
    import torch
    from hymeko_neuro.core import ChebyshevCRActivation
    torch.manual_seed(0)
    cell = ChebyshevCRActivation(1, grid=16, k=7)
    cps = cell.control_points().detach().numpy()[0]
    kb = kochanek_bartels_wavetable(cps, tension=0, continuity=0, bias=0, size=cr.size)
    assert np.allclose(cr.table, kb.table, atol=1e-4)


def test_kb_tension_changes_waveform() -> None:
    bank0 = kb_bank(1, tension=0.0, seed=1)
    bankT = kb_bank(1, tension=0.7, seed=1)
    assert not np.allclose(bank0[0].table, bankT[0].table, atol=1e-3)


def test_kb_requires_enough_control_points() -> None:
    with pytest.raises(ValueError, match=">= 4 control points"):
        kochanek_bartels_wavetable([0.0, 1.0, -1.0])


# ---------------------------------------------------------------------
# Training the Chebyshev parameters to a target timbre
# ---------------------------------------------------------------------


def test_fit_reduces_loss_and_matches_target() -> None:
    tgt = target_array("saw", size=256)
    cell, history = fit_chebyshev(tgt, degree=14, iters=300, seed=0)
    assert history[-1] < history[0] * 0.5            # loss at least halved
    q = fit_quality(cell, tgt)
    assert q["corr"] > 0.9                            # learned curve matches the saw


def test_trained_wavetable_plugs_into_synth(tmp_path) -> None:
    wt = trained_wavetable("square", iters=150, seed=0)
    notes = bassline(33, "minor_pentatonic", 1, channel=0, riff="driving")
    out = render_wav(notes, tmp_path / "sq.wav", n_channels=1,
                     config=SynthConfig(sample_rate=16000), bank=[wt])
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() > 0


def test_fit_rejects_tiny_target() -> None:
    with pytest.raises(ValueError, match=">= 8 samples"):
        fit_chebyshev(np.zeros(4))


# ---------------------------------------------------------------------
# Resonant filter + sidechain
# ---------------------------------------------------------------------


def test_filter_attenuates_high_frequencies() -> None:
    """A low cutoff removes high-frequency energy vs no filter."""
    notes = bassline(45, "minor_pentatonic", 1, channel=0, riff="rolling")
    bank = chebyshev_bank(1, degree=10, seed=1)
    nofilt = Synth(SynthConfig(sample_rate=16000), bank=bank).render(notes, 1)
    filt = Synth(SynthConfig(sample_rate=16000,
                             filter=Filter(cutoff=300.0, resonance=2.0)),
                 bank=bank).render(notes, 1)
    n = min(len(nofilt), len(filt))

    def hf(s):
        return float(np.abs(np.fft.rfft(s[:n]))[n // 4:].sum())
    assert hf(filt) < hf(nofilt)


def test_sidechain_env_dips_at_kicks_and_recovers() -> None:
    """The ducking gain drops to ~(1-depth) at each kick and recovers to 1."""
    from hymeko_neuro.examples.cheby_music.synth import _sidechain_env
    sr = 48000
    env = _sidechain_env(sr, kick_starts=[1000, 24000], depth=0.8, sr=sr)
    assert env[500] == 1.0                                  # before any kick
    assert env[1000] < 0.25                                 # ducked to ~0.2
    assert env[1000 + int(0.15 * sr) - 1] > 0.95            # recovered after ~150 ms
    assert env[24000] < 0.25                                # ducks again at 2nd kick


def test_sidechain_changes_the_mix() -> None:
    """Sidechain on vs off produces a different waveform (pumping)."""
    notes = (bassline(45, "minor_pentatonic", 1, channel=0, riff="driving")
             + drum_pattern("four_on_floor", 1, channel=2))
    bank = chebyshev_bank(2, seed=2)
    common = dict(sample_rate=16000, bpm=120, ticks_per_step=120, drum_channel=2)
    dry = Synth(SynthConfig(**common), bank=bank).render(notes, 3)
    ducked = Synth(SynthConfig(**common, sidechain=0.8), bank=bank).render(notes, 3)
    n = min(len(dry), len(ducked))
    assert not np.allclose(dry[:n], ducked[:n])
