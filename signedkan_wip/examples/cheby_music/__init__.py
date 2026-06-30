"""CR-Chebyshev music: each HSiKAN cell channel is a MIDI/audio voice.

Idea (user, 2026-06-29). A ``ChebyshevCRActivation`` holds one learnable,
band-limited curve *per channel*; sampling every channel along a shared time
axis ``t ∈ [-1, 1]`` turns each into a melodic contour — the cell *is* a
multi-track sequencer where channel = voice. Quantise to a scale → notes →
Standard MIDI File, or render through a small synth (LFO, mixing, output
pooling, distortion, noise) + a background drum beat to a WAV.

Public API re-exported here; submodules: ``scale``, ``sequencer``, ``beat``,
``midi``, ``synth``, ``viz``.
"""
from signedkan_wip.examples.cheby_music.beat import (
    DRUM_PITCHES,
    HAT_CLOSED,
    KICK,
    SNARE,
    make_beat,
)
from signedkan_wip.examples.cheby_music.midi import notes_to_midi, write_midi
from signedkan_wip.examples.cheby_music.oscillator import (
    Wavetable,
    chebyshev_bank,
    chebyshev_wavetable,
    kb_bank,
    kochanek_bartels_wavetable,
    morph,
    wavelet_wavetable,
)
from signedkan_wip.examples.cheby_music.patterns import (
    BAR,
    DRUM_PATTERNS,
    arp,
    bassline,
    drum_pattern,
)
from signedkan_wip.examples.cheby_music.scale import SCALES, Scale
from signedkan_wip.examples.cheby_music.sequencer import ChebyshevSequencer, Note
from signedkan_wip.examples.cheby_music.synth import (
    FORMANTS,
    LFO,
    Filter,
    Formant,
    Pooling,
    Synth,
    SynthConfig,
    midi_to_hz,
    render_wav,
)
from signedkan_wip.examples.cheby_music.train_wavetable import (
    TARGETS,
    fit_chebyshev,
    fit_quality,
    target_array,
    trained_wavetable,
)
from signedkan_wip.examples.cheby_music.viz import plot_piano_roll

__all__ = [
    "Scale", "SCALES", "Note", "ChebyshevSequencer",
    "make_beat", "KICK", "SNARE", "HAT_CLOSED", "DRUM_PITCHES",
    "Wavetable", "chebyshev_wavetable", "wavelet_wavetable", "morph", "chebyshev_bank",
    "kochanek_bartels_wavetable", "kb_bank",
    "DRUM_PATTERNS", "drum_pattern", "bassline", "arp", "BAR",
    "notes_to_midi", "write_midi",
    "Synth", "SynthConfig", "LFO", "Filter", "Formant", "FORMANTS", "Pooling",
    "render_wav", "midi_to_hz",
    "TARGETS", "target_array", "fit_chebyshev", "trained_wavetable", "fit_quality",
    "plot_piano_roll",
]
