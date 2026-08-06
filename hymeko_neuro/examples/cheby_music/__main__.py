"""CLI: a Chebyshev-CR / wavelet / Kochanek-Bartels wavetable synth + beats.

Voices (bass + arp) play through wavetables whose waveform is a Chebyshev-CR
curve — optionally morphed toward a wavelet, shaped by Kochanek-Bartels TCB
knobs, or *trained* to a target timbre. A named industrial/electronic drum
pattern backs them, through a resonant filter, ring-mod, distortion, pooling and
kick sidechain.

    python -m hymeko_neuro.examples.cheby_music --wav track.wav --out track.mid \
        --beat techno --bars 4 --bpm 130 --root 33 --bass driving --arp \
        --train-bass saw --train-lead vowel_a --kb-tension 0.4 --kb-bias -0.2 \
        --filter-cutoff 400 --filter-env 3500 --filter-res 6 \
        --ring-mod 0.25 --sidechain 0.6 --drive 0.5 --pool soft --seed 3 \
        --wavetable-plot tables.png
"""
from __future__ import annotations

import argparse

from hymeko_neuro.examples.cheby_music.midi import write_midi
from hymeko_neuro.examples.cheby_music.oscillator import (
    chebyshev_bank,
    kb_bank,
    morph,
    wavelet_wavetable,
)
from hymeko_neuro.examples.cheby_music.patterns import (
    DRUM_PATTERNS,
    arp,
    bassline,
    drum_pattern,
)
from hymeko_neuro.examples.cheby_music.scale import SCALES
from hymeko_neuro.examples.cheby_music.synth import (
    FORMANTS,
    LFO,
    Filter,
    Formant,
    Pooling,
    Synth,
    SynthConfig,
)
from hymeko_neuro.examples.cheby_music.train_wavetable import TARGETS, trained_wavetable


def _build_bank(a: argparse.Namespace):
    """Two wavetables (bass, lead). Priority: trained targets > KB-shaped >
    Chebyshev-CR (optionally morphed toward a wavelet)."""
    if a.train_bass or a.train_lead:
        return [
            trained_wavetable(a.train_bass or "saw", seed=a.seed),
            trained_wavetable(a.train_lead or "organ", seed=a.seed),
        ]
    if a.kb_tension or a.kb_continuity or a.kb_bias:
        return kb_bank(2, tension=a.kb_tension, continuity=a.kb_continuity,
                       bias=a.kb_bias, degree=a.degree, seed=a.seed)
    bank = chebyshev_bank(2, degree=a.degree, seed=a.seed)
    if a.wavelet:
        wl = wavelet_wavetable(a.wavelet)
        bank = [morph(w, wl, a.morph) for w in bank]
    return bank


def _add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--wav", default=None)
    ap.add_argument("--out", default=None, help="MIDI path")
    ap.add_argument("--wavetable-plot", default=None, help="PNG of the wavetables")
    ap.add_argument("--beat", default="techno", choices=sorted(DRUM_PATTERNS))
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--bpm", type=int, default=128)
    ap.add_argument("--root", type=int, default=33, help="bass root MIDI note")
    ap.add_argument("--scale", default="minor_pentatonic", choices=sorted(SCALES))
    ap.add_argument("--bass", default="rolling", choices=("rolling", "offbeat", "driving"))
    ap.add_argument("--arp", action="store_true", help="add an arpeggio voice")
    ap.add_argument("--degree", type=int, default=8, help="Chebyshev order (timbre)")
    # waveform shaping
    ap.add_argument("--wavelet", default=None, choices=("ricker", "morlet"))
    ap.add_argument("--morph", type=float, default=0.4, help="Cheby→wavelet morph [0,1]")
    ap.add_argument("--kb-tension", type=float, default=0.0, help="KB tension (-1,1)")
    ap.add_argument("--kb-continuity", type=float, default=0.0, help="KB continuity (-1,1)")
    ap.add_argument("--kb-bias", type=float, default=0.0, help="KB bias (-1,1)")
    ap.add_argument("--train-bass", default=None, choices=sorted(TARGETS),
                    help="train the bass wavetable to a target timbre")
    ap.add_argument("--train-lead", default=None, choices=sorted(TARGETS))
    # signal chain
    ap.add_argument("--pool", default="soft", choices=[p.value for p in Pooling])
    ap.add_argument("--drive", type=float, default=0.4)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--filter-cutoff", type=float, default=0.0, help="base cutoff Hz; 0=off")
    ap.add_argument("--filter-res", type=float, default=4.0, help="resonance Q")
    ap.add_argument("--filter-env", type=float, default=0.0, help="cutoff env amount Hz")
    ap.add_argument("--filter-decay", type=float, default=8.0)
    ap.add_argument("--ring-mod", type=float, default=0.0)
    ap.add_argument("--ring-ratio", type=float, default=1.5)
    ap.add_argument("--sidechain", type=float, default=0.0, help="duck by kick [0,1]")
    ap.add_argument("--formant", default=None, choices=sorted(FORMANTS),
                    help="vowel formant filter (scream/vocal)")
    ap.add_argument("--formant-mix", type=float, default=0.6)
    ap.add_argument("--lfo-rate", type=float, default=0.0)
    ap.add_argument("--vibrato", type=float, default=0.0)
    ap.add_argument("--tremolo", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_args(ap)
    a = ap.parse_args(argv)
    if not (a.wav or a.out or a.wavetable_plot):
        ap.error("nothing to do: pass --wav, --out, or --wavetable-plot")

    BASS, LEAD, DRUMS = 0, 1, 2
    notes = bassline(a.root, a.scale, a.bars, BASS, riff=a.bass)
    if a.arp:
        notes += arp(a.root, a.scale, a.bars, LEAD)
    notes += drum_pattern(a.beat, a.bars, DRUMS)
    n_channels = 3
    tps = 480 // 4                                  # 16th-note grid
    bank = _build_bank(a)

    if a.wavetable_plot:
        _plot_wavetables(bank, a, a.wavetable_plot)
        print(f"Wrote {a.wavetable_plot}")
    if a.out:
        out = write_midi(notes, n_channels, a.out, bpm=a.bpm,
                         ticks_per_step=tps, drum_channel=DRUMS)
        print(f"Wrote {out}  ({len(notes)} notes, beat={a.beat})")
    if a.wav:
        flt = (Filter(cutoff=a.filter_cutoff, resonance=a.filter_res,
                      env_amount=a.filter_env, env_decay=a.filter_decay)
               if a.filter_cutoff > 0 else None)
        cfg = SynthConfig(
            bpm=a.bpm, ticks_per_step=tps, seed=a.seed, drum_channel=DRUMS,
            pooling=Pooling(a.pool), drive=a.drive, noise=a.noise,
            filter=flt, ring_mod=a.ring_mod, ring_ratio=a.ring_ratio,
            sidechain=a.sidechain,
            formant=(Formant(vowel=a.formant, mix=a.formant_mix) if a.formant else None),
            lfo=LFO(rate_hz=a.lfo_rate, vibrato=a.vibrato, tremolo=a.tremolo),
        )
        out = Synth(cfg, bank=bank).write_wav(notes, n_channels, a.wav)
        print(f"Wrote {out}  (beat={a.beat}, pool={a.pool}, drive={a.drive}, "
              f"filter={a.filter_cutoff}, sidechain={a.sidechain})")
    return 0


def _plot_wavetables(bank, a, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for wt in bank:
        ax.plot(wt.table, lw=1.4, label=wt.name)
    bits = []
    if a.train_bass or a.train_lead:
        bits.append(f"trained({a.train_bass or '-'},{a.train_lead or '-'})")
    if a.kb_tension or a.kb_continuity or a.kb_bias:
        bits.append(f"KB(t={a.kb_tension},c={a.kb_continuity},b={a.kb_bias})")
    if a.wavelet:
        bits.append(f"morph→{a.wavelet}")
    ax.set_title("Synth wavetables (one cycle)"
                 + (" — " + ", ".join(bits) if bits else ""))
    ax.set_xlabel("table index (one period)")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
