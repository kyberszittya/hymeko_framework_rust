"""Piano-roll visualisation: the Chebyshev channel curves becoming notes."""
from __future__ import annotations

from pathlib import Path

from hymeko_neuro.examples.cheby_music.sequencer import ChebyshevSequencer, Note


def plot_piano_roll(seq: ChebyshevSequencer, notes: list[Note], path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = seq.contours()
    fig, (ax_c, ax_r) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[1, 1.4])
    cmap = plt.get_cmap("tab10")
    for c in range(seq.n_channels):
        ax_c.plot(range(seq.n_steps), y[:, c].tolist(), color=cmap(c), lw=1.5,
                  label=f"ch {c}")
    ax_c.set_title("CR-Chebyshev channel curves (each cell = one voice)")
    ax_c.set_ylabel("normalised value")
    ax_c.legend(ncol=seq.n_channels, fontsize=7, loc="upper right")
    ax_c.grid(alpha=0.2)

    for n in notes:
        ax_r.barh(n.pitch, n.duration, left=n.start, height=0.8,
                  color=cmap(n.channel % 10), edgecolor="white", linewidth=0.3)
    ax_r.set_title(f"Piano roll · {seq.scale.name} · {len(notes)} notes "
                   "(low pitches = drum channel)")
    ax_r.set_xlabel("step")
    ax_r.set_ylabel("MIDI pitch")
    ax_r.grid(alpha=0.2)
    fig.tight_layout()
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
