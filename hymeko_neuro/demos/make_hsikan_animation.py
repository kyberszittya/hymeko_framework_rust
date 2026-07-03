"""Generate an explanatory GIF of the HSiKAN forward pass.

Six stages, revealed left-to-right, of how HSiKAN turns a signed graph's
*structure* into a sign prediction:

    1. signed graph            (± relations)
    2. query edge u–v          (whose sign we predict)
    3. FIR                     (enumerate cycles/walks through the edge, per arity)
    4. HSiKAN-CR middle shell  (SignedKAN Catmull-Rom spline embedding per tuple)
    5. αₖ regime mix           (softmax over arities — the learned structural prior)
    6. CPML                    (combine → p(sign=+1) → predicted sign)

The αₖ vector is the **real learned regime** when ``--checkpoint`` is given
(default: the Bitcoin-OTC optuna-best), else a clearly-labelled illustrative
regime so the GIF renders offline. Schematic by design: the spline and graph
glyphs illustrate the mechanism; only αₖ carries a real number.

Dependency-free GIF assembly via Pillow (no imageio). Deterministic (fixed seed).

Run:
    PYTHONPATH=. .venv/Scripts/python.exe -m hymeko_neuro.demos.make_hsikan_animation
Output:
    demo_out/hsikan_anim/frame_*.png, hsikan.gif
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")  # headless: never touch a GUI/Tcl backend.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "architecture" / "hsikan"
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "hsikan" / "bitcoin_otc_optuna_best.pt"


def _force_utf8_console() -> None:
    """αₖ in prints must not crash a cp125x Windows console."""
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc is not None:
            rc(encoding="utf-8", errors="backslashreplace")

GREEN, RED, BLUE, PURPLE, INK, MUTE = "#1D9E75", "#D9534F", "#4472C4", "#7F77DD", "#222", "#bbb"
STAGE_TITLES = [
    "1 · Signed graph",
    "2 · Query edge",
    "3 · FIR: cycles / walks per arity",
    "4 · HSiKAN-CR shell (spline embed)",
    "5 · αₖ regime mix",
    "6 · CPML → sign",
]
STAGE_CAPTIONS = [
    "A signed (hyper)graph: + trust / − distrust relations.",
    "Pick a held-out edge u–v; predict its sign from structure alone.",
    "Enumerate the cycles & walks through u–v, grouped by arity k.",
    "Each tuple → a SignedKAN Catmull-Rom spline → a per-arity embedding.",
    "Mix per-arity embeddings by the learned αₖ — HSiKAN's structural prior.",
    "Combine → logit → p(sign=+1); here the prediction is positive.",
]


def load_alpha(checkpoint: Path | None) -> tuple[np.ndarray, list[str], str]:
    """Return (alpha, labels, source). Real learned αₖ from a checkpoint if
    available; else a labelled walks-dominant illustrative regime."""
    if checkpoint is not None and checkpoint.is_file():
        try:
            from hymeko_neuro.demos.seminar.compat import register_legacy_checkpoint_aliases
            from hymeko_neuro.experiments.demo.inference import load_bundle

            register_legacy_checkpoint_aliases()
            b = load_bundle(checkpoint, device="cpu")
            a = b.alpha_vector()
            if a is not None:
                return np.asarray(a, float), b.tuple_labels(), os.path.relpath(checkpoint, REPO_ROOT)
        except Exception as err:  # fall back, but say why (no silent failure).
            print(f"[hsikan-anim] checkpoint αₖ unavailable ({err}); using illustrative regime.")
    labels = ["c2", "c5", "w2", "w3", "w4"]
    alpha = np.array([0.12, 0.10, 0.30, 0.26, 0.22])  # walks-dominant, mirrors OTC notes
    return alpha, labels, "illustrative (no checkpoint)"


# ── per-stage glyph painters (each draws into its own axes) ────────────────
def _frame_box(ax: Any, active: bool, done: bool) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    edge = BLUE if active else (INK if done else MUTE)
    lw = 2.6 if active else 1.0
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                                fill=False, edgecolor=edge, linewidth=lw, alpha=1.0 if (active or done) else 0.35))


def _alpha_for(active: bool, done: bool) -> float:
    return 1.0 if active else (0.85 if done else 0.18)


def draw_graph(ax: Any, al: float) -> None:
    pos = {0: (0.25, 0.7), 1: (0.7, 0.78), 2: (0.5, 0.4), 3: (0.8, 0.35), 4: (0.2, 0.3)}
    edges = [(0, 1, +1), (1, 2, -1), (0, 2, +1), (2, 3, +1), (2, 4, -1), (0, 4, +1)]
    for u, v, s in edges:
        c = GREEN if s > 0 else RED
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=c, lw=2.2, alpha=al, zorder=1)
    for _, (x, y) in pos.items():
        ax.scatter([x], [y], s=210, color="white", edgecolor=INK, zorder=2, alpha=al)


def draw_query(ax: Any, al: float) -> None:
    draw_graph(ax, al * 0.5)
    u, v = (0.25, 0.7), (0.7, 0.78)
    ax.plot([u[0], v[0]], [u[1], v[1]], color=BLUE, lw=3.4, ls=(0, (4, 3)), alpha=al, zorder=3)
    ax.text(0.47, 0.86, "u–v  ?", color=BLUE, ha="center", fontsize=12, fontweight="bold", alpha=al)


def draw_fir(ax: Any, al: float) -> None:
    rows = [("k=2", 2), ("k=3", 3), ("k=5", 5)]
    for i, (lbl, k) in enumerate(rows):
        y = 0.78 - i * 0.26
        ax.text(0.06, y, lbl, color=INK, fontsize=10, alpha=al, va="center")
        cx = np.linspace(0.34, 0.92, k)
        cy = y + 0.06 * np.sin(np.linspace(0, np.pi, k))
        ax.plot(cx, cy, color=PURPLE, lw=1.6, alpha=al, zorder=1)
        ax.scatter(cx, cy, s=46, color="white", edgecolor=PURPLE, alpha=al, zorder=2)


def draw_shell(ax: Any, al: float) -> None:
    xs = np.linspace(-1, 1, 200)
    ys = 0.5 + 0.32 * np.tanh(3 * xs) * np.cos(2.2 * xs)  # Catmull-Rom-ish bend
    ax.plot((xs + 1) / 2, ys, color=PURPLE, lw=2.4, alpha=al)
    ctrl_x = np.linspace(0, 1, 6)
    ctrl_y = 0.5 + 0.32 * np.tanh(3 * (2 * ctrl_x - 1)) * np.cos(2.2 * (2 * ctrl_x - 1))
    ax.scatter(ctrl_x, ctrl_y, s=40, color=BLUE, alpha=al, zorder=3)
    ax.text(0.5, 0.12, "SignedKAN spline", color=INK, ha="center", fontsize=9, alpha=al)


def draw_alpha(ax: Any, al: float, alpha: np.ndarray, labels: list[str]) -> None:
    xs = np.arange(len(alpha))
    cols = [BLUE if lb.startswith("c") else "#ED7D31" for lb in labels]
    ax.bar(xs, alpha, color=cols, edgecolor=INK, linewidth=0.5, alpha=al, width=0.7)
    ax.set_xlim(-0.6, len(alpha) - 0.4)
    ax.set_ylim(0, max(0.4, float(alpha.max()) * 1.25))
    for x, a, lb in zip(xs, alpha, labels):
        ax.text(x, a + 0.01, f"{a:.2f}", ha="center", fontsize=7.5, alpha=al)
        ax.text(x, -0.045 * ax.get_ylim()[1] / 0.4, lb, ha="center", fontsize=8, color=INK, alpha=al)


def draw_cpml(ax: Any, al: float) -> None:
    xs = np.linspace(-6, 6, 200)
    ys = 1 / (1 + np.exp(-xs))
    ax.plot((xs + 6) / 12, ys, color=INK, lw=2.0, alpha=al)
    ax.axhline(0.5, color=MUTE, lw=0.8, ls=":", alpha=al)
    ax.scatter([0.78], [1 / (1 + np.exp(-3.4))], s=120, color=GREEN, edgecolor=INK, zorder=3, alpha=al)
    ax.text(0.5, 0.12, "p(+) = 0.97  →  +", color=GREEN, ha="center", fontsize=10,
            fontweight="bold", alpha=al)


def render(out_dir: Path, alpha: np.ndarray, labels: list[str], src: str,
           frames_per_stage: int, hold: int) -> tuple[list[Path], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    painters = [draw_graph, draw_query, draw_fir, draw_shell,
                lambda ax, al: draw_alpha(ax, al, alpha, labels), draw_cpml]
    n = len(painters)
    total = n * frames_per_stage + hold
    frame_paths: list[Path] = []

    for f in range(total):
        active = min(f // frames_per_stage, n - 1)
        is_hold = f >= n * frames_per_stage
        fig = plt.figure(figsize=(12, 3.4), dpi=110)
        fig.suptitle("How HSiKAN works — structure → sign", fontsize=14, fontweight="bold", y=0.99)
        gs = fig.add_gridspec(1, n, left=0.02, right=0.98, top=0.82, bottom=0.26, wspace=0.18)
        for i, paint in enumerate(painters):
            ax = fig.add_subplot(gs[0, i])
            done = (i < active) or is_hold
            now = (i == active) and not is_hold
            _frame_box(ax, now, done)
            paint(ax, _alpha_for(now, done))
            ax.set_title(STAGE_TITLES[i], fontsize=8.5,
                         color=BLUE if now else (INK if done else MUTE), pad=2)
        # arrows between stages (figure coords)
        for i in range(n - 1):
            x0 = 0.02 + (i + 1) * (0.96 / n) - 0.006
            arr = FancyArrowPatch((x0, 0.54), (x0 + 0.012, 0.54), transform=fig.transFigure,
                                  arrowstyle="-|>", mutation_scale=12,
                                  color=INK if i < active or is_hold else MUTE, lw=1.4)
            fig.add_artist(arr)
        cap = STAGE_CAPTIONS[active] if not is_hold else \
            "Structure alone predicts the sign — the cycles are the inductive prior."
        fig.text(0.5, 0.10, cap, ha="center", fontsize=10.5, color=INK)
        fig.text(0.5, 0.03, f"αₖ source: {src}", ha="center", fontsize=7.5, color=MUTE)

        p = out_dir / f"frame_{f:03d}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        frame_paths.append(p)

    gif_path = out_dir / "hsikan_forward.gif"
    _assemble_gif(frame_paths, gif_path, frames_per_stage)
    return frame_paths, gif_path


def _assemble_gif(frames: list[Path], gif_path: Path, frames_per_stage: int) -> None:
    """Assemble a looping GIF with Pillow (no imageio dependency)."""
    from PIL import Image

    adaptive = Image.ADAPTIVE  # type: ignore[attr-defined]
    imgs = [Image.open(p).convert("P", palette=adaptive) for p in frames]
    # ~0.85 s per stage; the final hold frame lingers longer.
    per = max(1, int(850 / frames_per_stage))
    durations = [per] * (len(imgs) - 1) + [2200]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=durations, loop=0, optimize=True)


# ── neuron-level KAN renderer (the "neural-wise" view) ─────────────────────
# In a KAN the learnable activation is on the EDGE (a spline φ_ij), and each
# neuron simply SUMS its incoming activated signals. HSiKAN's inner SignedKAN
# is exactly this, with signed edges. We draw a small 5→4→2 network and sweep a
# forward pulse, lighting each layer's edge-splines as the signal passes.
_LAYERS = [5, 4, 2]
_LAYER_LABELS = ["tuple\nfeatures", "SignedKAN\nhidden", "edge\nembed"]


def _neuron_positions() -> list[list[tuple[float, float]]]:
    xs = np.linspace(0.12, 0.88, len(_LAYERS))
    pos = []
    for li, n in enumerate(_LAYERS):
        ys = np.linspace(0.78, 0.22, n) if n > 1 else [0.5]
        pos.append([(float(xs[li]), float(y)) for y in ys])
    return pos


def _spline_glyph(ax: Any, cx: float, cy: float, scale: float, sign: int, al: float) -> None:
    """A tiny Catmull-Rom-ish activation curve centred at (cx, cy) — the KAN
    edge function φ. Colour encodes the edge sign."""
    t = np.linspace(-1, 1, 40)
    curve = np.tanh(2.4 * t) * (0.5 if sign > 0 else -0.5)
    gx = cx + t * scale
    gy = cy + curve * scale
    ax.plot(gx, gy, color=(GREEN if sign > 0 else RED), lw=1.3, alpha=al, zorder=4)


def render_neurons(out_dir: Path, frames_per_step: int, hold: int) -> tuple[list[Path], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pos = _neuron_positions()
    # signed edges between consecutive layers
    edges = []  # (layer, i, j, sign)
    for li in range(len(_LAYERS) - 1):
        for i in range(_LAYERS[li]):
            for j in range(_LAYERS[li + 1]):
                edges.append((li, i, j, int(rng.choice([-1, 1]))))

    n_steps = len(_LAYERS) - 1
    total = n_steps * frames_per_step + hold
    frame_paths: list[Path] = []

    for f in range(total):
        is_hold = f >= n_steps * frames_per_step
        step = min(f // frames_per_step, n_steps - 1)
        prog = ((f % frames_per_step) + 1) / frames_per_step
        if is_hold:
            step, prog = n_steps - 1, 1.0

        fig = plt.figure(figsize=(11, 5), dpi=110)
        fig.suptitle("HSiKAN — neuron view (KAN: spline activation on every edge)",
                     fontsize=14, fontweight="bold", y=0.97)
        ax = fig.add_axes((0.03, 0.10, 0.94, 0.80))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # edges + spline glyphs
        for (li, i, j, sign) in edges:
            active = (li < step) or (li == step)
            lit = (li < step) or (li == step and is_hold) or (li == step and prog >= 0.5)
            (x0, y0), (x1, y1) = pos[li][i], pos[li + 1][j]
            ax.plot([x0, x1], [y0, y1], color=(GREEN if sign > 0 else RED),
                    lw=1.5 if lit else 0.8, alpha=(0.85 if lit else 0.12), zorder=1)
            if active:
                mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                _spline_glyph(ax, mx, my, 0.032, sign, 0.95 if lit else 0.25)

        # travelling pulse on the active layer's edges
        if not is_hold:
            for (li, i, j, sign) in edges:
                if li != step:
                    continue
                (x0, y0), (x1, y1) = pos[li][i], pos[li + 1][j]
                px, py = x0 + prog * (x1 - x0), y0 + prog * (y1 - y0)
                ax.scatter([px], [py], s=26, color=BLUE, zorder=5, alpha=0.9)

        # neurons
        for li, layer in enumerate(pos):
            filled = (li <= step) or is_hold
            for (x, y) in layer:
                ax.add_patch(Circle((x, y), 0.028, facecolor=(BLUE if filled else "white"),
                                    edgecolor=INK, lw=1.4, alpha=1.0 if filled else 0.4, zorder=3))
            lx = layer[0][0]
            ax.text(lx, 0.06, _LAYER_LABELS[li], ha="center", fontsize=9,
                    color=(INK if filled else MUTE))
            ax.text(lx, 0.92, f"Σ  ({_LAYERS[li]})", ha="center", fontsize=8.5, color=MUTE)

        cap = ("Each neuron sums signals reshaped by a learnable spline φ on every "
               "incoming edge; green/red = signed relation."
               if not is_hold else
               "Forward pass complete: structure → edge embedding → sign. "
               "The splines (not the weights) are what HSiKAN learns.")
        fig.text(0.5, 0.015, cap, ha="center", fontsize=10.5, color=INK)

        p = out_dir / f"neuron_frame_{f:03d}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        frame_paths.append(p)

    gif_path = out_dir / "hsikan_neurons.gif"
    _assemble_gif(frame_paths, gif_path, frames_per_step)
    return frame_paths, gif_path


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["forward", "neurons", "both"], default="both",
                   help="forward = dataflow pipeline; neurons = KAN neuron view.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                   help="checkpoint to read real αₖ from; '' for illustrative.")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--frames-per-stage", type=int, default=5)
    p.add_argument("--hold", type=int, default=4)
    p.add_argument("--keep-frames", action="store_true",
                   help="keep the intermediate PNG frames (default: GIF only).")
    args = p.parse_args(argv)
    out = Path(args.out)

    def _finish(frames: list[Path], gif: Path, what: str) -> None:
        if not args.keep_frames:
            for fp in frames:
                fp.unlink(missing_ok=True)
        print(f"[hsikan-anim] {what}: {len(frames)} frames -> {gif}")

    if args.mode in ("forward", "both"):
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        alpha, labels, src = load_alpha(ckpt)
        frames, gif = render(out, alpha, labels, src, args.frames_per_stage, args.hold)
        print(f"[hsikan-anim] forward αₖ source={src}")
        _finish(frames, gif, "forward")
    if args.mode in ("neurons", "both"):
        frames, gif = render_neurons(out, args.frames_per_stage, args.hold)
        _finish(frames, gif, "neurons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
