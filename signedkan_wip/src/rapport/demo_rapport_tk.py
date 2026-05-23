"""Tkinter + matplotlib visualisation of the rapport-coherence loop.

Shows three panels:

* **Triangle graph** — agents as nodes, signed dyadic edges
  coloured by current sign (green = +, red = −) and opacity by
  current |w_ij|.
* **σ(triad)(t)** — scrolling time-series of the cycle coherence.
* **Behaviour log** — observation events + robot policy firings.

Bottom controls: ``Play / Pause``, ``Step``, ``Inject conflict``,
``Reset``.

Run:
    python -m signedkan_wip.src.rapport.demo_rapport_tk \\
        --coalition data/coalitions/triad_hri.hymeko

Plan: docs/plans/2026-05-18-rapport-coherence-demo-nagoya/.
"""
from __future__ import annotations

import argparse
import math
import sys
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from .coalition import Coalition, load_coalition  # noqa: E402
from .coherence import sigma_cycle  # noqa: E402
from .estimator import CoalitionEstimator  # noqa: E402
from .policy import PolicyEngine  # noqa: E402
from .simulator import (  # noqa: E402
    ConflictScenario, Simulator, SimulatorConfig,
)


# Layout for the three triadic agents on the canvas.
_TRIANGLE_POSITIONS: dict[str, tuple[float, float]] = {
    "alice": (-0.7, +0.5),
    "bob":   (+0.7, +0.5),
    "r1":    ( 0.0, -0.6),
}


class RapportDemoApp(tk.Tk):
    """Main Tk window — the visit's interactive demonstrator."""

    def __init__(self, coalition: Coalition, cycle_name: str = "triad",
                 frame_ms: int = 200) -> None:
        super().__init__()
        self.title(
            "Rapport-coherence demo — σ-cycle balance over a triadic HRI coalition"
        )
        self.geometry("1100x720")
        self.coalition = coalition
        self.cycle_name = cycle_name
        self.frame_ms = int(frame_ms)
        self.cycle = coalition.cycles[cycle_name]

        # Sim/est/policy + state.
        cfg = SimulatorConfig(n_frames=10_000)  # effectively unlimited
        self.simulator = Simulator(coalition, cfg)
        self.estimator = CoalitionEstimator(coalition, alpha=0.2)
        self.policy = PolicyEngine(coalition, cooldown_frames=15)

        self.t: int = 0
        self.playing: bool = True
        self.sigma_history: deque[float] = deque(maxlen=400)
        self.log_messages: deque[str] = deque(maxlen=12)
        self.recent_actions: deque[tuple[int, str]] = deque(maxlen=4)
        self.injected_conflicts: int = 0

        self._build_ui()
        self.after(self.frame_ms, self._step_loop)

    # ─── UI ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Header.
        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        n_pol = len(self.coalition.policies)
        n_rel = len(self.coalition.relations)
        ttk.Label(header,
                   text=(f"coalition: {self.coalition.name}  |  "
                         f"agents: {len(self.coalition.agents)}  |  "
                         f"relations: {n_rel}  |  "
                         f"policies: {n_pol}  |  "
                         f"cycle: {self.cycle_name}"),
                   font=("Helvetica", 9)).pack(anchor=tk.W)

        # Matplotlib figure with two axes.
        self.fig = plt.Figure(figsize=(10, 5.0), dpi=100)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.8])
        self.ax_graph = self.fig.add_subplot(gs[0, 0])
        self.ax_sigma = self.fig.add_subplot(gs[0, 1])
        self.fig.tight_layout(pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                          expand=True, padx=8, pady=4)

        # Controls.
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        self.play_btn = ttk.Button(ctrl, text="Pause", command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Step",
                    command=self._step_once).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Inject conflict",
                    command=self._inject_conflict).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Reset",
                    command=self._reset).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text=" σ:").pack(side=tk.LEFT, padx=(20, 2))
        self.sigma_label = ttk.Label(ctrl, text="—", font=("Helvetica", 12, "bold"))
        self.sigma_label.pack(side=tk.LEFT)
        ttk.Label(ctrl, text="  t:").pack(side=tk.LEFT, padx=(16, 2))
        self.t_label = ttk.Label(ctrl, text="0", font=("Helvetica", 10))
        self.t_label.pack(side=tk.LEFT)

        # Log panel.
        log_frame = ttk.LabelFrame(self, text="Behaviour log", padding=4)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        self.log_text = tk.Text(log_frame, height=8, font=("Courier", 9),
                                  state=tk.DISABLED, wrap=tk.NONE)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ─── step loop ──────────────────────────────────────────────────

    def _step_loop(self) -> None:
        if self.playing:
            self._step_once()
        self.after(self.frame_ms, self._step_loop)

    def _step_once(self) -> None:
        obs_list = self.simulator.step(self.t)
        weights = self.estimator.step(obs_list)
        sigma_val = sigma_cycle(weights, self.cycle)
        self.sigma_history.append(sigma_val)
        out = self.policy.step(self.t, {self.cycle_name: sigma_val})
        for act in out.actions:
            self.recent_actions.append((self.t, act))
            self._log(f"[t={self.t:4d}] policy fired: {act}")
            self.simulator.trigger_repair(self.t)
        for obs in obs_list[-3:]:  # log only last few per frame to avoid spam
            self._log(
                f"[t={self.t:4d}] obs {obs.kind:18s} {obs.src} → {obs.dst}"
            )
        self.t += 1
        self._redraw()
        self.t_label.config(text=str(self.t))
        self.sigma_label.config(
            text=f"{sigma_val:+.3f}",
            foreground=("#2d8a4f" if sigma_val > 0 else "#a82a35"),
        )

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.config(text="Play" if not self.playing else "Pause")

    def _inject_conflict(self) -> None:
        """Add a 50-frame alice↔bob conflict starting next frame."""
        start = self.t + 1
        end = start + 50
        new_scenarios = list(self.simulator.config.conflict_scenarios) + [
            ConflictScenario(start=start, end=end, agent_a="alice", agent_b="bob"),
        ]
        self.simulator.config.conflict_scenarios = tuple(new_scenarios)
        self.injected_conflicts += 1
        self._log(
            f"[t={self.t:4d}] *** injected conflict alice↔bob for 50 frames ***"
        )

    def _reset(self) -> None:
        self.t = 0
        self.sigma_history.clear()
        self.log_messages.clear()
        self.recent_actions.clear()
        self.injected_conflicts = 0
        cfg = SimulatorConfig(n_frames=10_000, seed=int(time_seed()))
        self.simulator = Simulator(self.coalition, cfg)
        self.estimator = CoalitionEstimator(self.coalition, alpha=0.2)
        self.policy.reset_cooldowns()
        self._log("[reset] state cleared")

    # ─── rendering ──────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._redraw_graph()
        self._redraw_sigma()
        self.canvas.draw_idle()

    def _redraw_graph(self) -> None:
        ax = self.ax_graph
        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        # Draw edges first.
        for r in self.coalition.relations.values():
            x0, y0 = _TRIANGLE_POSITIONS[r.src]
            x1, y1 = _TRIANGLE_POSITIONS[r.dst]
            w = self.estimator.weights[r.name]
            colour = "#2d8a4f" if w >= 0 else "#a82a35"
            alpha = max(0.15, min(1.0, abs(w)))
            ax.plot([x0, x1], [y0, y1], "-",
                     color=colour, lw=4, alpha=alpha)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.06, f"{w:+.2f}",
                     ha="center", va="bottom",
                     fontsize=8, color=colour,
                     fontweight="bold")
        # Draw agent nodes.
        for name, (x, y) in _TRIANGLE_POSITIONS.items():
            if name not in self.coalition.agents:
                continue
            kind = self.coalition.agents[name].kind
            face = "#cfeaf6" if kind == "human" else "#ffdda8"
            ax.scatter([x], [y], s=1500, c=face, edgecolor="black",
                        zorder=3, linewidth=1.5)
            ax.text(x, y, name, ha="center", va="center",
                     fontsize=11, fontweight="bold", zorder=4)
        # Show recent robot actions inset.
        if self.recent_actions:
            txt = "\n".join(f"  {act}  (t={tt})"
                              for tt, act in reversed(list(self.recent_actions)))
            ax.text(-1.15, -0.95, f"recent actions:\n{txt}",
                     ha="left", va="bottom", fontsize=7,
                     bbox=dict(facecolor="white", alpha=0.85, edgecolor="#888"))

    def _redraw_sigma(self) -> None:
        ax = self.ax_sigma
        ax.clear()
        if self.sigma_history:
            ts = list(range(self.t - len(self.sigma_history), self.t))
            ax.plot(ts, list(self.sigma_history), "-", color="#264a73", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5, linestyle="--")
        ax.axhline(-0.2, color="#ff8800", lw=0.5, linestyle=":")
        ax.axhline(-0.5, color="#cc4400", lw=0.5, linestyle=":")
        ax.axhline(-0.8, color="#990000", lw=0.5, linestyle=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("σ(triad) — Cartwright-Harary cycle balance", fontsize=10)
        ax.set_xlabel("time (frames)")
        ax.set_ylabel("σ")
        # Annotate the three policy thresholds.
        if self.sigma_history:
            x_left = ts[0]
            ax.text(x_left, -0.2, " repair", color="#ff8800", va="center", fontsize=8)
            ax.text(x_left, -0.5, " mediate", color="#cc4400", va="center", fontsize=8)
            ax.text(x_left, -0.8, " withdraw", color="#990000", va="center", fontsize=8)

    # ─── log ────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        self.log_messages.append(message)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "\n".join(self.log_messages))
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


def time_seed() -> int:
    import time
    return int(time.time() * 1000) & 0x7FFF_FFFF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coalition",
        default="data/coalitions/triad_hri.hymeko",
        help="Path to a coalition .hymeko file.",
    )
    ap.add_argument(
        "--cycle", default="triad",
        help="Name of the σ-cycle to monitor (must exist in the file).",
    )
    ap.add_argument(
        "--frame-ms", type=int, default=200,
        help="Frame interval in milliseconds (200ms = 5Hz).",
    )
    args = ap.parse_args()

    path = Path(args.coalition)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        print(f"coalition file not found: {path}", file=sys.stderr)
        sys.exit(2)

    coalition = load_coalition(path)
    if args.cycle not in coalition.cycles:
        print(f"cycle {args.cycle!r} not in coalition; available: "
              f"{list(coalition.cycles.keys())}", file=sys.stderr)
        sys.exit(2)
    app = RapportDemoApp(coalition, cycle_name=args.cycle,
                          frame_ms=args.frame_ms)
    app.mainloop()


if __name__ == "__main__":
    main()
