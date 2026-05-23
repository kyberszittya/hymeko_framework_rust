"""Generate the CPU-vs-GPU per-iteration scaling plot for KEPAF §VII.

Numbers are pulled from Table tab:layout-time (CPU baseline) and
Table tab:vulkan-bench (GPU). Hardcoded here so the plot is
self-contained and reproducible without re-running the bench.

Output: paper/kepaf_v1/figures/scaling.{pdf,png}
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = REPO / "paper" / "kepaf_v1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-iteration cost (ms), from §VII tables.
fixtures = [
    ("canonical",      31,    0.067,   0.117),
    ("MNIST adj.",     7882,  2624.0,  7.64),
    ("synthetic 1e4",  35000, 86965.0, 52.6),
]
labels  = [f[0] for f in fixtures]
n_levi  = np.array([f[1] for f in fixtures], dtype=float)
cpu_ms  = np.array([f[2] for f in fixtures], dtype=float)
gpu_ms  = np.array([f[3] for f in fixtures], dtype=float)
speedup = cpu_ms / gpu_ms

fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=120)

ax.loglog(n_levi, cpu_ms, "o-",
          color="#b02a2a", lw=2.0, ms=7,
          label="CPU NetworkX spring_layout (single thread)")
ax.loglog(n_levi, gpu_ms, "s-",
          color="#1b6ca8", lw=2.0, ms=7,
          label="GPU hymeko_compute force_directed (RTX 2070 SUPER)")

# O(N^2) reference line anchored at the canonical CPU point.
n_ref = np.array([20.0, 50000.0])
c_ref = cpu_ms[0] * (n_ref / n_levi[0]) ** 2
ax.loglog(n_ref, c_ref, "--", color="#888", lw=1.0,
          label=r"$O(|V_L|^2)$ reference")

# Annotate speed-up factor next to each fixture.
for i, (lab, n, cms, gms, sp) in enumerate(
    zip(labels, n_levi, cpu_ms, gpu_ms, speedup)
):
    if sp >= 100:
        ax.annotate(
            f"{sp:.0f}×",
            xy=(n, np.sqrt(cms * gms)),
            xytext=(8, 0),
            textcoords="offset points",
            color="#2e7d32",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

# Shade the regime where CPU >= 1 s/iter (interactively impractical).
ax.axhspan(1000.0, 1e8, color="#fde7e7", alpha=0.6, zorder=0)
ax.text(50, 5000, "CPU > 1 s/iter\n(non-interactive)",
        color="#7b2424", fontsize=9, alpha=0.85)

# Shade the regime where GPU stays under 1 frame at 60 fps.
ax.axhspan(0.001, 16.7, color="#e7f3e7", alpha=0.5, zorder=0)
ax.text(50, 0.02, "GPU under 16.7 ms/iter\n(60 fps frame budget)",
        color="#1f5d1f", fontsize=9, alpha=0.9)

ax.set_xlabel(r"Levi-graph size $|V_L|$")
ax.set_ylabel("per-iteration layout time (ms, log scale)")
ax.set_title("CPU vs. GPU Fruchterman-Reingold scaling")

# X-axis ticks at the fixture sizes, with labels.
ax.set_xticks(n_levi)
ax.set_xticklabels([f"{lab}\n({int(n)})" for lab, n in zip(labels, n_levi)],
                   fontsize=9)
ax.minorticks_off()

ax.grid(True, which="both", ls=":", color="#aaa", lw=0.5, alpha=0.5)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
fig.tight_layout()

out_pdf = OUT_DIR / "scaling.pdf"
out_png = OUT_DIR / "scaling.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=140)
plt.close(fig)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
