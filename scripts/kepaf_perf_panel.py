"""4-panel KEPAF Section 7 performance figure:
  (a) force_directed per-iter scaling (CPU vs GPU, log-log) — GPU side
      uses a 6-point density sweep over |V|, not just the 3 fixture
      sizes
  (b) signed_spmv per-call scaling (sub-frame across two orders of |V|)
  (c) FR convergence at synthetic-10k: RMS per-iter displacement vs iter
  (d) total convergence-sweep wall time, CPU vs GPU bar chart

Output: paper/kepaf_v1/figures/perf_panels.{pdf,png}
Plus a second figure perf_throughput.{pdf,png} reporting FR
effective FLOPS vs |V| as a sanity check on absolute GPU utilisation.

Re-runs the GPU layout binary with `dump_every` to capture per-iter
snapshots for the convergence panel, and at 6 sizes for the density
sweep on panel (a).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kepaf_benchmark import fixture_synthetic

LAYOUT_BIN = REPO / "target" / "release" / "examples" / "layout_from_json"
OUT_DIR = REPO / "paper" / "kepaf_v1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── (a) force_directed scaling: hard-coded from §VII tables ─────────
fr_fixtures = [
    ("canonical",    31,    0.067,   0.117),
    ("MNIST adj.",   7882,  2624.0,  7.64),
    ("synthetic 1e4",35000, 86965.0, 52.6),
]

# ─── (b) signed_spmv per-call: hard-coded from §VII tables ───────────
spmv_fixtures = [
    ("canonical",    31,    0.45),
    ("MNIST adj.",   1242,  0.37),
    ("synthetic 1e4",35000, 0.56),
]


def run_convergence(N: int = 10_000, M: int = 25_000, n_iter: int = 100,
                    dump_every: int = 1):
    """Run the GPU FR kernel with per-iter snapshots, return iter array
    and RMS per-iter displacement array."""
    if not LAYOUT_BIN.exists():
        raise SystemExit(f"missing binary: {LAYOUT_BIN}")

    G, name = fixture_synthetic(N=N, M=M)
    print(f"[convergence] {name}: |V_levi|={len(G)}, |E|={G.number_of_edges()}")
    labels = list(G.nodes())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    edges = [[label_to_id[u], label_to_id[v]] for u, v in G.edges()]
    payload = json.dumps({
        "n_nodes": len(labels),
        "n_iter": n_iter,
        "seed": 0,
        "dump_every": dump_every,
        "edges": edges,
    })
    proc = subprocess.run(
        [str(LAYOUT_BIN)], input=payload,
        capture_output=True, text=True, check=True,
    )
    out = json.loads(proc.stdout)
    print(f"  device={out['device']}  wall={out['wall_ms']:.1f}ms  "
          f"snaps={len(out['snapshots'])}")
    snaps = out["snapshots"]

    # RMS displacement between consecutive snapshots, normalised by the
    # bounding-box diagonal of the *first* snapshot (so the curve is
    # scale-invariant across fixtures).
    pos0 = np.asarray(snaps[0]["positions"], dtype=np.float64)
    bbox_diag = np.linalg.norm(pos0.max(0) - pos0.min(0)) or 1.0

    iters, rms = [], []
    for k in range(1, len(snaps)):
        pa = np.asarray(snaps[k - 1]["positions"], dtype=np.float64)
        pb = np.asarray(snaps[k]["positions"], dtype=np.float64)
        dx = pb - pa
        # Per-iter displacement: divide by gap to get per-iter RMS.
        gap = max(snaps[k]["iter"] - snaps[k - 1]["iter"], 1)
        rms_per_iter = float(np.sqrt(np.mean((dx ** 2).sum(1))) / gap / bbox_diag)
        iters.append(snaps[k]["iter"])
        rms.append(rms_per_iter)
    return np.asarray(iters), np.asarray(rms), name


# ─── GPU density sweep at 6 sizes for panel (a) ──────────────────────

def run_density_point(n: int, m_arcs: int, n_iter: int = 50, seed: int = 0):
    """Run the GPU kernel on a synthetic graph of (n_nodes, m_arcs);
    return per-iter time in ms (kernel-side wall, warmed-up)."""
    if not LAYOUT_BIN.exists():
        raise SystemExit(f"missing binary: {LAYOUT_BIN}")
    rng = np.random.default_rng(seed)
    edges = [[int(a), int(b)] for a, b in
             rng.integers(0, n, size=(m_arcs, 2))]
    payload = json.dumps({
        "n_nodes": n, "n_iter": n_iter, "seed": seed, "edges": edges,
    })
    proc = subprocess.run(
        [str(LAYOUT_BIN)], input=payload,
        capture_output=True, text=True, check=True,
    )
    out = json.loads(proc.stdout)
    return out["wall_ms"] / out["n_iter"]


density_sizes = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
density_arcs  = [int(2.5 * n) for n in density_sizes]
print("== density sweep ==")
gpu_density_ms = []
for n, m in zip(density_sizes, density_arcs):
    # Drop n_iter at large sizes to keep wall time reasonable; the
    # per-iter mean is what we report.
    iters = 50 if n <= 10_000 else 20 if n <= 30_000 else 10
    t = run_density_point(n, m, n_iter=iters)
    gpu_density_ms.append(t)
    print(f"  |V|={n:6}  |E|={m:7}   per-iter={t:.3f} ms  (n_iter={iters})")

# ─── Build the figure ─────────────────────────────────────────────────

print("== KEPAF performance panel ==")
conv_iter, conv_rms, conv_name = run_convergence()

plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2), dpi=120)
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

# (a) FR scaling
n_fr   = np.array([f[1] for f in fr_fixtures], dtype=float)
cpu_ms = np.array([f[2] for f in fr_fixtures], dtype=float)
gpu_ms = np.array([f[3] for f in fr_fixtures], dtype=float)
speedup = cpu_ms / gpu_ms

ax_a.loglog(n_fr, cpu_ms, "o-", color="#b02a2a", lw=2.0, ms=7,
            label="CPU NetworkX (3 fixtures)")
# GPU: 6-point density sweep (filled) + the 3 paper fixtures (open).
ax_a.loglog(density_sizes, gpu_density_ms, "s-", color="#1b6ca8",
            lw=1.6, ms=6, label="GPU sweep (7 sizes)")
ax_a.loglog(n_fr, gpu_ms, "D", color="#1b6ca8", ms=8,
            mfc="white", mew=1.6,
            label="GPU paper fixtures")
n_ref = np.array([20.0, 50000.0])
ax_a.loglog(n_ref, cpu_ms[0] * (n_ref / n_fr[0]) ** 2,
            "--", color="#888", lw=1.0, label=r"$O(|V_L|^2)$")
for n, cms, gms, sp in zip(n_fr, cpu_ms, gpu_ms, speedup):
    if sp >= 100:
        ax_a.annotate(f"{sp:.0f}×",
                      xy=(n, np.sqrt(cms * gms)),
                      xytext=(8, 0), textcoords="offset points",
                      color="#2e7d32", fontsize=10, fontweight="bold",
                      ha="left", va="center")
ax_a.axhspan(1000.0, 1e8, color="#fde7e7", alpha=0.5, zorder=0)
ax_a.axhspan(0.001, 16.7, color="#e7f3e7", alpha=0.5, zorder=0)
ax_a.set_xlabel(r"Levi-graph size $|V_L|$")
ax_a.set_ylabel("per-iteration time (ms, log)")
ax_a.set_title("(a) force_directed scaling")
ax_a.set_xticks(n_fr)
ax_a.set_xticklabels([f"{int(n)}" for n in n_fr], fontsize=9)
ax_a.minorticks_off()
ax_a.grid(True, which="both", ls=":", color="#aaa", lw=0.5, alpha=0.5)
ax_a.legend(loc="upper left", fontsize=8, framealpha=0.95)

# (b) SpMV scaling
n_sp  = np.array([f[1] for f in spmv_fixtures], dtype=float)
sp_ms = np.array([f[2] for f in spmv_fixtures], dtype=float)
ax_b.semilogx(n_sp, sp_ms, "D-", color="#1b6ca8", lw=2.0, ms=8)
ax_b.axhspan(0.0, 16.7, color="#e7f3e7", alpha=0.5, zorder=0)
ax_b.text(40, 14.0, "60 fps frame budget (16.7 ms)",
          color="#1f5d1f", fontsize=9, alpha=0.9)
for n, m in zip(n_sp, sp_ms):
    ax_b.annotate(f"{m:.2f} ms",
                  xy=(n, m), xytext=(0, 8),
                  textcoords="offset points",
                  ha="center", fontsize=9, color="#1b6ca8")
ax_b.set_xlabel(r"row count $|V|$")
ax_b.set_ylabel("per-call time (ms)")
ax_b.set_title("(b) signed_spmv scaling")
ax_b.set_xticks(n_sp)
ax_b.set_xticklabels([f"{int(n)}" for n in n_sp], fontsize=9)
ax_b.set_ylim(bottom=0, top=18)
ax_b.minorticks_off()
ax_b.grid(True, which="both", ls=":", color="#aaa", lw=0.5, alpha=0.5)

# (c) Convergence
ax_c.semilogy(conv_iter, conv_rms, "-", color="#2e7d32", lw=2.0)
ax_c.set_xlabel("iteration")
ax_c.set_ylabel(r"per-iter RMS displacement / bbox diag")
ax_c.set_title(f"(c) convergence: {conv_name}")
ax_c.grid(True, which="both", ls=":", color="#aaa", lw=0.5, alpha=0.5)
ax_c.text(
    0.97, 0.93,
    f"$|V_L|={int(len(conv_iter)) and 35000}$\n"
    f"final $\\Delta_{{rms}}\\approx{conv_rms[-1]:.2e}$",
    transform=ax_c.transAxes, ha="right", va="top",
    fontsize=9,
    bbox=dict(facecolor="white", edgecolor="#aaa", boxstyle="round,pad=0.3"),
)

# (d) Total convergence-sweep wall time (CPU vs GPU, log bar chart).
n_iter_sweep = np.array([100.0, 50.0, 20.0])  # iter counts used per fixture
cpu_total_s = cpu_ms * n_iter_sweep / 1000.0
gpu_total_s = gpu_ms * n_iter_sweep / 1000.0
x = np.arange(len(fr_fixtures))
w = 0.36
bars_cpu = ax_d.bar(x - w / 2, cpu_total_s, w, color="#b02a2a",
                    label="CPU total")
bars_gpu = ax_d.bar(x + w / 2, gpu_total_s, w, color="#1b6ca8",
                    label="GPU total")
ax_d.set_yscale("log")
ax_d.set_ylabel("total time (s, log)")
ax_d.set_title("(d) total convergence-sweep wall time")
ax_d.set_xticks(x)
ax_d.set_xticklabels(
    [f"{f[0]}\n({int(f[1])}, {int(it)} iter)"
     for f, it in zip(fr_fixtures, n_iter_sweep)],
    fontsize=8,
)
for xi, t_cpu, t_gpu in zip(x, cpu_total_s, gpu_total_s):
    ax_d.text(xi - w / 2, t_cpu * 1.5, f"{t_cpu:.2f}s" if t_cpu < 60
              else f"{t_cpu/60:.0f} min", ha="center", fontsize=8,
              color="#7b2424")
    ax_d.text(xi + w / 2, t_gpu * 1.5, f"{t_gpu*1000:.0f}ms"
              if t_gpu < 1 else f"{t_gpu:.2f}s",
              ha="center", fontsize=8, color="#1f5d1f")
ax_d.grid(True, axis="y", which="both", ls=":", color="#aaa", lw=0.5,
          alpha=0.5)
ax_d.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax_d.minorticks_off()

fig.tight_layout()
out_pdf = OUT_DIR / "perf_panels.pdf"
out_png = OUT_DIR / "perf_panels.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=140)
plt.close(fig)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")

# ─── Throughput figure (separate) ─────────────────────────────────────
# Effective FLOPS for force_directed at each fixture: pair-wise
# repulsion is N^2 inner steps × ~10 flops, plus N × M arc scans × ~5
# flops; we report the dominant term.
fr_flops = np.array([
    n * n * 10 + n * m_estimate * 5
    for n, m_estimate in zip(n_fr, [33.0, 13280.0, 100000.0])
])
gflops = fr_flops / (gpu_ms * 1e-3) / 1e9  # ms → s, divide

fig2, ax2 = plt.subplots(figsize=(5.4, 3.0), dpi=120)
ax2.semilogx(n_fr, gflops, "o-", color="#1b6ca8", lw=2.0, ms=8)
ax2.axhline(9000.0, ls="--", color="#888",
            label="device fp32 peak (9 TFLOPS)")
for n, g in zip(n_fr, gflops):
    if g < 1.0:
        # Sub-GFLOP/s point: dispatch-overhead-dominated, do not
        # round to 0 in the label.
        text = "dispatch\ndominated"
        color = "#888"
    elif g < 10.0:
        text = f"{g:.1f} GFLOP/s"
        color = "#1b6ca8"
    else:
        text = f"{g:.0f} GFLOP/s"
        color = "#1b6ca8"
    ax2.annotate(text, xy=(n, g), xytext=(0, 8),
                 textcoords="offset points", ha="center", fontsize=9,
                 color=color)
ax2.set_xticks(n_fr)
ax2.set_xticklabels([f"{int(n)}" for n in n_fr], fontsize=9)
ax2.minorticks_off()
ax2.set_xlabel(r"Levi-graph size $|V_L|$")
ax2.set_ylabel("effective throughput (GFLOP/s)")
ax2.set_title("Force-summation effective fp32 throughput")
ax2.set_yscale("log")
ax2.grid(True, which="both", ls=":", color="#aaa", lw=0.5, alpha=0.5)
ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig2.tight_layout()
out_pdf2 = OUT_DIR / "perf_throughput.pdf"
out_png2 = OUT_DIR / "perf_throughput.png"
fig2.savefig(out_pdf2)
fig2.savefig(out_png2, dpi=140)
plt.close(fig2)
print(f"wrote {out_pdf2}")
print(f"wrote {out_png2}")
