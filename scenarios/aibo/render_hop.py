"""Plot the contact-scheduled hop MPC: the ballistic flight + the bounded capturability Lyapunov.

Shows the AIBO (and human) planned hop — COM leaves the ground on a ballistic arc yet the
capturability Lyapunov stays inside the recoverable region — the PLANNED loss of static stability.
Writes a static PNG (Agg backend, headless).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .hop_mpc import CentroidalHopMPC, HopParams  # noqa: E402

_OUT = Path("reports/2026-07-27-aibo-hop")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, pr, color in [("AIBO (m=2 kg, z0=0.23)", HopParams(mass=2.0, z0=0.23, f_max=60), "#2266cc"),
                             ("human (m=15 kg, z0=0.645)", HopParams(mass=15.0, z0=0.645, f_max=400), "#cc6622")]:
        mpc = CentroidalHopMPC(p=pr, x_target=0.25)
        traj, f, sched = mpc.plan()
        t = np.arange(len(traj)) * mpc.dt
        flight = ~sched
        vcap = mpc.capture_lyapunov(traj)
        # (1) COM x-z trajectory: stance vs flight
        axes[0].plot(traj[:-1][sched, 0], traj[:-1][sched, 2], ".", color=color, ms=6, label=f"{label} stance")
        axes[0].plot(traj[:-1][flight, 0], traj[:-1][flight, 2], "-", color=color, lw=2.5, alpha=0.6,
                     label=f"{label} FLIGHT (ballistic)")
        axes[0].axhline(pr.z0, color=color, ls=":", lw=0.8, alpha=0.5)
        # (2) vertical force
        axes[1].plot(t[:-1], f[:, 1], color=color, lw=1.8, label=label)
        # (3) capturability Lyapunov
        axes[2].plot(t, vcap, color=color, lw=1.8, label=label)
    axes[0].set(title="COM trajectory — a PLANNED hop", xlabel="forward x (m)", ylabel="height z (m)")
    axes[0].legend(fontsize=7)
    fs = mpc.n_stance1 * mpc.dt
    ff = (mpc.n_stance1 + mpc.n_flight) * mpc.dt
    for ax, ttl, yl in ((axes[1], "ground force Fz — ZERO in flight (ballistic)", "Fz (N)"),
                        (axes[2], "capturability Lyapunov — BOUNDED (recoverable)", r"V$_{cap}$")):
        ax.axvspan(fs, ff, color="0.85", label="flight phase")
        ax.set(title=ttl, xlabel="time (s)", ylabel=yl)
        ax.legend(fontsize=7)
    axes[2].axhline(0.1, color="r", ls="--", lw=0.9)
    axes[2].text(0.01, 0.105, "recoverable region", color="r", fontsize=8)
    fig.suptitle("Contact-scheduled centroidal hop MPC — planned flight inside the Lyapunov region "
                 "(not the retracted exploit)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_OUT / "hop_mpc.png", dpi=110)
    print("wrote", _OUT / "hop_mpc.png")


if __name__ == "__main__":
    main()
