"""Plot the periodic running gait MPC — the COM bounce, the flight phases, bounded capturability.

Shows a continuous run (5 strides) for the AIBO and the human: the COM height bounces periodically
with a ballistic FLIGHT each stride, forward position advances steadily, and the capturability
Lyapunov stays inside the recoverable region every stride (the orbital stability of running).
Writes a static PNG (Agg backend, headless).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .hop_mpc import HopParams  # noqa: E402
from .running_mpc import RunningGaitMPC  # noqa: E402

_OUT = Path("reports/2026-07-27-aibo-hop")


def _shade_flight(ax, t, sched) -> None:
    on = None
    for k in range(len(sched)):
        if not sched[k] and on is None:
            on = t[k]
        if (sched[k] or k == len(sched) - 1) and on is not None:
            ax.axvspan(on, t[k], color="0.86", lw=0)
            on = None


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, pr, color in [("AIBO (m=2, z0=0.23)", HopParams(mass=2.0, z0=0.23, f_max=80), "#2266cc"),
                             ("human (m=15, z0=0.645)", HopParams(mass=15.0, z0=0.645, f_max=500), "#cc6622")]:
        mpc = RunningGaitMPC(p=pr, v_forward=0.6)
        traj, sched, _f = mpc.simulate(n_strides=5)
        t = np.arange(len(traj)) * mpc.dt
        vcap = mpc.capture_lyapunov(traj)
        axes[0].plot(t, traj[:, 2], color=color, lw=1.8, label=label)
        axes[0].axhline(pr.z0, color=color, ls=":", lw=0.8, alpha=0.5)
        axes[1].plot(t, traj[:, 0], color=color, lw=1.8, label=f"{label}  (v≈{traj[:,1].mean():.2f} m/s)")
        axes[2].plot(t, vcap, color=color, lw=1.8, label=label)
        if color == "#2266cc":
            for ax in axes:
                _shade_flight(ax, t[:-1], sched)
    axes[0].set(title="COM height — periodic bounce, FLIGHT = shaded", xlabel="time (s)", ylabel="z (m)")
    axes[1].set(title="forward position — steady run", xlabel="time (s)", ylabel="x (m)")
    axes[2].set(title="capturability Lyapunov — BOUNDED every stride", xlabel="time (s)", ylabel=r"V$_{cap}$")
    axes[2].axhline(0.1, color="r", ls="--", lw=0.9)
    axes[2].text(0.02, 0.105, "recoverable region", color="r", fontsize=8)
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Running as a periodic hopping gait (centroidal MPC) — flight phases inside the "
                 "capturability region", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_OUT / "running_mpc.png", dpi=110)
    print("wrote", _OUT / "running_mpc.png")


if __name__ == "__main__":
    main()
