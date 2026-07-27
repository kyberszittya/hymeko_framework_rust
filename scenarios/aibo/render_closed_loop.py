"""Plot closed-loop capturability MPC vs open-loop under a mid-run push — disturbance rejection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .closed_loop_mpc import ClosedLoopRunningMPC  # noqa: E402
from .hop_mpc import HopParams  # noqa: E402
from .running_mpc import RunningGaitMPC  # noqa: E402

_OUT = Path("reports/2026-07-27-aibo-hop")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    pr = HopParams(mass=2.0, z0=0.23, f_max=80)
    cl = ClosedLoopRunningMPC(p=pr, v_forward=0.6)
    traj, sched, _v = cl.simulate(n_strides=6, push_stride=3, push_dvx=0.4)
    t = np.arange(len(traj)) * cl.dt
    vcap = cl.capture_lyapunov(traj)
    # open-loop under the same push
    nom = RunningGaitMPC(p=pr, v_forward=0.6)
    f0, ztd, vztd = nom.plan_stride()
    prof = np.vstack([f0, np.zeros((nom.n_flight, 2))])
    x, m, g = np.array([0.0, 0.6, ztd, vztd]), pr.mass, pr.g
    ol = [x.copy()]
    for s in range(6):
        if s == 3:
            x[1] += 0.4
        for fx, fz in prof:
            x = x + cl.dt * np.array([x[1], fx / m, x[3], fz / m - g])
            ol.append(x.copy())
    ol = np.array(ol)
    t_push = 3 * (cl.n_stance + cl.n_flight) * cl.dt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t, traj[:, 1], "-", color="#2266cc", lw=2, label="closed-loop MPC (rejects push)")
    axes[0].plot(t[:len(ol)], ol[:, 1], "--", color="#cc6622", lw=2, label="open-loop plan (drifts)")
    axes[0].axhline(0.6, color="k", ls=":", lw=0.8, label="target speed")
    axes[0].axvline(t_push, color="r", lw=1, alpha=0.6)
    axes[0].text(t_push + 0.02, 0.95, "+0.4 m/s push", color="r", fontsize=8)
    axes[0].set(title="forward speed under a mid-run push", xlabel="time (s)", ylabel="vx (m/s)")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, vcap, color="#2266cc", lw=1.8, label="closed-loop capturability")
    axes[1].axvline(t_push, color="r", lw=1, alpha=0.6)
    axes[1].axhline(0.15, color="r", ls="--", lw=0.9)
    axes[1].text(0.02, 0.155, "recoverable region", color="r", fontsize=8)
    axes[1].set(title="capturability Lyapunov — BOUNDED through the push", xlabel="time (s)", ylabel=r"V$_{cap}$")
    axes[1].legend(fontsize=8)
    fig.suptitle("Closed-loop (receding-horizon) capturability MPC — feedback rejects the disturbance",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_OUT / "closed_loop_mpc.png", dpi=110)
    print("wrote", _OUT / "closed_loop_mpc.png")


if __name__ == "__main__":
    main()
