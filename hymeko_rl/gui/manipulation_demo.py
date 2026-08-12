"""Interactive manipulation demo GUI (matplotlib) — HyMeKo planar two-arm delivery.

Place a manipuland of any curriculum shape (coin / box / triangle / pentagon / hexagon / ellipse / capsule), click
anywhere to set the delivery target, pick a strategy (TD3 / k×n actor-critic / SAC), and press RUN to watch the two
planar arms reach a straddle grasp and carry the object to the target.

Honesty note (shown in the UI): the arms use the REAL calibrated 2R inverse kinematics and the REAL HyMeKo-generated
object geometry, but the carry is a *geometric preview* (kinematic transport), not a trained-policy physics rollout —
robust delivery of arbitrary shapes to arbitrary targets is the open research this framework targets. TD3 is the coin's
deployed policy; k×n actor-critic and SAC are the in-development architectures the strategy selector previews.

Run:  python -m hymeko_rl.gui.manipulation_demo
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button, RadioButtons

from hymeko_rl.coin_delivery.forward_displacement import _fingertip_geoms
from hymeko_rl.coin_delivery.theta_option.home_states import HOME_STATE_V1_GENERIC
from hymeko_rl.coin_delivery.theta_option.planar_arm_2r import PlanarArm2R, calibrate_arm
from hymeko_rl.coin_delivery.theta_option.planar_geometric_approach import CoinStraddleTargets
from hymeko_rl.env.object_spec import ObjectSpec, Shape, equal_area_regular_ngon_circumradius
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

# --- the shape menu (label → an ObjectSpec; all equal-area to the coin) ----------------------------------------------
_R = 0.02
SHAPES: list[tuple[str, ObjectSpec]] = [
    ("coin (circle)", ObjectSpec(shape=Shape.CYLINDER, radius=_R)),
    ("square", ObjectSpec(shape=Shape.BOX, radius=0.0177245, radius_y=0.0177245)),
    ("triangle", ObjectSpec(shape=Shape.TRIANGLE, radius=_R)),
    ("pentagon", ObjectSpec(shape=Shape.NGON, radius=_R, n_sides=5)),
    ("hexagon", ObjectSpec(shape=Shape.NGON, radius=_R, n_sides=6)),
    ("ellipse", ObjectSpec(shape=Shape.ELLIPSE, radius=0.025, radius_y=0.016)),
    ("capsule", ObjectSpec(shape=Shape.CAPSULE, radius=0.026533, radius_y=0.013266)),
]
STRATEGIES = ["TD3", "k×n actor-critic", "SAC"]
_STRATEGY_NOTE = {
    "TD3": "TD3 — the coin's deployed residual policy.",
    "k×n actor-critic": "k×n actor-critic — in development (Track C: actors × critics tensor).",
    "SAC": "SAC — the option-RL engine baseline.",
}


def object_boundary(spec: ObjectSpec, n_arc: int = 64) -> np.ndarray:
    """Closed 2-D outline (N×2) of the manipuland in its own frame, centred at the origin (yaw 0)."""
    s = spec.shape
    if s is Shape.CYLINDER:
        t = np.linspace(0, 2 * math.pi, n_arc, endpoint=False)
        return np.c_[spec.radius * np.cos(t), spec.radius * np.sin(t)]
    if s is Shape.BOX:
        rx, ry = spec.radius, (spec.radius_y if spec.radius_y is not None else spec.radius)
        return np.array([[rx, ry], [-rx, ry], [-rx, -ry], [rx, -ry]])
    if s in (Shape.TRIANGLE, Shape.NGON):
        n = 3 if s is Shape.TRIANGLE else int(spec.n_sides or 3)
        rr = equal_area_regular_ngon_circumradius(spec.radius, n)
        a = math.pi / 2 + np.arange(n) * 2 * math.pi / n           # apex up (matches the mesh generator)
        return np.c_[rr * np.cos(a), rr * np.sin(a)]
    if s is Shape.ELLIPSE:
        rx, ry = spec.radius, (spec.radius_y or spec.radius)
        t = np.linspace(0, 2 * math.pi, n_arc, endpoint=False)
        return np.c_[rx * np.cos(t), ry * np.sin(t)]
    # CAPSULE / stadium: two semicircular caps (radius b) at ±(a−b) joined by straight sides.
    a, b = spec.radius, (spec.radius_y or spec.radius)
    s0 = a - b
    right = [(s0 + b * math.cos(t), b * math.sin(t)) for t in np.linspace(-math.pi / 2, math.pi / 2, n_arc // 2)]
    left = [(-s0 + b * math.cos(t), b * math.sin(t)) for t in np.linspace(math.pi / 2, 3 * math.pi / 2, n_arc // 2)]
    return np.array(right + left)


def _place(outline: np.ndarray, centre: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return outline @ rot.T + centre


@dataclass
class Arms:
    left: PlanarArm2R
    right: PlanarArm2R

    @classmethod
    def calibrate(cls) -> "Arms":
        """Calibrate the two 2R arms from a fresh coin env's MuJoCo model (arms are object-independent)."""
        env = PlanarGraspEnv(**ObjectSpec(shape=Shape.CYLINDER, radius=_R).planar_env_kwargs())
        model = env.model
        gl, gr = _fingertip_geoms(model)
        coin = np.array([0.0, 0.16])

        def fk(q4: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
            d = mujoco.MjData(model)
            d.qpos[:4] = q4
            d.qpos[4:6] = coin
            d.qpos[6] = 0.0
            mujoco.mj_forward(model, d)
            return d.xanchor.copy(), d.geom_xpos[gl][:2].copy(), d.geom_xpos[gr][:2].copy()

        left = calibrate_arm(fk, shoulder_dof=0, elbow_dof=1, base_joint=0, tip_is_left=True)
        right = calibrate_arm(fk, shoulder_dof=2, elbow_dof=3, base_joint=2, tip_is_left=False)
        return cls(left, right)


def _reachable(arm: PlanarArm2R, tip: np.ndarray, prev: np.ndarray, tol: float = 2e-3) -> bool:
    """Can this arm's fingertip reach ``tip`` (IK round-trip within ``tol``)? (IK clamps outside the annulus.)"""
    _, _, got = arm.link_points(arm.ik_cont(tip, prev))
    return bool(np.linalg.norm(got - tip) < tol)


def _grippable(arms: Arms, off_l: np.ndarray, off_r: np.ndarray, c: np.ndarray) -> bool:
    z = np.zeros(2)
    return _reachable(arms.left, c + off_l, z) and _reachable(arms.right, c + off_r, z)


def grippable_region(arms: Arms) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Sample the both-arm delivery workspace (object centres where both fingertips can grip the coin straddle).
    Returns (points N×2, off_l, off_r)."""
    st = CoinStraddleTargets.for_object(np.zeros(2), _R).precontact()
    off_l, off_r = st.tip_left, st.tip_right
    xs, ys = np.linspace(-0.28, 0.28, 43), np.linspace(0.02, 0.34, 33)
    pts = np.array([[x, y] for y in ys for x in xs if _grippable(arms, off_l, off_r, np.array([x, y]))])
    return pts, off_l, off_r


def _plan(arms: Arms, spec: ObjectSpec, start: np.ndarray, target: np.ndarray,
          n_reach: int = 26, n_carry: int = 44) -> "tuple[list[dict[str, Any]], bool]":
    """A geometric reach→grip→carry preview: home → straddle the object → transport it as far toward the target as
    BOTH arms can still grip. Returns (frames, reached) — ``reached`` is False if the target is outside the
    both-arm workspace (the object stops at the reachable boundary, arms still gripping). Real 2R IK; kinematic carry."""
    home = HOME_STATE_V1_GENERIC.q
    qL, qR = home[0:2].copy(), home[2:4].copy()
    _, _, home_l = arms.left.link_points(qL)
    _, _, home_r = arms.right.link_points(qR)
    yaw = 0.0
    straddle = CoinStraddleTargets.for_object(start, spec.footprint_radius()).precontact()
    tip_l0, tip_r0 = straddle.tip_left, straddle.tip_right
    off_l, off_r = tip_l0 - start, tip_r0 - start                      # grip offsets relative to object centre
    frames: list[dict[str, Any]] = []

    def push(phase: str) -> None:
        _, _, tl = arms.left.link_points(qL)
        _, _, tr = arms.right.link_points(qR)
        frames.append({"qL": qL.copy(), "qR": qR.copy(), "tipL": tl, "tipR": tr,
                       "obj": obj.copy(), "yaw": yaw, "phase": phase})

    obj = start.copy()
    for i in range(n_reach + 1):                                       # HOME → straddle
        u = i / n_reach
        tl, tr = (1 - u) * home_l + u * tip_l0, (1 - u) * home_r + u * tip_r0
        qL, qR = arms.left.ik_cont(tl, qL), arms.right.ik_cont(tr, qR)
        push("reach")
    for _ in range(6):
        push("grip")                                                  # brief settle at the grasp
    reached = True
    for i in range(n_carry + 1):                                       # transport object as far as both arms can grip
        u = i / n_carry
        cand = (1 - u) * start + u * target
        if not (_reachable(arms.left, cand + off_l, qL) and _reachable(arms.right, cand + off_r, qR)):
            reached = False                                           # target beyond the both-arm workspace
            break
        obj = cand
        qL = arms.left.ik_cont(obj + off_l, qL)
        qR = arms.right.ik_cont(obj + off_r, qR)
        push("carry")
    for _ in range(8):
        push("done")
    return frames, reached


class Demo:
    def __init__(self) -> None:
        self.arms = Arms.calibrate()
        self.workspace, _, _ = grippable_region(self.arms)
        self.spec = SHAPES[0][1]
        self.strategy = STRATEGIES[0]
        self.start = np.array([0.0, 0.16])
        self.target = np.array([0.07, 0.08])            # inside the delivery workspace
        self._anim: Any = None
        self._reached = True

        self.fig = plt.figure(figsize=(11, 7))
        self.fig.canvas.manager.set_window_title("HyMeKo — manipulation demo")
        self.ax = self.fig.add_axes((0.30, 0.08, 0.66, 0.86))
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-0.42, 0.42)
        self.ax.set_ylim(-0.10, 0.48)
        self.ax.set_title("click anywhere in the workspace to set the delivery target", fontsize=10)

        ax_shape = self.fig.add_axes((0.02, 0.50, 0.24, 0.44))
        ax_shape.set_title("object shape", fontsize=10)
        self.rb_shape = RadioButtons(ax_shape, [s[0] for s in SHAPES])
        self.rb_shape.on_clicked(self._on_shape)

        ax_strat = self.fig.add_axes((0.02, 0.22, 0.24, 0.24))
        ax_strat.set_title("strategy", fontsize=10)
        self.rb_strat = RadioButtons(ax_strat, STRATEGIES)
        self.rb_strat.on_clicked(self._on_strategy)

        self.b_run = Button(self.fig.add_axes((0.02, 0.12, 0.115, 0.06)), "RUN ▶")
        self.b_run.on_clicked(self._on_run)
        self.b_reset = Button(self.fig.add_axes((0.145, 0.12, 0.115, 0.06)), "reset")
        self.b_reset.on_clicked(self._on_reset)

        self.status = self.fig.text(0.02, 0.03, "", fontsize=9, color="#333")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._draw_static()

    # -- rendering --------------------------------------------------------------------------------------------------
    def _draw_frame(self, fr: dict[str, Any] | None) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-0.42, 0.42)
        self.ax.set_ylim(-0.10, 0.48)
        # delivery workspace (object centres both arms can grip) — the region to click inside
        if len(self.workspace):
            self.ax.scatter(self.workspace[:, 0], self.workspace[:, 1], s=10, c="#8fd6a0", alpha=0.35, marker="s", edgecolors="none")
        # target zone
        self.ax.add_patch(plt.Circle(self.target, 0.04, color="#2a9d3f", alpha=0.20))
        self.ax.plot(*self.target, "x", color="#1a7d2f", ms=12, mew=3)
        obj_c = fr["obj"] if fr else self.start
        yaw = fr["yaw"] if fr else 0.0
        phase = fr["phase"] if fr else "ready"
        # object
        poly = _place(object_boundary(self.spec), obj_c, yaw)
        self.ax.add_patch(MplPolygon(poly, closed=True, facecolor="#d84b30", edgecolor="#7a2010", lw=1.5))
        # arms
        for arm, q, col in ((self.arms.left, fr["qL"] if fr else HOME_STATE_V1_GENERIC.q[0:2], "#2b6cb0"),
                            (self.arms.right, fr["qR"] if fr else HOME_STATE_V1_GENERIC.q[2:4], "#2b6cb0")):
            base, elbow, tip = arm.link_points(np.asarray(q))
            self.ax.plot([base[0], elbow[0], tip[0]], [base[1], elbow[1], tip[1]], "-", color=col, lw=4, solid_capstyle="round")
            self.ax.plot(*base, "s", color="#333", ms=7)
            self.ax.plot(*tip, "o", color="#e8a33d", ms=9)
        done = phase == "done"
        tag = ("   ✓ delivered" if (done and self._reached) else
               "   ⚠ target beyond both-arm workspace — carried as far as reachable" if done else "")
        self.ax.set_title(f"shape: {self._shape_label()}   ·   strategy: {self.strategy}   ·   phase: {phase}{tag}",
                          fontsize=10)
        self.fig.canvas.draw_idle()

    def _draw_static(self) -> None:
        self._draw_frame(None)
        self._set_status()

    def _shape_label(self) -> str:
        return next(lbl for lbl, sp in SHAPES if sp is self.spec)

    def _set_status(self, extra: str = "") -> None:
        self.status.set_text(f"{_STRATEGY_NOTE[self.strategy]}\nGeometric preview (real 2R IK + HyMeKo geometry; "
                             f"kinematic carry).  target=({self.target[0]:+.2f}, {self.target[1]:+.2f})  {extra}")

    # -- callbacks --------------------------------------------------------------------------------------------------
    def _on_shape(self, label: str) -> None:
        self.spec = next(sp for lbl, sp in SHAPES if lbl == label)
        self._stop()
        self._draw_frame(None)

    def _on_strategy(self, label: str) -> None:
        self.strategy = label
        self._set_status()
        self._draw_frame(None)

    def _on_click(self, event: Any) -> None:
        if event.inaxes is self.ax and event.xdata is not None:
            self.target = np.array([float(event.xdata), float(event.ydata)])
            self._stop()
            self._set_status()
            self._draw_frame(None)

    def _on_reset(self, _evt: Any) -> None:
        self._stop()
        self._draw_frame(None)

    def _on_run(self, _evt: Any) -> None:
        self._stop()
        try:
            frames, self._reached = _plan(self.arms, self.spec, self.start, self.target)
        except Exception as e:                                        # noqa: BLE001 — demo must not crash live
            self._set_status(f"(planning failed: {type(e).__name__})")
            return
        from matplotlib.animation import FuncAnimation
        self._anim = FuncAnimation(self.fig, lambda i: self._draw_frame(frames[i]),
                                   frames=len(frames), interval=45, repeat=False)
        self.fig.canvas.draw_idle()

    def _stop(self) -> None:
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None


def main() -> None:
    Demo()
    plt.show()


if __name__ == "__main__":
    main()
