"""hymeko_dashboard — PyQt5 visualization of the contextual flow + HTL monitor.

Subscribes to:
- ``/hymeko/grasping/diagnostics`` — the full V_global state + edge metadata
  (published by ``grasping_context_node`` at 10 Hz as a JSON string).
- ``/joint_states`` — to show robot arm motion.
- The hymeko output topics (``stability_margin`` / ``configuration`` /
  ``contact_force``) so we display per-output time series.

Displays:
1. Live V_global state table (vertex → value), colour-coded by recency.
2. Hypergraph diagram: 11 vertices placed in 2D, 6 signed hyperedges as
   arrows; arrows light up when their source vertex updates.
3. Time-series plots of the three hymeko grasping outputs.
4. Joint-state plot of the 6 UR arm joints (proves the arm is moving).
5. **HTL monitor** — evaluates a formula (passed via CLI / parameter)
   against the live signal stream; shows robustness ρ as a gauge and
   the satisfied/violated state.

Usage::

    ros2 run hymeko_ros2_demo dashboard_node
    ros2 run hymeko_ros2_demo dashboard_node \\
        --ros-args -p htl_formula:="G(stability_margin > 0.1)"
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import rclpy
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String

# HTL package — ships in signedkan_wip/src/htl.  Add that to sys.path
# so the dashboard finds it when run under a ROS workspace.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIGNEDKAN_SRC = _REPO_ROOT / "signedkan_wip" / "src"
if str(_SIGNEDKAN_SRC) not in sys.path:
    sys.path.insert(0, str(_SIGNEDKAN_SRC))

try:
    from htl import HtlMonitor, HypergraphEvent, satisfied  # type: ignore
    _HTL_OK = True
except Exception as exc:  # noqa: BLE001
    print(f"[dashboard] HTL import failed ({exc!r}); HTL panel disabled.")
    HtlMonitor = None  # type: ignore
    HypergraphEvent = None  # type: ignore
    _HTL_OK = False


# ----------------------------------------------------------------- ROS bridge


class RosBridge(Node):
    """rclpy node that pumps the latest values into a shared state dict.

    The PyQt main thread reads from ``self.state`` on a QTimer.  The
    executor spins in a background thread.  All cross-thread access
    goes through ``self._lock``.
    """

    HYMEKO_TOPICS = (
        ("stability_margin", "/hymeko/grasping/stability_margin"),
        ("configuration",    "/hymeko/grasping/configuration"),
        ("contact_force",    "/hymeko/grasping/contact_force"),
    )

    def __init__(self) -> None:
        super().__init__("hymeko_dashboard")
        self.declare_parameter("htl_formula", "")
        self.declare_parameter("history", 200)
        self.declare_parameter("monitor_horizon", 256)

        formula_str = self.get_parameter("htl_formula").value or ""
        history = int(self.get_parameter("history").value or 200)
        horizon = int(self.get_parameter("monitor_horizon").value or 256)

        self._lock = threading.Lock()
        self._t0 = time.time()

        # Latest scalar signals (per-key floats).
        self._signals: Dict[str, float] = {}
        # Time-series buffers per signal.
        self._series: Dict[str, Deque] = {
            name: deque(maxlen=history)
            for name, _ in self.HYMEKO_TOPICS
        }
        # Joint state buffer
        self._joints: Dict[str, Deque] = {}
        self._joint_t: Deque = deque(maxlen=history)
        # V_global + edges snapshot
        self._v_global: Dict[str, float] = {}
        self._edges: List[dict] = []
        # HTL trace
        self._htl_trace: Deque = deque(maxlen=history)

        # Subscriptions
        self.create_subscription(String, "/hymeko/grasping/diagnostics",
                                  self._on_diag, 10)
        for name, topic in self.HYMEKO_TOPICS:
            self.create_subscription(Float64, topic,
                                      self._make_float_cb(name), 10)
        self.create_subscription(JointState, "/joint_states",
                                  self._on_joint_state, 10)

        # HTL monitor (optional)
        self._monitor: Optional["HtlMonitor"] = None  # type: ignore
        self._formula = formula_str
        if _HTL_OK and formula_str.strip():
            try:
                self._monitor = HtlMonitor(formula_str, horizon=horizon)
                self.get_logger().info(
                    f"HTL monitor armed: {formula_str!r} horizon={horizon}"
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"HTL formula parse failed: {exc!r}")
                self._monitor = None
        elif formula_str.strip():
            self.get_logger().warn("HTL formula given but HTL import failed")

        self.get_logger().info("dashboard ROS bridge ready")

    # ─── subscriptions ─────────────────────────────────────────────

    def _on_diag(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._v_global = {
                k: float(v) for k, v in payload.get("v_global", {}).items()
            }
            self._edges = list(payload.get("edges", []))
        # Update HTL monitor on each diagnostics update.
        if self._monitor is not None:
            ev = HypergraphEvent(  # type: ignore
                t=time.time() - self._t0,
                scalar_signals={
                    **self._v_global,
                    # also expose the hymeko output names directly (lowercase)
                },
            )
            try:
                rho = self._monitor.observe(ev)
                sat = self._monitor.satisfied()
                with self._lock:
                    self._htl_trace.append((ev.t, rho, sat))
            except KeyError:
                # Formula references a signal we haven't seen yet — skip.
                pass

    def _make_float_cb(self, name: str):
        def _cb(msg: Float64) -> None:
            v = float(msg.data)
            t = time.time() - self._t0
            with self._lock:
                self._signals[name] = v
                self._series[name].append((t, v))
        return _cb

    def _on_joint_state(self, msg: JointState) -> None:
        t = time.time() - self._t0
        with self._lock:
            if not self._joints:
                for jn in msg.name:
                    self._joints[jn] = deque(maxlen=200)
            self._joint_t.append(t)
            for jn, pos in zip(msg.name, msg.position):
                if jn not in self._joints:
                    self._joints[jn] = deque(maxlen=200)
                self._joints[jn].append((t, float(pos)))

    # ─── snapshot for the UI thread ────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "signals": dict(self._signals),
                "series": {k: list(v) for k, v in self._series.items()},
                "joints": {k: list(v) for k, v in self._joints.items()},
                "v_global": dict(self._v_global),
                "edges": list(self._edges),
                "htl_trace": list(self._htl_trace),
                "formula": self._formula,
            }


# ----------------------------------------------------------------- PyQt UI


_VERTEX_LAYOUT = {
    # Approximate 2D layout for the 11 vertices of grasping_context.
    # x increases left→right, y increases top→bottom.
    "robot_pose":       (0.08, 0.15),
    "active_tool":      (0.08, 0.30),
    "active_payload":   (0.08, 0.45),
    "mode_parallel":    (0.08, 0.60),
    "grip_force":       (0.08, 0.78),
    "tool_params":      (0.30, 0.30),
    "payload_params":   (0.30, 0.55),
    "loaded_state":     (0.52, 0.40),
    "configuration":    (0.58, 0.62),
    "force_vector":     (0.78, 0.75),
    "stability_margin": (0.88, 0.45),
}

# Per-vertex mapping to the article's symbols (Section 4 + Eq. grasp_hyperedges).
# This is the cheat-sheet the dashboard surfaces so reviewers can
# trace each live value back to the paper.
_ARTICLE_SYMBOLS = {
    "robot_pose":       ("P_r",  "robot pose",              "input"),
    "active_tool":      ("ID_t", "active tool ID",          "input"),
    "active_payload":   ("ID_p", "active payload ID",       "input"),
    "mode_parallel":    ("M_g",  "grasp mode",              "input"),
    "grip_force":       ("F_g",  "grip force",              "input"),
    "tool_params":      ("T",    "tool params (e₁ out)",    "derived"),
    "payload_params":   ("L",    "payload params (e₂ out)", "derived"),
    "loaded_state":     ("S_l",  "loaded state (e₃ out)",   "output"),
    "configuration":    ("C_g",  "grasp configuration (e₄ out)", "output"),
    "force_vector":     ("F_l",  "contact force (e₅ out)",  "output"),
    "stability_margin": ("S_g",  "grasp success/stability (e₆ out)", "output"),
}


def _value_to_color(value: float, vmin: float = 0.0, vmax: float = 1.0
                     ) -> QtGui.QColor:
    """Map a value in ``[vmin, vmax]`` to a green → amber → red ramp.

    Mid-range (0.5) is amber; low (toward 0) is green; high (toward 1)
    is red.  Out-of-range values are clamped.
    """
    v = max(vmin, min(vmax, value))
    if v < 0.5:
        f = v / 0.5
        return QtGui.QColor(int(255 * f), 220, 30)
    f = (v - 0.5) / 0.5
    return QtGui.QColor(255, int(220 * (1 - f)), 30)


# ─── Dark theme palette ────────────────────────────────────────────────
_BG       = "#1a1c20"     # main window background
_PANEL    = "#22252b"     # group boxes
_BG_VIEW  = "#15171b"     # graphics view background
_TEXT     = "#e6e6e6"     # primary text
_TEXT_DIM = "#a0a4ac"     # secondary text
_ACCENT   = "#5ac8fa"     # highlights
_GRID     = "#3a3d44"
_EDGE_PEN = "#7a818b"
_VERT_BORDER = "#dde0e3"
_VERT_BORDER_ACTIVE = "#5ac8fa"


def _apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """Apply a dark Qt palette with high-contrast text."""
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(_BG))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(_TEXT))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(_PANEL))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(_BG))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(_TEXT))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(_PANEL))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(_TEXT))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(_ACCENT))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#000000"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(_PANEL))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(_TEXT))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{ background-color: {_BG}; color: {_TEXT}; }}
        QGroupBox {{
            border: 1px solid {_GRID};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 6px;
            color: {_TEXT};
            font-weight: bold;
            background-color: {_PANEL};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: {_ACCENT};
        }}
        QLabel {{ color: {_TEXT}; }}
        QTableWidget {{
            background-color: {_PANEL};
            color: {_TEXT};
            gridline-color: {_GRID};
            border: none;
        }}
        QHeaderView::section {{
            background-color: {_BG};
            color: {_TEXT};
            border: none;
            padding: 4px;
            font-weight: bold;
        }}
    """)


class HyperedgeView(QtWidgets.QGraphicsView):
    """A QGraphicsView showing the 11 grasping-context vertices + 6 edges.

    Live colour reflects each vertex's current scalar value.  Edges
    are arrows from inputs to output(s).  The most recently updated
    vertex (largest delta vs the previous tick) gets a cyan accent
    border so the reviewer can see which signal just changed.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene_ = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setMinimumHeight(420)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(_BG_VIEW)))
        self._vertex_nodes: Dict[str, QtWidgets.QGraphicsEllipseItem] = {}
        self._vertex_name_labels: Dict[str, QtWidgets.QGraphicsTextItem] = {}
        self._vertex_value_labels: Dict[str, QtWidgets.QGraphicsTextItem] = {}
        self._edge_items: List[QtWidgets.QGraphicsLineItem] = []
        self._last_values: Dict[str, float] = {}
        self._last_active: Optional[str] = None
        self._w = 620
        self._h = 380

    def populate(self, v_global: Dict[str, float],
                  edges: List[dict]) -> None:
        """Initial / re-build of the scene."""
        self.scene_.clear()
        self._vertex_nodes.clear()
        self._vertex_name_labels.clear()
        self._vertex_value_labels.clear()
        self._edge_items.clear()
        self._last_active = None

        w, h = self._w, self._h
        self.scene_.setSceneRect(0, 0, w, h)

        # Edges (drawn first, under vertices).
        for edge in edges:
            inputs = edge.get("inputs", [])
            outputs = edge.get("outputs", [])
            for out_name in outputs:
                if out_name not in _VERTEX_LAYOUT:
                    continue
                ox, oy = _VERTEX_LAYOUT[out_name]
                for in_name in inputs:
                    if in_name not in _VERTEX_LAYOUT:
                        continue
                    ix, iy = _VERTEX_LAYOUT[in_name]
                    line = QtWidgets.QGraphicsLineItem(
                        ix * w, iy * h, ox * w, oy * h,
                    )
                    pen = QtGui.QPen(QtGui.QColor(_EDGE_PEN), 1.6)
                    line.setPen(pen)
                    self.scene_.addItem(line)
                    self._edge_items.append(line)
                # Edge name midway.
                mx = ((sum(_VERTEX_LAYOUT[i][0] for i in inputs if i in _VERTEX_LAYOUT)
                       + ox) / (len(inputs) + 1))
                my = ((sum(_VERTEX_LAYOUT[i][1] for i in inputs if i in _VERTEX_LAYOUT)
                       + oy) / (len(inputs) + 1))
                lbl = QtWidgets.QGraphicsTextItem(edge.get("name", "?"))
                lbl.setDefaultTextColor(QtGui.QColor("#c0d4e0"))
                lbl.setPos(mx * w - 26, my * h - 9)
                f = QtGui.QFont("Sans", 8, QtGui.QFont.DemiBold)
                lbl.setFont(f)
                # Subtle outline so the text reads on top of the line.
                self.scene_.addItem(lbl)

        # Vertices.
        r = 16
        for name, (x, y) in _VERTEX_LAYOUT.items():
            ellipse = QtWidgets.QGraphicsEllipseItem(
                x * w - r, y * h - r, 2 * r, 2 * r,
            )
            value = float(v_global.get(name, 0.0))
            ellipse.setBrush(_value_to_color(value))
            ellipse.setPen(QtGui.QPen(QtGui.QColor(_VERT_BORDER), 1.6))
            self.scene_.addItem(ellipse)
            self._vertex_nodes[name] = ellipse

            # Two text items: the vertex name (with article symbol)
            # and the live numeric value.  Both white on dark.
            sym, _desc, _kind = _ARTICLE_SYMBOLS.get(name, (name, "", ""))
            name_lbl = QtWidgets.QGraphicsTextItem(f"{sym} · {name}")
            name_lbl.setDefaultTextColor(QtGui.QColor("#f0f3f7"))
            name_lbl.setPos(x * w - 56, y * h + r + 2)
            f = QtGui.QFont("Sans", 8, QtGui.QFont.DemiBold)
            name_lbl.setFont(f)
            self.scene_.addItem(name_lbl)
            self._vertex_name_labels[name] = name_lbl

            val_lbl = QtWidgets.QGraphicsTextItem(f"{value:.3f}")
            val_lbl.setDefaultTextColor(QtGui.QColor("#9be1ff"))
            val_lbl.setPos(x * w - 18, y * h - 6)
            vf = QtGui.QFont("Mono", 7, QtGui.QFont.Bold)
            val_lbl.setFont(vf)
            self.scene_.addItem(val_lbl)
            self._vertex_value_labels[name] = val_lbl

        self._last_values = dict(v_global)

    def update_values(self, v_global: Dict[str, float]) -> None:
        # Find which vertex changed the most this tick.
        best_delta = 0.0
        best_name: Optional[str] = None
        for name, ellipse in self._vertex_nodes.items():
            v = float(v_global.get(name, 0.0))
            prev = self._last_values.get(name, v)
            d = abs(v - prev)
            if d > best_delta:
                best_delta = d
                best_name = name
            ellipse.setBrush(_value_to_color(v))
            # Update inline value text.
            if name in self._vertex_value_labels:
                self._vertex_value_labels[name].setPlainText(f"{v:.3f}")

        # Apply / clear the "last updated" highlight border.
        if self._last_active is not None and self._last_active in self._vertex_nodes:
            self._vertex_nodes[self._last_active].setPen(
                QtGui.QPen(QtGui.QColor(_VERT_BORDER), 1.6)
            )
        if best_name is not None and best_delta > 1e-4:
            self._vertex_nodes[best_name].setPen(
                QtGui.QPen(QtGui.QColor(_VERT_BORDER_ACTIVE), 2.8)
            )
            self._last_active = best_name

        self._last_values = dict(v_global)

    @property
    def active_vertex(self) -> Optional[str]:
        return self._last_active


class ColorLegendWidget(QtWidgets.QWidget):
    """Horizontal gradient bar with min/mid/max tick labels.

    Reads:  value 0.0 = green, 0.5 = amber, 1.0 = red.  This
    exact mapping is what the hyperedge vertices use.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(46)

    def paintEvent(self, evt) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        bar_y, bar_h = 6, 18
        # Gradient via small column sweep.
        for i in range(w):
            v = i / max(1, w - 1)
            p.fillRect(i, bar_y, 1, bar_h, _value_to_color(v))
        # Tick labels.
        p.setPen(QtGui.QPen(QtGui.QColor(_TEXT)))
        f = QtGui.QFont("Sans", 8, QtGui.QFont.DemiBold)
        p.setFont(f)
        for v, label in ((0.0, "0.0  (low)"),
                          (0.5, "0.5  (mid)"),
                          (1.0, "1.0  (high)")):
            x = int(v * (w - 1))
            tx = max(2, min(w - 60, x - 22))
            p.drawText(tx, bar_y + bar_h + 14, label)
        p.end()


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, bridge: RosBridge) -> None:
        super().__init__()
        self.bridge = bridge
        self.setWindowTitle("HyMeKo — Grasping Context Live Dashboard")
        self.resize(1400, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # 1. Header
        title = QtWidgets.QLabel(
            "<h2 style='margin:0'>HyMeKo / Grasping Context (live)</h2>"
            "<div style='color:#666'>"
            "Multi-Contextual State Representation — MDPI Technologies"
            "</div>"
        )
        layout.addWidget(title, 0, 0, 1, 3)

        # 2. Hyperedge view + legend + active-state line + article table.
        self.hyperedge = HyperedgeView()
        self.legend = ColorLegendWidget()
        self.active_label = QtWidgets.QLabel("last updated: —")
        self.active_label.setStyleSheet(
            f"color: {_ACCENT}; font-weight: bold; padding: 4px;"
        )
        gbox_h = QtWidgets.QGroupBox(
            "Hypergraph V_global · 6 signed hyperedges  ·  "
            "(cyan border = vertex just updated)"
        )
        gh = QtWidgets.QVBoxLayout(gbox_h)
        gh.addWidget(self.hyperedge)
        # Legend row.
        legend_row = QtWidgets.QHBoxLayout()
        legend_lbl = QtWidgets.QLabel(
            "<b>Colour scale</b> — vertex value mapped to:"
        )
        legend_row.addWidget(legend_lbl)
        legend_row.addWidget(self.legend, stretch=1)
        gh.addLayout(legend_row)
        gh.addWidget(self.active_label)
        layout.addWidget(gbox_h, 1, 0, 2, 1)

        # 3. Output time series (top-right) — dark theme for pyqtgraph.
        pg.setConfigOptions(antialias=True, background=_BG_VIEW, foreground=_TEXT)
        self.plots: Dict[str, pg.PlotDataItem] = {}
        plot_widget = pg.GraphicsLayoutWidget()
        for i, (name, _) in enumerate(RosBridge.HYMEKO_TOPICS):
            p = plot_widget.addPlot(row=i, col=0, title=f"/hymeko/grasping/{name}")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("bottom", "t [s]")
            curve = p.plot(pen=pg.mkPen(color=("c" if i == 0 else "m" if i == 1 else "y"), width=2))
            self.plots[name] = curve
        gbox_ts = QtWidgets.QGroupBox("Contextual outputs")
        gts = QtWidgets.QVBoxLayout(gbox_ts)
        gts.addWidget(plot_widget)
        layout.addWidget(gbox_ts, 1, 1, 1, 2)

        # 4. Joint plot (bottom-right)
        self.joint_widget = pg.GraphicsLayoutWidget()
        self.joint_plot = self.joint_widget.addPlot(title="/joint_states (UR arm)")
        self.joint_plot.showGrid(x=True, y=True, alpha=0.3)
        self.joint_plot.setLabel("bottom", "t [s]")
        self.joint_plot.setLabel("left", "joint angle [rad]")
        self.joint_curves: Dict[str, pg.PlotDataItem] = {}
        gbox_j = QtWidgets.QGroupBox("Robot joint motion")
        gj = QtWidgets.QVBoxLayout(gbox_j)
        gj.addWidget(self.joint_widget)
        layout.addWidget(gbox_j, 2, 1, 1, 1)

        # 4.5  Article ↔ Live mapping table (paper symbol → file vertex → value).
        # Bigger fonts + clearer columns + alternating row colours.
        self.article_table = QtWidgets.QTableWidget()
        self.article_table.setColumnCount(5)
        self.article_table.setHorizontalHeaderLabels(
            ["paper", "kind", "file vertex", "description", "live value"]
        )
        self.article_table.setRowCount(len(_ARTICLE_SYMBOLS))
        self.article_table.verticalHeader().setVisible(False)
        self.article_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.article_table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.article_table.setAlternatingRowColors(True)
        self.article_table.setShowGrid(False)
        # Set explicit column resize modes so columns size sensibly.
        hh = self.article_table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        # Bigger fonts overall.
        table_font = QtGui.QFont("Sans", 10)
        self.article_table.setFont(table_font)
        header_font = QtGui.QFont("Sans", 10, QtGui.QFont.Bold)
        hh.setFont(header_font)
        # Sane minimum height.
        self.article_table.setMinimumHeight(290)
        self.article_table.setStyleSheet(f"""
            QTableWidget {{
                alternate-background-color: #2a2e35;
                background-color: {_PANEL};
                color: {_TEXT};
                border: 1px solid {_GRID};
            }}
            QTableWidget::item {{ padding: 6px 8px; }}
            QHeaderView::section {{
                background-color: #2c333d;
                color: {_ACCENT};
                border: none;
                padding: 6px 8px;
                font-weight: bold;
            }}
        """)
        kind_colors = {
            "input":   QtGui.QColor("#3a6f9a"),
            "derived": QtGui.QColor("#7a6a3a"),
            "output":  QtGui.QColor("#3a7a4a"),
        }
        self._article_rows: Dict[str, int] = {}
        for row, (vertex, (sym, desc, kind)) in enumerate(_ARTICLE_SYMBOLS.items()):
            # Column 0: paper symbol — large + cyan
            sym_item = QtWidgets.QTableWidgetItem(sym)
            sym_item.setFont(QtGui.QFont("Sans", 11, QtGui.QFont.Bold))
            sym_item.setForeground(QtGui.QBrush(QtGui.QColor(_ACCENT)))
            sym_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.article_table.setItem(row, 0, sym_item)
            # Column 1: kind (input/derived/output) as a colored chip
            kind_item = QtWidgets.QTableWidgetItem(f"  {kind}  ")
            kind_item.setBackground(QtGui.QBrush(kind_colors.get(kind, QtGui.QColor("#444"))))
            kind_item.setForeground(QtGui.QBrush(QtGui.QColor("#ffffff")))
            kind_item.setTextAlignment(QtCore.Qt.AlignCenter)
            kind_item.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.DemiBold))
            self.article_table.setItem(row, 1, kind_item)
            # Column 2: file vertex name (monospace for distinctness)
            vname_item = QtWidgets.QTableWidgetItem(vertex)
            vname_item.setFont(QtGui.QFont("Mono", 10, QtGui.QFont.DemiBold))
            self.article_table.setItem(row, 2, vname_item)
            # Column 3: description
            desc_item = QtWidgets.QTableWidgetItem(desc)
            desc_item.setForeground(QtGui.QBrush(QtGui.QColor(_TEXT_DIM)))
            self.article_table.setItem(row, 3, desc_item)
            # Column 4: live value (color-coded)
            v_item = QtWidgets.QTableWidgetItem("—")
            v_item.setFont(QtGui.QFont("Mono", 11, QtGui.QFont.Bold))
            v_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.article_table.setItem(row, 4, v_item)
            self._article_rows[vertex] = row
        self.article_table.resizeRowsToContents()
        # Slightly taller rows for readability.
        for row in range(self.article_table.rowCount()):
            self.article_table.setRowHeight(row, 28)
        gbox_art = QtWidgets.QGroupBox(
            "Article ↔ Live mapping  ·  §4 MDPI Technologies paper symbols → file vertices → current values"
        )
        ga = QtWidgets.QVBoxLayout(gbox_art)
        ga.addWidget(self.article_table)
        layout.addWidget(gbox_art, 3, 0, 1, 3)

        # 5. HTL panel (bottom-right corner)
        gbox_htl = QtWidgets.QGroupBox("HTL monitor (Hypergraph Temporal Logic)")
        ghtl = QtWidgets.QVBoxLayout(gbox_htl)
        self.formula_label = QtWidgets.QLabel(
            f"<b>formula:</b> <code>{bridge._formula or '(none)'}</code>"
        )
        self.formula_label.setStyleSheet("color: #333;")
        ghtl.addWidget(self.formula_label)
        self.rho_label = QtWidgets.QLabel(
            "<div style='font-size:24pt'>ρ = —</div>"
        )
        self.rho_label.setAlignment(QtCore.Qt.AlignCenter)
        ghtl.addWidget(self.rho_label)
        self.sat_label = QtWidgets.QLabel(
            "<div style='font-size:16pt;color:gray'>not evaluated</div>"
        )
        self.sat_label.setAlignment(QtCore.Qt.AlignCenter)
        ghtl.addWidget(self.sat_label)
        # ρ time-series
        self.htl_widget = pg.GraphicsLayoutWidget()
        self.htl_plot = self.htl_widget.addPlot(title="ρ(t)")
        self.htl_plot.showGrid(x=True, y=True, alpha=0.3)
        self.htl_plot.addLine(y=0, pen=pg.mkPen(color="r", style=QtCore.Qt.DashLine))
        self.htl_curve = self.htl_plot.plot(pen=pg.mkPen(color="g", width=2))
        ghtl.addWidget(self.htl_widget)
        layout.addWidget(gbox_htl, 2, 2, 1, 1)

        layout.setColumnStretch(0, 4)
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(2, 3)

        # Refresh timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)  # 10 Hz UI refresh
        self._populated = False

    def _refresh(self) -> None:
        snap = self.bridge.snapshot()

        # Populate hypergraph view once we have edge metadata.
        if not self._populated and snap["edges"]:
            self.hyperedge.populate(snap["v_global"], snap["edges"])
            self._populated = True
        if self._populated:
            self.hyperedge.update_values(snap["v_global"])

        # Update "last updated" label.
        active = self.hyperedge.active_vertex
        if active is not None:
            sym, desc, _ = _ARTICLE_SYMBOLS.get(active, (active, "", ""))
            cur_val = float(snap["v_global"].get(active, 0.0))
            self.active_label.setText(
                f"last updated:  <b>{sym}</b> · {active}  "
                f"<span style='color:#9be1ff'>{cur_val:.3f}</span>  "
                f"<span style='color:#a0a4ac'>({desc})</span>"
            )

        # Update article-mapping table values (column 4).
        v_global = snap["v_global"]
        for vertex, row in self._article_rows.items():
            v = v_global.get(vertex)
            if v is None:
                continue
            item = self.article_table.item(row, 4)
            if item is not None:
                item.setText(f"{float(v):+.4f}")
                # Color the cell background to match the hypergraph view;
                # the green/amber/red ramp is bright enough that black
                # foreground reads cleanly on all three.
                c = _value_to_color(float(v))
                item.setBackground(QtGui.QBrush(c))
                item.setForeground(QtGui.QBrush(QtGui.QColor("#101216")))

        # Update hymeko time-series
        for name, curve in self.plots.items():
            data = snap["series"].get(name, [])
            if data:
                t = np.array([p[0] for p in data])
                y = np.array([p[1] for p in data])
                curve.setData(t, y)

        # Joints
        joints = snap["joints"]
        if joints and not self.joint_curves:
            colors = ["r", "g", "b", "c", "m", "y"]
            for i, jn in enumerate(sorted(joints.keys())):
                self.joint_curves[jn] = self.joint_plot.plot(
                    pen=pg.mkPen(color=colors[i % len(colors)], width=1.5),
                    name=jn,
                )
            self.joint_plot.addLegend()
        for jn, curve in self.joint_curves.items():
            data = joints.get(jn, [])
            if data:
                t = np.array([p[0] for p in data])
                y = np.array([p[1] for p in data])
                curve.setData(t, y)

        # HTL panel
        trace = snap["htl_trace"]
        if trace:
            t_arr = np.array([r[0] for r in trace])
            rho_arr = np.array([r[1] for r in trace])
            # Replace inf with the max finite value for plotting
            finite = rho_arr[np.isfinite(rho_arr)]
            if finite.size:
                fmax = float(finite.max())
                fmin = float(finite.min())
                rho_arr = np.where(np.isinf(rho_arr) & (rho_arr > 0), fmax, rho_arr)
                rho_arr = np.where(np.isinf(rho_arr) & (rho_arr < 0), fmin, rho_arr)
            self.htl_curve.setData(t_arr, rho_arr)
            last_rho = trace[-1][1]
            last_sat = trace[-1][2]
            rho_display = (
                "+∞" if math.isinf(last_rho) and last_rho > 0
                else "−∞" if math.isinf(last_rho) and last_rho < 0
                else f"{last_rho:+.4f}"
            )
            color = "#2a7a2a" if last_sat else "#a02020"
            self.rho_label.setText(
                f"<div style='font-size:28pt;color:{color}'>ρ = {rho_display}</div>"
            )
            self.sat_label.setText(
                f"<div style='font-size:16pt;color:{color}'>"
                f"{'SATISFIED' if last_sat else 'VIOLATED'}</div>"
            )


# ----------------------------------------------------------------- main


def main(args=None):
    rclpy.init(args=args)
    bridge = RosBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)

    # Spin rclpy in a background thread; Qt owns the main thread.
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()

    try:
        app = QtWidgets.QApplication(sys.argv)
        _apply_dark_palette(app)
        dashboard = Dashboard(bridge)
        dashboard.show()
        rc = app.exec_()
    finally:
        executor.shutdown()
        bridge.destroy_node()
        rclpy.shutdown()
        t.join(timeout=1.0)
    sys.exit(rc)


if __name__ == "__main__":
    main()
