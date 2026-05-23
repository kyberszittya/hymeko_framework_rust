"""Single-command launcher for the Stage G' triadic-rapport demo.

Starts (in order, with adequate inter-start delay):
  1. gz sim     ── data/worlds/triad_hri.sdf
  2. ros_gz_bridge parameter_bridge  ── auto-generated from triad_hri.hymeko
  3. GzObserverNode
  4. RapportPipelineNode
  5. GzRobotControllerNode
  6. VisionSidecarNode    (CV-2 — r1's onboard-camera blob detector)
  7. RapportVizNode       (RViz markers)
  8. (Optional) RViz 2     ── with the preconfigured triad_hri.rviz layout

All child processes share a single stop-signal handler — Ctrl+C
brings down the whole stack cleanly.

Run:
    source /opt/ros/kilted/setup.bash
    source .venv-rapport-ros2/bin/activate
    python -m signedkan_wip.src.rapport_ros2.launch_triad

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# ─── Default paths (overridable via CLI) ─────────────────────────────


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COALITION = REPO_ROOT / "data" / "coalitions" / "triad_hri.hymeko"
DEFAULT_WORLD = REPO_ROOT / "data" / "worlds" / "triad_hri.sdf"
DEFAULT_MODELS_DIR = REPO_ROOT / "data" / "models"
DEFAULT_RVIZ_CONFIG = REPO_ROOT / "signedkan_wip" / "src" / "rapport_ros2" \
    / "rviz" / "triad_hri.rviz"


# ─── Process supervision ─────────────────────────────────────────────


class LaunchedStack:
    """Track child processes and reap them on exit."""

    def __init__(self) -> None:
        self.procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(self, name: str, cmd: list[str], *, env: dict | None = None,
              cwd: Path | str | None = None, settle_s: float = 0.0,
              stdout_path: str | None = None) -> None:
        out = (open(stdout_path, "w")
               if stdout_path else subprocess.DEVNULL)
        p = subprocess.Popen(
            cmd, env=env or os.environ.copy(),
            cwd=str(cwd) if cwd else None,
            stdout=out, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self.procs.append((name, p))
        print(f"[launch] started {name} (PID={p.pid})", flush=True)
        if settle_s > 0:
            time.sleep(settle_s)

    def reap(self) -> None:
        for name, p in reversed(self.procs):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGINT)
                except ProcessLookupError:
                    continue
        time.sleep(1.5)
        for name, p in reversed(self.procs):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def wait_any(self) -> None:
        """Block until any child exits."""
        while True:
            for name, p in self.procs:
                rc = p.poll()
                if rc is not None:
                    print(f"[launch] child {name} exited rc={rc}; tearing down")
                    return
            time.sleep(1.0)


# ─── The launch itself ──────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Launch the Stage G' triadic-rapport demo end-to-end.",
    )
    ap.add_argument("--coalition", default=str(DEFAULT_COALITION))
    ap.add_argument("--world", default=str(DEFAULT_WORLD))
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    ap.add_argument("--rviz", action="store_true",
                     help="Also launch RViz with the preconfigured layout.")
    ap.add_argument("--rviz-config", default=str(DEFAULT_RVIZ_CONFIG))
    ap.add_argument("--bridge-yaml", default="/tmp/triad_bridge.yaml")
    ap.add_argument("--log-dir", default="/tmp",
                     help="Where each component's stdout/stderr goes.")
    ap.add_argument("--gui", action="store_true",
                     help="Launch gz sim with its GUI (default: headless).")
    args = ap.parse_args(argv)

    # Generate the bridge YAML from the coalition file.
    print("[launch] generating ros_gz_bridge config from HyMeKo coalition")
    subprocess.run(
        [sys.executable, "-m", "signedkan_wip.src.rapport_ros2.bridge_config",
         args.coalition, "-o", args.bridge_yaml],
        check=True,
    )

    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = (
        f"{args.models_dir}:/opt/ros/kilted/share:"
        f"{env.get('GZ_SIM_RESOURCE_PATH', '')}"
    )
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    stack = LaunchedStack()
    try:
        # 1) gz sim
        gz_cmd = ["gz", "sim", "-r"]
        if not args.gui:
            gz_cmd.append("-s")
        gz_cmd.append(args.world)
        stack.spawn(
            "gz_sim", gz_cmd, env=env, settle_s=4.0,
            stdout_path=f"{args.log_dir}/triad_gz.log",
        )
        # 2) parameter bridge
        stack.spawn(
            "bridge",
            ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
             "--ros-args", "-p", f"config_file:={args.bridge_yaml}"],
            env=env, settle_s=2.5,
            stdout_path=f"{args.log_dir}/triad_bridge.log",
        )
        # 3-7) Python ROS 2 nodes
        for module, settle in [
            ("gz_observer_node", 1.0),
            ("rapport_pipeline_node", 1.0),
            ("gz_robot_controller_node", 1.0),
            ("vision_sidecar_node", 1.0),
            ("rapport_viz_node", 1.0),
        ]:
            stack.spawn(
                module,
                [sys.executable, "-m",
                 f"signedkan_wip.src.rapport_ros2.{module}",
                 "--coalition", args.coalition],
                env=env, settle_s=settle,
                stdout_path=f"{args.log_dir}/triad_{module}.log",
            )

        # 8) Optional RViz
        if args.rviz:
            cfg = Path(args.rviz_config)
            rviz_cmd = ["rviz2"]
            if cfg.exists():
                rviz_cmd += ["-d", str(cfg)]
            stack.spawn(
                "rviz2", rviz_cmd, env=env, settle_s=0.5,
                stdout_path=f"{args.log_dir}/triad_rviz.log",
            )

        print("[launch] all components running. Ctrl+C to stop.")
        print(f"[launch] log dir: {args.log_dir}")
        stack.wait_any()
    except KeyboardInterrupt:
        print("[launch] SIGINT — bringing down stack")
    finally:
        stack.reap()
        print("[launch] all components stopped")


if __name__ == "__main__":
    main()
