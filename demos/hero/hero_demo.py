"""Hero demo — one authored ``.hymeko`` model -> many faithful targets, gated.

Thesis: *we make AI systems structurally accountable*. A single signed-typed
hypergraph model is compiled once, **validated** (the gate), and only then fanned
out to many emitters (URDF / SDF / MJCF / DOT / Mermaid / Gazebo). A malformed
model is **rejected before any target is trusted**.

This is a thin orchestrator over the real ``hymeko`` CLI — it does not
re-implement parse / query / transform (CLAUDE.md s6.5 #3). Scenarios are a
data-driven constant list (``SCENARIOS``); adding one is a single entry.

Run::

    cargo build -p hymeko_cli        # produces target/debug/hymeko[.exe]
    python demos/hero/hero_demo.py   # validates + emits into demos/hero/out/
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from learner_parity import structural_parity

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Gate verdict (pure — unit-tested without the binary)
# --------------------------------------------------------------------------- #
class GateStatus(Enum):
    CLEAN = "clean"
    WARNINGS = "warnings"
    REJECTED = "rejected"


@dataclass(frozen=True)
class GateVerdict:
    status: GateStatus
    detail: str

    @property
    def trusted(self) -> bool:
        """A model may be emitted iff the gate did not reject it."""
        return self.status is not GateStatus.REJECTED


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def parse_validate_output(text: str) -> GateVerdict:
    """Verdict from the validator *message alone* (no exit code available).

    Fail-closed: an unrecognised message is treated as REJECTED — never trust an
    output we cannot read. Used as the message-only fallback; the authoritative
    path is :func:`verdict_from_run`, which honours the process exit code.

    Postconditions: REJECTED on a failure/❌ marker; WARNINGS if it compiled with
    warnings; CLEAN on a success/valid marker; else REJECTED.
    """
    low = text.lower()
    if "❌" in text or "compilation failed" in low or "failed" in low:
        return GateVerdict(GateStatus.REJECTED, _first_line(text))
    if "warning" in low:
        return GateVerdict(GateStatus.WARNINGS, _first_line(text))
    if "✅" in text or "is valid" in low or "valid" in low:
        return GateVerdict(GateStatus.CLEAN, _first_line(text))
    return GateVerdict(GateStatus.REJECTED, "unrecognised validator output")


def verdict_from_run(text: str, returncode: int) -> GateVerdict:
    """Authoritative gate verdict: the CLI **exit code** decides trust.

    ``hymeko validate`` exits non-zero iff the model fails to compile/resolve
    (warnings keep exit 0). So a non-zero code is REJECTED regardless of message;
    on a zero code the message only splits CLEAN vs WARNINGS. This is more robust
    than string-matching and cannot be fooled by an odd success message.

    Preconditions: ``text`` is combined stdout+stderr; ``returncode`` is the
    process exit code.
    """
    if returncode != 0:
        return GateVerdict(GateStatus.REJECTED, _first_line(text) or f"compile failed (exit {returncode})")
    status = GateStatus.WARNINGS if "warning" in text.lower() else GateStatus.CLEAN
    return GateVerdict(status, _first_line(text))


# --------------------------------------------------------------------------- #
# Scenario model (data-driven)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Target:
    fmt: str               # format name passed to the CLI
    ext: str               # output file extension
    root_token: str        # sanity token expected in a successful emission
    via: str = "emit"      # "emit" (built-in registry) or "transform" (templates)


def emit_args(source: Path, model_name: str, target: Target) -> list[str]:
    """CLI argument list for emitting `target` from `source` (pure; unit-tested).

    `emit` covers the built-in registry (urdf/sdf/mjcf/dot/mermaid/gazebo);
    `transform` covers template-only formats such as `torch_dataflow`.
    """
    if target.via == "transform":
        return ["transform", "-t", target.fmt, str(source), "--transforms-dir", "transforms"]
    return ["emit", "-f", target.fmt, str(source), "-n", model_name]


@dataclass(frozen=True)
class HeroScenario:
    scenario_id: str
    label: str
    source: Path
    model_name: str
    targets: tuple[Target, ...]
    kind: str = "robot"   # "robot" (kinematic) or "learner" (neural)


@dataclass(frozen=True)
class EmitResult:
    target: Target
    path: Path | None
    ok: bool
    n_bytes: int
    detail: str


@dataclass(frozen=True)
class HeroReport:
    scenario: HeroScenario
    gate: GateVerdict
    emits: tuple[EmitResult, ...]


# Kinematic models all support these targets (verified 2026-06-15).
KINEMATIC_TARGETS: tuple[Target, ...] = (
    Target("urdf", "urdf", "<robot"),
    Target("sdf", "sdf", "<sdf"),
    Target("mjcf", "mjcf", "<mujoco"),
    Target("dot", "dot", "digraph"),
    Target("mermaid", "mmd", "flowchart"),
)

# The same accountable IR also generates the *learner*: a neural model →
# torch_dataflow (the Gömb cascade) → a runnable nn.Module. The demo emits +
# gates the module *source*; running it (which would pull torch) is left to the
# user, so the demo stays dependency-free.
NEURAL_TARGETS: tuple[Target, ...] = (
    Target("torch_dataflow", "py", "import torch", via="transform"),
    Target("dot", "dot", "digraph"),
)

_ROB = REPO_ROOT / "data" / "robotics"
_NN = REPO_ROOT / "data" / "nn"

SCENARIOS: tuple[HeroScenario, ...] = (
    HeroScenario(
        "fanuc_arm", "FANUC LR Mate 200iD arm",
        _ROB / "sim" / "dual_fanuc" / "fanuc_lrmate200id.hymeko", "fanuc", KINEMATIC_TARGETS,
        kind="robot",
    ),
    HeroScenario(
        "anthropomorphic_arm", "Anthropomorphic 6-DoF arm",
        _ROB / "anthropomorphic_arm.hymeko", "arm", KINEMATIC_TARGETS,
        kind="robot",
    ),
    HeroScenario(
        "simple_net", "Simple MLP (learner)",
        _NN / "simple_net.hymeko", "SimpleNet", NEURAL_TARGETS,
        kind="learner",
    ),
    HeroScenario(
        "mnist_resmlp", "MNIST ResMLP (learner)",
        _NN / "mnist_resmlp_3.hymeko", "MnistResMlp", NEURAL_TARGETS,
        kind="learner",
    ),
    # The Gömb cascade (mixed-arity HSiKAN) — the cognitive stack's judgement
    # model, authored in .hymeko and emitted as a torch.nn.Module.
    HeroScenario(
        "gomb_hsikan", "Gömb HSiKAN cascade (learner)",
        _NN / "hsikan_mixed.hymeko", "GombHSiKAN", NEURAL_TARGETS,
        kind="learner",
    ),
    # Minimal Soma vision path: patch-graph conv → walk-conv → classifier.
    HeroScenario(
        "soma_vision", "Soma vision classifier (learner)",
        _NN / "soma_vision.hymeko", "SomaVision", NEURAL_TARGETS,
        kind="learner",
    ),
)

# Self-contained malformed model (no includes, a dangling joint endpoint) for the
# gate-rejection act: `ghost_link` is referenced but never declared.
BROKEN_TWIN = """broken_robot_description {}
broken_robot {
    base_link {}
    @j: base_link {
        (+ base_link, - ghost_link);
    }
}
"""


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def find_hymeko() -> Path | None:
    """Locate the built ``hymeko`` CLI binary, or None if it isn't built yet."""
    for name in ("hymeko.exe", "hymeko"):
        cand = REPO_ROOT / "target" / "debug" / name
        if cand.exists():
            return cand
    return None


class HeroDemo:
    """Drives the ``hymeko`` CLI over a scenario: gate, then emit every target.

    Invariants: an emission is only written when the gate trusts the model AND
    the output carries the target's root token (so a silent CLI failure never
    leaves a half-baked artifact on disk).
    """

    def __init__(self, hymeko_bin: Path, out_dir: Path) -> None:
        self._bin = hymeko_bin
        self._out = out_dir

    def _run(self, *args: str) -> tuple[str, int]:
        # cwd = repo root so `--transforms-dir transforms` and relative `@"..."`
        # includes resolve regardless of where the demo is invoked from.
        proc = subprocess.run(
            [str(self._bin), *args], capture_output=True, text=True, check=False,
            cwd=str(REPO_ROOT),
        )
        return proc.stdout + proc.stderr, proc.returncode

    def validate(self, source: Path) -> GateVerdict:
        text, code = self._run("validate", str(source))
        return verdict_from_run(text, code)

    def validate_text(self, source_text: str, stem: str) -> GateVerdict:
        """Validate an inline source by writing it next to the out dir first."""
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / f"{stem}.hymeko"
        path.write_text(source_text, encoding="utf-8")
        return self.validate(path)

    def emit(self, scenario: HeroScenario, target: Target) -> EmitResult:
        out_text, code = self._run(*emit_args(scenario.source, scenario.model_name, target))
        ok = code == 0 and target.root_token in out_text
        if not ok:
            return EmitResult(target, None, False, 0, _first_line(out_text) or f"exit {code}")
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / f"{scenario.scenario_id}.{target.ext}"
        path.write_text(out_text, encoding="utf-8")
        return EmitResult(target, path, True, len(out_text.encode("utf-8")), "ok")

    def run(self, scenario: HeroScenario) -> HeroReport:
        gate = self.validate(scenario.source)
        emits = tuple(self.emit(scenario, t) for t in scenario.targets) if gate.trusted else ()
        return HeroReport(scenario, gate, emits)


def _format_report(report: HeroReport) -> str:
    lines = [f"## [{report.scenario.kind}] {report.scenario.label}  [{report.scenario.scenario_id}]",
             f"   gate: {report.gate.status.value} — {report.gate.detail}"]
    for e in report.emits:
        mark = "ok " if e.ok else "FAIL"
        size = f"{e.n_bytes:>6} B" if e.ok else e.detail
        lines.append(f"   [{mark}] {e.target.fmt:<8} {size}")
    if not report.emits and report.gate.trusted:
        lines.append("   (no targets emitted)")
    return "\n".join(lines)


def main() -> int:
    hymeko = find_hymeko()
    if hymeko is None:
        print("hymeko CLI not built. Run:  cargo build -p hymeko_cli")
        return 1
    out_dir = REPO_ROOT / "demos" / "hero" / "out"
    demo = HeroDemo(hymeko, out_dir)

    n_robot = sum(1 for s in SCENARIOS if s.kind == "robot")
    n_learner = sum(1 for s in SCENARIOS if s.kind == "learner")
    print("HyMeKo hero demo — one accountable IR, many faithful targets")
    print(f"  {n_robot} robot + {n_learner} learner scenarios, one A1–A5 gate\n")
    for scenario in SCENARIOS:
        report = demo.run(scenario)
        print(_format_report(report))
        # Structural parity for learners: the emitted torch module must realise
        # every layer the .hymeko declared (gate-framed faithfulness).
        if scenario.kind == "learner":
            torch_emit = next(
                (e for e in report.emits if e.target.fmt == "torch_dataflow" and e.ok and e.path),
                None,
            )
            if torch_emit and torch_emit.path is not None:
                par = structural_parity(
                    scenario.source.read_text(encoding="utf-8"),
                    torch_emit.path.read_text(encoding="utf-8"),
                )
                mark = "faithful" if par.faithful else f"MISSING {par.missing}"
                print(f"   parity: {mark} ({len(par.attrs)}/{par.n_layers} layers realised)")
        print()

    # The accountability act: a malformed model is rejected by the gate.
    broken = demo.validate_text(BROKEN_TWIN, "broken_twin")
    print("## Accountability gate (broken twin: dangling joint endpoint)")
    print(f"   gate: {broken.status.value} — {broken.detail}")
    print(f"   trusted for emission? {broken.trusted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
