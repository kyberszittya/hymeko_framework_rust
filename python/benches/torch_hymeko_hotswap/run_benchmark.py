"""End-to-end benchmark: HyMeKo → torch_dataflow → train, with and
without entropy-feedback recompile cycles.

**What this actually tests.** The full pipeline: subprocess-call
`hymeko compile --format torch_dataflow` to emit a `nn.Module` from a
`.hymeko` description, import it, train it, and periodically invoke
the entropy/rewrite pipeline *during* training. With the current
stub runtime, the hypergraph layers are plain `nn.Linear`s, so this
measures plumbing (does compile+rebuild+weight-transfer preserve
training signal?), not the research claim (does entropy-driven
rewriting produce better representations?).

**Three arms** — same data, same seed, same budget:

1. `baseline`           — train the compiled model straight through.
   No interventions. This is the ground truth for "did the task
   converge under pure Adam on this architecture."
2. `optimizer_restart`  — every `--swap-at` epochs, rebuild the Adam
   optimizer (weights unchanged, optimizer momentum reset). Isolates
   how much of any "entropy feedback" effect is really just an
   SGDR-style optimizer restart.
3. `hymeko_recompile`   — every `--swap-at` epochs:
      a) log structural entropy via `hymeko entropy --json`,
      b) log the current rewrite proposal via `hymeko rewrite --json`,
      c) subprocess-recompile the .hymeko source to Python,
      d) import the freshly-compiled module, instantiate,
      e) `transfer_compatible_weights` from old to new,
      f) rebuild the Adam optimizer.
   This is the "entropy feedback" arm: the entropy signal is
   monitored and the pipeline goes through its full round-trip. The
   .hymeko itself doesn't change during training (the proposer's
   output on simple_net has cross edges and wouldn't round-trip if
   emitted), so arm 3 reduces to "arm 2 plus HyMeKo recompile
   overhead." Any divergence between arm 2 and arm 3 is either
   plumbing loss or compile-reproducibility drift.

**Honest scope.** The test answers: *"Does the HyMeKo
compile-and-transfer cycle preserve training signal?"* If yes
(arm 3 ≈ arm 2), the pipeline is a reliable foundation for future
entropy-driven rewrites. If no, there's a plumbing bug.

The test does **not** answer: *"Does entropy-driven rewriting
improve learning?"* That requires either (a) partial-row/col weight
transfer so shape changes preserve signal, or (b) real `ehk_torch`
layers so the "hypergraph" claim has teeth — both tracked in
`docs/quality/benchmark_plan.md`.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "python" / "ehk_torch_stub" / "src"))

from ehk_torch_stub import transfer_compatible_weights  # noqa: E402


# ─── HyMeKo compile helpers ─────────────────────────────────────────


def hymeko_bin() -> str:
    return str(_REPO / "target" / "release" / "hymeko")


def hymeko_compile_to_py(
    source: Path, class_name: str, out_py: Path,
) -> None:
    """Call `hymeko compile --format torch_dataflow` to emit a Python
    module describing the network. Raises on non-zero exit."""
    res = subprocess.run(
        [hymeko_bin(), "compile",
         "--format", "torch_dataflow",
         "-n", class_name,
         "-o", str(out_py),
         str(source)],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"hymeko compile failed (rc={res.returncode}):\n"
            f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
        )


def hymeko_query_json(subcommand: str, source: Path, *extra: str) -> dict:
    """Run `hymeko <subcommand> --json <source>` and return the parsed
    JSON. Used for entropy + rewrite monitoring."""
    res = subprocess.run(
        [hymeko_bin(), subcommand, str(source), "--json", *extra],
        capture_output=True, text=True, check=True,
    )
    return json.loads(res.stdout)


def import_generated_module(py_path: Path, class_name: str) -> Callable[[], nn.Module]:
    """Load the generated Python module and return a zero-arg factory
    for its declared class. Each call to the factory produces a fresh
    instance."""
    spec = importlib.util.spec_from_file_location(
        f"hymeko_gen_{class_name.lower()}_{py_path.stat().st_mtime_ns}",
        py_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return lambda: cls()


# ─── Task + training ────────────────────────────────────────────────


def make_dataset(n: int, seed: int, d_in: int = 3, d_out: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic regression — matches simple_net's 3→2
    shape. y0 = sin(x0) + cos(x1); y1 = tanh(x2² − x0·x1)."""
    assert d_in == 3 and d_out == 2, "current task is hard-coded 3→2"
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=g)
    y0 = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    y1 = torch.tanh(x[:, 2] ** 2 - x[:, 0] * x[:, 1])
    return x, torch.stack([y0, y1], dim=1)


@dataclass
class TrainRecord:
    arm: str
    seed: int
    epoch: int
    train_loss: float
    val_loss: float
    notes: str = ""


@dataclass
class ArmStats:
    final_val_losses: list[float] = field(default_factory=list)
    transfer_key_counts: list[int] = field(default_factory=list)
    entropy_trajectory: list[list[dict]] = field(default_factory=list)
    proposal_trajectory: list[list[dict]] = field(default_factory=list)
    wall_seconds: list[float] = field(default_factory=list)


def train_epoch(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                optim: torch.optim.Optimizer) -> float:
    model.train()
    optim.zero_grad()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    optim.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float(nn.functional.mse_loss(model(x), y).item())


# ─── Arms ───────────────────────────────────────────────────────────


def run_baseline(
    factory: Callable[[], nn.Module],
    epochs: int, lr: float, seed: int,
    x_tr, y_tr, x_va, y_va,
) -> tuple[list[TrainRecord], ArmStats]:
    torch.manual_seed(seed)
    model = factory()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    records, stats = [], ArmStats()

    t0 = time.time()
    for e in range(epochs):
        tl = train_epoch(model, x_tr, y_tr, optim)
        vl = eval_loss(model, x_va, y_va)
        records.append(TrainRecord("baseline", seed, e, tl, vl))
    stats.wall_seconds.append(time.time() - t0)
    stats.final_val_losses.append(records[-1].val_loss)
    return records, stats


def run_optimizer_restart(
    factory: Callable[[], nn.Module],
    epochs: int, swap_at: int, lr: float, seed: int,
    x_tr, y_tr, x_va, y_va,
) -> tuple[list[TrainRecord], ArmStats]:
    torch.manual_seed(seed)
    model = factory()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    records, stats = [], ArmStats()

    t0 = time.time()
    for e in range(epochs):
        if e > 0 and e % swap_at == 0:
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            records.append(TrainRecord(
                "optimizer_restart", seed, e, 0.0, 0.0, notes="restart"))
        tl = train_epoch(model, x_tr, y_tr, optim)
        vl = eval_loss(model, x_va, y_va)
        records.append(TrainRecord("optimizer_restart", seed, e, tl, vl))
    stats.wall_seconds.append(time.time() - t0)
    stats.final_val_losses.append(records[-1].val_loss)
    return records, stats


def run_hymeko_recompile(
    source: Path, class_name: str, tmp_dir: Path,
    epochs: int, swap_at: int, lr: float, seed: int,
    x_tr, y_tr, x_va, y_va,
) -> tuple[list[TrainRecord], ArmStats]:
    torch.manual_seed(seed)

    # Fresh compile for this run so we don't depend on a shared cache.
    py_path = tmp_dir / f"{class_name}_seed{seed}.py"
    hymeko_compile_to_py(source, class_name, py_path)
    factory = import_generated_module(py_path, class_name)

    model = factory()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    records, stats = [], ArmStats()

    t0 = time.time()
    per_seed_entropy: list[dict] = []
    per_seed_proposal: list[dict] = []

    for e in range(epochs):
        if e > 0 and e % swap_at == 0:
            entropy = hymeko_query_json("entropy", source)
            proposal = hymeko_query_json("rewrite", source)
            per_seed_entropy.append({"epoch": e, "scopes": entropy})
            per_seed_proposal.append({"epoch": e, "proposal": proposal})

            # Recompile from the same source (the .hymeko itself is
            # unchanged during training), rebuild, transfer weights.
            py_path = tmp_dir / f"{class_name}_seed{seed}_ep{e}.py"
            hymeko_compile_to_py(source, class_name, py_path)
            new_factory = import_generated_module(py_path, class_name)
            new_model = new_factory()
            report = transfer_compatible_weights(model, new_model)
            stats.transfer_key_counts.append(len(report.transferred))
            model = new_model
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            records.append(TrainRecord(
                "hymeko_recompile", seed, e, 0.0, 0.0,
                notes=f"recompile, transferred={len(report.transferred)}"))

        tl = train_epoch(model, x_tr, y_tr, optim)
        vl = eval_loss(model, x_va, y_va)
        records.append(TrainRecord("hymeko_recompile", seed, e, tl, vl))

    stats.wall_seconds.append(time.time() - t0)
    stats.final_val_losses.append(records[-1].val_loss)
    stats.entropy_trajectory.append(per_seed_entropy)
    stats.proposal_trajectory.append(per_seed_proposal)
    return records, stats


# ─── Orchestration ──────────────────────────────────────────────────


@dataclass
class BenchConfig:
    source: Path
    class_name: str
    seeds: int = 5
    epochs: int = 200
    swap_at: int = 50
    lr: float = 1e-2
    n_train: int = 512
    n_val: int = 128
    out_dir: Path = Path("data/benchmarks")
    tmp_dir: Path = Path("/tmp/hymeko_hotswap_bench")


def run_all(cfg: BenchConfig) -> tuple[list[TrainRecord], dict[str, ArmStats]]:
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compile once for baseline + optimizer_restart (they share a
    # factory). hymeko_recompile arm does its own per-seed compile.
    shared_py = cfg.tmp_dir / f"{cfg.class_name}_shared.py"
    hymeko_compile_to_py(cfg.source, cfg.class_name, shared_py)
    shared_factory = import_generated_module(shared_py, cfg.class_name)

    all_records: list[TrainRecord] = []
    arm_stats: dict[str, ArmStats] = {
        "baseline": ArmStats(),
        "optimizer_restart": ArmStats(),
        "hymeko_recompile": ArmStats(),
    }

    for seed in range(cfg.seeds):
        x_tr, y_tr = make_dataset(cfg.n_train, seed)
        x_va, y_va = make_dataset(cfg.n_val, seed + 100_000)

        for arm, runner in (
            ("baseline",
                lambda s=seed, xt=x_tr, yt=y_tr, xv=x_va, yv=y_va:
                    run_baseline(shared_factory, cfg.epochs, cfg.lr, s, xt, yt, xv, yv)),
            ("optimizer_restart",
                lambda s=seed, xt=x_tr, yt=y_tr, xv=x_va, yv=y_va:
                    run_optimizer_restart(shared_factory, cfg.epochs, cfg.swap_at, cfg.lr,
                                           s, xt, yt, xv, yv)),
            ("hymeko_recompile",
                lambda s=seed, xt=x_tr, yt=y_tr, xv=x_va, yv=y_va:
                    run_hymeko_recompile(cfg.source, cfg.class_name, cfg.tmp_dir,
                                          cfg.epochs, cfg.swap_at, cfg.lr,
                                          s, xt, yt, xv, yv)),
        ):
            recs, st = runner()
            all_records.extend(recs)
            arm_stats[arm].final_val_losses.extend(st.final_val_losses)
            arm_stats[arm].wall_seconds.extend(st.wall_seconds)
            arm_stats[arm].transfer_key_counts.extend(st.transfer_key_counts)
            arm_stats[arm].entropy_trajectory.extend(st.entropy_trajectory)
            arm_stats[arm].proposal_trajectory.extend(st.proposal_trajectory)

    return all_records, arm_stats


def write_csv(records: list[TrainRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "epoch", "train_loss", "val_loss", "notes"])
        for r in records:
            w.writerow([r.arm, r.seed, r.epoch,
                        f"{r.train_loss:.6f}", f"{r.val_loss:.6f}", r.notes])


def print_summary(cfg: BenchConfig, arm_stats: dict[str, ArmStats]) -> None:
    print()
    print(f"Source:    {cfg.source}")
    print(f"Class:     {cfg.class_name}")
    print(f"Seeds:     {cfg.seeds}")
    print(f"Epochs:    {cfg.epochs}  (swap-at={cfg.swap_at})")
    print()
    print(f"  {'arm':<22}  {'mean val loss':>14}  {'± std':>10}  {'wall s/run':>10}")
    for arm in ("baseline", "optimizer_restart", "hymeko_recompile"):
        s = arm_stats[arm]
        if not s.final_val_losses:
            continue
        mean_l = statistics.mean(s.final_val_losses)
        std_l  = statistics.stdev(s.final_val_losses) if len(s.final_val_losses) > 1 else 0.0
        mean_t = statistics.mean(s.wall_seconds)
        print(f"  {arm:<22}  {mean_l:>14.5f}  {std_l:>10.5f}  {mean_t:>10.2f}")

    rc = arm_stats["hymeko_recompile"]
    if rc.transfer_key_counts:
        print()
        print(f"hymeko_recompile: transferred {statistics.mean(rc.transfer_key_counts):.1f} "
              f"keys per recompile ({len(rc.transfer_key_counts)} recompiles total)")
    if rc.entropy_trajectory:
        # All seeds should log the same IR entropy since training doesn't
        # mutate the .hymeko source. Sanity-check by flattening.
        flat = [e for seed_traj in rc.entropy_trajectory for e in seed_traj]
        if flat:
            first_scope = flat[0]["scopes"][0] if flat[0]["scopes"] else None
            if first_scope:
                print(f"Structural entropy on `{first_scope['scope_name']}` during training:")
                print(f"  h_sign={first_scope['h_sign']:.4f}, "
                      f"h_total={first_scope['h_total']:.4f}  "
                      f"(static — .hymeko source doesn't change during training)")

    print()
    print("Interpretation:")
    b = arm_stats["baseline"].final_val_losses
    o = arm_stats["optimizer_restart"].final_val_losses
    h = arm_stats["hymeko_recompile"].final_val_losses
    if not (b and o and h):
        return
    mb, mo, mh = statistics.mean(b), statistics.mean(o), statistics.mean(h)
    print(f"  baseline vs optimizer_restart gap: {mo - mb:+.5f}")
    print(f"  optimizer_restart vs hymeko_recompile gap: {mh - mo:+.5f}")
    oh_gap = abs(mh - mo)
    sigma = statistics.stdev(h) if len(h) > 1 else 1.0
    if oh_gap < 0.5 * sigma:
        print(f"  ✓ hymeko_recompile ≈ optimizer_restart (gap < 0.5σ) — "
              f"the recompile + weight-transfer cycle preserves training")
        print(f"    signal; no measurable loss vs. bare optimizer restart.")
    else:
        print(f"  ! hymeko_recompile drifts from optimizer_restart by {oh_gap:.5f} "
              f"(≥ 0.5σ = {0.5 * sigma:.5f})")
        print(f"    — may indicate weight-transfer loss or compile "
              f"non-determinism; investigate.")


# ─── CLI ────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--source", type=Path,
                    default=_REPO / "data" / "nn" / "simple_net.hymeko",
                    help="HyMeKo source to compile + train.")
    ap.add_argument("--name", default="SimpleNet",
                    help="Class name for the generated nn.Module.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--swap-at", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--out-dir", type=Path, default=Path("data/benchmarks"))
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    if not Path(hymeko_bin()).exists():
        print(f"error: hymeko binary not found at {hymeko_bin()} — "
              f"run `cargo build -p hymeko_cli --release` first.", file=sys.stderr)
        return 1

    cfg = BenchConfig(
        source=args.source.resolve(),
        class_name=args.name,
        seeds=args.seeds, epochs=args.epochs, swap_at=args.swap_at,
        lr=args.lr, n_train=args.n_train, n_val=args.n_val,
        out_dir=args.out_dir,
    )

    t0 = time.time()
    records, arm_stats = run_all(cfg)
    elapsed = time.time() - t0

    if not args.no_csv:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = cfg.out_dir / f"hymeko_hotswap_{stamp}.csv"
        write_csv(records, csv_path)
        print(f"Wrote {len(records)} records to {csv_path}")

    print_summary(cfg, arm_stats)
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
