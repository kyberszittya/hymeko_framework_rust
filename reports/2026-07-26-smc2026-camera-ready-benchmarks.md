# SMC 2026 camera-ready measurements — data run

**Date:** 2026-07-26 (JST) · **Run dir:** `paper/smc2026/data/camera_ready_2026-07-26/`
**Machine-readable output:** `paper/smc2026/data/camera_ready_measurements.json`
**Harness commit:** `538bf37a` (on branch `bench/smc2026-camera-ready`, off base `7e7e0bc4`)

This is a data-only run for the accepted IEEE SMC 2026 HyMeKo paper (submission 1953),
following `01_analysis/articles/smc2026/conf/smc_conf_01/reviews/BENCHMARK_PROMPT.md`.
No `.tex` or existing `paper/` artefact was edited; all numbers come from **one session,
one machine, one toolchain** and are **not** mixed with the prior Table I / Fig. 4 run.

## ⚠️ Commensurability caveat (load-bearing)

The paper's original numbers were produced on **x86_64 with the AVX2 lexer backend**.
This session ran on an **Apple M5 Pro (aarch64)**, where **AVX2 does not exist** — the
parser's AVX2/SSE2 paths are `#[cfg(target_arch = "x86_64")]`-gated, so the lexer runs
its **scalar** path here. The build config "release, AVX2 lexer, single-threaded" from
the paper **cannot be reproduced on this machine.** This is exactly why the prompt
mandates re-running every benchmark together: the new numbers are internally
commensurate, and must **replace** the old ones as a set — never be interleaved with the
x86/AVX2 Table I / Fig. 4 values.

## Machine / toolchain

| | |
|---|---|
| CPU | Apple M5 Pro, 18 cores, 48 GB RAM |
| OS | macOS 26.5.2 (Darwin 25.5.0, arm64) |
| rustc / cargo | 1.96.1 (31fca3adb 2026-06-26) |
| Build | release (opt-level 3, lto thin, codegen-units 1), single measurement thread, **scalar lexer** |
| Python | 3.9.6 · xacro 2.1.1 · MuJoCo 3.10.0 |

## Harness status (discovery, per prompt policy)

| Harness | Path | Runs? | Invocation |
|---|---|---|---|
| 5-fixture workflow bench | `hymeko_query/tests/bench_workflow.rs` (`integration` test) | ✅ | `cargo test -p hymeko_query --release --test integration bench_end_to_end_workflow -- --test-threads=1` |
| COO-assembly bench (`bench_coo.csv` gen) | `hymeko_core/tests/benchmarks/bench_coo_builder_random.rs` | ✅ | smoke re-run OK; used as the shape reference for the new sweep |
| Scaling harness | `hymeko_bench` bin `bench_scaling` + `scripts/scaling/fixtures` | ✅ | `cargo run --release -p hymeko_bench --bin bench_scaling -- --fixtures scripts/scaling/fixtures --reps 15` |
| Cross-format invariant suite | test assertions in `hymeko_query/tests/*` (no single callable API) | n/a | re-implemented as a structured content diff for Task 4 |
| **new** memory harness | `hymeko_bench` bin `bench_memory` | ✅ | added this run (Task 1) |
| **new** scaling-limit sweep | `hymeko_bench` bin `bench_scaling_sweep` | ✅ | added this run (Task 2) |

No harness required a compatibility repair. The memory/scaling instrumentation is **new**
(the prompt named `dhat-rs`/`stats_alloc`; implemented in-tree as a dependency-free
tracking global allocator to avoid adding a Cargo dependency to a `CORE.YAML`-pinned tree).

## Results

### Task 1 — measured memory footprint + ρ (COMPLETE)

Counts from the pipeline's canonical `HyperGraphView` (Levi incidence); ρ_measured =
modeled-IR bytes / raw `Vec<Vec<usize>>` adjacency bytes; ρ_predicted = 1 + (n+m)/(m·d̄).

| fixture | n | m | d̄ | n/m | peak heap | B/incidence | ρ_meas | ρ_pred |
|---|---|---|---|---|---|---|---|---|
| mini_arm | 73 | 13 | 0.23 | 5.62 | 115.2 KB | 1144 | 10.21 | 29.67 |
| anthropomorphic_arm | 140 | 31 | 0.87 | 4.52 | 245.3 KB | 263.6 | 7.41 | 7.33 |
| anthropomorphic_using | 140 | 31 | 0.87 | 4.52 | 244.3 KB | 263.6 | 7.41 | 7.33 |
| robot_4wh | 148 | 30 | 0.87 | 4.93 | 253.5 KB | 282.2 | 7.91 | 7.85 |
| robot_4wh_using | 148 | 30 | 0.87 | 4.93 | 253.0 KB | 282.2 | 7.91 | 7.85 |

- **Peak heap 115–254 KB** across the five fixtures (the measured memory footprint R37 asked for).
- **n/m ∈ [4.5, 5.6]** — the evidence for Prop. 4's `n = O(m log n)` assumption.
- **Honest limitation:** these fixtures are **low-arity** (d̄ ≤ 0.9 ≪ log n), so ρ is
  legitimately **far from 1** (6.7–10.2), consistent with the paper's own hedge that
  `ρ → 1` needs a high-arity family. **Do not claim ρ → 1 on the robot fixtures.**

### Task 1b — 5-fixture workflow rerun (COMPLETE)

n=30 per fixture, this session. Compile medians: mini_arm 0.253 ms, anthropomorphic_arm
0.364 ms, anthropomorphic_using 0.315 ms, robot_4wh 0.307 ms, robot_4wh_using 0.282 ms.
Full per-format medians in the JSON (`task1b_workflow_rerun`). Raw:
`raw/task1b_workflow_benchmark_raw.csv` (150 rows).

### Task 2 — synthetic scaling sweep + scalability limit (COMPLETE)

19 points, random signed hypergraphs, **10² → 1.07×10⁷ incidences**.

- **compile** vs nnz: exponent **0.896** (R² 0.987) — near-linear.
- **project** (star/Levi expansion → COO) vs nnz: exponent **0.944** (R² 0.993).
- **emit** vs nnz (URDF, robot-like chain/tree/humanoid/quadruped corpus, ≤10⁴ links):
  exponent 0.724 (R² 0.897).
- **~0.65 µs / non-zero** at the largest size, roughly constant across five decades.

**`limit_encountered`:** largest successfully processed = **10,738,629 incidences**
(65 536 V / 32 768 E), compile 6.47 s, **compile peak heap ≈ 7.73 GB**. The binding
resource is **compile-stage peak heap** (parse+intern+lower building the full IR +
interned source), growing ~linearly at **~0.72 KB/nnz**; it crossed the 7 GB soft ceiling
here. Runtime was **not** the constraint. **No hard crash / OOM / stack limit was
observed** — the next rung (~19M nnz, extrapolated ~14 GB) was deliberately **not run** to
respect the 16 GB working-memory budget. Extrapolated 16 GB ceiling ≈ 22M nnz (estimate).

### Task 3 — xacro wall-clock reference (COMPLETE)

Hand-authored `moveo.xacro` semantically **equivalent** to `anthropomorphic_arm`
(structured diff: 7 links, 7 joints, masses, joint types, parent/child, axes, and limits
all match — **zero residual differences**).

| subject | median | IQR | basis |
|---|---|---|---|
| xacro → URDF | **18.44 ms** | 0.61 ms | fresh Python subprocess, n=30 |
| HyMeKo CLI → URDF | **2.30 ms** | 0.21 ms | fresh `hymeko` subprocess, n=30 |
| HyMeKo all-six formats | **0.509 ms** | — | in-process, shared compile (workflow rerun) |

HyMeKo CLI is **~8× faster** than xacro for an equivalent URDF (both startup-inclusive),
and its **all-six cost is ~1.29× its one-format cost** in-process (the shared-compile
"1.4× vs 6×" claim).

### Task 4 — drift-detection experiment (COMPLETE — **POSITIVE, scoped**)

Property: **per-link mass + link/body set**. Pipelines from the same `H`:

- **HyMeKo** co-emits URDF + MJCF → all structural invariants **hold**.
- **Pairwise** HyMeKo→URDF→[MuJoCo 3.10.0 import/save]→MJCF → **3 invariants fire**:
  link-count parity (7→6), link name-set (base_link missing), per-link mass map.

The pairwise chain **silently drops the root link `base_link` (25 kg, 65 % of total mass)**
— MuJoCo folds the fixed-root child into the static worldbody, and no converter error is
raised. `claimed_advantage_not_observed = false` (the advantage **is** observed).

**Caveats (recorded so the paper states exactly what was shown):** (a) the pairwise chain
needed a manual empty-`<link name="world"/>` fixup just to load (MuJoCo rejected the raw
HyMeKo URDF's undefined `world` parent); (b) MuJoCo's fixed-root folding is documented
behaviour — this shows pairwise conversion is **not invariant-preserving**, not that
MuJoCo has a bug; (c) HyMeKo's MJCF is a flat body list (no in-body joints), so it keeps
the structural invariants consistent without claiming a physically richer MJCF. Scope: one
fixture, one converter, one property class.

## File ownership / commits

- **No `.tex` or existing `paper/` artefact modified.** Existing `bench_coo.csv`,
  `workflow_benchmark.csv`, `scaling_results.csv` untouched (Apr mtimes).
- New data written **only** under `paper/smc2026/data/` (the contract output +
  `camera_ready_2026-07-26/` run dir with raw CSVs, logs, and per-task harness scripts).
- **`/paper/` is `.gitignore`d in this repo by design** (paper artefacts live in the
  separate paper repo), so the data is on disk at the contract path but not tracked here.
  The committed **harness (`538bf37a`)** reproduces every number deterministically. I did
  **not** `git add -f` against the repo convention.

## Tests / static checks

- `hymeko_bench`: 14 unit/integration tests pass (3× no flakes); clippy clean on the new
  code (pre-existing `parser` lints are core, untouched); `cargo fmt --check` clean.
- Python analysis scripts: `ruff check` clean; deterministic re-run reproduces the JSON.
- JSON validates; every aggregate traced to raw (compile median, largest nnz verified).

## Completeness

- **Complete:** Task 1, Task 1b, Task 2, Task 3, Task 4. **Partial/Blocked:** none.
- **Strongest supported result:** near-linear scaling to **10.7M incidences** with a
  measured memory-bound limit (~0.72 KB/nnz), plus the scoped drift positive.
- **Most important limitation:** aarch64 scalar lexer ≠ the paper's x86/AVX2 origin run —
  the whole set must replace, not augment, the old numbers.
