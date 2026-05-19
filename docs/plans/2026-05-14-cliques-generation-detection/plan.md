# Clique generation and detection — foundation

**Date:** 2026-05-14
**Status:** plan v1.
**Prerequisite for:** `docs/plans/2026-05-14-gomb-np-hard-approximation/plan.md`
(Stage 1 and 2 of the NP-hard approximation pivot both need
planted-clique ground truth and a benchmark harness, neither of
which exists yet).

## Why this exists

The cliques demo (`signedkan_wip/src/demo/cliques.py`) currently has:

- A **noise-based generator** that flips edge signs i.i.d. — produces
  no structural signal (we confirmed AUC = 0.5 on Gömb predictor).
- A **faction-based generator** that produces a learnable signal but
  no ground-truth balanced cliques to recover.
- **`enumerate_balanced_cliques`** — Bron-Kerbosch + balance check.
  Exact but exponential. Treated as ground truth by default, but its
  scaling makes large-network benchmarks slow.

To make any claim that "Gömb approximates max-balanced-clique" or
"Gömb recovers planted balanced cliques", we first need:

1. A **planted-clique generator** that places known balanced cliques
   of specified sizes into a network with controlled noise.
2. A **detection benchmark harness** that measures recall / precision
   / wall-time across detectors (exact, greedy, spectral, learned).
3. **Baselines** that aren't just "us vs. ourselves" — Bron-Kerbosch
   on small networks (exact), triangle-density heuristic (cheap), and
   a spectral / SDP relaxation (classical approximation).

This is foundation work. It does not on its own ship a research
result; it makes the Stage-1/2 NP-hard claims measurable.

## CORE.YAML items touched

**Empty list.** All work under `signedkan_wip/src/demo/cliques*.py`
and adjacent test / experiment files. No CORE crate, no pinned-dep
changes.

## Scope

### 1. Planted-clique generator

`signedkan_wip/src/demo/cliques_planted.py`:

```python
def make_planted_balanced_cliques(
    n_robots: int,
    clique_sizes: list[int],         # e.g. [5, 4, 3] → 3 planted cliques
    area_size: float = 10.0,
    comm_range: float = 4.0,
    noise_prob: float = 0.05,
    placement: str = "uniform",      # or "spatial_clustered" so cliques co-locate
    seed: int = 0,
) -> PlantedRobotNetworkBundle:
    """Plant K balanced cliques into a synthetic robot network.

    - Generate base RobotNetworkBundle with n_factions=2 (provides
      ambient signal for Gömb to learn from).
    - For each requested clique_size c_i:
        - pick c_i robots (uniformly or spatially clustered)
        - add ALL C(c_i, 2) internal edges with sign assignments such
          that the σ-product around every triangle is +1 (balanced)
        - sign assignment options:
            - "all positive" (easiest): every internal edge = +1
            - "split factions" (medium): edges within sub-cluster = +1,
              across = -1, with an even number of cross-edges
        - mark these edges as "planted" in the bundle for recall
          metrics later
    - Apply observation noise (noise_prob) to NON-planted edges only,
      so the planted cliques remain ground truth.
```

Returns a `PlantedRobotNetworkBundle` that extends `RobotNetworkBundle`
with:

- `planted_cliques: list[Clique]` — the cliques we actually planted.
- `planted_edge_mask: np.ndarray (n_edges,) bool` — True iff the edge
  is part of a planted clique.

### 2. Detection benchmark harness

`signedkan_wip/src/demo/cliques_bench.py`:

```python
@dataclass
class DetectorResult:
    detector_name: str
    cliques: list[Clique]
    wall_time_s: float
    peak_rss_mb: float | None

def benchmark_detectors(
    bundle: PlantedRobotNetworkBundle,
    detectors: list[Detector],
    min_size: int = 3,
    max_size: int = 8,
) -> list[DetectorResult]

def recall_against_planted(
    detected: list[Clique],
    planted: list[Clique],
    overlap_threshold: float = 0.5,
) -> dict[str, float]:
    """Jaccard-style overlap matching: each planted clique is matched
    to the detected clique with highest Jaccard overlap; recall is the
    fraction of planted cliques matched above ``overlap_threshold``.
    Precision counts detected cliques that match SOME planted one."""
```

Detectors as a Strategy interface:

- `BronKerboschDetector` — exact, exponential, ground truth on small.
- `GreedyBalancedDetector` — for each high-degree seed, grow greedily
  while σ-product stays +1. Poly.
- `TriangleDensityDetector` — rank vertices by triangle-balance score,
  expand top-K seeds via local neighbours.
- `SpectralBalancedDetector` — signed-Laplacian eigenvectors → sign
  clusters → balance check.

Each detector is a plain class with `.detect(bundle, ...) -> list[Clique]`.

### 3. Sweep + reports

`signedkan_wip/experiments/cliques_detection_sweep_2026_05_14.py`:

- For `n_robots ∈ {30, 50, 100, 200, 500}` × `clique_sizes` profiles
  × 5 seeds, run all 4 detectors and emit JSONL.
- Plot: recall vs `n_robots`, wall-time vs `n_robots`, recall vs
  noise_prob.
- Report: `reports/2026-05-14-cliques-detection-foundation.md` —
  the baseline numbers all later NP-hard claims must improve upon.

## Test strategy

- **Unit:** generator places exactly the requested clique sizes;
  planted cliques really are balanced (σ-product = +1) and really are
  cliques (all internal edges present); detector strategy classes
  return `list[Clique]` shape.
- **Integration:**
  - On a hand-built bundle with one planted 4-clique,
    `BronKerboschDetector` recovers it; `recall_against_planted`
    reports 1.0.
  - On a 30-robot bundle with [5, 4, 3] planted, all four detectors
    run end-to-end and emit results.
- **Performance:**
  - `BronKerboschDetector` on n=50 must finish in < 10 s (sanity).
  - All approximate detectors on n=200 must finish in < 5 s
    (the whole point of approximation).

## Performance budget

- Sweep: ~5 sizes × 3 clique-size profiles × 4 detectors × 5 seeds
  = 300 runs. Most ≤ 5 s; Bron-Kerbosch at n=500 may exceed budget,
  in which case it's marked TIMEOUT and excluded from that cell.
- Peak RSS: < 1 GB.
- Wall: < 1 h on CPU.

## Risk anticipation

- **Bron-Kerbosch combinatorial blow-up on dense networks.**
  Mitigation: set a hard 60 s timeout per detection call, mark
  failed cells in the JSONL.
- **Planted cliques get fragmented by noise.** If `noise_prob` is too
  high relative to clique size, the planted balanced structure is
  destroyed before the detector sees it. Mitigation: noise applies
  only to non-planted edges; the planted edges stay clean. Document
  this in the generator docstring as an honest-by-construction
  decision (not a leak).
- **Spectral baseline tricky to tune.** Signed-Laplacian has known
  failure modes on graphs without a clear bipartition.
  Mitigation: implement it from the Kunegis et al. 2010 paper
  faithfully; if it under-performs on our SBMs, that is itself the
  research story (Gömb beats spectral on signed-clique recovery).

## Rollback path

Self-contained — drop the new modules + the new experiment script.
Existing cliques demo (descriptive Bron-Kerbosch + faction predictor)
keeps working.

## Why no TikZ/PDF/Mermaid plan

Same rationale as the cliques v1 plan
(`docs/plans/2026-05-14-comm-cliques-demo/plan.md`). Foundation
benchmarking, not a CORE-touching architectural change. Upgrade to
four-format if the resulting numbers feed into a paper.

## Empty-plan-dir hygiene

If abandoned, delete `docs/plans/2026-05-14-cliques-generation-detection/`.

## Order of work

1. `cliques_planted.py` — generator + bundle dataclass.
2. `test_cliques_planted.py` — invariants.
3. `cliques_bench.py` — detector strategy interface + four detectors.
4. `test_cliques_bench.py` — detector smoke + recall metric correctness.
5. `cliques_detection_sweep_2026_05_14.py` — JSONL emitter.
6. `reports/2026-05-14-cliques-detection-foundation.md` — baseline
   numbers.
7. **GO / NO-GO for NP-hard Stage 1**: if the baselines are clean and
   recall metrics are sensible, the NP-hard approximation claim has
   ground to stand on. If not, fix the foundation before proceeding.

## Connection to the NP-hard plan

The NP-hard plan
(`docs/plans/2026-05-14-gomb-np-hard-approximation/plan.md`) requires:

- **Stage 1 (faction recovery)** — uses this plan's
  `PlantedRobotNetworkBundle` to generate networks where faction
  ground truth and clique ground truth are both known.
- **Stage 2 (balanced-clique extraction)** — uses this plan's
  `recall_against_planted` to measure whether Gömb-greedy actually
  recovers the largest balanced cliques, vs. just *some* balanced
  cliques.
- **Baselines** for Stage 1 + 2 are exactly the four detectors built
  here.

Without this foundation, the NP-hard plan's "beats baseline X" claims
have no baseline X to point at.
