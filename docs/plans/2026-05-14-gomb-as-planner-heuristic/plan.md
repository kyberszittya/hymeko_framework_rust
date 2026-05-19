# Gömb as a planner heuristic — direction stub

**Date:** 2026-05-14
**Status:** stub (intentionally brief — revisit after Niitsuma talk).

## The hypothesis

> Gömb's σ-product features over signed graphs can serve as a *learned
> heuristic* for graph search (A\*, D\*), competitive with classical
> heuristics on **non-grid topological** or **signed** planning graphs.

A* / D* lite operate on state-transition graphs and depend on a
heuristic `h(n)` estimating cost-to-go. Standard heuristics
(Euclidean, Manhattan) lose information on non-grid or signed
domains. Gömb's cycle pool computes structural features in poly time
that could feed a learned `h(n)`.

## Three angles, ranked by feasibility

### Angle A — Learned heuristic for A* on topological maps

**Status:** strongest prior-art angle. *Neural A\** (Yonetani et al.,
ICML 2021) uses CNNs over grid maps; Gömb's structural prior should
beat them on non-grid maps (corridors, hierarchical waypoints) where
cycle structure carries information.

**Setup:** train Gömb on a corpus of topological maps with
ground-truth shortest paths; use Gömb's vertex embeddings to predict
cost-to-go; plug into A*. Compare expansion counts and wall-time
against Euclidean / Manhattan / Neural-A\*-CNN baselines.

**Effort:** 1-2 weeks for a single-map-family demo.

### Angle B — Signed-graph navigation (risk-aware planning)

**Status:** the most natural fit for HSiKAN's architecture. On a map
where edges carry +/− desirability (smooth/rough terrain,
safe/contested area, ally/enemy zone), the planner seeks
shortest-path-while-staying-balanced. Gömb's cycle pool directly
captures path balance.

**Setup:** synthetic signed grid worlds + risk-aware A\* baseline
(chance-constrained MDP literature). Gömb provides a learned
balance-aware admissibility correction.

**Effort:** 2 weeks. Less prior art to compare against directly.

### Angle C — Hierarchical planning with Gömb-discovered regions

**Status:** **the angle that ties the cliques work together** —
hierarchy emerges from Gömb-detected balanced cliques, A\* runs on
the abstracted region graph. This is the integrated story for the
Niitsuma narrative — clique-detection-as-region-discovery + A* as
between-region planner.

**Setup:** large signed graph (50-500 robots / nodes), Gömb extracts
balanced cliques → abstract region graph → A* between regions.
Compare against flat A* on the original graph.

**Effort:** 2 weeks, but builds on the NP-hard Stage-2 work.

## Where it likely doesn't beat A*

- Pure shortest-path on Euclidean grids — JPS (Jump Point Search) is
  sub-microsecond.
- D\* lite incremental replanning — engineering-optimal data structures
  there; a learned model adds latency.

## Prerequisites

1. **Cliques generation/detection foundation**
   (`docs/plans/2026-05-14-cliques-generation-detection/plan.md`) —
   needed for any region-extraction work.
2. **NP-hard Stage 2** (balanced-clique extraction via Gömb features)
   — Angle C builds directly on this.
3. **Public planning-graph corpus selection** (e.g., MovingAI grid
   benchmarks for Angle A; custom topological maps for Angle B/C).

## Order of decision

After the Niitsuma talk:

1. If the talk audience reacts most strongly to the clique extraction
   story → pursue **Angle C** (it's the integrated narrative).
2. If the audience cares about classical planning benchmarks →
   **Angle A** (head-to-head against Neural A\* is a paper-shaped
   story).
3. If the audience cares about safety / risk-aware navigation →
   **Angle B** (signed-graph navigation is a less-crowded space).

## Empty-plan-dir hygiene

This is a *direction*, not a build commitment. If never pursued,
delete `docs/plans/2026-05-14-gomb-as-planner-heuristic/`.

## CORE.YAML items touched (when actually built)

Likely empty — Angles A and C reuse existing infrastructure. Angle B
*might* need a new risk-MDP baseline as a dependency. Re-assess at
build time.

## Slide-shaped narrative for the Niitsuma talk

One sentence per angle:

- "Gömb's vertex embeddings are a learned heuristic for A\* on
  topological maps — competitive with Neural A\* but with the
  inductive bias to exploit cycle structure."
- "On signed planning graphs (risk-aware navigation), Gömb's
  σ-product features give A\* a balance-aware admissibility
  correction classical heuristics cannot."
- "Gömb-detected balanced cliques are *automatically-discovered
  regions* for hierarchical planning — A\* between regions, trivial
  inside-region planning. Closes the loop between the cliques
  research and the planner narrative."
