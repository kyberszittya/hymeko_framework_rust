# Chicken-aggression detection — pipeline scaffold

End-to-end signed-graph approach to identifying aggressor birds in
group housing, built on HSiKAN.  Provides synthetic data for
algorithm iteration plus a clean drop-in point for real video / pose
tracks.

## Scope (per Éva Rampasek collaboration brief)

- Detect / track / mitigate aggressive behaviour in chicken flocks.
- Three sub-tasks the brief mentions:
  1. **Pose detection** (per-frame keypoints) — handled
     **out-of-tree** by DeepLabCut or SLEAP; we ingest their CSV
     output, no model training here.
  2. **Behaviour classification** (per-bird, per-frame: aggressive
     event vs. neutral) — handled by `interactions.detect_peck_events`
     using kinematic features, or by a small handcrafted classifier
     trained on labelled bouts.
  3. **Aggressor identification** (per-bird across a recording window:
     "is this bird the one initiating the conflicts?") — handled by
     `aggressor.HSiKANAggressorClassifier` on the signed interaction
     graph.  This is the on-thesis contribution.

## Pipeline

```
[ raw video ]
    ↓                                   (DeepLabCut / SLEAP — out of tree)
[ per-frame pose CSV ]
    ↓                                   chicken/interactions.py
[ Trajectories (T × N × 2/3) ]
    ↓                                   detect_peck_events / detect_proximity_events
[ list[InteractionEvent] ]
    ↓                                   trajectories_to_signed_graph
[ SignedGraph (vertices = birds,        signs: -1=peck, +1=proximity)
              edges = aggregated pairs)
    ↓                                   chicken/aggressor.py
[ per-bird P(aggressor) ]
```

## Quick start (synthetic)

```bash
# Simulate a flock + ground-truth aggressors.
python -m signedkan_wip.src.chicken.simulator \
    --n-birds 40 --n-frames 800 --n-aggressors 6 --seed 0

# End-to-end: simulate, build signed graph, train aggressor classifier,
# evaluate against ground-truth labels.
python -m signedkan_wip.src.chicken.aggressor \
    --n-birds 40 --n-frames 800 --n-aggressors 6 \
    --seeds 0 1 2 --hidden 16 --n-epochs 200

# Use the kinematic peck-detector (more realistic; the detector
# overfires but signs aggregate correctly).
python -m signedkan_wip.src.chicken.aggressor \
    --n-birds 40 --n-frames 800 --n-aggressors 6 \
    --seeds 0 1 2 --hidden 16 --n-epochs 200 --use-detector
```

3-seed synthetic results at $40$ birds / $800$ frames / $6$
aggressors:

| event source     | AUC  | F1m  |
|------------------|------|------|
| ground truth     | 1.00 | 1.00 |
| kinematic detector | 1.00 | 1.00 |

Smaller flocks (e.g. $20$ birds / $300$ frames / $3$ aggressors)
give noisier results (AUC $\approx 0.72$) — there are simply too few
cycles in a $13$-edge signed graph for HSiKAN to lock onto.

## Real-data plug-in point

To run on real chicken video, build a ``Trajectories`` object from
the DeepLabCut / SLEAP CSV.  Expected schema:

| frame | bird_id | x   | y   | heading_rad |
|-------|---------|-----|-----|-------------|
| 0     | 0       | 1.2 | 0.3 | 0.7         |
| 0     | 1       | 0.4 | 1.1 | -1.4        |
| 0     | 2       | ... | ... | ...         |
| 1     | 0       | ... | ... | ...         |

Position units should be metres (or any consistent unit; the
``peck_radius`` and ``proximity_radius`` thresholds in
``detect_peck_events`` / ``detect_proximity_events`` are in the same
unit).  Heading is optional but improves peck detection.

A ``Trajectories.from_csv`` helper will be added once we know the
exact column names DeepLabCut / SLEAP exports for this project.

## Tunables to revisit on real data

- `peck_radius` (default $0.18$ m) — distance threshold for a peck.
  In real chickens this is roughly the head-to-body extension.
- `approach_speed_thresh` (default $0.06$ m/s) — minimum speed along
  the bird-to-bird direction.  Real pecks are typically faster
  ($> 0.3$ m/s).  Tighten once labelled data is available.
- `proximity_radius` (default $0.45$ m) — defines a peaceful-edge.
- `proximity_kwargs.stride` (default $8$ frames) — proximity event
  subsampling.  Lower values increase graph density but can swamp the
  signal with redundant peaceful edges.
- `aggregator` (`sum` / `majority`) in
  `trajectories_to_signed_graph` — how to combine multiple events on
  the same bird-pair.  `sum` says "any peck makes the edge negative",
  `majority` requires a majority of events to be pecks.

## Unsupervised aggressor scoring (no labels needed)

The supervised classifier in `aggressor.py` needs aggressor labels
to train.  When Éva first ships raw video with no annotations,
the **unsupervised path** in `unsupervised.py` works on the
signed graph alone:

```bash
python -m signedkan_wip.src.chicken.unsupervised \
    --seeds 0 1 2 3 4 5 6 7 \
    --n-birds 40 --n-frames 800 --n-aggressors 6 \
    --hidden 8 --n-epochs 200 --use-detector
```

Four scorers reported side-by-side (8-seed synthetic, kinematic
detector, no GT events):

| scorer                                       | mean AUC | std    |
|----------------------------------------------|----------|--------|
| baseline (negative-out-degree)               | 0.638    | 0.164  |
| **Cartwright-Harary (cycle-balance fraction)** | 0.586  | **0.107**  |
| HSiKAN (self-supervised edge prediction)     | 0.653    | 0.155  |
| **rank-ensemble of all three**               | **0.693** | **0.120** |

The Cartwright-Harary scorer is **pure topology, no training**: for
each vertex $v$, score = fraction of incident k-cycles that are
*balanced* (sign product $= +1$ — Heider 1946 / Cartwright-Harary
1956).  It has the lowest variance of any single scorer — cycle
balance is structural and doesn't depend on training noise.

A counter-intuitive subtlety: the score is **balanced**, not
**unbalanced**.  An aggressor with two victims forms a triangle
$(\text{aggr}, v_1, v_2)$ with signs $(-, -, +)$ — sign product
$= +1$, **balanced**.  Aggressors generate clusters of structurally-
balanced "victim cliques", exactly the pattern Heider predicted for
stable hostile groups.  The ensemble combines this Heider-anchored
signal with the simple negative-degree baseline and the
HSiKAN-self-supervised signal for a more robust ranking.

All three are noisy at $40$ birds; longer recordings give more
peck events and higher AUC.  **Ensemble is the recommended
unsupervised default** — it's the most stable across seeds and
recovers from cases where either pure-baseline or pure-HSiKAN
fails on a particular configuration.

The supervised version reaches AUC = 1.00 on the same data as
soon as a few labelled aggressors are available, so the
unsupervised path is the bootstrap, not the final answer.

## Bootstrap from raw video — without any annotations

If Éva's only deliverable is raw video (no pose, no labels), the
practical path:

1. **Detection + tracking** — use a generic person/bird detector
   (e.g. YOLOv8 with the default COCO weights, which include a
   `bird` class) plus IoU tracking (SORT / ByteTrack).  Output:
   per-frame bounding boxes + persistent IDs.  No
   chicken-specific labels needed.
2. **Box-centroid trajectories** — convert each (frame, bird_id,
   box) into ``(frame, bird_id, x_centre, y_centre)`` and feed
   into the existing ``Trajectories(positions=..., heading=None)``.
3. **Kinematic peck detection** — `detect_peck_events` operates
   on positions only; heading is optional and only refines the
   src/dst attribution within a peck.  Without heading, the
   detector still fires on close-approach + speed-spike events.
4. **Unsupervised aggressor scoring** — run
   ``unsupervised.hsikan_self_supervised_score`` to get a
   per-bird ranking with no labels at all.
5. **Validation / labelling** — Éva scrubs the top-$k$ bird-IDs
   and confirms or corrects.  Each correction produces a
   labelled training example for the supervised classifier in
   `aggressor.py`.

For higher-fidelity pose (e.g. for distinguishing peck vs
preening), DeepLabCut needs ~20 hand-labelled frames.  But for
**aggression detection** specifically, COCO-bird-detection +
centroid trajectories appear sufficient on visual inspection of
typical commercial-flock videos.

## What's missing for production use

- **Real-data ingestion** — `Trajectories.from_csv` once we know the
  DeepLabCut / SLEAP column conventions; `Trajectories.from_yolo`
  for the bootstrap path.
- **Mitigation interface** — once an aggressor is identified, what
  intervention does the system trigger?  (Cage rotation, isolation,
  feed rebalance — out of scope until Éva confirms the actuation
  loop.)
- **Online operation** — the current pipeline is batch (process a
  recording window, then classify).  Real deployment likely wants a
  sliding-window / online variant; the SignedKAN encoder supports
  this with minimal change (just rebuild M_vt per window).
- **Pose-detection training** — only needed for fine-grained
  behaviour discrimination (preening vs feeding etc).  Aggression
  detection works without it.

## Anatomical hypergraph (HyMeKo schema)

The chicken body itself is encoded as a HyMeKo hypergraph at
`data/anatomy/chicken_anatomy.hymeko`:

- 12 keypoints (vertices): beak, eye_L/R, comb, neck, breast,
  back, tail, wing_L/R, foot_L/R
- bones (rigid, sign $+1$) — beak-eye, eye-comb, breast-back, …
- flexible joints (sign $-1$) — neck, tail base, wing roots,
  hip joints
- kinematic chains (multi-vertex hyperedges) — head, torso,
  legs, wings

```bash
hymeko validate data/anatomy/chicken_anatomy.hymeko        # ✓
python -m scripts.hymeko_to_signed_graph \
    data/anatomy/chicken_anatomy.hymeko --enumerate --ks 4 6 8 10
```

Star-expansion stats: $47$ regular edges with $\sigma$ split
$32 / 15$, mean arity $2.47$.  Cycle counts:

| $k$  | cycles | dominant structure                       |
|------|--------|-------------------------------------------|
| $4$  | $20$   | local triangles around head & shoulder    |
| $6$  | $47$   | region-level loops (head, torso, legs)    |
| $8$  | $53$   | inter-region loops (head ↔ leg via spine) |
| $10$ | $20$   | full-body chains                          |

The k=6 sweet spot matches the four named kinematic chains
(head + torso + legs + wings).  HSiKAN's $\alpha_k$ readout
on per-frame pose data should naturally weight $k=6$ — **a
testable prediction once we have real pose CSVs**.

The anatomy hypergraph is **per-bird, per-frame**.  Per-flock
interactions (peck / proximity) are stacked **on top** in the
signed-interaction graph.  The two-level structure is exactly
what HyMeKo's hierarchical-hypergraph projection is designed
for: anatomy lives in the factor view, social interactions
live in the dataflow view.

## Files

- `simulator.py` — agent-based chicken-flock simulator.
- `interactions.py` — trajectories → events → signed graph.
- `aggressor.py` — supervised HSiKAN aggressor classifier (needs
  per-bird aggressor labels).
- `unsupervised.py` — unsupervised aggressor scoring (baseline
  + Cartwright-Harary cycle-balance + HSiKAN-self-supervised +
  rank-ensemble), all label-free.
- `__init__.py` — public API re-exports.
- `README.md` — this file.

External:
- `data/anatomy/meta_anatomy.hymeko` — meta-types for animal
  anatomy hypergraphs.
- `data/anatomy/chicken_anatomy.hymeko` — chicken-specific
  12-keypoint skeleton.
