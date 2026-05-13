# Signed-graph link prediction — demo

A Gradio frontend for loading a trained HSiKAN / MixedAritySignedKAN
checkpoint and inspecting its test-set predictions on a signed graph
(Bitcoin Alpha, Bitcoin OTC, Slashdot, Epinions, …).

For each (u, v) pair, the model outputs:

- a **probability `p(+) ∈ [0, 1]`** that the edge sign is positive,
- a **predicted sign** (`+` if `p(+) > 0.5`, `−` otherwise).

See [Using the predictions](#using-the-predictions) below for the
application patterns this signal supports — vetting, anomaly detection,
cold-start ranking, uncertainty triage, and subgraph forensics.

## Layout

| File | Role |
| --- | --- |
| `checkpoint.py` | Save / load format. Pickles `state_dict + cfg + meta + (inference_bundle?) + (classifier_module?)`. Format version 2. |
| `inference.py` | `load_bundle(...)` and `predict_test_edges(...)`. Computes AUC / F1 / accuracy from the bundled test inputs. |
| `plotting.py` | matplotlib + NetworkX figures: ROC curve, α_κ bar chart, signed subgraph view. |
| `registry.py` | Loader for the catalogue YAML (the dropdown shown in the GUI). |
| `models.yaml` | Catalogue of canonical "best" checkpoints — HSiKAN, Gömb, baselines. |
| `kinematic.py` | URDF → signed-kinematic-graph bundle + URDF registry loader. |
| `kinematic_plotting.py` | 2-D NetworkX layout + cycle-arity bar chart for the Kinematic tab. |
| `kinematic_models.yaml` | Catalogue of in-repo URDFs (13 entries: parallel mechanisms + chains + humanoids + trees). |
| `kinematic_classifier.py` | Trained `GraphLevelHSiKAN` family classifier — pre-train CLI + load + predict. |
| `cliques.py` | Synthetic robot communication network generator + balanced-clique enumeration. |
| `cliques_plotting.py` | 2-D spatial layout with sign-coloured edges + clique convex-hull overlay. |
| `gui.py` | Gradio app — entry point. |

## Install

The demo depends on Gradio, matplotlib, NetworkX, pandas (declared in the
root `pyproject.toml` `demo` dependency-group):

```bash
uv sync --group ml --group demo
```

## Model catalogue

The GUI's primary input is a dropdown of canonical configurations defined
in `models.yaml`. Each entry pairs a stable id with the path it would be
saved to plus the exact command that produces it:

| Framework | Dataset | AUC (n-seed) | id |
| --- | --- | --- | --- |
| HSiKAN | bitcoin_alpha | 0.9959 (10) | `hsikan_bitcoin_alpha_optuna` |
| HSiKAN | bitcoin_otc | 0.9933 (10) | `hsikan_bitcoin_otc_optuna` |
| HSiKAN | slashdot | 0.9067 (5) | `hsikan_slashdot_edge_cr` |
| HSiKAN | slashdot | 0.9070 (5) | `hsikan_slashdot_edge_cr_kernel_on` |
| HSiKAN | epinions | 0.8464 (5) | `hsikan_epinions_edge_cr` |
| HSiKAN | bitcoin_alpha | 0.9845 (5) | `hsikan_joint_mix` |
| HymeKo-Gömb | bitcoin_otc | 0.9118 (5) | `gomb_bitcoin_otc_full` |
| HymeKo-Gömb | slashdot | 0.9031 (5) | `gomb_slashdot_full` |
| SGCN | bitcoin_alpha | — | `sgcn_bitcoin_alpha` |

Each entry in the dropdown is tagged:
- **available** (✓): the `.pt` file exists at the resolved path.
- **`[NOT TRAINED]`**: the file is missing; the info panel prints the
  exact `train_cmd` so you can reproduce it.

Override the registry location with `HYMEKO_MODEL_REGISTRY=/path/to/file.yaml`
to point at a private catalogue (e.g. one tracking your own training
output directory).

## Save a checkpoint

`run_final_cell.py` writes the demo format when given `--save-checkpoint`:

```bash
PYTHONPATH=. uv run python -m signedkan_wip.src.run_final_cell \
    --dataset bitcoin_alpha \
    --model HSiKAN \
    --hidden 8 \
    --n-epochs 80 \
    --seed 0 \
    --save-checkpoint ./checkpoints/bitcoin_alpha_seed0.pt
```

The file bundles the precomputed test-set inputs (`inference_bundle`) so
the GUI doesn't have to re-enumerate cycles / walks on load.

Supported datasets via `--save-checkpoint`: `bitcoin*`, `slashdot`,
`epinions`, `sbm_*`, `wikisigned`, `wiki_elec`, `wiki_conflict`, `mesh_*`.

## Launch

Default binds to `0.0.0.0:7860` so the GUI is reachable from any host on
the local network:

```bash
PYTHONPATH=. uv run python -m signedkan_wip.src.demo.gui
```

Connect from another machine on the LAN at `http://<server-ip>:7860`.
Find the host's address with `ip -4 addr show | grep inet` (look for
your `eth0` / `wlp*` / `enp*` interface).

### Flags

| Flag | Env var | Default | Purpose |
| --- | --- | --- | --- |
| `--host` | `HYMEKO_DEMO_HOST` | `0.0.0.0` | Bind address. Set to `127.0.0.1` to restrict to localhost only. |
| `--port` | `HYMEKO_DEMO_PORT` | `7860` | TCP port. |
| `--share` | — | off | Also expose via Gradio's `gradio.live` public tunnel. |

Examples:

```bash
# localhost only
HYMEKO_DEMO_HOST=127.0.0.1 PYTHONPATH=. uv run python -m signedkan_wip.src.demo.gui

# pick a non-default port
PYTHONPATH=. uv run python -m signedkan_wip.src.demo.gui --port 8080

# public tunnel (Gradio-hosted; URL printed on launch)
PYTHONPATH=. uv run python -m signedkan_wip.src.demo.gui --share
```

### Firewall

Linux hosts behind `ufw` need the port opened on the LAN interface:

```bash
sudo ufw allow from 192.168.0.0/16 to any port 7860 proto tcp
```

Adjust the subnet to match your network. **Do not** `ufw allow 7860`
without a source restriction — the demo has no authentication and binding
to `0.0.0.0` plus an unrestricted firewall rule exposes it to the public
internet if the host has a routable IP.

## Tabs

### Single model

1. Pick **Source: Catalogue** (default) and choose a model from the
   dropdown — or switch to **Source: Upload** and pick a `.pt`.
2. Choose device (`cpu` / `cuda` if available).
3. **Load & Predict** runs inference on the bundled test set and renders:
   - Model + dataset metadata (n_params, tuple_specs, seed, …).
   - Metrics block (AUC, F1-macro, accuracy, confusion matrix).
   - ROC curve.
   - α_κ bar chart over the mixed arities (walks vs. cycles colour-coded).
   - Top-50 test-edge predictions table (probability, prediction, true sign).
4. Pick a row index + radius and **Render subgraph** to draw the signed
   k-hop neighbourhood of that edge. Blue = positive, red = negative,
   black = the focus edge, yellow = its endpoints.

### Kinematic graph

A different application surface from signed-link prediction. Pick a
robot URDF (or upload one), and the tab parses it into a signed
kinematic graph:

- **vertices** = rigid links
- **edges** = movable joints
- **sign** = joint kind (`+` rotational / revolute / continuous,
  `−` translational / prismatic)

It renders:
- A 2-D NetworkX layout of the kinematic structure (joint type
  colour-coded).
- A cycle-arity bar chart (k = 3..6) — open chains have no bars, a
  4-bar linkage spikes at k=4, a Stewart / delta platform spikes at
  k=6.
- A summary heuristic for the mechanism topology (`open chain` /
  `tree` / `4-bar / planar parallel` / `Stewart / delta / spatial
  parallel` / `mixed serial-parallel`).

This is the structural front-end for HSiKAN's α_κ to consume.

**v0.5 adds a trained classifier.** If
`checkpoints/kinematic/family_classifier_k{4,6}.pt` are on disk, the
Analyse output also reports the **predicted kinematic family**
(`four_bar` / `stewart` / `delta_3rrr` / `serial`) with a softmax
confidence and per-class probabilities. The classifier is a small
`GraphLevelHSiKAN` per dominant arity; it discriminates Stewart from
Delta at k=6 by topology, not just cycle count.

Pre-train the classifiers with:

```bash
PYTHONPATH=. uv run python -m signedkan_wip.src.demo.kinematic_classifier \
    --arities 4 6 --n-train 80 --n-epochs 60
```

(~30 s/arity on CPU; checkpoints are ~10 kB each.)

Catalogue lives in `kinematic_models.yaml` and currently ships 8
URDFs (MoveIt arm, scaling-study chain / humanoid / tree fixtures,
plus a 2-link smoke fixture).

### Robot communication cliques

A second applied surface, distinct from signed-link prediction and from
the kinematic graph. The narrative: **a multi-robot communication
network is a signed graph**, and **a balanced clique is a stable
communication team**.

- vertices = robots
- edges    = pairwise comm attempts (within `comm_range`)
- sign     = `+` reliable / trusted; `−` jammed / lost / distrusted

Cartwright-Harary (1956): a signed graph is *balanced* iff every cycle
has an even number of negative edges — equivalently, the σ-product
around every cycle equals `+1`. HSiKAN learns σ-products natively in
its cycle pool, so this is the exact inductive bias the model wants
on this task.

The tab exposes six sliders (`n_robots`, `comm_range`, `noise_prob`,
`seed`, `min/max clique size`). Click *Generate & analyse* to:
1. spawn a deterministic synthetic network in 2-D,
2. enumerate balanced cliques up to `max_size`,
3. render the spatial layout (blue = reliable edge, red dashed =
   jammed) and a second view with balanced cliques shaded as convex
   hulls.

Use case framing: this is a **structural-balance demo for
multi-robot coordination**. Predicting which subsets of robots will
form stable teams (balanced cliques) under noisy / adversarial RF
conditions is the kind of signed-graph problem HSiKAN's architecture
is uniquely suited for. v1 of this tab is descriptive (generator +
enumeration); v0.5 will train a small HSiKAN on a corpus of synthetic
networks and predict edge signs from positions alone.

### Compare two models

A and B each have their own source toggle, so you can compare
"catalogued SOTA" against "my latest experiment upload" in one click.
The two checkpoints must be on the same dataset.

View:
- Combined ROC plot (both models overlaid).
- Side-by-side α_κ bars.
- Paired-Δ summary (Δ-AUC, Δ-F1, Δ-accuracy).

Useful for ablation comparisons (e.g. cycle-only vs. joint-mix).

## Using the predictions

### Per-dataset domain context

| Dataset | Node | Edge meaning | `+` means | `−` means |
| --- | --- | --- | --- | --- |
| `bitcoin_alpha`, `bitcoin_otc` | Trader account | Trust rating | "I trust this user" | "I don't trust" |
| `slashdot` | User account | Tag relationship | "friend" tag | "foe" tag |
| `epinions` | Reviewer | Trust between reviewers | trusts | distrusts |

### Application patterns

1. **Vetting new connections.** Given a candidate counterparty, query
   the model for `p(+)`. High-confidence positive (`> 0.95`) → extend
   trust by default; high-confidence negative (`< 0.05`) → review
   before engaging.
2. **Anomaly / abuse detection.** When the *observed* sign disagrees
   with the *predicted* sign at high confidence (e.g. observed `+` but
   `p(+) < 0.1`), the edge is structurally surprising — a candidate
   for manual review (compromised accounts, sybil rings, sockpuppetry,
   coordinated downvoting).
3. **Cold-start ranking.** Sort candidate edges by `p(+)` to surface
   connections in order of predicted positivity. The α_κ bar chart
   shows which structural primitives (k-cycles, walks) the model
   relied on — sanity-check that the mix matches the regime your graph
   resembles.
4. **Uncertainty triage.** Predictions near `p ≈ 0.5` are "I don't
   know". Route those to human review rather than auto-acting on them.
5. **Subgraph forensics.** For any edge in the predictions table, the
   subgraph viz renders the local signed neighbourhood that informed
   the decision — useful for explaining a prediction to a stakeholder
   (*"every neighbour rated this user `−`, so the model predicts `−`"*).

### How to read `p(+)`

| `p(+)` | Interpretation | Suggested action |
| ---: | --- | --- |
| `> 0.95` | High-confidence positive | Trust by default; spot-check |
| `0.80 – 0.95` | Likely positive | Default to trust; verify context |
| `0.50 – 0.80` | Weakly positive | Manual review or extra signal |
| `0.20 – 0.50` | Weakly negative | Manual review |
| `0.05 – 0.20` | Likely negative | Restrict by default |
| `< 0.05` | High-confidence negative | Restrict; investigate |

### Caveats before deploying anywhere

- These are **research checkpoints**, not production-ready models.
  Trained on small public datasets (3 k – 130 k nodes), not on your
  network.
- A model trained on Bitcoin Alpha will not generalise to a
  billion-edge platform without retraining.
- **High AUC does not imply calibration.** Apply Platt scaling /
  isotonic regression before treating probabilities as actionable
  thresholds.
- A high-confidence prediction reflects the model's view based on
  **observed structure**, not ground truth — it can be wrong.
- The evaluation uses the standard transductive split. Treat the
  AUC printed under Metrics as a benchmark number, not as expected
  real-world precision.
- **Do not use a signed-link model as the sole decision-maker for
  consequential actions** (suspending accounts, denying loans,
  rejecting transactions). Use it as one input among many.

The GUI mirrors this guide on the **How to use these predictions**
tab; module-level constants `PREDICTIONS_CALLOUT_MD` and `HOW_TO_USE_MD`
in `gui.py` are the single source of truth.

## Engineering caveats

- **Pickled `cfg` and `classifier_module`.** The checkpoint pickles the
  config dataclass and (for Bitcoin / OTC paths) the external classifier
  module. Renaming the model class or its config dataclass breaks reload.
  If you refactor, re-save your checkpoints.
- **Model is re-imported by string.** `model_class` is a qualified import
  path, e.g. `signedkan_wip.src.mixed_arity_signedkan.model.MixedAritySignedKAN`.
  Moving the class invalidates older checkpoints.
- **`inference_bundle` is required for `predict_test_edges`.** If you
  saved without it, re-train with `--save-checkpoint` to capture the
  precomputed test inputs.
- **Round-trip tests** live at
  `signedkan_wip/tests/test_demo_checkpoint.py`. They cover save / load
  with and without the bundle + external classifier.

## See also

- `signedkan_wip/src/run_final_cell.py` — training entry point.
- `signedkan_wip/src/mixed_arity_signedkan/` — model definition.
- `.env.example` (repo root) — runtime knobs that may affect what
  checkpoints actually contain (e.g. `HSIKAN_MIXED_TUPLES`,
  `HSIKAN_TOPK_MODE`).
