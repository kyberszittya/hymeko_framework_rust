"""Gradio frontend for the demo.

Run with::

    uv sync --group ml --group demo
    PYTHONPATH=. uv run python -m signedkan_wip.src.demo.gui

Sections:

  1. **Single model** — upload a checkpoint, see model + dataset
     metadata, test-set metrics, ROC plot, α_κ bar chart, and a
     sample of per-edge predictions.
  2. **Subgraph** — pick an edge from the predictions table and
     render the local signed-graph neighbourhood.
  3. **Compare two models** — load two checkpoints, view paired metrics
     and side-by-side ROC + α plots.

Checkpoints must be saved by ``run_final_cell.py --save-checkpoint``
(includes the precomputed test-set inference bundle).
"""
from __future__ import annotations

import math
import traceback
from typing import Any

import numpy as np
import pandas as pd

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover — user-facing message
    raise SystemExit(
        "gradio is not installed. Run `uv sync --group ml --group demo` first."
    ) from e

from .inference import (
    ModelBundle, PredictionResult, load_bundle, predict_test_edges,
)
from .plotting import alpha_figure, roc_figure, subgraph_figure
from .registry import (
    ModelEntry, dropdown_choices, find_by_id, load_registry,
)
from .kinematic import (
    URDFEntry, find_urdf_by_id, load_urdf_bundle, load_urdf_registry,
    topology_signature, urdf_dropdown_choices,
)
from .kinematic_classifier import PRETRAINED_DIR, predict_family
from .kinematic_plotting import cycle_arity_figure, kinematic_graph_figure
from .cliques import (
    balance_summary, enumerate_balanced_cliques, make_robot_network,
)
from .cliques_plotting import cliques_figure, network_figure


# Short callout shown above the per-edge predictions table.
PREDICTIONS_CALLOUT_MD = (
    "**Reading the table.** `u`, `v` are the two graph nodes whose "
    "edge sign we predict. `true` is the held-out test label "
    "(`+` = trust / friend, `−` = distrust / foe). `pred` is the "
    "model's predicted sign; `p(+)` is its probability that the sign "
    "is positive — values near 0.5 mean the model is uncertain. "
    "See the **How to use these predictions** tab for the full "
    "interpretation guide and the application catalogue."
)


# Long-form guide. Module-level so the README + tests can verify a single
# source of truth.
HOW_TO_USE_MD = """## How to use these predictions

A signed-link-prediction model takes two nodes `(u, v)` from a graph
and outputs, for the (possibly unobserved) edge between them:

- a **probability `p(+) ∈ [0, 1]`** that the relationship is positive,
- a **predicted sign** (`+` if `p(+) > 0.5`, `−` otherwise).

This signal is useful wherever you need to estimate the polarity of an
unobserved relationship from the structure of observed ones.

### Per-dataset domain context

| Dataset | Node | Edge meaning | `+` means | `−` means |
| --- | --- | --- | --- | --- |
| `bitcoin_alpha`, `bitcoin_otc` | Trader account | Trust rating between traders | "I trust this user" | "I don't trust" |
| `slashdot` | User account | Tag relationship | "friend" tag | "foe" tag |
| `epinions` | Reviewer | Trust between reviewers | trusts | distrusts |

### Application patterns

1. **Vetting new connections.** Given a candidate counterparty, query the
   model for `p(+)`. High-confidence positive (`> 0.95`) → reasonable to
   extend trust by default; high-confidence negative (`< 0.05`) → review
   before engaging.

2. **Anomaly / abuse detection.** When the *observed* sign of an edge
   disagrees with the model's *predicted* sign **at high confidence**
   (observed `+` but `p(+) < 0.1`, or observed `−` but `p(+) > 0.9`), the
   edge is structurally surprising — a candidate for manual review
   (compromised accounts, sybil rings, sockpuppetry, coordinated
   downvoting).

3. **Cold-start ranking.** Sort candidate edges by `p(+)` to surface
   connections in order of predicted positivity. The α_κ bar chart shows
   which structural patterns (k-cycles `c2…c5`, walks `w2…w5`) the model
   relied on — sanity-check that the mix matches the regime your graph
   resembles.

4. **Uncertainty triage.** Predictions near `p ≈ 0.5` are the model
   saying "I don't know". Route those to human review rather than
   auto-acting on them.

5. **Subgraph forensics.** For any single edge in the predictions table,
   the **Subgraph viz** below renders the local signed neighbourhood
   that informed the decision. Useful for explaining a prediction to a
   stakeholder: *"this user is predicted negative because every
   neighbour rated them `−`"*.

### How to read `p(+)`

| `p(+)` | Interpretation | Suggested action |
| ---: | --- | --- |
| `> 0.95` | High-confidence positive | Trust by default; spot-check |
| `0.80 – 0.95` | Likely positive | Default to trust; verify context |
| `0.50 – 0.80` | Weakly positive | Manual review or extra signal |
| `0.20 – 0.50` | Weakly negative | Manual review |
| `0.05 – 0.20` | Likely negative | Restrict by default |
| `< 0.05` | High-confidence negative | Restrict; investigate |

### Caveats — read before deploying anywhere

- These are **research checkpoints**, not production-ready models. They
  were trained on small public datasets (3 k – 130 k nodes), not on your
  network.
- Predictions are only as good as the training graph. A model trained
  on Bitcoin Alpha's small-network dynamics will not generalise to a
  billion-edge social platform without retraining.
- **High AUC does not imply calibration.** The probabilities are roughly
  informative but should be calibrated (Platt scaling / isotonic
  regression) before being used as actionable thresholds.
- A high-confidence prediction reflects the model's view **based on
  observed structure**, not on ground truth. The model can be wrong.
- The current evaluation uses the standard transductive split. Treat
  the AUC printed in the Metrics block as a benchmark number, not as
  expected real-world precision.
- **Do not use a signed-link model as the sole decision-maker for
  consequential actions** (suspending an account, denying a service,
  rejecting a transaction). Use it as one input among many.

### Reproducibility

Each catalogued checkpoint embeds the inference bundle, so the
**Test AUC** printed under the metrics matches the seed-0 number in the
corresponding report (e.g.
`reports/2026-05-13-bitcoin-optuna-best-10seed.md`). The α_κ vector is
the softmax of the model's learned arity-mixture logits — it gives a
window into *why* the model decided what it did, by showing which
structural primitives it weighed most.
"""


# ─── State containers (per-session via gr.State) ────────────────────


def _format_meta(bundle: ModelBundle) -> str:
    """Markdown summary of model + dataset metadata."""
    m = bundle.meta
    lines = [
        f"**Dataset:** `{m.dataset}`",
        f"**Vertices:** {bundle.n_nodes:,}",
        f"**Edges:** {bundle.graph.edges.shape[0]:,}",
        f"**Model:** `{type(bundle.model).__name__}` "
        f"({bundle.n_params:,} params)",
        f"**Tuple specs:** "
        + ", ".join(_tuple_spec_str(s) for s in m.tuple_specs),
        f"**Seed:** {m.seed}  |  **Epochs:** {m.n_epochs}",
    ]
    if m.test_auc is not None:
        lines.append(f"**Train-time test AUC (saved):** {m.test_auc:.4f}")
    if bundle.inference_bundle is None:
        lines.append(
            "⚠ checkpoint has no `inference_bundle` — predictions disabled. "
            "Re-save via `run_final_cell.py --save-checkpoint`."
        )
    return "\n\n".join(lines)


def _tuple_spec_str(spec: Any) -> str:
    """Convert a (kind, k, walk_len) tuple spec to a label like c2 / w3."""
    try:
        kind, k, walk_len = spec[0], spec[1], spec[2]
    except (IndexError, TypeError):
        return str(spec)
    if kind == "walk":
        return f"w{walk_len}"
    return f"c{k}"


def _format_metrics(pred: PredictionResult, n_params: int) -> str:
    c = pred.confusion()
    rows = [
        f"**Test AUC:** **{pred.auc:.4f}**  |  "
        f"**F1 (macro):** {pred.f1_macro:.4f}  |  "
        f"**Accuracy:** {pred.accuracy:.4f}",
        f"**Confusion:** TP = {c['tp']:,}, TN = {c['tn']:,}, "
        f"FP = {c['fp']:,}, FN = {c['fn']:,}",
        f"**N test edges:** {pred.edges.shape[0]:,}  |  "
        f"**Params:** {n_params:,}",
    ]
    return "\n\n".join(rows)


def _predictions_dataframe(pred: PredictionResult, n: int = 50) -> "pd.DataFrame":
    """Return a sorted preview of N test predictions for the table view.

    Sorted by predicted-prob descending so the highest-confidence
    predictions appear first. The 'correct' column is True when the
    predicted sign matches the true sign.
    """
    n = min(n, pred.edges.shape[0])
    order = np.argsort(-pred.predicted_prob)[:n]
    rows = []
    for i in order:
        u, v = int(pred.edges[i, 0]), int(pred.edges[i, 1])
        true_s = "+" if pred.true_signs[i] == 1 else "−"
        pred_s = "+" if pred.predicted_sign[i] == 1 else "−"
        prob = float(pred.predicted_prob[i])
        ok = bool(pred.true_signs[i] == pred.predicted_sign[i])
        rows.append({
            "row_idx": int(i),
            "u": u, "v": v,
            "true": true_s, "pred": pred_s,
            "p(+)": round(prob, 4),
            "correct": ok,
        })
    return pd.DataFrame(rows)


# ─── Callback: load single model ────────────────────────────────────


def _resolve_source(source: str, file_obj, registry_id: str,
                     entries: list[ModelEntry]) -> tuple[str | None, str | None]:
    """Return ``(checkpoint_path, error_markdown)``.

    Exactly one of the two is non-None.
    """
    if source == "Catalogue":
        if not registry_id:
            return None, "ℹ pick a model from the catalogue dropdown."
        entry = find_by_id(entries, registry_id)
        if entry is None:
            return None, f"⚠ unknown registry id `{registry_id}`."
        if not entry.available:
            return None, (
                f"⚠ `{entry.id}` is not trained yet. Run:\n\n"
                f"```bash\n{entry.train_cmd}\n```"
            )
        return str(entry.path), None
    # Upload path.
    if file_obj is None:
        return None, "ℹ upload a checkpoint to begin."
    return file_obj.name, None


def _cb_load_single(checkpoint_path, device: str):
    """Gradio callback: load + predict in one click.

    Parameters
    ----------
    checkpoint_path
        A filesystem path string. The caller resolves it from either an
        upload (`file.name`) or a registry lookup (`entry.path`).
    """
    if checkpoint_path is None:
        return (
            "⚠ no checkpoint selected.",
            None, None, None, None, None,
        )
    try:
        bundle = load_bundle(checkpoint_path, device=device)
    except Exception as e:
        return (
            f"### Load failed\n```\n{traceback.format_exc()}\n```",
            None, None, None, None, None,
        )
    meta_md = _format_meta(bundle)
    alpha = bundle.alpha_vector()
    alpha_fig = alpha_figure(alpha, labels=bundle.tuple_labels(),
                              title=f"αₖ ({bundle.dataset})")
    if bundle.inference_bundle is None:
        return (meta_md, "ℹ predictions disabled — no inference bundle.",
                None, alpha_fig, pd.DataFrame(), bundle)
    try:
        pred = predict_test_edges(bundle)
    except Exception as e:
        return (
            meta_md,
            f"### Predict failed\n```\n{traceback.format_exc()}\n```",
            None, alpha_fig, pd.DataFrame(), bundle,
        )
    metrics_md = _format_metrics(pred, bundle.n_params)
    roc_fig = roc_figure(pred, title=f"ROC — {bundle.dataset}")
    pred_df = _predictions_dataframe(pred, n=50)
    # Stash both bundle and prediction in state.
    return meta_md, metrics_md, roc_fig, alpha_fig, pred_df, (bundle, pred)


# ─── Callback: subgraph viz from a selected row ─────────────────────


def _cb_render_subgraph(state, row_idx_input, radius_input):
    if state is None:
        return None, "ℹ load a checkpoint first."
    bundle, pred = state
    try:
        row_idx = int(row_idx_input)
    except (TypeError, ValueError):
        return None, "ℹ row_idx must be an integer."
    if row_idx < 0 or row_idx >= pred.edges.shape[0]:
        return None, f"ℹ row_idx must be in [0, {pred.edges.shape[0]})."
    u, v = int(pred.edges[row_idx, 0]), int(pred.edges[row_idx, 1])
    fig = subgraph_figure(
        edges=bundle.graph.edges,
        signs=bundle.graph.signs,
        focus_u=u, focus_v=v,
        pred_prob=float(pred.predicted_prob[row_idx]),
        true_sign=int(pred.true_signs[row_idx]),
        radius=int(radius_input),
        title=f"{bundle.dataset} subgraph (radius={int(radius_input)})",
    )
    return fig, f"edge ({u}, {v})  row_idx={row_idx}"


# ─── Callback: two-model comparison ─────────────────────────────────


def _cb_compare(path_a: str | None, path_b: str | None, device: str):
    """Two-model comparison callback. Paths resolved by the caller."""
    if path_a is None or path_b is None:
        return (
            "ℹ choose A and B (catalogue or upload) to compare.",
            None, None, None,
        )
    try:
        ba = load_bundle(path_a, device=device)
        bb = load_bundle(path_b, device=device)
    except Exception:
        return (
            f"### Load failed\n```\n{traceback.format_exc()}\n```",
            None, None, None,
        )
    if ba.dataset != bb.dataset:
        return (
            f"⚠ dataset mismatch: A on `{ba.dataset}`, B on `{bb.dataset}`.",
            None, None, None,
        )
    if ba.inference_bundle is None or bb.inference_bundle is None:
        return (
            "⚠ one of the checkpoints has no inference bundle.",
            None, None, None,
        )
    pa = predict_test_edges(ba)
    pb = predict_test_edges(bb)
    delta = pa.auc - pb.auc
    summary = "\n\n".join([
        f"**Dataset:** `{ba.dataset}`",
        f"**A:** AUC **{pa.auc:.4f}**, F1 {pa.f1_macro:.4f}, "
        f"params {ba.n_params:,}, model `{type(ba.model).__name__}`",
        f"**B:** AUC **{pb.auc:.4f}**, F1 {pb.f1_macro:.4f}, "
        f"params {bb.n_params:,}, model `{type(bb.model).__name__}`",
        f"**Δ (A − B):** **{delta:+.4f}**",
    ])

    # Combined ROC.
    import matplotlib.pyplot as plt
    fig_roc, ax = plt.subplots(figsize=(5, 5))
    fpr_a, tpr_a = pa.roc_curve_xy
    fpr_b, tpr_b = pb.roc_curve_xy
    ax.plot(fpr_a, tpr_a, lw=2, label=f"A  AUC={pa.auc:.4f}")
    ax.plot(fpr_b, tpr_b, lw=2, label=f"B  AUC={pb.auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, alpha=0.5)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — A vs B on {ba.dataset}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig_roc.tight_layout()

    # α side-by-side.
    aa = ba.alpha_vector()
    ab = bb.alpha_vector()
    if aa is None and ab is None:
        fig_alpha = alpha_figure(None)
    else:
        n_max = max(
            len(aa) if aa is not None else 0,
            len(ab) if ab is not None else 0,
        )

        def _pad(a):
            if a is None:
                return np.zeros(n_max)
            pad_n = n_max - len(a)
            return np.concatenate([a, np.zeros(pad_n)]) if pad_n > 0 else a

        aa_p, ab_p = _pad(aa), _pad(ab)
        labels = (ba.tuple_labels()
                  if len(ba.tuple_labels()) == n_max
                  else [f"k{i}" for i in range(n_max)])
        fig_alpha, ax = plt.subplots(figsize=(7, 4))
        xs = np.arange(n_max)
        w = 0.4
        ax.bar(xs - w / 2, aa_p, w, label="A", color="#4472C4", edgecolor="black")
        ax.bar(xs + w / 2, ab_p, w, label="B", color="#ED7D31", edgecolor="black")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("α (softmax weight)")
        ax.set_title("αₖ — A vs B")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig_alpha.tight_layout()
    return summary, fig_roc, fig_alpha, (ba, bb, pa, pb)


# ─── Build the Gradio app ───────────────────────────────────────────


def _format_entry_summary(entry: ModelEntry) -> str:
    """Markdown for the catalogue info panel under the dropdown."""
    lines = [f"### {entry.label}"]
    auc = entry.auc
    if auc is not None:
        std = entry.metrics.get("test_auc_std")
        n_seeds = entry.n_seeds
        if std is not None:
            lines.append(
                f"**AUC:** {auc:.4f} ± {float(std):.4f}"
                + (f" (n={n_seeds} seeds)" if n_seeds else "")
            )
        else:
            lines.append(f"**AUC:** {auc:.4f}"
                         + (f" (n={n_seeds} seeds)" if n_seeds else ""))
    bits = [f"`{entry.framework}` on `{entry.dataset}`"]
    if entry.n_params:
        bits.append(f"{entry.n_params:,} params")
    lines.append(" · ".join(bits))
    if entry.notes:
        lines.append(entry.notes)
    if not entry.available:
        lines.append(
            "**Status:** ⚠ not trained yet. Train with:\n\n"
            f"```bash\n{entry.train_cmd}\n```"
        )
    else:
        lines.append(f"**Status:** ✓ available at `{entry.path}`")
    if entry.report:
        lines.append(f"**Report:** `{entry.report}`")
    return "\n\n".join(lines)


def _toggle_source(source: str):
    """Show/hide catalogue + upload widgets when the radio flips."""
    if source == "Catalogue":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


def _on_catalogue_pick(registry_id: str, entries: list[ModelEntry]) -> str:
    if not registry_id:
        return "ℹ pick a model from the dropdown above."
    entry = find_by_id(entries, registry_id)
    if entry is None:
        return f"⚠ unknown id `{registry_id}`."
    return _format_entry_summary(entry)


def build_app():
    entries = load_registry()
    choices = dropdown_choices(entries)
    default_id = choices[0][1] if choices else None
    catalogue_available = bool(entries)
    default_source = "Catalogue" if catalogue_available else "Upload"

    with gr.Blocks(title="HyMeKo signed link prediction — demo") as app:
        gr.Markdown(
            "# HyMeKo signed-link-prediction demo\n"
            "Pick a catalogued model from the dropdown — HSiKAN, HymeKo-Gömb, "
            "or a baseline — or upload your own `.pt` saved by "
            "`run_final_cell.py --save-checkpoint`. Inspect the model's "
            "predictions on the test split."
        )
        if catalogue_available:
            n_avail = sum(1 for e in entries if e.available)
            gr.Markdown(
                f"**Catalogue:** {len(entries)} entries "
                f"({n_avail} available, {len(entries) - n_avail} not yet trained). "
                f"Edit `signedkan_wip/src/demo/models.yaml` to add more, or "
                f"set `HYMEKO_MODEL_REGISTRY=...` to use a different file."
            )
        else:
            gr.Markdown(
                "ℹ No model registry found at "
                "`signedkan_wip/src/demo/models.yaml`. Falling back to "
                "upload-only mode."
            )

        with gr.Tabs():
            with gr.Tab("Single model"):
                state_single = gr.State(value=None)

                source_single = gr.Radio(
                    choices=["Catalogue", "Upload"],
                    value=default_source,
                    label="Checkpoint source",
                )
                with gr.Group(visible=(default_source == "Catalogue")) as cat_group_s:
                    cat_dd_s = gr.Dropdown(
                        choices=choices, value=default_id,
                        label="Catalogued model",
                        info="Pick from the canonical 'best' configs. "
                             "[NOT TRAINED] entries print their train command.",
                    )
                    cat_info_s = gr.Markdown(
                        _on_catalogue_pick(default_id, entries)
                        if default_id else ""
                    )
                with gr.Group(visible=(default_source == "Upload")) as up_group_s:
                    ckpt_in = gr.File(
                        label="Checkpoint (.pt)",
                        file_types=[".pt", ".pth", ".ckpt"],
                    )

                with gr.Row():
                    device_in = gr.Dropdown(
                        choices=["cpu", "cuda"],
                        value="cpu", label="Device",
                    )
                    load_btn = gr.Button("Load & Predict", variant="primary")

                meta_md = gr.Markdown("ℹ select a checkpoint and click Load.")
                metrics_md = gr.Markdown("")
                with gr.Row():
                    roc_plot = gr.Plot(label="ROC")
                    alpha_plot = gr.Plot(label="α_κ")
                gr.Markdown(PREDICTIONS_CALLOUT_MD)
                pred_table = gr.Dataframe(
                    headers=["row_idx", "u", "v", "true", "pred", "p(+)", "correct"],
                    label="Top-50 highest-confidence predictions",
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("### Subgraph viz")
                with gr.Row():
                    row_idx_in = gr.Number(
                        label="row_idx from the table above",
                        value=0, precision=0,
                    )
                    radius_in = gr.Slider(
                        minimum=1, maximum=3, step=1, value=1,
                        label="BFS radius",
                    )
                    render_btn = gr.Button("Render subgraph")
                sub_info = gr.Markdown("")
                sub_plot = gr.Plot(label="Local signed neighbourhood")

                # Wiring.
                source_single.change(
                    _toggle_source, inputs=[source_single],
                    outputs=[cat_group_s, up_group_s],
                )
                cat_dd_s.change(
                    lambda rid: _on_catalogue_pick(rid, entries),
                    inputs=[cat_dd_s], outputs=[cat_info_s],
                )

                def _load_unified(source, registry_id, file_obj, device):
                    path, err = _resolve_source(source, file_obj, registry_id, entries)
                    if err is not None:
                        return (err, None, None, None, None, None)
                    return _cb_load_single(path, device)

                load_btn.click(
                    _load_unified,
                    inputs=[source_single, cat_dd_s, ckpt_in, device_in],
                    outputs=[meta_md, metrics_md, roc_plot, alpha_plot,
                             pred_table, state_single],
                )
                render_btn.click(
                    _cb_render_subgraph,
                    inputs=[state_single, row_idx_in, radius_in],
                    outputs=[sub_plot, sub_info],
                )

            with gr.Tab("Compare two models"):
                state_pair = gr.State(value=None)

                gr.Markdown("### Model A")
                source_a = gr.Radio(
                    choices=["Catalogue", "Upload"],
                    value=default_source,
                    label="Source — A",
                )
                with gr.Group(visible=(default_source == "Catalogue")) as cat_group_a:
                    cat_dd_a = gr.Dropdown(
                        choices=choices, value=default_id,
                        label="Catalogued model — A",
                    )
                with gr.Group(visible=(default_source == "Upload")) as up_group_a:
                    ckpt_a = gr.File(label="Checkpoint A",
                                       file_types=[".pt", ".pth", ".ckpt"])

                gr.Markdown("### Model B")
                source_b = gr.Radio(
                    choices=["Catalogue", "Upload"],
                    value=default_source,
                    label="Source — B",
                )
                with gr.Group(visible=(default_source == "Catalogue")) as cat_group_b:
                    cat_dd_b = gr.Dropdown(
                        choices=choices, value=default_id,
                        label="Catalogued model — B",
                    )
                with gr.Group(visible=(default_source == "Upload")) as up_group_b:
                    ckpt_b = gr.File(label="Checkpoint B",
                                       file_types=[".pt", ".pth", ".ckpt"])

                with gr.Row():
                    device_cmp = gr.Dropdown(
                        choices=["cpu", "cuda"], value="cpu", label="Device",
                    )
                    cmp_btn = gr.Button("Compare", variant="primary")
                cmp_summary = gr.Markdown(
                    "ℹ choose A and B (catalogue or upload) on the same dataset."
                )
                with gr.Row():
                    cmp_roc = gr.Plot(label="ROC — A vs B")
                    cmp_alpha = gr.Plot(label="αₖ — A vs B")

                source_a.change(_toggle_source, inputs=[source_a],
                                  outputs=[cat_group_a, up_group_a])
                source_b.change(_toggle_source, inputs=[source_b],
                                  outputs=[cat_group_b, up_group_b])

                def _compare_unified(src_a, rid_a, fobj_a,
                                       src_b, rid_b, fobj_b, device):
                    pa, err_a = _resolve_source(src_a, fobj_a, rid_a, entries)
                    if err_a is not None:
                        return ("**A:** " + err_a, None, None, None)
                    pb, err_b = _resolve_source(src_b, fobj_b, rid_b, entries)
                    if err_b is not None:
                        return ("**B:** " + err_b, None, None, None)
                    return _cb_compare(pa, pb, device)

                cmp_btn.click(
                    _compare_unified,
                    inputs=[source_a, cat_dd_a, ckpt_a,
                            source_b, cat_dd_b, ckpt_b, device_cmp],
                    outputs=[cmp_summary, cmp_roc, cmp_alpha, state_pair],
                )

            with gr.Tab("Kinematic graph"):
                urdf_entries = load_urdf_registry()
                urdf_choices = urdf_dropdown_choices(urdf_entries)
                default_urdf_id = urdf_choices[0][1] if urdf_choices else None

                k4_ckpt = PRETRAINED_DIR / "family_classifier_k4.pt"
                k6_ckpt = PRETRAINED_DIR / "family_classifier_k6.pt"
                n_trained = sum(1 for p in (k4_ckpt, k6_ckpt) if p.is_file())
                gr.Markdown(
                    "Pick a robot URDF (or upload one). The tab parses it "
                    "into a signed kinematic graph — vertices are links, "
                    "edges are movable joints, sign is rotational (+) vs. "
                    "translational (−) — shows the cycle-arity profile "
                    "HSiKAN's α_κ would consume, and (if classifier "
                    "checkpoints are present) predicts the kinematic "
                    f"family. **{n_trained}/2 classifiers trained.** "
                    "Pre-train with: "
                    "`PYTHONPATH=. uv run python -m signedkan_wip.src.demo.kinematic_classifier`."
                )

                with gr.Row():
                    urdf_dd = gr.Dropdown(
                        choices=urdf_choices, value=default_urdf_id,
                        label="Catalogued URDF",
                        info="Edit `signedkan_wip/src/demo/kinematic_models.yaml` "
                             "to add more, or upload below to bypass the catalogue.",
                    )
                    urdf_upload = gr.File(
                        label="…or upload a .urdf",
                        file_types=[".urdf", ".xml"],
                    )
                    urdf_btn = gr.Button("Analyse", variant="primary")

                kin_summary_md = gr.Markdown(
                    "ℹ pick a URDF and click Analyse."
                )
                with gr.Row():
                    kin_graph_plot = gr.Plot(label="Kinematic graph")
                    kin_cycle_plot = gr.Plot(label="Cycle arities")

                def _kin_analyse(entry_id, upload_file):
                    """Either: catalogued id → entry.path, or upload → file.name."""
                    if upload_file is not None:
                        path = upload_file.name
                        name = upload_file.name.split("/")[-1].rsplit(".", 1)[0]
                    elif entry_id:
                        entry = find_urdf_by_id(urdf_entries, entry_id)
                        if entry is None:
                            return ("⚠ unknown URDF id.", None, None)
                        if not entry.available:
                            return (
                                f"⚠ `{entry.id}` is in the catalogue but the "
                                f"file is missing at `{entry.path}`.",
                                None, None,
                            )
                        path = entry.path
                        name = entry.id
                    else:
                        return ("ℹ pick a URDF or upload one.", None, None)
                    try:
                        bundle = load_urdf_bundle(path, name=name)
                    except Exception:
                        return (
                            f"### URDF parse failed\n```\n{traceback.format_exc()}\n```",
                            None, None,
                        )
                    sig = topology_signature(bundle)
                    bal = bundle.balance_summary()
                    parts = [
                        f"**`{bundle.name}`** — *{sig}*",
                        f"- {bundle.n_links} links, {bundle.n_joints} movable joints "
                        f"({bundle.n_revolute} revolute, {bundle.n_prismatic} prismatic)",
                        f"- joint-sign mix: {bal['n_pos']} positive (+) / "
                        f"{bal['n_neg']} negative (−), "
                        f"pos-fraction {bal['pos_fraction']:.3f}",
                    ]
                    n_cycles_total = sum(bundle.cycle_counts.values())
                    if n_cycles_total == 0:
                        parts.append(
                            "- **no closed kinematic loops** — HSiKAN's "
                            "cycle pool would be empty at every arity. "
                            "Walks-only or vertex-aggregation models are "
                            "the right choice here."
                        )
                    else:
                        cc = bundle.cycle_counts
                        cycle_str = ", ".join(
                            f"k={k}: {v}" for k, v in sorted(cc.items()) if v
                        )
                        parts.append(
                            f"- closed loops present: {cycle_str} — "
                            f"the α_κ vector tells the model which arity "
                            f"to weight."
                        )

                    # v0.5 — trained classifier prediction.
                    try:
                        result = predict_family(bundle)
                    except Exception:
                        result = None
                    if result is not None:
                        if result.rule_based:
                            parts.append(
                                f"### Kinematic-family prediction\n"
                                f"**`{result.predicted_family}`** "
                                f"(rule-based; "
                                f"{'cycles absent → serial' if result.arity_used is None else f'k={result.arity_used} classifier missing'})"
                            )
                        else:
                            prob_bars = "\n".join(
                                f"- `{f:11}`  {p:.4f}  {'█' * int(p * 24)}"
                                for f, p in zip(result.class_labels, result.probs)
                            )
                            parts.append(
                                f"### Kinematic-family prediction\n"
                                f"**`{result.predicted_family}`** "
                                f"(confidence {result.confidence:.3f}, "
                                f"k={result.arity_used} classifier)\n\n"
                                f"{prob_bars}\n\n"
                                f"_Pretrained `GraphLevelHSiKAN` consumes "
                                f"the bundle's cycle-pool features for "
                                f"arity k={result.arity_used} and outputs "
                                f"a softmax over the four families "
                                f"(`four_bar`, `stewart`, `delta_3rrr`, "
                                f"`serial`)._"
                            )
                    return ("\n\n".join(parts),
                            kinematic_graph_figure(bundle),
                            cycle_arity_figure(bundle))

                urdf_btn.click(
                    _kin_analyse,
                    inputs=[urdf_dd, urdf_upload],
                    outputs=[kin_summary_md, kin_graph_plot, kin_cycle_plot],
                )

            with gr.Tab("Robot communication cliques"):
                gr.Markdown(
                    "A multi-robot communication network is a signed "
                    "graph: vertices are robots, edges are pairwise "
                    "communication attempts, sign is **+** (reliable "
                    "link / trusted) or **−** (jammed / lost / "
                    "distrusted). Cartwright-Harary structural balance "
                    "theory: a signed graph is *balanced* iff every "
                    "cycle has an even number of negative edges — "
                    "equivalently, the σ-product around every cycle "
                    "equals +1. **A balanced clique is a stable "
                    "communication team.** HSiKAN learns σ-products "
                    "in its cycle pool by construction, so this is the "
                    "inductive bias the demo is set up to highlight."
                )

                with gr.Row():
                    n_robots_in = gr.Slider(
                        label="n_robots", minimum=4, maximum=40,
                        value=14, step=1,
                    )
                    comm_range_in = gr.Slider(
                        label="comm_range", minimum=1.0, maximum=8.0,
                        value=3.5, step=0.1,
                    )
                    noise_prob_in = gr.Slider(
                        label="noise_prob  (P(edge flipped to −))",
                        minimum=0.0, maximum=0.5, value=0.10, step=0.01,
                    )
                with gr.Row():
                    cliques_seed_in = gr.Number(
                        label="seed", value=0, precision=0,
                    )
                    min_clique_in = gr.Slider(
                        label="min clique size", minimum=3, maximum=6,
                        value=3, step=1,
                    )
                    max_clique_in = gr.Slider(
                        label="max clique size", minimum=3, maximum=8,
                        value=6, step=1,
                    )
                    cliques_btn = gr.Button(
                        "Generate & analyse", variant="primary")

                cliques_summary_md = gr.Markdown(
                    "ℹ Move the sliders, then click *Generate & analyse*."
                )
                with gr.Row():
                    cliques_net_plot = gr.Plot(label="Network (spatial)")
                    cliques_cliques_plot = gr.Plot(
                        label="Balanced cliques (shaded)")

                def _generate_and_analyse(
                    n_robots, comm_range, noise_prob, seed,
                    min_size, max_size,
                ):
                    try:
                        bundle = make_robot_network(
                            n_robots=int(n_robots),
                            area_size=10.0,
                            comm_range=float(comm_range),
                            noise_prob=float(noise_prob),
                            seed=int(seed),
                            name=f"network_seed{int(seed)}",
                        )
                        bs = balance_summary(bundle)
                        cliques = enumerate_balanced_cliques(
                            bundle,
                            min_size=int(min_size),
                            max_size=int(max_size),
                            limit=20,
                        )
                    except Exception:
                        return (
                            f"### Generation failed\n```\n"
                            f"{traceback.format_exc()}\n```",
                            None, None,
                        )

                    parts = [
                        f"**Network:** `{bundle.name}`  ·  "
                        f"{bs['n_robots']} robots  ·  "
                        f"{bs['n_edges']} edges  "
                        f"({bs['n_positive']} +, {bs['n_negative']} −, "
                        f"neg-fraction {bs['negative_fraction']:.3f})  ·  "
                        f"mean degree {bs['mean_degree']:.2f}",
                        f"**Balanced cliques found:** "
                        f"{len(cliques)}  (size {int(min_size)}…{int(max_size)})",
                    ]
                    if cliques:
                        rows = []
                        for ci, c in enumerate(cliques[:10]):
                            members = ", ".join(f"r{m}" for m in c.members)
                            n_neg = sum(1 for s in c.signs if s == -1)
                            rows.append(
                                f"  - **#{ci+1}**  size={c.size}  "
                                f"σ-prod=+1  ({n_neg} negative edges "
                                f"out of {len(c.signs)})  ·  "
                                f"members: `{{{members}}}`"
                            )
                        parts.append("**Top balanced cliques:**\n" + "\n".join(rows))
                    else:
                        parts.append(
                            "_No balanced cliques in the requested "
                            "size range. Try raising `comm_range`, "
                            "lowering `noise_prob`, or relaxing "
                            "`min clique size`._"
                        )
                    return (
                        "\n\n".join(parts),
                        network_figure(bundle),
                        cliques_figure(bundle, cliques),
                    )

                cliques_btn.click(
                    _generate_and_analyse,
                    inputs=[n_robots_in, comm_range_in, noise_prob_in,
                            cliques_seed_in, min_clique_in, max_clique_in],
                    outputs=[cliques_summary_md, cliques_net_plot,
                             cliques_cliques_plot],
                )

            with gr.Tab("How to use these predictions"):
                gr.Markdown(HOW_TO_USE_MD)

        gr.Markdown(
            "---\n"
            "**Provenance.** Checkpoints round-trip via "
            "`signedkan_wip.src.demo.checkpoint` (format_version 2). "
            "Predictions use the precomputed test-set inference bundle "
            "stored alongside the model weights. The catalogue lives at "
            "`signedkan_wip/src/demo/models.yaml`."
        )
    return app


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description="Launch the HyMeKo signed-link-prediction demo GUI."
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("HYMEKO_DEMO_HOST", "0.0.0.0"),
        help="Bind address. '0.0.0.0' exposes on the LAN (default); "
             "'127.0.0.1' restricts to localhost. Override with "
             "HYMEKO_DEMO_HOST.",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HYMEKO_DEMO_PORT", "7860")),
    )
    ap.add_argument(
        "--share", action="store_true",
        help="Also expose via Gradio's public tunnel (gradio.live).",
    )
    args = ap.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
