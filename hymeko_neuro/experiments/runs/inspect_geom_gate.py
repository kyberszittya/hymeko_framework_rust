"""Inspect the trained geometric-attention gate — decide the readout fork.

The 5-seed A/B (``reports/2026-06-17-geometric-attention-head.md``) found the
geometric (quaternion+Clifford) readout adds nothing net over the bilinear head
(tie on alpha, regress on otc). Before sinking another lever (sign-aware values)
into the readout, this runs the discriminating test the report flagged but did
not yet do: train ``geom_attn`` on the small bitcoin fixtures with the same tuned
recipe and dump, per dataset/seed, the gate diagnostic from
``summarise_gate`` — σ(gate), signed-weight saturation, and the pool-vs-h_v
residual magnitude.

Reading:
  * ``w_abs_mean`` → 0  ⇒ score collapse: the pool carries no triad-specific
    signal (the readout cannot help no matter what value it pools).
  * ``pool_to_hv`` → 0  ⇒ residual swamp: the pool is informative but tiny next
    to ``h_v``, so ``h_v + pool ≈ h_v`` (the residual gain is the bottleneck).
  * neither → 0         ⇒ the readout *is* used; the value content is the lever
    (sign-aware values), so pushing the head is warranted.

    python -m hymeko_neuro.experiments.runs.inspect_geom_gate \
        > reports/geom_gate_inspect_20260617.jsonl
"""
from __future__ import annotations

import json
import sys

from hymeko_neuro.hyperedge.geometric_triad_attention import summarise_gate

from .run_hsikan_rotor import GeomTrainedState, TrainConfig, run

# Match the tuned recipe that produced the regressing 5-seed geom_attn numbers.
TUNED = TrainConfig(weight_decay=1e-4, grad_clip=1.0, class_weight=True,
                    early_stop=True)
DATASETS = ("bitcoin_alpha", "bitcoin_otc")
SEEDS = (0,)


def inspect_cell(dataset: str, seed: int) -> dict:
    """Train one geom_attn cell and return run-result merged with gate stats."""
    captured: dict = {}

    def on_trained(st: GeomTrainedState) -> None:
        captured.update(summarise_gate(st.pool, st.h_v, st.temb,
                                       st.inc_vertex, st.inc_triad, st.inc_balance))

    r = run(dataset, seed=seed, embed="rotor", shuffle=False, head="geom_attn",
            dedup=True, geom_channel="both", cfg=TUNED, on_trained=on_trained)
    return {**r, **captured}


def main() -> None:
    rows = [inspect_cell(d, s) for d in DATASETS for s in SEEDS]
    for row in rows:
        print(json.dumps(row))
    # Human-readable table to stderr (stdout stays a clean jsonl stream).
    hdr = (f"{'dataset':<14} {'seed':>4} {'AUROC':>7} {'gate_q':>7} "
           f"{'|w|mean':>8} {'%dead':>6} {'%sat':>6} {'pool/hv':>8} {'wv':>7}")
    print(hdr, file=sys.stderr)
    for r in rows:
        print(f"{r['dataset']:<14} {r['seed']:>4} {r['test_auroc']:>7.4f} "
              f"{r['gate_sigma']:>7.3f} {r['w_abs_mean']:>8.4f} "
              f"{r['w_frac_dead']:>6.2f} {r['w_frac_saturated']:>6.2f} "
              f"{r['pool_to_hv']:>8.4f} {r['wv_norm']:>7.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
