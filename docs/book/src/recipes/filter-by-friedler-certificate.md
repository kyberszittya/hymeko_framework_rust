# Filter NAS sweep outputs by Friedler certificate

The HSIKAN, Gömb, and cortical sweep drivers
(`run_hsikan_msg_sweep.py`, `run_gomb_msg_sweep.py`,
`run_cortical_msg_sweep.py`) all stamp the Friedler certificate on
every JSONL row they emit. Each row therefore carries:

| field | type | meaning |
| --- | --- | --- |
| `canonical_full_status` | `"PASS"` / `"FAIL"` | full-schema canonical Friedler S1..S5 |
| `extension_full_status` | `"PASS"` / `"FAIL"` | full-schema extension bundle (E-NoExcess, E-WellFormed, E-ConsumedHasProducer) |
| `canonical_abb_status`  | `"PASS"` / `"FAIL"` / null | canonical S1..S5 on the ABB-selected sub-schema (null when no ABB result) |
| `extension_abb_status`  | `"PASS"` / `"FAIL"` / null | extension on the sub-schema |
| `strict_no_excess`      | `bool` | engine mode used |

Filtering with `jq` is one line per question.

## "Show only canonical-feasible training runs"

```bash
cat sweep_results.jsonl \
  | jq -c 'select(.canonical_abb_status == "PASS")'
```

## "Show only canonical AND extension-feasible (the strictest filter)"

```bash
cat sweep_results.jsonl \
  | jq -c 'select(.canonical_abb_status == "PASS"
                 and .extension_abb_status == "PASS")'
```

## "Show runs where the engine and the canonical reading agreed"

The Phase 4 audit established that on by-product or disposal-sink
fixtures, the engine's strict-no-excess feasibility can diverge
from canonical S4. To keep only the "no divergence" runs:

```bash
cat sweep_results.jsonl \
  | jq -c 'select(.canonical_abb_status == "PASS"
                 and .extension_abb_status == "PASS"
                 and .canonical_full_status == "PASS")'
```

## "Compare AUC distributions by certificate"

```bash
cat sweep_results.jsonl \
  | jq -r '[.canonical_abb_status, .test_auroc] | @tsv' \
  | sort | datamash -g 1 mean 2 sstdev 2
```

## "Find runs that the canonical audit would have rejected"

```bash
cat sweep_results.jsonl \
  | jq -c 'select(.canonical_abb_status == "FAIL")'
```

Useful for sanity-checking that strict-mode sweeps didn't slip
disposal-sink-shaped architectures through.

## Generating the JSONL

Each driver appends one row per (selection, seed) when given
`--output path.jsonl`:

```bash
python -m signedkan_wip.experiments.runs.run_hsikan_msg_sweep \
    --pgraph data/hsikan/sweep_msg_byproduct_dominated.hymeko \
    --algorithm abb \
    --dataset bitcoin_alpha \
    --seeds 0 1 2 3 4 \
    --train \
    --output reports/hsikan_byproduct_filter.jsonl

cat reports/hsikan_byproduct_filter.jsonl \
  | jq -c 'select(.canonical_abb_status == "PASS")' \
  | wc -l
```

## See also

- [`reports/2026-05-19-pgraph-axiom-witness-phase4.md`](../../../reports/2026-05-19-pgraph-axiom-witness-phase4.md) — what the certificate fields actually attest to.
- [`reports/2026-05-20-pgraph-nas-byproduct-filter-phase11.md`](../../../reports/2026-05-20-pgraph-nas-byproduct-filter-phase11.md) — empirical demonstration that strict-mode certificate filtering produces +0.061 AUC on Bitcoin Alpha.
- [`reports/2026-05-19-pgraph-multi-objective-hsikan-gomb-phase10.md`](../../../reports/2026-05-19-pgraph-multi-objective-hsikan-gomb-phase10.md) — multi-objective weights as an orthogonal lever.
