# Abbreviations & symbols

Conventions used across **SignedKAN / HSiKAN** experiments, **reports**, and **Rust** crates.

## Datasets & splits

| Token | Meaning |
|-------|---------|
| **BA** / `bitcoin_alpha` | Bitcoin Alpha trust signed graph (link prediction). |
| **OTC** / `bitcoin_otc` | Bitcoin OTC trust signed graph. |
| **Slashdot** | Slashdot Zoo signed social graph (large). |
| **Epinions** | Epinions signed trust network (benchmark harness). |

## Models & architectures

| Token | Meaning |
|-------|---------|
| **HSiKAN** | Hypergraph / signed-cycle–aware KAN-style link predictor (`hsikan_mixed`, variants by arity). |
| **SignedKAN** | Signed-graph KAN baseline (`signedkan_L1`, etc.). |
| **SGCN** | Signed Graph Convolutional Network baseline (`sgcn_balance`). |
| **SiGAT** | Signed attention GAT-style baseline (`sigat_attn`). |
| **MLP / GCN** | Blind MLP / GCN baselines in phase panels (`mlp_blind`, `gcn_blind`). |
| **Gömb** | HymeKo-Gömb three-shell cascade (outer Clifford-FIR / middle HSiKAN-CR / inner CPML-MLP). |
| **edge_cr** | Slashdot (and Epinions) **Catmull–Rom per-edge highway** HSiKAN configuration used as a **strong reference** row (`run_label: edge_cr`). |

## Tuple & cycle notation

| Notation | Meaning |
|----------|---------|
| **c*k* ** | Simple cycle feature of length *k* (e.g. `c3`, `c4`). |
| **w*k* ** | Walk / non-simple closed feature of length *k* (e.g. `w2`, `w3`). |
| **`c3,c4,w2,w3`** | Joint mix **tuple set** for Bitcoin joint runs (walks + cycles in the same mixer). |
| **k=3, k=4 lean** | Phase‑8 **mixed leanest** configuration: arities `{3,4}` only, `max_k4` cap — **not** the full joint tuple set above. |

## Algorithms & infra

| Token | Meaning |
|-------|---------|
| **ABB** | Accelerated branch-and-bound (per-vertex or global top‑\(K\) cycle enumeration). |
| **top‑\(K\) / top‑\(m\)** | Per-graph or **per-vertex** cap on enumerated cycles fed to HSiKAN. |
| **CPG** | Cycle / pattern groupings used in Epinions “kitchen sink” configs (see `reports/` walk study). |
| **SGT** | External / baseline reference curves (`sgt_*.jsonl` in results). |
| **JSONL** | One JSON object per line — standard for append-only experiment logs. |

## Metrics

| Symbol | Meaning |
|--------|---------|
| **AUC** | ROC–AUC on held-out **test** edges unless a run explicitly logs validation. |
| **\(\Delta\)** pp | Mean paired **percentage-point** accuracy shift (thesis IV suite) or analogous paired delta — see [Mathematics](./mathematics.md). |
| **F1 / F1m** | Macro or binary F1 as logged per run. |

## Environment knobs (frequent)

| Prefix | Role |
|--------|------|
| `HSIKAN_*` | HSiKAN / enumeration / ABB / tuple / protocol flags (centralised in `RuntimeConfig` where migrated). |
| `HYMEKO_*` | Broader HyMeKo tooling flags. |

Full list: [Reference: env vars](../reference-env-vars.md).
