# Legacy launcher scripts

Scripts here have been **migrated** to the new YAML-config + runner
framework (`hymeko_neuro/experiments/lib/` +
`hymeko_neuro/experiments/configs/`). They are kept for:

1. **Benchmark parity audit trail** — until the YAML config has
   been verified to reproduce the prior numerical result, the
   original script is the falsifier.
2. **Git history reachability** — the moved file lets `git mv`-style
   blame work cleanly.

| legacy script | replaced by | parity status |
|---|---|---|
| `run_slashdot_edge_cr_5seed_2026_05_09.sh` | `configs/slashdot_edge_cr_5seed.yaml` | pending parity check |
| `run_epinions_edge_cr_5seed_2026_05_09.sh` | `configs/epinions_edge_cr_5seed.yaml` | pending parity check |
| `run_btc_alpha_otc_sota_gate.sh` | `configs/bitcoin_alpha_edge_cr_5seed.yaml` (BA portion) | pending parity check |

**Delete an entry from this directory only after** the corresponding
parity check passes (`scripts/runner_parity_check.py` returns 0).
