### bitcoin_alpha

| method | AUC (5-seed) | params | fwd CPU ms | fwd CUDA ms | tuned | source |
|---|---:|---:|---:|---:|:--:|---|
| HSiKAN-lean | — | 15219 | 106.33 | 29.89 | — | this-bench (latency only) |
| HSiKAN-optuna | 0.9959 ± 0.0011 | 30487 | — | — | yes | bitcoin_optuna_best_5seed_2026_05_13.jsonl |
| HSiKAN-leanest | 0.8281 ± 0.0101 | 61059 | — | — | no | phase8_bitcoin_5seed.json |
| HSiKAN-joint | — | 61059 | 415.62 | 29.34 | — | this-bench (latency only) |
| SignedKAN-L1 | 0.7449 ± 0.0234 | 121729 | — | — | no | phase8_bitcoin_5seed.json |
| MLP-blind | 0.8919 ± 0.0073 | 123169 | 0.60 | 0.85 | no | phase8_bitcoin_5seed.json |
| GCN-blind | 0.8710 ± 0.0172 | 125281 | — | — | no | phase8_bitcoin_5seed.json |
| SiGAT | 0.9033 ± 0.0080 | 134465 | 85.92 | 208.75 | no | phase8_bitcoin_5seed.json |
| SGCN | 0.9294 ± 0.0101 | 135585 | 15.46 | 4.95 | no | phase8_bitcoin_5seed.json |
| SGT | — | 148465 | 196.59 | 500.08 | — | this-bench (latency only) |
