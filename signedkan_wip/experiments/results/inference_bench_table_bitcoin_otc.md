### bitcoin_otc

| method | AUC (5-seed) | params | fwd CPU ms | fwd CUDA ms | tuned | source |
|---|---:|---:|---:|---:|:--:|---|
| HSiKAN-lean | — | 23611 | 95.51 | 24.62 | — | this-bench (latency only) |
| HSiKAN-optuna | 0.9933 ± 0.0023 | 23815 | — | — | yes | bitcoin_optuna_best_5seed_2026_05_13.jsonl |
| HSiKAN-leanest | 0.8506 ± 0.0165 | 94627 | — | — | no | phase8_bitcoin_5seed.json |
| HSiKAN-joint | — | 94627 | 309.67 | 31.75 | — | this-bench (latency only) |
| SignedKAN-L1 | 0.8020 ± 0.0123 | 188865 | — | — | no | phase8_bitcoin_5seed.json |
| MLP-blind | 0.9077 ± 0.0087 | 190305 | 0.55 | 0.88 | no | phase8_bitcoin_5seed.json |
| GCN-blind | 0.9055 ± 0.0126 | 192417 | — | — | no | phase8_bitcoin_5seed.json |
| SiGAT | 0.9322 ± 0.0044 | 201601 | 91.75 | 202.97 | no | phase8_bitcoin_5seed.json |
| SGCN | 0.9421 ± 0.0060 | 202721 | 20.37 | 5.12 | no | phase8_bitcoin_5seed.json |
| SGT | — | 215601 | 212.37 | 469.23 | — | this-bench (latency only) |
