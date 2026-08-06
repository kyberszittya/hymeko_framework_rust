---
name: project-nagare-holonomy-line
description: "NAGARE = hymeko_nagare holonomy local-learning line; state as of 2026-07-04 (package promoted, FD-tested kernel, frozen fixture, parity narrowed not closed)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 20d6b9e1-d4b0-4eb1-a367-e0fbfa45c47b
---

NAGARE (hymeko_nagare) holonomy local-learning line, consolidated 2026-07-04:

- **Standing results (2026-07-01/02, all committed):** entropy-pool local learner matches backprop-like accuracy on moons/spiral/xor with 60 vs 2,836 params and ~23–25× faster forward; entropy *gate* ablation NEGATIVE (constant gate wins loss on all 12 stress rows); **fitted projection gate** is the first positive gate result (best loss 0.2506 on the hard spiral few-shot row); Chebyshev-deploy classifier beats PyTorch 1.7–1.9× at 5 MiB vs 553 MiB.
- **2026-07-04 (report `reports/2026-07-04-nagare-holonomy-package.md`):** stack promoted into `hymeko_nagare/src/holonomy/` package (datasets/features/pooling/projection/metrics/learner) on new `ops/project_alpha_mix` kernel (FD-tested fwd+bwd incl. grad_basis); seed-53 datasets frozen as FNV-hashed fixture; promotion proven **bit-identical** vs committed JSON. Parity re-run post-fusion: found+fixed a real defect (fused kernel was single-threaded vs rayon `linear_forward` → 3× slower); after fix Nagare 64–84 µs vs PyTorch 38–43 same-day → **PyTorch still ~1.6–2× faster on the unbatched entropy-feedback shape; caveat narrowed, not closed**. Allocations halved (2.43 MB/fwd), RSS 9 MiB vs 636 MiB.
- **Open:** flamegraph the parity gap (serial global_pool vs scalar accumulation) before more kernel work; fused backward still serial; extraction to `nagare-holonomy-learn` sibling repo — prerequisites all met, awaiting Hajdu's go/no-go; next science test = corrupted hyperedges/shuffled order/multi-class to isolate what the projection gate actually learns.
- Gotcha: `docs/plans/` is **gitignored by policy** (IP-strategy, Csapó co-author line) — plans live on disk only. PyTorch fixture regeneration is byte-deterministic under pinned torch 2.12.
