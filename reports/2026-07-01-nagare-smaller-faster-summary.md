# Is Nagare Smaller and Faster?

Created-at: 2026-07-01 16:03 JST

## Short Answer

Yes, on the small native Chebyshev-deploy classifier forwards, Nagare is both smaller and faster than PyTorch CPU.

The stronger statement is:

> Nagare is already much smaller. It is faster when the computation is native and allocation-light. For unfused composed kernels, it can still be smaller but slower until temporary materialization is removed.

## Best Positive Result

Exact PyTorch-vs-Nagare fixture on synthetic moons, spiral, and XOR:

- Model: `2 -> 32 -> 2`
- Activation: direct Chebyshev deploy, `k=5`, Chebyshev-domain rescale only (`scale=0.5`)
- Samples: `n=256`
- Same inputs and weights in both engines
- Logit parity: about `1e-7`

| task | max abs logits | PyTorch median | Nagare median | Nagare speedup |
| --- | ---: | ---: | ---: | ---: |
| moons | 7.45e-8 | 4.05 us | 2.18 us | 1.86x |
| spiral | 8.94e-8 | 5.43 us | 3.10 us | 1.75x |
| xor | 1.04e-7 | 4.27 us | 2.54 us | 1.68x |

Memory:

| engine | peak RSS |
| --- | ---: |
| PyTorch process tree | 552.89 MiB |
| Nagare release executable | 5.39 MiB |

Raw report:

- `reports/2026-07-01-nagare-pytorch-synthetic-cheby-compare.md`

## Native CR/Chebyshev Result

Native Nagare CR and Chebyshev-CR kernels were tested on small descriptor-like shapes.

| case | shape | CR train | Cheb-CR train | Cheb deploy |
| --- | --- | ---: | ---: | ---: |
| tiny descriptor | `n=1, c=16` | 1.1000 us | 2.1000 us | 0.4000 us |
| small descriptions | `n=8, c=32` | 1.0750 us | 0.8875 us | 0.2375 us |
| toy point batch | `n=4608, c=32` | 1.0717 us | 1.0013 us | 0.4043 us |

Peak RSS for this release run: `7.80 MiB`.

Important technical note: direct Chebyshev deploy initially underperformed because it allocated a temporary vector per scalar. After removing that allocation, deploy became the fastest path. A later correction removed the toy `tanh` bound and replaced it with Chebyshev-domain rescale only, which is better aligned with HSiKAN curve-activation policy.

Raw report:

- `reports/2026-07-01-nagare-cr-cheby-small-perf.md`

## Caveat: Unfused Entropy Feedback

The exact global-pool + entropy-feedback parity fixture showed a different result:

| task | PyTorch median | Nagare median |
| --- | ---: | ---: |
| moons | 60.49 us | 94.61 us |
| rings | 55.51 us | 93.24 us |
| xor | 62.36 us | 91.54 us |

But memory still strongly favored Nagare:

| engine | allocation measure | allocation bytes/forward | peak RSS |
| --- | --- | ---: | ---: |
| PyTorch CPU | profiler CPU memory events | 4,926,736 | 663.05 MiB process tree |
| Nagare/Rust | instrumented Vec estimate | 4,812,672 | 11.00 MiB process |

The cause is not numerical overhead. Parity was good at about `1e-7`. The cause is materialization: the current Rust entropy-feedback path builds the wide broadcast tensor `[h, pooled, entropy]` before the update linear. That erases the advantage of the smaller runtime.

Raw report:

- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity.md`

## Interpretation

Nagare has two clear advantages for small deployed CPU workloads:

- Very low resident memory compared with PyTorch.
- Lower latency when the operator is native and avoids temporary buffers.

The path to making the entropy-feedback model also faster is now concrete:

1. Fuse broadcast-context construction with the update linear.
2. Avoid allocating `update_x`.
3. Stream pooled context and entropy directly into the update kernel.
4. Re-run the exact PyTorch parity fixture after fusion.

## Verdict

For small Chebyshev-deploy classifiers: **Nagare is smaller and faster**.

For unfused entropy-feedback/global-pool: **Nagare is smaller, but not faster yet**.

The evidence points to a kernel-fusion issue, not a limitation of the Nagare approach.
