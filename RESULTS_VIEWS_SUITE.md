# Thesis IV entropy-regularization suite — overnight 2026-04-23 results

All paired comparisons of the form  **Δ = (treatment − baseline) val-acc**,
matched seeds, same optimiser config. Percentage-point units (pp); a Δ of
+0.100 means a one-tenth-of-a-percent accuracy gain.

| column       | meaning |
|--------------|---------|
| Δ pp         | mean accuracy shift, with ▲/▼ for sign and * for p-value |
| t            | paired two-sided t-statistic |
| W/L/T        | wins / losses / ties across seeds |
| σ base → σ treat | stdev of val-acc per arm (drop = variance-reducing) |
| ΔH           | final spectral entropy shift (treat − baseline) |
| sig          | `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `.` p<0.10 |


## Anchor reference

The original overnight-1 finding this suite is probing:
- **MNIST** plain-MLP (`mnist_small`), scalar_entropy / dataflow, 33 seeds × 15 epochs → **Δ = +0.149 pp**, t=+2.88, p≈0.007, W/L=22/11.
- **MNIST** plain-MLP, scalar_entropy / factor, 33 seeds × 15 epochs → **Δ = +0.135 pp**, t=+2.53, W/L=19/13.


## All runs (sorted by dataset, arm, view)

| dataset        | view     | arm              |   n | ep  | Δ pp          |    t  | W/L/T     | σ base → σ treat              | ΔH     |
|----------------|----------|------------------|-----|-----|---------------|-------|-----------|-------------------------------|--------|
| mnist_small    | dataflow | entropy_adaptive |  33 |  15 | ▲ +0.209 *** | +3.99 | 25/7/1    | 0.407 → 0.267 (-0.140) | +0.011 |
| mnist_small    | dataflow | entropy_target   |  33 |  15 | ▲ +0.229 *** | +4.42 | 24/8/1    | 0.407 → 0.340 (-0.067) | +0.011 |
| mnist_small    | dataflow | entropy_target_ka |  33 |  15 | ▲ +0.025     | +0.76 | 20/12/1   | 0.407 → 0.394 (-0.013) | +0.042 |
| mnist_small    | dataflow | entropy_telgarsky |  33 |  15 | ▲ +0.040     | +1.25 | 17/15/1   | 0.407 → 0.379 (-0.028) | -0.092 |
| mnist_small    | dataflow | entropy_unified  |  33 |  15 | ▲ +0.214 *** | +3.82 | 23/10/0   | 0.407 → 0.303 (-0.104) | +0.006 |
| mnist_small    | dataflow | scalar_entropy   |  15 |  30 | ▲ +0.165 *   | +2.45 | 10/5/0    | 0.285 → 0.163 (-0.122) | -0.044 |
| mnist_small    | factor   | scalar_entropy   |  15 |  30 | ▲ +0.157 .   | +1.84 | 10/5/0    | 0.285 → 0.177 (-0.107) | -0.273 |
| mnist_small    | dataflow | scalar_entropy_normalized |  33 |  15 | ▲ +0.232 *** | +3.86 | 25/6/2    | 0.407 → 0.249 (-0.159) | +0.011 |
| mnist_small    | dataflow | structural_composite |  33 |  15 | ▲ +0.021     | +0.49 | 19/13/1   | 0.407 → 0.381 (-0.026) | +0.024 |
| mnist_small    | dataflow | total_combined   |  33 |  15 | ▲ +0.209 *** | +4.39 | 29/4/0    | 0.407 → 0.299 (-0.109) | +0.009 |
| mnist_resnet_20 | dataflow | entropy_target   |  33 |   5 | ▲ +0.040     | +1.16 | 18/14/1   | 0.328 → 0.325 (-0.002) | -0.002 |
| mnist_resnet_20 | dataflow | kl_trajectory    |  66 |   5 | ▲ +0.049 .   | +1.98 | 36/28/2   | 0.329 → 0.310 (-0.019) | -0.000 |
| mnist_resnet_20 | dataflow | scalar_entropy   |  66 |   5 | ▲ +0.046 .   | +1.76 | 36/29/1   | 0.329 → 0.286 (-0.043) | -0.005 |
| mnist_resnet_20 | dataflow | scalar_entropy   |  33 |   5 | ▲ +0.070 .   | +1.89 | 18/15/0   | 0.328 → 0.294 (-0.033) | -0.006 |
| mnist_resnet_20 | dataflow | scalar_entropy_normalized |  33 |   5 | ▲ +0.054 .   | +1.70 | 20/11/2   | 0.328 → 0.337 (+0.009) | -0.002 |
| fashion_mnist  | dataflow | entropy_target   |  33 |  15 | ▼ -0.022     | -0.72 | 17/16/0   | 0.347 → 0.333 (-0.014) | -0.003 |
| fashion_mnist  | dataflow | scalar_entropy   |  33 |  15 | ▲ +0.039     | +1.13 | 19/13/1   | 0.347 → 0.307 (-0.040) | -0.014 |
| fashion_mnist  | factor   | scalar_entropy   |  33 |  15 | ▼ -0.009     | -0.24 | 17/16/0   | 0.347 → 0.333 (-0.014) | -0.038 |
| fashion_mnist  | dataflow | scalar_entropy_normalized |  33 |  15 | ▼ -0.027     | -0.73 | 16/15/2   | 0.347 → 0.299 (-0.048) | -0.004 |
| kmnist         | dataflow | entropy_target   |  33 |  15 | ▲ +0.004     | +0.08 | 15/18/0   | 0.714 → 0.715 (+0.001) | -0.000 |
| kmnist         | dataflow | kl_trajectory    |  33 |   5 | ▼ -0.029     | -1.00 | 15/17/1   | 0.898 → 0.844 (-0.053) | -0.000 |
| kmnist         | dataflow | scalar_entropy   |  33 |  15 | ▼ -0.029     | -0.33 | 15/17/1   | 0.714 → 0.701 (-0.014) | -0.004 |
| kmnist         | factor   | scalar_entropy   |  33 |  15 | ▲ +0.097     | +1.28 | 18/15/0   | 0.714 → 0.683 (-0.031) | -0.007 |
| kmnist         | dataflow | scalar_entropy_normalized |  33 |  15 | ▲ +0.042     | +0.82 | 18/14/1   | 0.714 → 0.632 (-0.082) | -0.000 |
| cifar10        | dataflow | scalar_entropy   |  15 |  30 | ▼ -0.095     | -0.43 | 8/7/0     | 0.530 → 0.758 (+0.228) | -0.213 |
| cifar10        | dataflow | scalar_entropy   |  15 |  20 | ▲ +0.043     | +0.28 | 10/5/0    | 0.613 → 0.499 (-0.114) | -0.143 |
| cifar10        | factor   | scalar_entropy   |  15 |  20 | ▲ +0.073     | +0.39 | 7/8/0     | 0.613 → 0.651 (+0.039) | -0.465 |
| two_moons      | dataflow | kl_trajectory    | 100 |  50 | ▼ -0.008     | -0.46 | 16/21/63  | 0.703 → 0.704 (+0.002) | +0.043 |
| two_moons      | factor   | kl_trajectory    | 100 |  50 | ▼ -0.016     | -0.77 | 18/29/53  | 0.703 → 0.688 (-0.015) | +0.098 |
| two_moons      | dataflow | scalar_entropy   | 300 |  50 | ▲ +0.003     | +0.35 | 65/66/169 | 0.704 → 0.718 (+0.015) | -0.189 |
| two_moons      | dataflow | scalar_entropy   | 100 |  50 | ▲ +0.016     | +0.92 | 25/22/53  | 0.703 → 0.702 (-0.000) | -0.188 |
| two_moons      | factor   | scalar_entropy   | 100 |  50 | ▲ +0.026     | +1.43 | 26/20/54  | 0.703 → 0.681 (-0.021) | -0.156 |
| spirals        | dataflow | entropy_adaptive | 100 |  50 | ▲ +0.724 *** | +3.73 | 52/12/36  | 2.333 → 1.339 (-0.994) | -0.248 |
| spirals        | dataflow | entropy_target   | 100 |  50 | ▲ +0.662 *** | +3.75 | 51/12/37  | 2.333 → 1.558 (-0.774) | -0.246 |
| spirals        | dataflow | entropy_target   | 100 |  50 | ▲ +0.630 *** | +3.58 | 50/15/35  | 2.333 → 1.716 (-0.616) | -0.263 |
| spirals        | dataflow | entropy_target   | 100 |  50 | ▲ +0.652 *** | +3.71 | 52/16/32  | 2.333 → 1.683 (-0.650) | -0.256 |
| spirals        | dataflow | entropy_target   | 100 |  50 | ▲ +0.662 *** | +3.75 | 51/12/37  | 2.333 → 1.558 (-0.774) | -0.246 |
| spirals        | dataflow | entropy_target   | 100 |  50 | ▲ +0.688 *** | +3.62 | 50/14/36  | 2.333 → 1.347 (-0.986) | -0.236 |
| spirals        | dataflow | entropy_target   | 100 |  50 | ▲ +0.040     | +0.68 | 20/21/59  | 2.333 → 2.148 (-0.184) | +0.155 |
| spirals        | dataflow | entropy_target_ka | 100 |  50 | ▲ +0.662 *** | +3.75 | 51/12/37  | 2.333 → 1.558 (-0.774) | -0.246 |
| spirals        | dataflow | entropy_telgarsky | 100 |  50 | ▲ +0.598 **  | +3.33 | 58/12/30  | 2.333 → 1.709 (-0.623) | -0.069 |
| spirals        | dataflow | entropy_unified  | 100 |  50 | ▲ +0.630 *** | +3.58 | 50/15/35  | 2.333 → 1.716 (-0.616) | -0.263 |
| spirals        | dataflow | kl_trajectory    | 100 |  50 | ▲ +0.294 *** | +3.79 | 39/19/42  | 2.333 → 2.179 (-0.153) | +0.085 |
| spirals        | factor   | kl_trajectory    | 100 |  50 | ▲ +0.066     | +0.90 | 29/33/38  | 2.333 → 2.399 (+0.067) | +0.141 |
| spirals        | dataflow | scalar_entropy   | 300 |  50 | ▲ +0.433 *** | +5.39 | 136/57/107 | 2.452 → 1.928 (-0.525) | -0.310 |
| spirals        | dataflow | scalar_entropy   | 100 |  50 | ▲ +0.412 *** | +3.45 | 43/21/36  | 2.333 → 1.800 (-0.532) | -0.301 |
| spirals        | factor   | scalar_entropy   | 100 |  50 | ▲ +0.580 **  | +3.24 | 52/20/28  | 2.333 → 1.691 (-0.642) | -0.263 |
| spirals        | dataflow | scalar_entropy_normalized | 100 |  50 | ▲ +0.624 *** | +3.54 | 52/14/34  | 2.333 → 1.600 (-0.733) | -0.254 |
| spirals        | dataflow | structural_composite | 100 |  50 | ▲ +0.454 *** | +4.51 | 48/14/38  | 2.333 → 1.992 (-0.341) | -0.227 |
| spirals        | dataflow | total_combined   | 100 |  50 | ▲ +0.582 **  | +3.33 | 49/15/36  | 2.333 → 1.693 (-0.640) | -0.269 |
| circles        | dataflow | entropy_adaptive | 100 |  50 | ▼ -0.020     | -0.94 | 19/26/55  | 1.223 → 1.222 (-0.001) | -0.165 |
| circles        | dataflow | entropy_target   | 100 |  50 | · +0.000     | +0.00 | 25/25/50  | 1.223 → 1.225 (+0.003) | -0.162 |
| circles        | dataflow | entropy_target   | 100 |  50 | ▼ -0.026     | -1.07 | 22/25/53  | 1.223 → 1.238 (+0.016) | -0.180 |
| circles        | dataflow | entropy_target   | 100 |  50 | ▼ -0.014     | -0.61 | 22/23/55  | 1.223 → 1.229 (+0.007) | -0.172 |
| circles        | dataflow | entropy_target   | 100 |  50 | · +0.000     | +0.00 | 25/25/50  | 1.223 → 1.225 (+0.003) | -0.162 |
| circles        | dataflow | entropy_target   | 100 |  50 | ▼ -0.006     | -0.34 | 19/21/60  | 1.223 → 1.238 (+0.015) | -0.144 |
| circles        | dataflow | entropy_target   | 100 |  50 | ▲ +0.018     | +1.38 | 21/11/68  | 1.223 → 1.207 (-0.016) | +0.064 |
| circles        | dataflow | entropy_target_ka | 100 |  50 | · +0.000     | +0.00 | 25/25/50  | 1.223 → 1.225 (+0.003) | -0.162 |
| circles        | dataflow | entropy_telgarsky | 100 |  50 | ▼ -0.046 *   | -2.01 | 24/39/37  | 1.223 → 1.182 (-0.040) | -0.053 |
| circles        | dataflow | entropy_unified  | 100 |  50 | ▼ -0.026     | -1.07 | 22/25/53  | 1.223 → 1.238 (+0.016) | -0.180 |
| circles        | dataflow | kl_trajectory    | 100 |  50 | ▼ -0.016     | -0.71 | 25/22/53  | 1.223 → 1.219 (-0.003) | +0.026 |
| circles        | factor   | kl_trajectory    | 100 |  50 | ▼ -0.046 .   | -1.88 | 25/33/42  | 1.223 → 1.214 (-0.008) | +0.048 |
| circles        | dataflow | scalar_entropy   | 100 |  50 | ▼ -0.060 *   | -2.30 | 22/32/46  | 1.223 → 1.230 (+0.007) | -0.213 |
| circles        | factor   | scalar_entropy   | 100 |  50 | ▼ -0.030     | -1.14 | 29/30/41  | 1.223 → 1.229 (+0.007) | -0.152 |
| circles        | dataflow | scalar_entropy_normalized | 100 |  50 | ▼ -0.016     | -0.77 | 19/25/56  | 1.223 → 1.222 (-0.000) | -0.165 |
| circles        | dataflow | structural_composite | 100 |  50 | ▼ -0.010     | -0.45 | 19/25/56  | 1.223 → 1.217 (-0.006) | -0.165 |
| circles        | dataflow | total_combined   | 100 |  50 | ▼ -0.038     | -1.60 | 20/27/53  | 1.223 → 1.219 (-0.004) | -0.191 |
| mnist_capsnet  | dataflow | entropy_adaptive |  33 |  10 | ▼ -0.002     | -0.07 | 15/17/1   | 0.192 → 0.187 (-0.005) | -0.002 |
| mnist_capsnet  | dataflow | entropy_target   |  33 |  10 | ▼ -0.012     | -0.52 | 14/18/1   | 0.192 → 0.193 (+0.001) | -0.002 |
| svhn           | dataflow | entropy_target   |  15 |  20 | ▼ -0.117     | -0.58 | 7/8/0     | 0.668 → 0.610 (-0.058) | -0.003 |
| mnist_capsnet  | dataflow | entropy_target_ka |  33 |  10 | ▼ -0.024     | -1.50 | 13/19/1   | 0.192 → 0.211 (+0.019) | +0.000 |
| mnist_capsnet  | dataflow | entropy_telgarsky |  33 |  10 | ▲ +0.010     | +0.45 | 19/14/0   | 0.192 → 0.191 (-0.001) | -0.005 |
| mnist_capsnet  | dataflow | entropy_unified  |  33 |  10 | ▲ +0.021     | +1.08 | 21/12/0   | 0.192 → 0.197 (+0.005) | -0.004 |
| iris           | dataflow | kl_trajectory    | 100 | 100 | ▲ +0.026     | +0.45 | 3/2/95    | 3.456 → 3.481 (+0.025) | +0.004 |
| wine           | dataflow | kl_trajectory    | 100 | 100 | ▼ -0.022     | -1.00 | 0/1/99    | 1.422 → 1.419 (-0.003) | +0.001 |
| breast_cancer  | dataflow | kl_trajectory    | 100 | 100 | ▼ -0.007     | -0.38 | 3/4/93    | 0.605 → 0.565 (-0.040) | +0.001 |
| digits         | dataflow | kl_trajectory    | 100 | 100 | ▼ -0.029 *   | -2.01 | 9/20/71   | 0.468 → 0.458 (-0.010) | +0.007 |
| gaussian_quants | dataflow | kl_trajectory    | 100 |  50 | ▼ -0.026     | -1.15 | 24/28/48  | 1.771 → 1.737 (-0.034) | +0.030 |
| mnist_capsnet  | dataflow | kl_trajectory    |  33 |   5 | ▲ +0.001     | +0.36 | 14/12/7   | 0.214 → 0.208 (-0.006) | +0.002 |
| iris           | factor   | kl_trajectory    | 100 | 100 | ▼ -0.026     | -0.33 | 4/5/91    | 3.456 → 3.410 (-0.046) | +0.005 |
| wine           | factor   | kl_trajectory    | 100 | 100 | ▲ +0.000     | +0.00 | 2/2/96    | 1.422 → 1.349 (-0.072) | +0.002 |
| breast_cancer  | factor   | kl_trajectory    | 100 | 100 | ▲ +0.042 .   | +1.75 | 7/2/91    | 0.605 → 0.580 (-0.025) | +0.001 |
| digits         | factor   | kl_trajectory    | 100 | 100 | ▲ +0.002     | +0.19 | 14/12/74  | 0.468 → 0.481 (+0.013) | +0.008 |
| gaussian_quants | factor   | kl_trajectory    | 100 |  50 | ▼ -0.002     | -0.07 | 33/30/37  | 1.771 → 1.759 (-0.012) | +0.045 |
| iris           | dataflow | scalar_entropy   | 100 | 100 | ▼ -0.053     | -0.38 | 8/11/81   | 3.456 → 3.506 (+0.050) | -0.098 |
| wine           | dataflow | scalar_entropy   | 100 | 100 | ▼ -0.044     | -1.42 | 0/2/98    | 1.422 → 1.416 (-0.006) | -0.020 |
| breast_cancer  | dataflow | scalar_entropy   | 100 | 100 | ▲ +0.014     | +0.50 | 7/6/87    | 0.605 → 0.566 (-0.039) | -0.014 |
| digits         | dataflow | scalar_entropy   | 100 | 100 | ▲ +0.047 **  | +2.73 | 29/13/58  | 0.468 → 0.449 (-0.019) | -0.034 |
| gaussian_quants | dataflow | scalar_entropy   | 100 |  50 | ▼ -0.014     | -0.51 | 30/33/37  | 1.771 → 1.760 (-0.011) | -0.111 |
| mnist_highway_10 | dataflow | scalar_entropy   |  33 |  15 | ▲ +0.027     | +0.80 | 18/14/1   | 0.242 → 0.236 (-0.006) | -0.058 |
| mnist_highway_20 | dataflow | scalar_entropy   |  33 |  15 | ▼ -0.031     | -1.02 | 14/18/1   | 0.247 → 0.224 (-0.024) | -0.065 |
| fashion_mnist_highway_20 | dataflow | scalar_entropy   |  33 |  15 | ▲ +0.008     | +0.16 | 16/15/2   | 0.361 → 0.352 (-0.008) | -0.062 |
| fashion_mnist_resnet_20 | dataflow | scalar_entropy   |  33 |  15 | ▼ -0.010     | -0.18 | 17/16/0   | 0.363 → 0.341 (-0.022) | -0.007 |
| emnist_letters | dataflow | scalar_entropy   |  33 |  15 | ▲ +0.170     | +1.24 | 18/15/0   | 0.941 → 0.684 (-0.257) | -0.020 |
| mnist_capsnet  | dataflow | scalar_entropy   |  33 |  10 | ▼ -0.057 *   | -2.14 | 14/18/1   | 0.192 → 0.227 (+0.035) | -0.027 |
| fashion_mnist_capsnet | dataflow | scalar_entropy   |  33 |  10 | ▼ -0.004     | -0.14 | 15/17/1   | 0.273 → 0.308 (+0.034) | -0.003 |
| svhn           | dataflow | scalar_entropy   |  15 |  20 | ▼ -0.137     | -0.55 | 8/7/0     | 0.668 → 0.736 (+0.069) | -0.002 |
| iris           | factor   | scalar_entropy   | 100 | 100 | ▼ -0.158     | -1.03 | 10/15/75  | 3.456 → 3.360 (-0.096) | -0.120 |
| wine           | factor   | scalar_entropy   | 100 | 100 | ▼ -0.067 .   | -1.75 | 0/3/97    | 1.422 → 1.413 (-0.009) | -0.042 |
| breast_cancer  | factor   | scalar_entropy   | 100 | 100 | ▲ +0.056 .   | +1.72 | 15/7/78   | 0.605 → 0.556 (-0.049) | -0.058 |
| digits         | factor   | scalar_entropy   | 100 | 100 | ▼ -0.056 *   | -2.56 | 17/35/48  | 0.468 → 0.461 (-0.007) | -0.175 |
| gaussian_quants | factor   | scalar_entropy   | 100 |  50 | ▲ +0.022     | +0.79 | 31/27/42  | 1.771 → 1.749 (-0.022) | -0.070 |
| emnist_letters | factor   | scalar_entropy   |  33 |  15 | ▲ +0.187     | +1.00 | 20/13/0   | 0.941 → 0.683 (-0.257) | -0.037 |
| mnist_capsnet  | factor   | scalar_entropy   |  33 |  10 | ▼ -0.001     | -0.04 | 15/17/1   | 0.192 → 0.155 (-0.037) | -0.060 |
| fashion_mnist_capsnet | factor   | scalar_entropy   |  33 |  10 | ▲ +0.004     | +0.10 | 18/15/0   | 0.273 → 0.308 (+0.034) | -0.024 |
| svhn           | factor   | scalar_entropy   |  15 |  20 | ▼ -0.322     | -1.42 | 5/10/0    | 0.668 → 0.885 (+0.217) | -0.079 |
| emnist_letters | dataflow | scalar_entropy_normalized |  33 |  15 | ▲ +0.058     | +0.43 | 17/16/0   | 0.941 → 0.776 (-0.164) | -0.003 |
| mnist_capsnet  | dataflow | scalar_entropy_normalized |  33 |  10 | ▼ -0.002     | -0.07 | 15/17/1   | 0.192 → 0.187 (-0.005) | -0.002 |
| svhn           | dataflow | scalar_entropy_normalized |  15 |  20 | ▼ -0.047     | -0.30 | 6/9/0     | 0.668 → 0.669 (+0.001) | -0.001 |
| mnist_capsnet  | dataflow | structural_composite |  33 |  10 | ▼ -0.002     | -0.07 | 15/17/1   | 0.192 → 0.187 (-0.005) | -0.002 |
| mnist_capsnet  | dataflow | total_combined   |  33 |  10 | ▲ +0.021     | +0.86 | 20/10/3   | 0.192 → 0.175 (-0.017) | -0.007 |

## Summary

- Total experiments analysed: **111**

- Positive and significant (p < 0.05, Δ > 0): **23** `mnist_small/dataflow/entropy_adaptive`, `mnist_small/dataflow/entropy_target`, `mnist_small/dataflow/entropy_unified`, `mnist_small/dataflow/scalar_entropy`, `mnist_small/dataflow/scalar_entropy_normalized`, `mnist_small/dataflow/total_combined`, `spirals/dataflow/entropy_adaptive`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target_ka`, `spirals/dataflow/entropy_telgarsky`, `spirals/dataflow/entropy_unified`, `spirals/dataflow/kl_trajectory`, `spirals/dataflow/scalar_entropy`, `spirals/dataflow/scalar_entropy`, `spirals/factor/scalar_entropy`, `spirals/dataflow/scalar_entropy_normalized`, `spirals/dataflow/structural_composite`, `spirals/dataflow/total_combined`, `digits/dataflow/scalar_entropy`.

- Positive but only marginal (0.05 ≤ p < 0.10): **7** `mnist_small/factor/scalar_entropy`, `mnist_resnet_20/dataflow/kl_trajectory`, `mnist_resnet_20/dataflow/scalar_entropy`, `mnist_resnet_20/dataflow/scalar_entropy`, `mnist_resnet_20/dataflow/scalar_entropy_normalized`, `breast_cancer/factor/kl_trajectory`, `breast_cancer/factor/scalar_entropy`.

- Negative and significant: **5** `circles/dataflow/entropy_telgarsky`, `circles/dataflow/scalar_entropy`, `digits/dataflow/kl_trajectory`, `mnist_capsnet/dataflow/scalar_entropy`, `digits/factor/scalar_entropy`.

- Variance-reducing (σ treat < 0.9 × σ base): **32** `mnist_small/dataflow/entropy_adaptive`, `mnist_small/dataflow/entropy_target`, `mnist_small/dataflow/entropy_unified`, `mnist_small/dataflow/scalar_entropy`, `mnist_small/factor/scalar_entropy`, `mnist_small/dataflow/scalar_entropy_normalized`, `mnist_small/dataflow/total_combined`, `mnist_resnet_20/dataflow/scalar_entropy`, `mnist_resnet_20/dataflow/scalar_entropy`, `fashion_mnist/dataflow/scalar_entropy`, `fashion_mnist/dataflow/scalar_entropy_normalized`, `kmnist/dataflow/scalar_entropy_normalized`, `cifar10/dataflow/scalar_entropy`, `spirals/dataflow/entropy_adaptive`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target`, `spirals/dataflow/entropy_target_ka`, `spirals/dataflow/entropy_telgarsky`, `spirals/dataflow/entropy_unified`, `spirals/dataflow/scalar_entropy`, `spirals/dataflow/scalar_entropy`, `spirals/factor/scalar_entropy`, `spirals/dataflow/scalar_entropy_normalized`, `spirals/dataflow/structural_composite`, `spirals/dataflow/total_combined`, `emnist_letters/dataflow/scalar_entropy`, `emnist_letters/factor/scalar_entropy`, `mnist_capsnet/factor/scalar_entropy`, `emnist_letters/dataflow/scalar_entropy_normalized`.
