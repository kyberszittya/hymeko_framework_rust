## Leakage metrics — baselines (5-seed, mean over datasets)

`L = max(0, shuffle−0.5)` (residual leak), `Δ = real − shuffle` (audit drop). `L_topo` uses the topology-shuffle variant; `L≈L_topo≈0` ⇒ no structural leak.

| method | real | shuffle (strict) | L | Δ | L_topo |
|---|---|---|---|---|---|
| dadsgnn | 0.862±0.075 | 0.522±0.037 | 0.022 | 0.340 | 0.023 |
| sesgformer | 0.873±0.077 | 0.545±0.046 | 0.045 | 0.328 | 0.043 |
| sgcl | 0.877±0.083 | 0.524±0.043 | 0.024 | 0.353 | 0.021 |
| sgcn | 0.870±0.078 | 0.523±0.042 | 0.023 | 0.346 | 0.026 |
| sgt | 0.886±0.080 | 0.532±0.028 | 0.032 | 0.354 | 0.030 |
| sigat | 0.888±0.079 | 0.515±0.006 | 0.015 | 0.373 | 0.017 |
| sigformer | 0.884±0.083 | 0.522±0.033 | 0.022 | 0.362 | 0.016 |

## Leakage metrics — by method × dataset

| method | dataset | real | shuffle | L | Δ | leaks? |
|---|---|---|---|---|---|---|
| dadsgnn | bitcoin_alpha | 0.878±0.023 | 0.538±0.026 | 0.038 | 0.340 | no |
| dadsgnn | bitcoin_otc | 0.888±0.013 | 0.508±0.025 | 0.008 | 0.380 | no |
| dadsgnn | epinions | 0.908±0.017 | 0.484±0.020 | 0.000 | 0.424 | no |
| dadsgnn | reddit_body | 0.757±0.004 | 0.561±0.024 | 0.061 | 0.195 | **yes** |
| dadsgnn | slashdot | 0.881±0.008 | 0.521±0.005 | 0.021 | 0.360 | no |
| sesgformer | bitcoin_alpha | 0.888±0.014 | 0.565±0.023 | 0.065 | 0.323 | **yes** |
| sesgformer | bitcoin_otc | 0.908±0.009 | 0.554±0.038 | 0.054 | 0.354 | **yes** |
| sesgformer | epinions | 0.919±0.006 | 0.500±0.025 | 0.000 | 0.419 | no |
| sesgformer | reddit_body | 0.765±0.007 | 0.591±0.021 | 0.091 | 0.174 | **yes** |
| sesgformer | slashdot | 0.884±0.005 | 0.515±0.005 | 0.015 | 0.369 | no |
| sgcl | bitcoin_alpha | 0.901±0.011 | 0.545±0.024 | 0.045 | 0.356 | no |
| sgcl | bitcoin_otc | 0.915±0.013 | 0.514±0.012 | 0.014 | 0.401 | no |
| sgcl | epinions | 0.922±0.005 | 0.478±0.009 | 0.000 | 0.445 | no |
| sgcl | reddit_body | 0.760±0.006 | 0.569±0.020 | 0.069 | 0.191 | **yes** |
| sgcl | slashdot | 0.885±0.003 | 0.515±0.005 | 0.015 | 0.370 | no |
| sgcn | bitcoin_alpha | 0.891±0.011 | 0.543±0.021 | 0.043 | 0.348 | no |
| sgcn | bitcoin_otc | 0.910±0.007 | 0.510±0.013 | 0.010 | 0.400 | no |
| sgcn | epinions | 0.910±0.005 | 0.481±0.011 | 0.000 | 0.430 | no |
| sgcn | reddit_body | 0.759±0.007 | 0.568±0.019 | 0.068 | 0.191 | **yes** |
| sgcn | slashdot | 0.878±0.007 | 0.514±0.007 | 0.014 | 0.364 | no |
| sgt | bitcoin_alpha | 0.897±0.005 | 0.514±0.019 | 0.014 | 0.383 | no |
| sgt | bitcoin_otc | 0.912±0.017 | 0.521±0.044 | 0.021 | 0.391 | no |
| sgt | epinions | 0.943±0.003 | 0.541±0.026 | 0.041 | 0.401 | no |
| sgt | reddit_body | 0.775±0.003 | 0.567±0.024 | 0.067 | 0.208 | **yes** |
| sgt | slashdot | 0.903±0.002 | 0.516±0.013 | 0.016 | 0.387 | no |
| sigat | bitcoin_alpha | 0.906±0.008 | 0.513±0.035 | 0.013 | 0.393 | no |
| sigat | bitcoin_otc | 0.911±0.007 | 0.516±0.034 | 0.016 | 0.395 | no |
| sigat | epinions | 0.944±0.005 | 0.512±0.008 | 0.012 | 0.432 | no |
| sigat | reddit_body | 0.778±0.006 | 0.523±0.019 | 0.023 | 0.255 | no |
| sigat | slashdot | 0.899±0.005 | 0.510±0.007 | 0.010 | 0.389 | no |
| sigformer | bitcoin_alpha | 0.897±0.012 | 0.522±0.034 | 0.022 | 0.375 | no |
| sigformer | bitcoin_otc | 0.913±0.008 | 0.507±0.022 | 0.007 | 0.407 | no |
| sigformer | epinions | 0.938±0.008 | 0.503±0.012 | 0.003 | 0.435 | no |
| sigformer | reddit_body | 0.767±0.005 | 0.569±0.003 | 0.069 | 0.199 | **yes** |
| sigformer | slashdot | 0.903±0.003 | 0.510±0.006 | 0.010 | 0.394 | no |

## Leakage metrics — cycle/HSiKAN method across the reachability lattice

Leakage (`L > 0`) appears only at `full` (held-out sign reachable); `topo` keeps real signal but shuffles to chance ⇒ cycle topology is a clean feature.

| rule | dataset | real | shuffle | L | Δ | leaks? |
|---|---|---|---|---|---|---|
| strict | bitcoin_alpha | 0.500±0.000 | 0.500±0.000 | 0.000 | 0.000 | no |
| strict | bitcoin_otc | 0.500±0.000 | 0.500±0.000 | 0.000 | 0.000 | no |
| topo | bitcoin_alpha | 0.804±0.024 | 0.474±0.037 | 0.000 | 0.330 | no |
| topo | bitcoin_otc | 0.840±0.022 | 0.490±0.025 | 0.000 | 0.350 | no |
| full | bitcoin_alpha | 0.890±0.032 | 0.743±0.057 | 0.243 | 0.147 | **yes** |
| full | bitcoin_otc | 0.864±0.016 | 0.664±0.041 | 0.164 | 0.200 | **yes** |
