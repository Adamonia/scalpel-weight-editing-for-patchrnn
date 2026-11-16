# Why is noise-to-signal higher in M3, M4, M13 than in M5, M8, M12?

All numbers are read from `out/trace_M*.json` (step1.py), `out/readout_M*.json` (step2.py) and `out/weights.json` (step3.py). No gradient training. K = 10 fresh N(0,1) draws (seeds 0..9) in the untrained message-embedding rows against the zeroed ("clean") model, on all 7608 test samples. Group A = {M3, M4, M13}, group B = {M5, M8, M12}. One trained model per architecture, n = 6: nothing here is a general claim about architectures.

## 0. Checks

| Model | heads (registry) | layers | untrained rows | hit samples | max abs gap, zeroed manual forward vs cached `untrained_zero` (all 7608) |
|---|---|---|---|---|---|
| M3 | 1 | 1 | 14872 | 4328 | 2.7e-06 |
| M4 | 2 | 1 | 14872 | 4328 | 0.0e+00 |
| M5 | 8 | 1 | 14872 | 4328 | 0.0e+00 |
| M8 | 16 | 1 | 14872 | 4328 | 3.0e-06 |
| M12 | 8 | 2 | 14872 | 4328 | 2.9e-06 |
| M13 | 1 | 2 | 14872 | 4328 | 6.2e-06 |

Head-count check by function. The checkpoint shapes cannot give the head count (in_proj_weight is (3d, d) for any number of heads). Instead the clean message branch was recomputed with H = 1, 2, 4, 8, 16 on 256 hit samples; max abs margin gap to the cached margin:

| Model | registry H | H=1 | H=2 | H=4 | H=8 | H=16 |
|---|---|---|---|---|---|---|
| M3 | 1 | 0.0e+00 | 1.8e+00 | 3.5e+00 | 4.6e+00 | 6.1e+00 |
| M4 | 2 | 5.9e+00 | 0.0e+00 | 5.6e+00 | 7.1e+00 | 6.9e+00 |
| M5 | 8 | 7.6e+00 | 6.3e+00 | 6.8e+00 | 0.0e+00 | 3.1e+00 |
| M8 | 16 | 1.0e+01 | 1.0e+01 | 9.0e+00 | 7.5e+00 | 0.0e+00 |
| M12 | 8 | 1.2e+01 | 1.0e+01 | 8.0e+00 | 0.0e+00 | 3.6e+00 |
| M13 | 1 | 0.0e+00 | 7.4e+00 | 5.8e+00 | 7.0e+00 | 8.2e+00 |

The registry head count is the only value that reproduces the cached margins. (This tests the head count the weights were trained with, not whether the head split is "meaningful".)

## S1. Stage trace of the message branch

Stages: `emb` = embedding + position; per layer l: `Ll.a` attention output (after out_proj), `Ll.n1` after norm1(x+a), `Ll.ff` feed-forward output, `Ll.n2` after norm2 (layer output); `pool` = final LayerNorm of the CLS vector; classifier `c0` = LayerNorm(384) of [twin_b, twin_a, msg], `c2` = Linear(384,384), `c3` = LayerNorm(384); `margin` = logit1 - logit0. Stage vectors are taken at the CLS position (the only position that is read out); for `emb` the CLS vector is unchanged by construction, so its sequence-level numbers are given in a separate table. Statistics are over the 4328 hit samples (the other samples have change 0) and the 10 draws. "Change" = noise minus clean. `ratio_raw` = mean |change| / mean |clean|. `ratio_centered` = mean |change| / mean |clean - mean over samples of clean|, i.e. relative to the part of the clean activation that varies between samples (the signal). Amplification = ratio at this stage / ratio at the previous stage that carries a change. Mark: A>B means all of M3, M4, M13 are above all of M5, M8, M12 at this stage.

**PRIMARY. Noise-to-signal at each stage, same definition as the margin NSR: rms over samples of the sd over draws (norm of the per-coordinate sd vector) / rms over samples of |clean centered|**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 0.298 | 0.373 | 0.239 | 0.278 | 0.317 | 0.263 | overlap |
| L0.n1 | 0.294 | 0.402 | 0.248 | 0.289 | 0.322 | 0.273 | overlap |
| L0.ff | 0.256 | 0.405 | 0.226 | 0.241 | 0.278 | 0.285 | overlap |
| L0.n2 | 0.317 | 0.427 | 0.249 | 0.275 | 0.306 | 0.267 | overlap |
| L1.a | - | - | - | - | 0.279 | 0.306 | n/a |
| L1.n1 | - | - | - | - | 0.282 | 0.322 | n/a |
| L1.ff | - | - | - | - | 0.256 | 0.328 | n/a |
| L1.n2 | - | - | - | - | 0.289 | 0.359 | n/a |
| pool | 0.316 | 0.425 | 0.249 | 0.274 | 0.288 | 0.358 | A>B |
| c0 | 0.246 | 0.313 | 0.191 | 0.209 | 0.217 | 0.266 | A>B |
| c2 | 0.290 | 0.374 | 0.209 | 0.241 | 0.253 | 0.335 | A>B |
| c3 | 0.266 | 0.361 | 0.191 | 0.222 | 0.233 | 0.305 | A>B |
| margin | 0.257 | 0.334 | 0.189 | 0.206 | 0.210 | 0.285 | A>B |

**Stage-to-stage amplification of the primary noise-to-signal**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.n1 | 0.99 | 1.08 | 1.04 | 1.04 | 1.02 | 1.04 | overlap |
| L0.ff | 0.87 | 1.01 | 0.91 | 0.83 | 0.86 | 1.05 | overlap |
| L0.n2 | 1.24 | 1.05 | 1.10 | 1.14 | 1.10 | 0.94 | overlap |
| L1.a | - | - | - | - | 0.91 | 1.14 | n/a |
| L1.n1 | - | - | - | - | 1.01 | 1.05 | n/a |
| L1.ff | - | - | - | - | 0.91 | 1.02 | n/a |
| L1.n2 | - | - | - | - | 1.13 | 1.09 | n/a |
| pool | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | overlap |
| c0 | 0.78 | 0.74 | 0.77 | 0.76 | 0.75 | 0.74 | overlap |
| c2 | 1.18 | 1.20 | 1.09 | 1.16 | 1.17 | 1.26 | A>B |
| c3 | 0.91 | 0.96 | 0.91 | 0.92 | 0.92 | 0.91 | overlap |
| margin | 0.97 | 0.92 | 0.99 | 0.93 | 0.90 | 0.93 | overlap |

**Same with means of norms instead of rms: mean sd-norm / mean |clean centered|**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 0.282 | 0.305 | 0.237 | 0.280 | 0.299 | 0.158 | overlap |
| L0.n1 | 0.272 | 0.316 | 0.239 | 0.283 | 0.301 | 0.162 | overlap |
| L0.ff | 0.249 | 0.327 | 0.220 | 0.250 | 0.272 | 0.169 | overlap |
| L0.n2 | 0.271 | 0.334 | 0.223 | 0.258 | 0.288 | 0.159 | overlap |
| L1.a | - | - | - | - | 0.275 | 0.273 | n/a |
| L1.n1 | - | - | - | - | 0.262 | 0.263 | n/a |
| L1.ff | - | - | - | - | 0.237 | 0.282 | n/a |
| L1.n2 | - | - | - | - | 0.251 | 0.288 | n/a |
| pool | 0.271 | 0.333 | 0.223 | 0.258 | 0.250 | 0.287 | A>B |
| c0 | 0.200 | 0.237 | 0.163 | 0.187 | 0.184 | 0.209 | A>B |
| c2 | 0.248 | 0.294 | 0.187 | 0.224 | 0.219 | 0.269 | A>B |
| c3 | 0.236 | 0.286 | 0.179 | 0.213 | 0.208 | 0.253 | A>B |
| margin | 0.238 | 0.279 | 0.185 | 0.211 | 0.195 | 0.246 | A>B |

**Mean norm of the sd vector over draws (noise around its own mean)**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 1.707 | 3.005 | 1.580 | 2.004 | 1.947 | 1.298 | overlap |
| L0.n1 | 1.111 | 1.872 | 0.970 | 1.263 | 1.387 | 0.720 | overlap |
| L0.ff | 2.544 | 4.759 | 1.671 | 2.288 | 0.551 | 0.399 | overlap |
| L0.n2 | 2.150 | 3.035 | 1.607 | 2.041 | 1.500 | 0.689 | overlap |
| L1.a | - | - | - | - | 1.873 | 3.615 | n/a |
| L1.n1 | - | - | - | - | 1.943 | 1.915 | n/a |
| L1.ff | - | - | - | - | 2.587 | 2.978 | n/a |
| L1.n2 | - | - | - | - | 2.302 | 2.659 | n/a |
| pool | 2.171 | 3.070 | 1.615 | 2.056 | 2.316 | 2.687 | overlap |
| c0 | 2.203 | 3.151 | 1.630 | 2.084 | 2.350 | 2.742 | overlap |
| c2 | 4.718 | 8.531 | 2.619 | 3.778 | 4.329 | 6.703 | A>B |
| c3 | 1.703 | 3.144 | 1.094 | 1.624 | 1.983 | 2.695 | overlap |
| margin | 0.555 | 0.816 | 0.426 | 0.608 | 0.653 | 0.693 | overlap |

The tables below use the change relative to the clean (zeroed) model, as asked. This includes the systematic shift between the zeroed model and the mean of the draws, so it is larger than the draw-to-draw sd.

**Where the primary noise-to-signal is built (read from the primary table).** Factors are ratios of the primary NSR between stages.

| Model | NSR L0.a | NSR L0.n2 | NSR L1.a | NSR last layer out | NSR pool | NSR margin (hit samples) | L0.n2 / L0.a | layer 1 factor | pool / L0.a (all after first attention) | margin / pool (classifier) |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 0.298 | 0.317 | - | 0.317 | 0.316 | 0.257 | 1.06 | - | 1.06 | 0.81 |
| M4 | 0.373 | 0.427 | - | 0.427 | 0.425 | 0.334 | 1.14 | - | 1.14 | 0.79 |
| M5 | 0.239 | 0.249 | - | 0.249 | 0.249 | 0.189 | 1.04 | - | 1.04 | 0.76 |
| M8 | 0.278 | 0.275 | - | 0.275 | 0.274 | 0.206 | 0.99 | - | 0.99 | 0.75 |
| M12 | 0.317 | 0.306 | 0.279 | 0.289 | 0.288 | 0.210 | 0.97 | 0.95 | 0.91 | 0.73 |
| M13 | 0.263 | 0.267 | 0.306 | 0.359 | 0.358 | 0.285 | 1.02 | 1.34 | 1.36 | 0.80 |

**Bootstrap over test samples (step5.py), NSR of the margin, all 7608 samples.** 95% intervals over 1000 resamples; they cover test-sample uncertainty only, not the choice of the 10 draws and not the training run.

| Model | NSR | 95% interval |
|---|---|---|
| M3 | 0.189 | 0.183 - 0.195 |
| M4 | 0.248 | 0.240 - 0.255 |
| M5 | 0.139 | 0.135 - 0.143 |
| M8 | 0.152 | 0.148 - 0.156 |
| M12 | 0.157 | 0.152 - 0.161 |
| M13 | 0.210 | 0.203 - 0.216 |

Group ratio (mean NSR of A / mean NSR of B): median 1.44, 95% interval 1.42 - 1.47. Share of resamples with min(A) > max(B): 1.00.

**Mean norm of the clean activation**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 7.41 | 14.82 | 7.75 | 8.42 | 7.25 | 9.46 | overlap |
| L0.n1 | 11.20 | 11.28 | 11.24 | 11.22 | 11.35 | 11.58 | overlap |
| L0.ff | 10.73 | 16.89 | 8.81 | 10.30 | 3.03 | 4.48 | overlap |
| L0.n2 | 11.38 | 11.39 | 11.38 | 11.37 | 11.31 | 11.03 | overlap |
| L1.a | - | - | - | - | 7.77 | 18.37 | n/a |
| L1.n1 | - | - | - | - | 11.33 | 11.30 | n/a |
| L1.ff | - | - | - | - | 11.98 | 11.10 | n/a |
| L1.n2 | - | - | - | - | 11.38 | 11.37 | n/a |
| pool | 11.49 | 11.56 | 11.44 | 11.44 | 11.47 | 11.52 | A>B |
| c0 | 19.63 | 19.56 | 19.63 | 19.59 | 19.59 | 19.56 | overlap |
| c2 | 47.76 | 48.27 | 42.17 | 39.89 | 37.29 | 42.96 | A>B |
| c3 | 18.42 | 18.17 | 18.96 | 19.06 | 19.00 | 18.19 | A<B |
| margin | 2.60 | 3.57 | 2.46 | 3.38 | 3.63 | 3.12 | overlap |

**Mean norm of the clean activation, centered over samples**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 6.05 | 9.86 | 6.66 | 7.14 | 6.50 | 8.22 | overlap |
| L0.n1 | 4.08 | 5.92 | 4.06 | 4.46 | 4.61 | 4.43 | overlap |
| L0.ff | 10.24 | 14.55 | 7.60 | 9.14 | 2.02 | 2.36 | overlap |
| L0.n2 | 7.94 | 9.09 | 7.20 | 7.91 | 5.21 | 4.33 | overlap |
| L1.a | - | - | - | - | 6.82 | 13.25 | n/a |
| L1.n1 | - | - | - | - | 7.41 | 7.29 | n/a |
| L1.ff | - | - | - | - | 10.91 | 10.54 | n/a |
| L1.n2 | - | - | - | - | 9.17 | 9.24 | n/a |
| pool | 8.02 | 9.23 | 7.24 | 7.97 | 9.25 | 9.35 | overlap |
| c0 | 10.99 | 13.28 | 9.99 | 11.13 | 12.79 | 13.14 | overlap |
| c2 | 19.01 | 29.02 | 14.01 | 16.86 | 19.74 | 24.91 | overlap |
| c3 | 7.21 | 11.01 | 6.11 | 7.64 | 9.52 | 10.66 | overlap |
| margin | 2.33 | 2.93 | 2.31 | 2.89 | 3.34 | 2.82 | overlap |

**Mean norm of the change**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 1.339 | 1.880 | 1.355 | 1.869 | 1.714 | 0.744 | overlap |
| L0.n1 | 0.871 | 1.175 | 0.829 | 1.173 | 1.216 | 0.409 | overlap |
| L0.ff | 1.827 | 3.236 | 1.292 | 1.915 | 0.465 | 0.236 | overlap |
| L0.n2 | 1.588 | 1.968 | 1.283 | 1.762 | 1.300 | 0.385 | overlap |
| L1.a | - | - | - | - | 1.947 | 2.786 | n/a |
| L1.n1 | - | - | - | - | 1.923 | 1.469 | n/a |
| L1.ff | - | - | - | - | 2.264 | 2.241 | n/a |
| L1.n2 | - | - | - | - | 2.120 | 2.038 | n/a |
| pool | 1.603 | 1.994 | 1.289 | 1.775 | 2.132 | 2.059 | overlap |
| c0 | 1.627 | 2.047 | 1.301 | 1.799 | 2.164 | 2.101 | overlap |
| c2 | 3.413 | 5.552 | 2.023 | 3.141 | 3.774 | 5.081 | overlap |
| c3 | 1.216 | 2.040 | 0.840 | 1.353 | 1.720 | 2.009 | overlap |
| margin | 0.388 | 0.499 | 0.321 | 0.492 | 0.546 | 0.504 | overlap |

**Ratio |change| / |clean| (raw)**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 0.181 | 0.127 | 0.175 | 0.222 | 0.236 | 0.079 | overlap |
| L0.n1 | 0.078 | 0.104 | 0.074 | 0.105 | 0.107 | 0.035 | overlap |
| L0.ff | 0.170 | 0.192 | 0.147 | 0.186 | 0.153 | 0.053 | overlap |
| L0.n2 | 0.140 | 0.173 | 0.113 | 0.155 | 0.115 | 0.035 | overlap |
| L1.a | - | - | - | - | 0.251 | 0.152 | n/a |
| L1.n1 | - | - | - | - | 0.170 | 0.130 | n/a |
| L1.ff | - | - | - | - | 0.189 | 0.202 | n/a |
| L1.n2 | - | - | - | - | 0.186 | 0.179 | n/a |
| pool | 0.139 | 0.173 | 0.113 | 0.155 | 0.186 | 0.179 | overlap |
| c0 | 0.083 | 0.105 | 0.066 | 0.092 | 0.110 | 0.107 | overlap |
| c2 | 0.071 | 0.115 | 0.048 | 0.079 | 0.101 | 0.118 | overlap |
| c3 | 0.066 | 0.112 | 0.044 | 0.071 | 0.090 | 0.110 | overlap |
| margin | 0.149 | 0.140 | 0.131 | 0.146 | 0.150 | 0.161 | overlap |

**Ratio |change| / |clean centered| (signal-relative)**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.a | 0.221 | 0.191 | 0.203 | 0.262 | 0.264 | 0.091 | overlap |
| L0.n1 | 0.213 | 0.199 | 0.204 | 0.263 | 0.264 | 0.092 | overlap |
| L0.ff | 0.179 | 0.222 | 0.170 | 0.210 | 0.230 | 0.100 | overlap |
| L0.n2 | 0.200 | 0.217 | 0.178 | 0.223 | 0.249 | 0.089 | overlap |
| L1.a | - | - | - | - | 0.285 | 0.210 | n/a |
| L1.n1 | - | - | - | - | 0.260 | 0.202 | n/a |
| L1.ff | - | - | - | - | 0.207 | 0.213 | n/a |
| L1.n2 | - | - | - | - | 0.231 | 0.221 | n/a |
| pool | 0.200 | 0.216 | 0.178 | 0.223 | 0.231 | 0.220 | overlap |
| c0 | 0.148 | 0.154 | 0.130 | 0.162 | 0.169 | 0.160 | overlap |
| c2 | 0.180 | 0.191 | 0.144 | 0.186 | 0.191 | 0.204 | overlap |
| c3 | 0.169 | 0.185 | 0.137 | 0.177 | 0.181 | 0.188 | overlap |
| margin | 0.167 | 0.171 | 0.139 | 0.170 | 0.163 | 0.179 | overlap |

**Stage-to-stage amplification of ratio_centered**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.n1 | 0.96 | 1.04 | 1.00 | 1.01 | 1.00 | 1.02 | overlap |
| L0.ff | 0.84 | 1.12 | 0.83 | 0.80 | 0.87 | 1.08 | overlap |
| L0.n2 | 1.12 | 0.97 | 1.05 | 1.06 | 1.09 | 0.89 | overlap |
| L1.a | - | - | - | - | 1.14 | 2.37 | n/a |
| L1.n1 | - | - | - | - | 0.91 | 0.96 | n/a |
| L1.ff | - | - | - | - | 0.80 | 1.05 | n/a |
| L1.n2 | - | - | - | - | 1.11 | 1.04 | n/a |
| pool | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | overlap |
| c0 | 0.74 | 0.71 | 0.73 | 0.73 | 0.73 | 0.73 | overlap |
| c2 | 1.21 | 1.24 | 1.11 | 1.15 | 1.13 | 1.28 | A>B |
| c3 | 0.94 | 0.97 | 0.95 | 0.95 | 0.95 | 0.92 | overlap |
| margin | 0.99 | 0.92 | 1.01 | 0.96 | 0.90 | 0.95 | overlap |

**Stage-to-stage amplification of ratio_raw**

| stage | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| L0.n1 | 0.43 | 0.82 | 0.42 | 0.47 | 0.45 | 0.45 | overlap |
| L0.ff | 2.19 | 1.84 | 1.99 | 1.78 | 1.43 | 1.49 | overlap |
| L0.n2 | 0.82 | 0.90 | 0.77 | 0.83 | 0.75 | 0.66 | overlap |
| L1.a | - | - | - | - | 2.18 | 4.35 | n/a |
| L1.n1 | - | - | - | - | 0.68 | 0.86 | n/a |
| L1.ff | - | - | - | - | 1.11 | 1.55 | n/a |
| L1.n2 | - | - | - | - | 0.99 | 0.89 | n/a |
| pool | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | overlap |
| c0 | 0.59 | 0.61 | 0.59 | 0.59 | 0.59 | 0.60 | A>B |
| c2 | 0.86 | 1.10 | 0.72 | 0.86 | 0.92 | 1.10 | overlap |
| c3 | 0.92 | 0.98 | 0.92 | 0.90 | 0.89 | 0.93 | A>B |
| margin | 2.26 | 1.25 | 2.95 | 2.05 | 1.66 | 1.46 | overlap |

**Sequence level (all 200 positions, Frobenius norm per sample), embedding and layer 0.** Shows how much change enters the layer and how much of it is at the CLS position.

| stage (seq ratio) | M3 | M4 | M5 | M8 | M12 | M13 | A vs B |
|---|---|---|---|---|---|---|---|
| emb | 0.085 | 0.081 | 0.080 | 0.080 | 0.081 | 0.081 | overlap |
| L0.a | 0.075 | 0.029 | 0.041 | 0.042 | 0.230 | 0.259 | overlap |
| L0.n1 | 0.087 | 0.071 | 0.084 | 0.081 | 0.117 | 0.139 | overlap |
| L0.ff | 0.073 | 0.040 | 0.089 | 0.081 | 0.129 | 0.192 | overlap |
| L0.n2 | 0.090 | 0.040 | 0.082 | 0.077 | 0.123 | 0.154 | overlap |

**Layer-0 CLS attention output: where the change comes from.** Exact split of change of a = out_proj(sum_h w_h V_h): value path = out_proj(w_new (V_new - V_old)), weight path = out_proj((w_new - w_old) V_old). The CLS query itself does not change at layer 0 (the CLS input is not touched). Norms are means over hit samples and draws.

| Model | |change of a| | |value path| | |weight path| | cos(value, weight) | value / total | weight / total |
|---|---|---|---|---|---|---|
| M3 | 1.339 | 1.142 | 0.645 | -0.020 | 0.85 | 0.48 |
| M4 | 1.880 | 0.628 | 1.703 | 0.047 | 0.33 | 0.91 |
| M5 | 1.355 | 0.889 | 0.961 | 0.017 | 0.66 | 0.71 |
| M8 | 1.869 | 1.185 | 1.401 | 0.010 | 0.63 | 0.75 |
| M12 | 1.714 | 1.077 | 1.299 | 0.001 | 0.63 | 0.76 |
| M13 | 0.744 | 0.422 | 0.610 | -0.015 | 0.57 | 0.82 |

**Branch contribution to the clean margin variance (all 7608 samples).** The classifier has LayerNorms, so the shares need not add to 1. "msg varies only" = margin with twin_b and twin_a features fixed at their means.

| Model | var(margin) | var, msg varies only | share msg | var, code varies only | share code | R2 full vs msg-only | R2 full vs code-only |
|---|---|---|---|---|---|---|---|
| M3 | 8.472 | 7.813 | 0.922 | 0.476 | 0.056 | 0.960 | 0.142 |
| M4 | 12.783 | 11.577 | 0.906 | 2.319 | 0.181 | 0.922 | 0.195 |
| M5 | 8.539 | 7.359 | 0.862 | 0.801 | 0.094 | 0.927 | 0.221 |
| M8 | 13.013 | 10.833 | 0.832 | 2.021 | 0.155 | 0.890 | 0.250 |
| M12 | 16.443 | 14.237 | 0.866 | 2.258 | 0.137 | 0.907 | 0.216 |
| M13 | 11.718 | 11.011 | 0.940 | 1.011 | 0.086 | 0.954 | 0.155 |

## S2. Readout gain at the pooled vector

g_i = gradient of the margin with respect to the pooled message vector (128-d) at the clean point, per sample (input gradient, no weights trained). Alongness = |change . g/|g|| / |change| per sample. For a random direction in 128-d the mean absolute cosine is 0.071. Margin sd = per-sample sd over draws, rms over all 7608 samples. Decomposition: NSR = (rms |dp| / rms |p centered|) x (gain on noise / gain on signal), where gain on noise = margin sd / rms |dp| and gain on signal = clean margin sd / rms |p centered| (all 7608 samples).

| Model | mean |g| | mean |dp| | mean |along g| | mean |orthogonal| | |along|/|dp| (ratio of means) | mean per-sample |cos| | margin sd, actual | margin sd, along-g part only | corr(linear, actual change) |
|---|---|---|---|---|---|---|---|---|---|
| M3 | 0.31 | 1.60 | 1.273 | 0.925 | 0.79 | 0.70 | 0.550 | 0.520 | 0.990 |
| M4 | 0.26 | 1.99 | 1.475 | 1.257 | 0.74 | 0.66 | 0.885 | 0.590 | 0.870 |
| M5 | 0.34 | 1.29 | 0.972 | 0.797 | 0.75 | 0.64 | 0.406 | 0.400 | 0.998 |
| M8 | 0.38 | 1.77 | 1.305 | 1.129 | 0.74 | 0.61 | 0.548 | 0.546 | 0.994 |
| M12 | 0.37 | 2.13 | 1.478 | 1.429 | 0.69 | 0.57 | 0.636 | 0.611 | 0.985 |
| M13 | 0.28 | 2.06 | 1.638 | 1.182 | 0.80 | 0.73 | 0.718 | 0.594 | 0.963 |

| Model | rms |dp| (all) | rms |p centered| | relative change at pooled | gain on noise | gain on signal | gain ratio noise/signal | NSR (= product) |
|---|---|---|---|---|---|---|---|
| M3 | 2.174 | 8.937 | 0.243 | 0.253 | 0.326 | 0.776 | 0.189 |
| M4 | 3.669 | 9.978 | 0.368 | 0.241 | 0.358 | 0.674 | 0.248 |
| M5 | 1.594 | 8.143 | 0.196 | 0.255 | 0.359 | 0.709 | 0.139 |
| M8 | 1.990 | 8.838 | 0.225 | 0.276 | 0.408 | 0.675 | 0.152 |
| M12 | 2.367 | 9.842 | 0.241 | 0.269 | 0.412 | 0.652 | 0.157 |
| M13 | 2.907 | 9.985 | 0.291 | 0.247 | 0.343 | 0.720 | 0.210 |

| Model | participation ratio of dp | top-1 PC share of dp | top-5 share of dp | participation ratio, clean pooled | top-5 share, clean pooled | cos(g, clean centered pooled) | |mean dp over draws| | |dp - mean over draws| |
|---|---|---|---|---|---|---|---|---|
| M3 | 1.6 | 0.76 | 0.97 | 1.6 | 0.99 | -0.13 | 0.85 | 1.64 |
| M4 | 1.8 | 0.72 | 0.98 | 1.8 | 0.99 | -0.18 | 1.86 | 2.31 |
| M5 | 1.7 | 0.75 | 0.92 | 1.6 | 0.96 | -0.12 | 0.72 | 1.26 |
| M8 | 2.0 | 0.69 | 0.90 | 1.7 | 0.95 | -0.15 | 1.05 | 1.65 |
| M12 | 2.3 | 0.63 | 0.88 | 1.7 | 0.95 | -0.12 | 1.22 | 1.89 |
| M13 | 1.8 | 0.69 | 0.98 | 1.6 | 0.98 | -0.08 | 1.27 | 2.05 |

## S3. Attention shape (CLS query row, the row that feeds pooling)

Rows: 200 keys (CLS + 199 message positions, padding not masked); maximal entropy = 5.30 nats. Entropy and max weight are means over samples (hit samples) and heads. Mass on untrained = sum of the attention weights on untrained key positions; distribution over samples and heads (hit samples; "fresh" pools the 10 draws, "clean" is the zeroed model). Chance level (untrained keys / keys, mean over hit samples) = 0.017. "Max single" = largest weight on any one untrained token. Per-head values are in the json.

| Model | layer | rows | entropy mean | entropy per-head range | max weight mean | mass U mean | sd | p90 | p99 | max single U weight mean | p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 0 | fresh | 2.65 | 2.65-2.65 | 0.29 | 0.143 | 0.159 | 0.361 | 0.728 | 0.106 | 0.646 |
| M3 | 0 | clean | 2.72 | 2.72-2.72 | 0.28 | 0.100 | 0.097 | 0.217 | 0.513 | 0.065 | 0.355 |
| M4 | 0 | fresh | 2.71 | 2.49-2.93 | 0.38 | 0.067 | 0.190 | 0.189 | 0.962 | 0.063 | 0.949 |
| M4 | 0 | clean | 2.84 | 2.65-3.04 | 0.35 | 0.002 | 0.004 | 0.004 | 0.018 | 0.001 | 0.017 |
| M5 | 0 | fresh | 2.23 | 1.35-4.75 | 0.44 | 0.084 | 0.188 | 0.286 | 0.914 | 0.074 | 0.880 |
| M5 | 0 | clean | 2.28 | 1.39-4.75 | 0.43 | 0.033 | 0.090 | 0.087 | 0.462 | 0.027 | 0.397 |
| M8 | 0 | fresh | 1.51 | 0.61-3.74 | 0.59 | 0.093 | 0.230 | 0.330 | 0.994 | 0.087 | 0.989 |
| M8 | 0 | clean | 1.53 | 0.63-3.77 | 0.58 | 0.032 | 0.121 | 0.046 | 0.748 | 0.029 | 0.710 |
| M12 | 0 | fresh | 1.28 | 0.75-2.27 | 0.63 | 0.101 | 0.236 | 0.401 | 0.990 | 0.094 | 0.984 |
| M12 | 0 | clean | 1.33 | 0.77-2.31 | 0.61 | 0.027 | 0.101 | 0.047 | 0.596 | 0.024 | 0.530 |
| M12 | 1 | fresh | 2.02 | 0.80-4.74 | 0.51 | 0.108 | 0.227 | 0.407 | 0.978 | 0.098 | 0.969 |
| M12 | 1 | clean | 2.03 | 0.72-4.74 | 0.51 | 0.116 | 0.243 | 0.449 | 0.989 | 0.108 | 0.986 |
| M13 | 0 | fresh | 1.30 | 1.30-1.30 | 0.65 | 0.059 | 0.178 | 0.141 | 0.952 | 0.055 | 0.927 |
| M13 | 0 | clean | 1.36 | 1.36-1.36 | 0.64 | 0.003 | 0.008 | 0.011 | 0.040 | 0.003 | 0.032 |
| M13 | 1 | fresh | 3.27 | 3.27-3.27 | 0.31 | 0.047 | 0.133 | 0.123 | 0.757 | 0.042 | 0.713 |
| M13 | 1 | clean | 3.33 | 3.33-3.33 | 0.30 | 0.018 | 0.073 | 0.026 | 0.411 | 0.017 | 0.410 |

## S4. Weights (message branch and classifier)

Frobenius / spectral norm. in_proj is split into Q, K, V blocks (128 x 128 each). LN gain = mean of the LayerNorm weight. No scale factor is applied to the embeddings in the forward code. Classifier input order is [twin_b, twin_a, msg]; "msg columns" are columns 256..383 of Linear(384,384).

| tensor | M3 | M4 | M5 | M8 | M12 | M13 |
|---|---|---|---|---|---|---|
| layer0 Q | 8.03 / 1.40 | 8.24 / 2.20 | 8.17 / 2.04 | 8.48 / 2.99 | 9.07 / 2.91 | 8.74 / 1.77 |
| layer0 K | 7.81 / 2.07 | 8.26 / 2.42 | 8.28 / 1.63 | 8.51 / 1.58 | 8.79 / 1.77 | 8.67 / 2.14 |
| layer0 V | 8.54 / 3.09 | 8.65 / 2.49 | 8.31 / 1.85 | 8.45 / 1.70 | 8.48 / 1.64 | 8.52 / 2.42 |
| layer0 out_proj | 7.37 / 2.97 | 7.30 / 2.18 | 7.05 / 2.56 | 7.22 / 2.93 | 7.06 / 2.18 | 7.18 / 1.84 |
| layer0 linear1 (512x128) | 13.91 / 3.82 | 14.77 / 4.25 | 13.65 / 3.52 | 13.77 / 3.78 | 14.05 / 2.17 | 14.48 / 2.90 |
| layer0 linear2 (128x512) | 7.49 / 2.43 | 8.39 / 3.42 | 7.14 / 2.04 | 7.33 / 2.25 | 7.78 / 1.88 | 8.10 / 2.89 |
| layer0 norm1 gain mean | 1.004 | 1.006 | 1.003 | 1.002 | 1.004 | 1.010 |
| layer0 norm2 gain mean | 1.002 | 1.001 | 1.002 | 1.002 | 1.001 | 0.979 |
| layer1 Q | - | - | - | - | 8.67 / 2.51 | 8.28 / 1.48 |
| layer1 K | - | - | - | - | 8.67 / 1.83 | 8.22 / 1.95 |
| layer1 V | - | - | - | - | 8.42 / 1.69 | 8.41 / 2.42 |
| layer1 out_proj | - | - | - | - | 7.04 / 2.33 | 7.05 / 2.08 |
| layer1 linear1 (512x128) | - | - | - | - | 13.61 / 2.84 | 13.80 / 3.10 |
| layer1 linear2 (128x512) | - | - | - | - | 7.26 / 2.17 | 7.41 / 2.23 |
| layer1 norm1 gain mean | - | - | - | - | 1.001 | 1.000 |
| layer1 norm2 gain mean | - | - | - | - | 1.002 | 1.000 |
| final norm gain mean | 1.013 | 1.020 | 1.009 | 1.011 | 1.012 | 1.014 |
| classifier LN0 gain mean | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 | 0.997 |
| classifier LN3 gain mean | 0.984 | 0.976 | 0.989 | 0.989 | 0.988 | 0.979 |
| LN0 gain mean, msg part | 1.013 | 1.020 | 1.009 | 1.011 | 1.012 | 1.014 |
| LN0 gain mean, twin part | 0.992 | 0.990 | 0.996 | 0.994 | 0.993 | 0.989 |
| |w1 - w0| of last linear | 0.515 | 0.452 | 0.580 | 0.603 | 0.592 | 0.478 |
| classifier Linear(384,384) | 12.21 / 3.82 | 12.69 / 4.60 | 11.93 / 3.24 | 12.02 / 3.50 | 12.07 / 3.75 | 12.31 / 3.94 |
|   msg columns | 7.45 / 2.83 | 7.95 / 3.41 | 7.13 / 2.41 | 7.20 / 2.54 | 7.30 / 2.76 | 7.68 / 3.17 |
|   twin columns | 9.67 / 2.63 | 9.89 / 3.19 | 9.56 / 2.26 | 9.62 / 2.49 | 9.62 / 2.60 | 9.63 / 2.44 |
| classifier Linear(384,2) | 0.69 / 0.58 | 0.67 / 0.59 | 0.71 / 0.58 | 0.71 / 0.57 | 0.71 / 0.58 | 0.67 / 0.58 |
| trained msg rows, mean norm | 11.30 | 11.29 | 11.29 | 11.29 | 11.29 | 11.30 |
| untrained rows (original N(0,1)), mean norm | 11.29 | 11.29 | 11.30 | 11.29 | 11.29 | 11.28 |
| position embedding, mean row norm | 11.27 | 11.27 | 11.18 | 11.30 | 11.29 | 11.30 |
| CLS token norm | 10.98 | 10.23 | 11.80 | 11.46 | 10.39 | 12.20 |
| trained rows (count) | 65958 | 65958 | 65958 | 65958 | 65958 | 65958 |

## S5. Clean signal and noise term (absolute units)

Noise sd = per-sample sd of the margin over the 10 draws, rms over all 7608 samples (numerator). Clean sd = sd of the clean margin over samples (denominator). F1 and AUC in percent, clean = zeroed model, threshold margin > 0.

| Model | F1 clean | AUC clean | F1 fresh mean (sd) | F1 gap | clean margin sd | noise sd rms (numerator) | NSR | sd of msg-only clean margin | noise / msg-only sd | noise sd rms, hit only | clean sd, hit only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 74.35 | 89.56 | 73.48 (0.22) | 0.87 | 2.911 | 0.550 | 0.189 | 2.795 | 0.197 | 0.729 | 2.836 |
| M4 | 73.40 | 87.94 | 72.04 (0.33) | 1.36 | 3.575 | 0.885 | 0.248 | 3.402 | 0.260 | 1.174 | 3.519 |
| M5 | 73.82 | 88.89 | 73.02 (0.15) | 0.79 | 2.922 | 0.406 | 0.139 | 2.713 | 0.150 | 0.538 | 2.847 |
| M8 | 72.80 | 87.69 | 72.27 (0.15) | 0.53 | 3.607 | 0.548 | 0.152 | 3.291 | 0.167 | 0.727 | 3.533 |
| M12 | 74.12 | 88.95 | 73.68 (0.25) | 0.44 | 4.055 | 0.636 | 0.157 | 3.773 | 0.169 | 0.843 | 4.010 |
| M13 | 73.33 | 87.81 | 72.50 (0.23) | 0.82 | 3.423 | 0.718 | 0.210 | 3.318 | 0.216 | 0.951 | 3.333 |

- noise sd rms: group A mean 0.718 (range 0.550-0.885), group B mean 0.530 (range 0.406-0.636).
- clean margin sd: group A mean 3.303 (range 2.911-3.575), group B mean 3.528 (range 2.922-4.055).
- NSR: group A mean 0.215 (range 0.189-0.248), group B mean 0.149 (range 0.139-0.157).

## S6. Sample composition (number of untrained tokens in the window)

Noise sd = rms over samples of the per-sample sd over 10 draws. NSR(all) uses the clean sd of all 7608 samples; NSR(group) uses the clean sd inside the group. Ratio to 1 token vs sqrt(k): independent equal-size token contributions would give sqrt(k) for exactly k tokens; the 3+ group has a mean of about 7 tokens.

| Model | untrained tokens | samples | margin sd rms | median | NSR (all) | NSR (group) | mean |dp| at pooled | sd / sd(1 token) |
|---|---|---|---|---|---|---|---|---|
| M3 | 1 | 1799 | 0.662 | 0.339 | 0.227 | 0.233 | 1.36 | 1.00 |
| M3 | 2 | 970 | 0.746 | 0.435 | 0.256 | 0.256 | 1.66 | 1.13 |
| M3 | 3+ | 1559 | 0.789 | 0.495 | 0.271 | 0.287 | 1.85 | 1.19 |
| M4 | 1 | 1799 | 1.078 | 0.355 | 0.302 | 0.314 | 1.56 | 1.00 |
| M4 | 2 | 970 | 1.230 | 0.589 | 0.344 | 0.351 | 2.12 | 1.14 |
| M4 | 3+ | 1559 | 1.242 | 0.709 | 0.347 | 0.345 | 2.41 | 1.15 |
| M5 | 1 | 1799 | 0.510 | 0.300 | 0.175 | 0.180 | 1.12 | 1.00 |
| M5 | 2 | 970 | 0.561 | 0.377 | 0.192 | 0.195 | 1.35 | 1.10 |
| M5 | 3+ | 1559 | 0.554 | 0.400 | 0.190 | 0.197 | 1.45 | 1.09 |
| M8 | 1 | 1799 | 0.665 | 0.470 | 0.184 | 0.192 | 1.51 | 1.00 |
| M8 | 2 | 970 | 0.745 | 0.546 | 0.206 | 0.207 | 1.80 | 1.12 |
| M8 | 3+ | 1559 | 0.782 | 0.610 | 0.217 | 0.221 | 2.06 | 1.18 |
| M12 | 1 | 1799 | 0.745 | 0.441 | 0.184 | 0.189 | 1.80 | 1.00 |
| M12 | 2 | 970 | 0.900 | 0.563 | 0.222 | 0.224 | 2.25 | 1.21 |
| M12 | 3+ | 1559 | 0.910 | 0.592 | 0.224 | 0.224 | 2.44 | 1.22 |
| M13 | 1 | 1799 | 0.843 | 0.396 | 0.246 | 0.259 | 1.70 | 1.00 |
| M13 | 2 | 970 | 1.015 | 0.567 | 0.296 | 0.298 | 2.23 | 1.20 |
| M13 | 3+ | 1559 | 1.025 | 0.577 | 0.299 | 0.305 | 2.37 | 1.22 |

Exactly one untrained token: NSR(group) A = 0.233, 0.314, 0.259, B = 0.180, 0.192, 0.189; min A 0.233 vs max B 0.192.

## Conclusion

n = 6 models, one trained model per architecture, head counts from the loader registry (confirmed by function only: only the registry H reproduces the cached margins, section 0). Nothing below is a claim about architectures in general. A perfect 3-vs-3 split on one metric has probability 0.1 by chance for six values (2 of 20 splits), and many metrics were looked at, so a split on one metric is weak evidence.

**Short answer.** The extra noise-to-signal of M3, M4, M13 is a larger relative change that arrives at the pooled message vector. It is not a stronger readout, not a weaker clean signal, not sharper attention, and not feed-forward or norm amplification. Where in the network that larger relative change is created differs between the models, so the grouping {M3, M4, M13} vs {M5, M8, M12} holds at the pooled vector and after it, but not at every stage before it.

**The grouping.** The margin NSR (all samples) is A = 0.189, 0.248, 0.210 and B = 0.139, 0.152, 0.157 (bootstrap 95% intervals do not overlap; group ratio 1.44, interval 1.42-1.47, test-sample uncertainty only). Primary stage NSR (hit samples): A and B overlap at every stage of layer 0 and at every stage of layer 1 (for M12, M13). The first stage where all three of A lie above all three of B is `pool` (A 0.316, 0.425, 0.358; B 0.249, 0.274, 0.288) and it stays separated through c0, c2, c3 and the margin. Inside layer 0, M12 (0.317 at L0.a) is above M3 (0.298), M13 (0.263) is below M8 (0.278) and M5 (0.239) is lowest, so the split does not exist there.

**Where the difference is made, by model.**
- One-layer models (M3, M4 vs M5, M8): the NSR at the first attention output is already higher (0.298, 0.373 vs 0.239, 0.278), i.e. about 75% of the log difference (group means of the log ratio) at pool (0.316, 0.425 vs 0.249, 0.274) is there; the rest comes from small extra factors after it (pool / L0.a = 1.06, 1.14 vs 1.04, 0.99). M3 vs M8 differ little at L0.a (0.298 vs 0.278).
- Two-layer models: M12 and M13 have similar NSR at the end of layer 0 (0.306 vs 0.267, M13 lower). Layer 1 then lowers M12 (factor 0.95) and raises M13 (factor 1.34; L1.a alone 0.306 vs 0.279, and the norm of the clean attention output at L1.a is 18.4 for M13 vs 7.8 for M12). So for M13 the difference is created in layer 1, for M12 it is not.
- Within attention the two one-layer models of A do not share a path: at the layer-0 CLS attention output the change is mostly the value path in M3 (value 0.85 of |change|, weight path 0.48) and mostly the attention-weight path in M4 (0.33 vs 0.91). No common attention property of M3, M4, M13 was found.

**Readout (S2).** The noise change at the pooled vector lies mostly along the margin gradient in all six models: |along|/|change| 0.74-0.80 for A and 0.69-0.75 for B against 0.071 for a random direction. Both the noise and the clean pooled vector live in about 2 dimensions in every model (participation ratio 1.6-2.3), which is why the alignment is high. NSR = relative change at pooled x gain ratio: relative change A 0.243, 0.368, 0.291 vs B 0.196, 0.225, 0.241 (group means 0.301 vs 0.221, factor 1.36); gain ratio noise/signal A 0.776, 0.674, 0.720 vs B 0.709, 0.675, 0.652 (0.723 vs 0.679, factor 1.06; overlap between groups). The absolute gain is lower in A (mean |g| 0.31, 0.26, 0.28 vs 0.34, 0.38, 0.37), for noise and signal alike. The classifier after `pool` lowers the NSR in all models, a little less in A (margin/pool 0.81, 0.79, 0.80 vs 0.76, 0.75, 0.73). Part of that is that the message branch carries more of the margin variance in A (share 0.92, 0.91, 0.94 vs 0.86, 0.83, 0.87).

**Candidate explanations**

| | verdict | deciding number |
|---|---|---|
| (a) amplified more inside attention | undecided | One-layer pair: yes, NSR at L0.a 0.298, 0.373 vs 0.239, 0.278. Layer 0 of all six: no (overlap; M12 0.317, M13 0.263). M13: the rise is in layer 1 (factor 1.34 vs M12 0.95). M3 and M4 use different paths (value 0.85 vs 0.33 of the change). |
| (b) amplified inside feed-forward or norms | contradicted as the main cause | Factors n1, ff, n2 are 0.8-1.25 in all models and overlap between groups (L0.a to L0.n2: 1.06, 1.14, 1.02 for A; 1.04, 0.99, 0.97 for B). The largest single step is M13 layer 1, mostly attention plus a small ff part (1.14, 1.05, 1.02, 1.09). |
| (c) by the classifier readout | supported only as a small contributor | Gain ratio noise/signal 0.723 vs 0.679 (factor 1.06) against total NSR factor 1.44; relative change at pooled 0.301 vs 0.221 (factor 1.36) carries the rest. Margin/pool 0.79-0.81 vs 0.73-0.76. |
| (d) weaker clean signal | contradicted as the main cause | Clean margin sd: A 2.91, 3.58, 3.42 vs B 2.92, 3.61, 4.06 (means 3.30 vs 3.53, factor 1.07). Noise sd rms: A 0.55, 0.89, 0.72 vs B 0.41, 0.55, 0.64 (means 0.718 vs 0.530, factor 1.35). M3 and M5 have the same clean sd (2.91, 2.92) and NSR 0.189 vs 0.139. In log terms about 80% numerator, 20% denominator. |
| (e) sharper attention | contradicted | CLS-row entropy (fresh): M3 2.65, M4 2.71 are the most diffuse; M13 layer 0 is sharp (1.30), like M12 (1.28) and M8 (1.51). Mass on untrained tokens p99: A 0.73, 0.96, 0.95 vs B 0.91, 0.99, 0.99. Every model has a few samples where one untrained token takes almost all attention (max single weight p99 0.65-0.99), but this is not specific to A. |
| (f) no systematic property, within one-model-per-architecture variation | undecided, not excluded | The A/B split exists at pool and at the margin but not at layer-0 stages, and the mechanism differs between M3, M4 and M13. Metrics that also split 3 vs 3: mean |g|, norm of |w1 - w0|, spectral norm of the V block (A 3.09, 2.49, 2.42 vs B 1.85, 1.70, 1.64) and of the msg columns of the classifier (2.83, 3.41, 3.17 vs 2.41, 2.54, 2.76), message share of margin variance. These are correlates in six models; none was tested as a cause, and the V-norm split in particular has chance probability 0.1. |

**S6 check.** The ordering holds on samples with exactly one untrained token (NSR within group A 0.233, 0.314, 0.259; B 0.180, 0.192, 0.189), so sample composition (number of untrained tokens) does not create it. The margin sd grows only 1.09-1.22 times from one token to three or more (mean about 7 tokens), well below sqrt(k): one token already gives most of the noise.

**Not tested here:** the effect of the training run (one model per architecture), noise scales other than N(0,1), and any causal edit of attention or V weights. The stage NSR uses hit samples; the margin NSR in S5 uses all 7608 samples (hit-only margin NSR is higher, see the stage table).
