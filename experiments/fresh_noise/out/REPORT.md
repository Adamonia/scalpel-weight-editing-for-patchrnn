# Test of hypothesis H on six word-level Transformer security-patch classifiers

All numbers are read from `out/t1t2_M*.json` and `out/t3t4_M*.json`, written by `step1.py` and `step3.py`. F1 and AUC in percent. No gradient training. Test set: all 7608 samples; holdout = part 2.

## 0. Setup and checks

| Model | heads | layers | untrained rows | samples with untrained token | max abs gap orig | max abs gap zero | bias shift (zero) | F1 orig | F1 zero | gain F1 (all) | gain F1 (holdout) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 1 | 1 | 14872 | 4328 (57%) | 0.0e+00 | 0.0e+00 | 0.000 | 73.22 | 74.35 | 1.14 | 1.58 |
| M4 | 2 | 1 | 14872 | 4328 (57%) | 0.0e+00 | 0.0e+00 | 0.000 | 72.37 | 73.40 | 1.03 | 1.22 |
| M5 | 8 | 1 | 14872 | 4328 (57%) | 0.0e+00 | 0.0e+00 | 0.000 | 73.57 | 73.82 | 0.25 | 0.16 |
| M8 | 16 | 1 | 14872 | 4328 (57%) | 0.0e+00 | 0.0e+00 | 0.000 | 72.16 | 72.80 | 0.64 | 0.88 |
| M12 | 8 | 2 | 14872 | 4328 (57%) | 0.0e+00 | 0.0e+00 | 0.000 | 73.85 | 74.12 | 0.27 | -0.31 |
| M13 | 1 | 2 | 14872 | 4328 (57%) | 0.0e+00 | 0.0e+00 | 0.000 | 72.74 | 73.33 | 0.58 | 0.78 |

Layers are counted from the checkpoint keys. The head count is NOT stored in the checkpoint (the attention weight shapes are the same for any head count); it comes from the loader registry (`scalpel/models.py`), so it is taken from the project, not verified from the weights. The recomputed orig and zero margins equal the cached ones (gap 0). The recentered zero edit reproduces the stated gains (M3 +1.14, M4 +1.03, M8 +0.64, M13 +0.58, M5 +0.25, M12 +0.27). Recenter used here: subtract the median over all samples of (new margin - orig margin).

## T1. Fresh noise in the untrained rows

Replacement rows: independent N(0, scale^2) draws, seeds 0..K-1, bias recentered as for the zero edit. Gap = F1(zero) - mean F1(fresh). Percentile = position of the original model F1 among the fresh-noise F1 values (ties count half). sd is over draws.

### scale 1.0, K=30 (all 7608)

| Model | F1 zero | F1 orig | F1 fresh mean | F1 fresh sd | gap zero-fresh | orig - fresh mean | orig percentile | AUC zero | AUC fresh mean | AUC gap |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 74.35 | 73.22 | 73.46 | 0.24 | 0.89 | -0.25 | 17 | 89.558 | 88.837 | 0.721 |
| M4 | 73.40 | 72.37 | 72.00 | 0.29 | 1.40 | 0.36 | 87 | 87.940 | 86.931 | 1.010 |
| M5 | 73.82 | 73.57 | 73.07 | 0.20 | 0.74 | 0.49 | 100 | 88.887 | 88.464 | 0.423 |
| M8 | 72.80 | 72.16 | 72.28 | 0.26 | 0.51 | -0.12 | 30 | 87.692 | 87.312 | 0.380 |
| M12 | 74.12 | 73.85 | 73.66 | 0.19 | 0.46 | 0.19 | 87 | 88.952 | 88.601 | 0.351 |
| M13 | 73.33 | 72.74 | 72.51 | 0.18 | 0.82 | 0.23 | 97 | 87.810 | 87.179 | 0.631 |

### scale 0.5, K=10 (all 7608)

| Model | F1 zero | F1 orig | F1 fresh mean | F1 fresh sd | gap zero-fresh | orig - fresh mean | orig percentile | AUC zero | AUC fresh mean | AUC gap |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 74.35 | 73.22 | 74.31 | 0.08 | 0.05 | -1.09 | 0 | 89.558 | 89.468 | 0.089 |
| M4 | 73.40 | 72.37 | 73.35 | 0.05 | 0.05 | -0.99 | 0 | 87.940 | 87.913 | 0.028 |
| M5 | 73.82 | 73.57 | 73.82 | 0.08 | -0.00 | -0.25 | 0 | 88.887 | 88.845 | 0.042 |
| M8 | 72.80 | 72.16 | 72.80 | 0.16 | -0.00 | -0.64 | 0 | 87.692 | 87.685 | 0.007 |
| M12 | 74.12 | 73.85 | 74.11 | 0.18 | 0.01 | -0.25 | 10 | 88.952 | 88.873 | 0.079 |
| M13 | 73.33 | 72.74 | 73.13 | 0.17 | 0.20 | -0.39 | 0 | 87.810 | 87.649 | 0.161 |

### scale 2.0, K=10 (all 7608)

| Model | F1 zero | F1 orig | F1 fresh mean | F1 fresh sd | gap zero-fresh | orig - fresh mean | orig percentile | AUC zero | AUC fresh mean | AUC gap |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 74.35 | 73.22 | 69.12 | 0.42 | 5.23 | 4.09 | 100 | 89.558 | 84.508 | 5.050 |
| M4 | 73.40 | 72.37 | 66.87 | 0.33 | 6.53 | 5.50 | 100 | 87.940 | 82.302 | 5.638 |
| M5 | 73.82 | 73.57 | 68.37 | 0.35 | 5.45 | 5.20 | 100 | 88.887 | 84.993 | 3.894 |
| M8 | 72.80 | 72.16 | 67.72 | 0.54 | 5.08 | 4.44 | 100 | 87.692 | 83.721 | 3.971 |
| M12 | 74.12 | 73.85 | 70.38 | 0.20 | 3.74 | 3.47 | 100 | 88.952 | 86.409 | 2.544 |
| M13 | 73.33 | 72.74 | 69.14 | 0.28 | 4.19 | 3.61 | 100 | 87.810 | 84.455 | 3.355 |

### Holdout (part 2), scale 1.0

| Model | F1 zero | F1 fresh mean | F1 fresh sd | gap | F1 orig | orig percentile |
|---|---|---|---|---|---|---|
| M3 | 74.25 | 73.34 | 0.36 | 0.91 | 72.67 | 10 |
| M4 | 72.84 | 71.50 | 0.34 | 1.34 | 71.62 | 63 |
| M5 | 74.20 | 73.54 | 0.30 | 0.66 | 74.04 | 93 |
| M8 | 72.63 | 72.03 | 0.36 | 0.61 | 71.75 | 20 |
| M12 | 74.23 | 73.82 | 0.28 | 0.41 | 74.54 | 100 |
| M13 | 73.61 | 72.69 | 0.29 | 0.92 | 72.83 | 67 |

Dose-response: gap F1(zero) - F1(fresh) by scale.

| Model | 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| M3 | 0.05 | 0.89 | 5.23 |
| M4 | 0.05 | 1.40 | 6.53 |
| M5 | -0.00 | 0.74 | 5.45 |
| M8 | -0.00 | 0.51 | 5.08 |
| M12 | 0.01 | 0.46 | 3.74 |
| M13 | 0.20 | 0.82 | 4.19 |

## T2. Clean-margin density and noise-to-signal (scale 1.0, K=30)

Clean = zeroed model (recentered). sd_c = sd of the clean margin over the 7608 samples. Noise sd = per-sample sd of the margin over the 30 fresh draws. rms = root mean square over samples. Flip rate = share of predictions that differ between zero and a fresh draw (mean over draws).

| Model | sd_c | |m|<0.25 sd_c | |m|<0.5 sd_c | |m|<1.0 sd_c | noise sd / sd_c (rms, all) | (rms, hit samples) | flip rate all (%) | flip rate hit (%) | sd(orig-zero) | sd(fresh-zero) |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 2.91 | 13.7% | 27.9% | 57.8% | 0.191 | 0.253 | 2.72 | 4.79 | 0.565 | 0.572 |
| M4 | 3.58 | 11.7% | 22.9% | 51.7% | 0.248 | 0.328 | 2.89 | 5.08 | 1.007 | 0.994 |
| M5 | 2.92 | 15.3% | 30.3% | 62.7% | 0.138 | 0.183 | 2.51 | 4.41 | 0.436 | 0.428 |
| M8 | 3.61 | 13.4% | 27.3% | 56.9% | 0.154 | 0.204 | 2.49 | 4.37 | 0.599 | 0.605 |
| M12 | 4.06 | 13.7% | 27.4% | 59.9% | 0.157 | 0.208 | 3.06 | 5.39 | 0.701 | 0.707 |
| M13 | 3.42 | 11.9% | 24.3% | 54.1% | 0.208 | 0.276 | 3.32 | 5.84 | 0.747 | 0.771 |

## T3. Attention of the first encoder layer on untrained tokens (original models)

Samples with at least one untrained token. The manual layer-0 forward reproduces the cached margins (max abs gap in the table). Mass = attention weight on untrained key positions. The CLS query is the one that feeds the feature in a 1-layer model. Two denominators: share of all 200 positions (padding is not masked in these models, so padding takes part of the mass) and share of real (non-pad) positions, where ratio = (mass on untrained / mass on real positions) / (untrained share of real positions). Ratio > 1 = attends more than chance.

| Model | forward gap | untrained share all pos | untrained share real pos | CLS mass on untrained | CLS ratio (all pos) | CLS ratio (real only) | all-query ratio (all pos) | all-query ratio (real only) |
|---|---|---|---|---|---|---|---|---|
| M3 | 8.2e-06 | 0.0170 | 0.127 | 0.1421 | 8.35 | 1.18 | 1.32 | 0.89 |
| M4 | 0.0e+00 | 0.0170 | 0.127 | 0.0672 | 3.95 | 1.20 | 1.20 | 0.90 |
| M5 | 0.0e+00 | 0.0170 | 0.127 | 0.0848 | 4.98 | 0.99 | 1.25 | 0.86 |
| M8 | 0.0e+00 | 0.0170 | 0.127 | 0.0934 | 5.49 | 0.98 | 1.14 | 0.83 |
| M12 | 0.0e+00 | 0.0170 | 0.127 | 0.0990 | 5.81 | 0.97 | 4.59 | 1.02 |
| M13 | 1.1e-05 | 0.0170 | 0.127 | 0.0536 | 3.15 | 0.58 | 8.16 | 1.22 |

Per head, CLS query, ratio to share of all positions:

- M3: 8.35
- M4: 4.86, 3.04
- M5: 8.42, 2.48, 8.82, 0.28, 7.54, 3.74, 1.71, 6.86
- M8: 3.48, 7.32, 8.37, 7.81, 1.54, 6.96, 2.77, 7.88, 7.55, 2.64, 1.70, 2.77, 2.83, 6.57, 8.84, 8.75
- M12: 5.62, 3.38, 8.61, 7.57, 7.43, 3.01, 8.33, 2.55
- M13: 3.15

## T4. Averaging across heads (multi-head models)

Change of each head output at the CLS attention output of layer 0 (the only place where the per-head split is exact, because out_proj is linear in the concatenated heads). It is before residual, LayerNorm and FF, and in M12 also before layer 1, so it is NOT the pooled representation. Reference for independent equal-norm vectors: cosine 0, norm ratio 1/sqrt(H). Parallel vectors: cosine 1, ratio 1. "orig" = original rows minus zero rows; "fresh" = mean over fresh draws 0..4 minus zero rows.

| Model | heads | cos (orig) | ratio (orig) | cos (fresh) | ratio (fresh) | 1/sqrt(H) | mean head change norm (orig) | norm of summed change (orig) |
|---|---|---|---|---|---|---|---|---|
| M4 | 2 | 0.018 | 0.918 | 0.013 | 0.919 | 0.707 | 0.991 | 1.908 |
| M5 | 8 | 0.013 | 0.684 | 0.013 | 0.685 | 0.354 | 0.253 | 1.363 |
| M8 | 16 | 0.003 | 0.628 | 0.003 | 0.625 | 0.250 | 0.206 | 1.877 |
| M12 | 8 | 0.004 | 0.736 | 0.003 | 0.736 | 0.354 | 0.303 | 1.671 |


## Rank correlations over the six models (from out/summary.json)

| pair | Spearman rho | p (n=6, not corrected) |
|---|---|---|
| gap_vs_heads | -0.71 | 0.12 |
| gap_vs_layers | -0.41 | 0.41 |
| nsr_vs_heads | -0.71 | 0.12 |
| gain_vs_gap | 0.66 | 0.16 |
| gap_vs_density05 | -0.37 | 0.47 |
| nsr_vs_gap | 0.71 | 0.11 |

Negative rho for gap_vs_heads means: more heads, smaller gap (H direction). With n = 6 these are descriptive only.

## What each test shows and does not show

- Recentering: the bias shift for the zero edit (median over all samples of new minus orig margin) is 0.000 in all models, because 43% of samples contain no untrained token and have change 0. So the "recenter" is a no-op here; the reproduced gains match the stated ones. Fresh-noise draws use the same rule.
- T1 separates a model-level noise cost (F1 zero minus mean F1 fresh) from the luck of the one trained init (orig minus fresh mean). It uses one trained model per architecture, so architecture and training run are confounded; sd over draws is 0.18-0.29 F1 and says nothing about training-run variation.
- The original models are N(0,1) draws, so scale 0.5 and 2.0 are not comparable with the orig percentile (percentile 0 / 100 there is expected, not informative).
- T2 uses the zeroed model as the "clean" signal; it is a proxy, not the true noise-free model. Margin change of orig vs zero differs between models (sd 0.44 for M5, 1.01 for M4), so "similar size" holds only roughly.
- T3 shows where layer-0 attention goes; it does not show how much the value vectors of those tokens matter downstream. For 2-layer models layer 0 is only the first stage.
- T4 split is exact only at the layer-0 CLS attention output (see above). M3 and M13 have one head: no averaging to test.

## Verdicts

**H (few heads, one layer are hurt more by random untrained rows): undecided, leaning weakly supported for heads, not supported for layers.**
- For: at scale 1 the F1 cost of random rows (gap) is 0.89 / 1.40 / 0.82 (M3, M4, M13: 1-2 heads) against 0.74 / 0.51 / 0.46 (M5, M12, M8: 8-16 heads). Noise-to-signal of the margin is 0.19-0.25 for 1-2 heads and 0.14-0.16 for 8-16 heads. Spearman gap vs heads -0.71 (p 0.12, n = 6).
- Against / unclear: layers do not separate the models (M13 with 2 layers has gap 0.82, as large as M3; rho -0.41). M4 (2 heads) is the extreme, M3 (1 head) is lower. At noise scale 2 the gaps are 3.7-6.5 and not ordered by heads (M3 5.23, M5 5.45). Differences between gaps (0.5 to 1.4 F1) are a few times the draw sd (0.2-0.3), and no run-to-run variation of training is available.
- Luck of the original init matters: orig F1 is at percentile 100 (M5), 97 (M13), 87 (M4, M12) of the fresh draws, and 17 / 30 for M3 / M8. So part of the small observed gain of M5 (0.25) is a lucky init (orig is +0.49 above fresh mean), and M3's observed gain is partly bad luck of the opposite sign. Observed gain vs fresh-noise gap: rho 0.66 (p 0.16).
- Mechanism not supported: T4 head changes are almost uncorrelated (mean cosine 0.003-0.018, as independent noise would be), but the total change of the CLS attention output is not smaller with more heads (norm of summed change 1.91 for M4, 1.36 M5, 1.88 M8, 1.67 M12). The ratio of summed norm to sum of norms (0.63-0.74 for 8-16 heads, 0.92 for 2 heads) is above the equal-norm orthogonal reference (0.25-0.35, 0.71), which means head changes have unequal sizes. T3: CLS attention on untrained tokens relative to real tokens is near chance (ratio 0.97-1.20, M13 0.58), slightly above 1 only for M3 and M4. No clear attention explanation.

**R1 (same noise, but some models have more samples near the threshold): contradicted as an explanation of the gap ordering, undecided otherwise.** The share of samples with |clean margin| < 0.5 sd is 22.9-30.3%; the models with the largest gaps (M4 22.9%, M13 24.3%) have the fewest near-threshold samples, M5 the most (30.3%) with a mid gap (rho gap vs density -0.37, p 0.47). What does differ is noise-to-signal (M4 0.25, M13 0.21 against M5 0.14), i.e. the same noise moves the margin more relative to its spread in some models. That points to transmission of noise, not to density.

Not proven: any claim about architectures in general. n = 6, one trained model each, heads taken from the registry.
