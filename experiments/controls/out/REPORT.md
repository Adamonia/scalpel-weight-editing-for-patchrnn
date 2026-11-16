# Scalpel control experiments (word-level models M0, M2, M3, M4, M5, M8, M12, M13)

All numbers below were computed by the scripts in `/home/claude/work/ext/` and read from `/home/claude/work/ext/out/`. Units: F1, accuracy, precision, recall in percent points (pp). "Effect" = method minus `orig` at threshold 0, on the eval half. `UZ` = `untrained_zero`. "rc" = label-free recenter (fit-half mean margin set equal to the fit-half mean of `orig`). Format "mean [p2.5, p97.5]" is over 50 random stratified 50/50 splits (seeds 0..49). The splits overlap, so these percentiles show split-to-split variation; they are NOT independent-sample confidence intervals. The only real intervals are the paired bootstrap ones (2,000 resamples of the 7,608 samples).

## 0. Verification of `orig`

| model | pred = cached base_pred | pred = archive predicted.txt | acc full test % | holdout F1 (mine / summary_all) | holdout acc (mine / summary_all) | holdout idx from DEV_SEED = bundle part |
|---|---|---|---|---|---|---|
| 0 | 7608/7608 | 7607/7608 | 83.123 | 73.748 / 73.748 | 83.044 / 83.044 | True |
| 2 | 7608/7608 | 7608/7608 | 83.438 | 73.319 / 73.319 | 83.623 / 83.623 | True |
| 3 | 7608/7608 | 7608/7608 | 81.690 | 72.671 / 72.671 | 81.572 / 81.572 | True |
| 4 | 7608/7608 | 7608/7608 | 81.559 | 71.619 / 71.619 | 81.020 / 81.020 | True |
| 5 | 7608/7608 | 7608/7608 | 81.138 | 74.041 / 74.041 | 81.493 / 81.493 | True |
| 8 | 7608/7608 | 7608/7608 | 81.611 | 71.754 / 71.754 | 81.414 / 81.414 | True |
| 12 | 7608/7608 | 7608/7608 | 81.572 | 74.542 / 74.542 | 82.098 / 82.098 | True |
| 13 | 7608/7608 | 7608/7608 | 80.967 | 72.831 / 72.831 | 81.230 / 81.230 | True |

One mismatch: M0, sample 7513, margin +0.0015 (my prediction and the cached one are 1 and `predicted.txt` has 0). This is a float-noise case at the decision boundary. All other models match `predicted.txt` 7608/7608. Holdout F1 and accuracy of `orig` equal the summary_all.json values for all 8 models (to the 3 printed decimals). Message-stage recompute vs cached features: max abs margin gap 0.0 for all models.

## 1. Does zeroing untrained rows beat the placebos?

Placebos: both zero the same NUMBER of message rows as `untrained_zero` (about 14.8k rows), chosen from rows that were trained, 20 seeds each. `uniform` = uniformly at random among trained rows. `freqmatch` = each untrained row gets a distinct random trained row with the same number of occurrences in the test messages (if that frequency bucket is used up: the nearest available frequency). Achieved total test occurrences vs the untrained rows: M0 13506 vs 14750 (same for the others up to 1 %; freqmatch is about 8 % below the target; uniform placebos have about 2.7x the occurrences). **freqmatch is the fairer control**: untrained rows are rare in the test set (about 1 occurrence each), while a uniform trained row is often a frequent token, so uniform placebos remove far more input signal and are damaged by exposure, not by the kind of row. freqmatch equalizes exposure. It is still not perfect: a trained row carries a small learned signal that an untrained row does not.

F1 effect (pp), rc. Placebo cells: mean over seeds [min, max over the 20 seeds], each seed averaged over the 50 splits.

| model | orig F1 (eval-half mean) | UZ effect | freqmatch placebo | uniform placebo | freqmatch seeds with effect >= UZ | UZ - freqmatch placebo mean (per split) | bootstrap 95 % full set, UZ - freqmatch |
|---|---|---|---|---|---|---|---|
| 0 | 73.77 | -0.00 [-0.60, +0.52] | -0.52 [-0.71, -0.22] | -5.12 [-8.20, -3.19] | 0/20 | +0.52 [-0.24, +1.47] | [-0.23, +1.31] |
| 2 | 72.99 | +0.19 [-0.39, +0.69] | +0.01 [-0.29, +0.23] | -3.55 [-7.03, -1.03] | 1/20 | +0.18 [-0.37, +0.72] | [-0.55, +0.91] |
| 3 | 73.16 | +1.13 [+0.73, +1.60] | +0.21 [+0.04, +0.41] | -3.75 [-7.61, -0.41] | 0/20 | +0.92 [+0.34, +1.58] | [+0.39, +1.68] |
| 4 | 72.23 | +1.36 [+0.82, +1.92] | +0.39 [+0.19, +0.59] | -4.25 [-8.46, -1.25] | 0/20 | +0.96 [+0.34, +1.90] | [+0.27, +1.70] |
| 5 | 73.59 | +0.18 [-0.21, +0.57] | -0.33 [-0.45, -0.23] | -3.77 [-6.96, -1.13] | 0/20 | +0.52 [-0.05, +1.06] | [+0.00, +1.13] |
| 8 | 72.08 | +0.67 [+0.12, +1.24] | -0.19 [-0.33, +0.07] | -3.75 [-7.71, -0.72] | 0/20 | +0.87 [+0.29, +1.39] | [+0.28, +1.58] |
| 12 | 73.83 | +0.31 [-0.22, +0.80] | -0.13 [-0.31, +0.09] | -3.73 [-7.29, -1.22] | 0/20 | +0.44 [-0.27, +0.92] | [-0.09, +1.16] |
| 13 | 72.62 | +0.59 [+0.09, +0.97] | -0.13 [-0.27, +0.02] | -4.26 [-8.91, -0.97] | 0/20 | +0.72 [+0.01, +1.36] | [+0.14, +1.51] |

Threshold-free effects (AUC and average precision, no recenter needed; mean over 50 splits; placebo as above):

| model | orig AUC | UZ dAUC | freqmatch dAUC | uniform dAUC | orig AP | UZ dAP | freqmatch dAP | uniform dAP |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8776 | -0.0004 | -0.0008 [-0.0017, +0.0005] | -0.0424 [-0.0662, -0.0244] | 0.7954 | -0.0231 | -0.0169 [-0.0196, -0.0152] | -0.1095 [-0.1412, -0.0836] |
| 2 | 0.8791 | +0.0033 | +0.0040 [+0.0033, +0.0047] | -0.0196 [-0.0440, -0.0023] | 0.8163 | +0.0037 | +0.0033 [+0.0022, +0.0041] | -0.0301 [-0.0673, -0.0036] |
| 3 | 0.8877 | +0.0074 | +0.0011 [+0.0005, +0.0016] | -0.0305 [-0.0558, -0.0068] | 0.8206 | +0.0104 | +0.0034 [+0.0024, +0.0045] | -0.0431 [-0.0863, -0.0063] |
| 4 | 0.8692 | +0.0095 | +0.0015 [+0.0003, +0.0029] | -0.0329 [-0.0643, -0.0096] | 0.7940 | +0.0193 | +0.0075 [+0.0053, +0.0110] | -0.0428 [-0.0911, -0.0097] |
| 5 | 0.8848 | +0.0038 | -0.0003 [-0.0008, +0.0004] | -0.0281 [-0.0540, -0.0070] | 0.8245 | +0.0043 | -0.0006 [-0.0013, +0.0000] | -0.0410 [-0.0808, -0.0088] |
| 8 | 0.8720 | +0.0051 | +0.0007 [-0.0001, +0.0013] | -0.0283 [-0.0588, -0.0077] | 0.8084 | +0.0054 | +0.0006 [-0.0002, +0.0013] | -0.0395 [-0.0820, -0.0084] |
| 12 | 0.8854 | +0.0034 | -0.0003 [-0.0009, +0.0006] | -0.0289 [-0.0567, -0.0081] | 0.8203 | +0.0049 | -0.0011 [-0.0022, -0.0001] | -0.0412 [-0.0849, -0.0108] |
| 13 | 0.8731 | +0.0046 | +0.0009 [+0.0001, +0.0014] | -0.0328 [-0.0668, -0.0105] | 0.7992 | +0.0082 | +0.0013 [+0.0005, +0.0023] | -0.0452 [-0.0982, -0.0105] |

Without the recenter the F1 effects are (UZ / freqmatch mean / uniform mean): M0 +0.13 / -0.08 / -4.13; M2 +0.40 / +0.19 / -3.67; M3 +1.08 / +0.17 / -4.02; M4 +1.04 / +0.03 / -5.37; M5 +0.20 / -0.23 / -3.53; M8 +0.65 / -0.19 / -4.30; M12 +0.23 / -0.08 / -3.94; M13 +0.61 / -0.17 / -4.45.

## 2. Does it beat `thr_only` and `lstsq_head`?

Label-using baselines are fitted on the fit half only. `lstsq_head`: ridge least squares of a +-1 target on standardized features of the input of the last linear layer (ridge by inner 80/20 split, MSE), then best-F1 threshold on the fit half (`+thr`). `lstsq_head_argmax` = threshold 0 (a one-hot two-output fit with argmax is the same linear rule). Paired columns: UZ(rc) minus the baseline, per split, "mean [p2.5, p97.5] / fraction of splits with UZ better".

| model | UZ (rc) | thr_only | lstsq_head+thr | lstsq_head argmax | UZ+thr | UZ - thr_only | UZ - lstsq_head+thr | lstsq_head dAUC |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.00 | +0.19 | -0.11 | -5.93 | +0.61 | -0.19 [-0.95, +0.54] / 0.32 | +0.11 [-0.89, +1.38] / 0.64 | -0.0021 |
| 2 | +0.19 | +0.15 | +0.21 | -3.40 | +0.94 | +0.04 [-0.68, +0.77] / 0.56 | -0.03 [-0.95, +1.25] / 0.46 | +0.0025 |
| 3 | +1.13 | +0.19 | +0.20 | -0.73 | +1.05 | +0.94 [+0.44, +1.71] / 1.00 | +0.93 [+0.11, +1.81] / 0.98 | -0.0008 |
| 4 | +1.36 | -0.17 | -0.31 | -1.84 | +1.25 | +1.53 [+0.93, +2.17] / 1.00 | +1.67 [+0.80, +2.42] / 1.00 | -0.0010 |
| 5 | +0.18 | +0.29 | +0.15 | -1.54 | +0.40 | -0.11 [-0.92, +0.57] / 0.44 | +0.03 [-0.89, +0.98] / 0.52 | -0.0017 |
| 8 | +0.67 | +0.14 | +0.12 | -1.30 | +0.48 | +0.53 [-0.06, +1.43] / 0.96 | +0.55 [+0.03, +1.13] / 0.98 | -0.0005 |
| 12 | +0.31 | -0.08 | +0.17 | -1.69 | +0.11 | +0.39 [-0.25, +1.08] / 0.86 | +0.14 [-0.61, +1.14] / 0.56 | -0.0017 |
| 13 | +0.59 | -0.06 | +0.10 | -1.31 | +0.32 | +0.65 [+0.04, +1.29] / 0.98 | +0.49 [-0.33, +1.35] / 0.90 | -0.0007 |

(F1 effects are means over 50 splits. `lstsq_head` dAUC is the AUC change of the refitted head; UZ dAUC is in table 1.)

## 3. Does imputation beat zero?

Mean imputation = mean of all trained message rows (pad row excluded). kNN imputation = mean of trained rows sharing the longest common prefix (>= 5 characters); if none, mean of trained rows with the same character shape (hex / number / snake_case / word). This is a string heuristic, not a semantic neighbour (untrained rows are random vectors, so embedding-space neighbours are not available). Share of untrained rows by source for M3: prefix 8136, shape 6736, global 0. Paired columns: zero minus imputation (per split), "mean [p2.5, p97.5] / fraction where zero is better".

| model | zero F1 | mean F1 | kNN F1 | zero - mean | zero - kNN | dAUC zero / mean / kNN | dAP zero / mean / kNN | |mean trained row| / mean row norm |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.00 | +0.01 | -0.02 | -0.01 [-0.12, +0.15] / 0.34 | +0.02 [-0.32, +0.41] / 0.52 | -0.0004 / -0.0002 / -0.0004 | -0.0231 / -0.0225 / -0.0127 | 0.0047 |
| 2 | +0.19 | +0.20 | +0.29 | -0.01 [-0.11, +0.08] / 0.28 | -0.10 [-0.54, +0.20] / 0.38 | +0.0033 / +0.0034 / +0.0030 | +0.0037 / +0.0038 / +0.0037 | 0.0050 |
| 3 | +1.13 | +1.11 | +1.11 | +0.02 [-0.12, +0.13] / 0.60 | +0.02 [-0.32, +0.31] / 0.54 | +0.0074 / +0.0075 / +0.0067 | +0.0104 / +0.0106 / +0.0094 | 0.0100 |
| 4 | +1.36 | +1.37 | +1.00 | -0.01 [-0.07, +0.02] / 0.04 | +0.36 [+0.07, +0.87] / 1.00 | +0.0095 / +0.0095 / +0.0074 | +0.0193 / +0.0193 / +0.0145 | 0.0088 |
| 5 | +0.18 | +0.17 | -0.17 | +0.02 [-0.05, +0.07] / 0.44 | +0.35 [+0.09, +0.65] / 1.00 | +0.0038 / +0.0038 / +0.0025 | +0.0043 / +0.0044 / +0.0032 | 0.0052 |
| 8 | +0.67 | +0.67 | +0.50 | -0.00 [-0.05, +0.06] / 0.18 | +0.17 [-0.08, +0.38] / 0.84 | +0.0051 / +0.0051 / +0.0036 | +0.0054 / +0.0054 / +0.0038 | 0.0051 |
| 12 | +0.31 | +0.35 | -0.03 | -0.04 [-0.22, +0.05] / 0.34 | +0.34 [-0.02, +0.72] / 0.96 | +0.0034 / +0.0035 / +0.0032 | +0.0049 / +0.0051 / +0.0054 | 0.0060 |
| 13 | +0.59 | +0.64 | +0.63 | -0.05 [-0.18, +0.09] / 0.26 | -0.04 [-0.38, +0.25] / 0.38 | +0.0046 / +0.0049 / +0.0031 | +0.0082 / +0.0085 / +0.0055 | 0.0090 |

The last column explains the first result: trained message rows have the same norm as untrained ones (about 11.3, the size of a random N(0,1) 128-vector), so training moved them little, and the mean of the trained rows is nearly the zero vector. Zero and mean imputation are almost the same edit.

## 4. Stability across the 50 splits (F1 effect, pp; mean [p2.5, p97.5] / fraction of splits with a positive effect)

| model | UZ (rc) | UZ (no rc) | rare_zero (rc) | prior | thr_only | UZ+thr | lstsq_head+thr |
|---|---|---|---|---|---|---|---|
| 0 | -0.00 [-0.60, +0.52] / 0.52 | +0.13 [-0.37, +0.71] / 0.64 | -0.73 [-1.61, -0.01] / 0.04 | -0.24 [-0.60, +0.12] / 0.06 | +0.19 [-0.74, +0.71] / 0.80 | +0.61 [-0.09, +1.16] / 0.94 | -0.11 [-1.19, +0.71] / 0.50 |
| 2 | +0.19 [-0.39, +0.69] / 0.76 | +0.40 [-0.11, +0.99] / 0.88 | +0.44 [-0.55, +1.32] / 0.84 | -2.04 [-2.62, -1.60] / 0.00 | +0.15 [-0.39, +0.64] / 0.72 | +0.94 [+0.30, +1.74] / 1.00 | +0.21 [-1.06, +0.94] / 0.72 |
| 3 | +1.13 [+0.73, +1.60] / 1.00 | +1.08 [+0.73, +1.58] / 1.00 | +1.42 [+0.85, +2.05] / 1.00 | -0.41 [-1.40, +0.21] / 0.14 | +0.19 [-0.48, +0.55] / 0.76 | +1.05 [+0.37, +1.57] / 1.00 | +0.20 [-0.49, +0.90] / 0.70 |
| 4 | +1.36 [+0.82, +1.92] / 1.00 | +1.04 [+0.58, +1.56] / 1.00 | +1.70 [+1.13, +2.53] / 1.00 | -1.08 [-2.06, -0.36] / 0.00 | -0.17 [-0.61, +0.19] / 0.26 | +1.25 [+0.42, +1.90] / 1.00 | -0.31 [-1.03, +0.34] / 0.22 |
| 5 | +0.18 [-0.21, +0.57] / 0.80 | +0.20 [-0.15, +0.60] / 0.88 | -0.04 [-0.66, +0.56] / 0.46 | +0.58 [-0.22, +1.36] / 0.94 | +0.29 [-0.38, +0.90] / 0.82 | +0.40 [-0.08, +0.94] / 0.90 | +0.15 [-0.78, +1.14] / 0.68 |
| 8 | +0.67 [+0.12, +1.24] / 0.98 | +0.65 [+0.12, +1.22] / 1.00 | +0.48 [-0.30, +0.97] / 0.92 | -0.30 [-1.07, +0.59] / 0.16 | +0.14 [-0.63, +0.49] / 0.74 | +0.48 [-0.31, +1.10] / 0.92 | +0.12 [-0.45, +0.79] / 0.62 |
| 12 | +0.31 [-0.22, +0.80] / 0.84 | +0.23 [-0.26, +0.81] / 0.80 | +0.16 [-0.48, +0.78] / 0.66 | +0.08 [-0.50, +0.54] / 0.56 | -0.08 [-0.65, +0.49] / 0.40 | +0.11 [-0.52, +0.67] / 0.62 | +0.17 [-0.85, +0.91] / 0.68 |
| 13 | +0.59 [+0.09, +0.97] / 0.98 | +0.61 [+0.09, +0.96] / 1.00 | +0.39 [-0.33, +0.91] / 0.88 | -0.17 [-0.70, +0.60] / 0.32 | -0.06 [-0.71, +0.38] / 0.50 | +0.32 [-0.37, +0.84] / 0.84 | +0.10 [-0.88, +0.69] / 0.62 |

Accuracy / precision / recall effect of UZ (rc), mean over splits:

| model | UZ acc | UZ prec | UZ rec | prior acc | prior prec | prior rec |
|---|---|---|---|---|---|---|
| 0 | -0.01 | -0.04 | +0.03 | +0.18 | +1.29 | -1.63 |
| 2 | -0.06 | -0.69 | +0.89 | -0.41 | +2.28 | -5.10 |
| 3 | +0.89 | +1.52 | +0.67 | +1.61 | +7.62 | -8.12 |
| 4 | +1.16 | +2.37 | +0.29 | +0.82 | +5.32 | -6.92 |
| 5 | +0.19 | +0.36 | -0.07 | +2.30 | +7.82 | -7.64 |
| 8 | +0.50 | +0.89 | +0.46 | +1.36 | +6.65 | -6.37 |
| 12 | +0.27 | +0.45 | +0.13 | +1.36 | +4.90 | -5.54 |
| 13 | +0.51 | +0.88 | +0.22 | +1.26 | +4.78 | -5.73 |

Paired bootstrap on all 7,608 samples (2,000 resamples, seed 777; F1 effect in pp, 95 % percentile interval; rc = mean margin on all samples set to the `orig` mean):

| model | orig F1 (full test) | UZ (rc) | UZ (no rc) | rare_zero (rc) | impute mean (rc) | impute kNN (rc) | freqmatch placebo mean (rc) | uniform placebo mean (rc) |
|---|---|---|---|---|---|---|---|---|
| 0 | 73.88 | -0.03 [-0.70, +0.65] | +0.07 [-0.52, +0.65] | -0.85 [-1.80, +0.11] | -0.01 [-0.69, +0.67] | -0.08 [-0.72, +0.60] | -0.56 [-1.26, +0.09] | -5.15 [-5.97, -4.38] |
| 2 | 73.18 | +0.19 [-0.51, +0.87] | +0.41 [-0.23, +1.04] | +0.39 [-0.44, +1.20] | +0.21 [-0.49, +0.87] | +0.34 [-0.34, +1.02] | +0.02 [-0.56, +0.59] | -3.59 [-4.21, -2.93] |
| 3 | 73.22 | +1.22 [+0.68, +1.75] | +1.13 [+0.65, +1.63] | +1.44 [+0.70, +2.16] | +1.18 [+0.65, +1.73] | +1.19 [+0.60, +1.77] | +0.18 [-0.27, +0.64] | -3.76 [-4.18, -3.35] |
| 4 | 72.37 | +1.41 [+0.76, +2.05] | +1.04 [+0.51, +1.57] | +1.70 [+0.83, +2.59] | +1.41 [+0.76, +2.05] | +1.04 [+0.38, +1.69] | +0.42 [-0.18, +1.04] | -4.23 [-4.78, -3.68] |
| 5 | 73.57 | +0.23 [-0.22, +0.69] | +0.24 [-0.22, +0.71] | -0.01 [-0.63, +0.61] | +0.22 [-0.22, +0.68] | -0.14 [-0.64, +0.35] | -0.33 [-0.73, +0.10] | -3.72 [-4.18, -3.26] |
| 8 | 72.16 | +0.64 [+0.10, +1.15] | +0.64 [+0.12, +1.15] | +0.41 [-0.31, +1.15] | +0.64 [+0.10, +1.15] | +0.51 [-0.07, +1.09] | -0.27 [-0.77, +0.20] | -3.77 [-4.33, -3.26] |
| 12 | 73.85 | +0.38 [-0.16, +0.92] | +0.26 [-0.26, +0.77] | +0.16 [-0.55, +0.87] | +0.36 [-0.18, +0.90] | +0.03 [-0.49, +0.56] | -0.18 [-0.64, +0.30] | -3.70 [-4.22, -3.18] |
| 13 | 72.74 | +0.57 [+0.04, +1.11] | +0.58 [+0.07, +1.12] | +0.29 [-0.46, +1.02] | +0.59 [+0.09, +1.13] | +0.59 [+0.03, +1.20] | -0.24 [-0.78, +0.27] | -4.32 [-4.81, -3.82] |

## 5. Holm correction of the 24 published comparisons (12 models x 2 rounds)

Source: `/home/claude/work/results/summary_all.json` (holdout, 3,804 samples). `p_f1_up` is one-sided (share of bootstrap resamples with a positive F1 change); converted to two-sided as 2*min(p, 1-p). Bootstrap resolution: B = 10,000 (`N_BOOT` in scalpel/report.py), so a reported 0 means p < 1/B; its two-sided value is bounded here by 2/B = 0.0002 (used in Holm, so adjusted values for those are lower bounds on the true level, conservative). M15 round 1 changed no prediction (d_f1 = 0, p_f1_up = 0 is a tie artefact); I set its p to 1. Resamples with a zero difference count as "not positive", which inflates the one-sided p for rows with many ties (e.g. M7 round 2). McNemar (exact, two-sided, accuracy-type test on fixed/broken pairs) is corrected in a second family of 24; a reported 0.000000 is replaced by 1e-6.

| round | model | d_f1 pp | p two-sided (F1) | Holm p (F1) | sig at 0.05 (F1) | McNemar p | Holm p (McNemar) | sig at 0.05 (McNemar) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | -0.864 | 0.0360 | 0.7200 | no | 0.171565 | 1.0000 | no |
| 1 | 2 | +0.475 | 0.4336 | 1.0000 | no | 0.675277 | 1.0000 | no |
| 1 | 3 | -0.295 | 0.6592 | 1.0000 | no | 0.000036 | 0.0007 | yes |
| 1 | 4 | +0.427 | 0.5180 | 1.0000 | no | 0.000000 | 0.0000 | yes |
| 1 | 5 | +0.397 | 0.5142 | 1.0000 | no | 0.000004 | 0.0001 | yes |
| 1 | 6 | -4.293 | 0.0002 | 0.0048 | yes | 0.000025 | 0.0005 | yes |
| 1 | 7 | -1.996 | 0.0086 | 0.1806 | no | 0.000946 | 0.0155 | yes |
| 1 | 8 | +0.430 | 0.4590 | 1.0000 | no | 0.000007 | 0.0001 | yes |
| 1 | 12 | +0.096 | 0.8516 | 1.0000 | no | 0.000912 | 0.0155 | yes |
| 1 | 13 | +0.153 | 0.7862 | 1.0000 | no | 0.699216 | 1.0000 | no |
| 1 | 14 | +0.114 | 0.6906 | 1.0000 | no | 1.000000 | 1.0000 | no |
| 1 | 15 | +0.000 | 1.0000 | 1.0000 | no | 1.000000 | 1.0000 | no |
| 2 | 0 | +0.447 | 0.1530 | 1.0000 | no | 0.694004 | 1.0000 | no |
| 2 | 2 | +0.563 | 0.2332 | 1.0000 | no | 0.926078 | 1.0000 | no |
| 2 | 3 | +1.816 | 0.0010 | 0.0230 | yes | 0.006734 | 0.1010 | no |
| 2 | 4 | +1.495 | 0.0010 | 0.0230 | yes | 0.000074 | 0.0013 | yes |
| 2 | 5 | +0.357 | 0.5578 | 1.0000 | no | 0.000006 | 0.0001 | yes |
| 2 | 6 | -0.001 | 0.9170 | 1.0000 | no | 1.000000 | 1.0000 | no |
| 2 | 7 | +0.054 | 0.2764 | 1.0000 | no | 0.500000 | 1.0000 | no |
| 2 | 8 | +0.783 | 0.0388 | 0.7372 | no | 0.023646 | 0.3310 | no |
| 2 | 12 | -0.273 | 0.4690 | 1.0000 | no | 0.714026 | 1.0000 | no |
| 2 | 13 | +0.676 | 0.0748 | 1.0000 | no | 0.048725 | 0.6334 | no |
| 2 | 14 | +0.114 | 0.6906 | 1.0000 | no | 1.000000 | 1.0000 | no |
| 2 | 15 | -0.076 | 0.8378 | 1.0000 | no | 0.283607 | 1.0000 | no |

Significant after Holm, F1 change: R1 M6 (-4.29 pp), R2 M3 (+1.82 pp), R2 M4 (+1.50 pp). Significant before correction (F1): R1 M0, R1 M6, R1 M7, R2 M3, R2 M4, R2 M8.
Significant after Holm, McNemar: R1 M3, R1 M4, R1 M5, R1 M6, R1 M7, R1 M8, R1 M12, R2 M4, R2 M5. McNemar tests whether fixed != broken, in either direction and about accuracy; it can be significant when F1 does not move (e.g. prior-correction rounds trade precision for recall).
Note M14 round 1 and round 2 are the same edit with the same numbers, counted twice as asked.

## What failed, what I could not verify, deviations

- torch was not installed in the sandbox; I installed it with pip (default PyPI index; the CPU-wheel index was blocked). This changes the environment but nothing in the project.
- Step 1 does not use the sealed-holdout flow. A temp Session per model (ext/tmp/S_M{n}) was created only to load model, vocabulary, training counts and cached features. All variant margins are computed by my code on a deep copy of the model. The cached before/after code-branch features are reused (message edits cannot change them); only samples that contain an edited token inside the model window get a new message feature. Orig recomputed through the same path reproduces the cache with max gap 0.0.
- The 50 splits and the bootstrap use all 7,608 test samples. The diag and guard parts were used by the agent to accept or refuse the published edits, and the recipes may have been shaped by earlier results (I could not check this), so the full-set numbers are not a clean out-of-sample test of the recipes. The label-using baselines are fit on the fit half only.
- Splits overlap: percentiles over splits are not confidence intervals. The 20 placebo seeds share the same splits.
- Placebo freqmatch is only approximately matched: same row count, same test frequency per row where possible; total test occurrences are about 8 % below the untrained rows (frequency buckets run out). Training counts and token shape are not matched. The uniform placebo is unfair (about 2.7x the occurrences) and is shown for completeness. 20 seeds were run for every model (no reduction was needed).
- `rare_zero` (train count <= 10) zeroes 76.9k of about 80.8k message rows (about 95 % of the vocabulary), because training counts are very skewed; it is close to removing the whole message vocabulary except frequent tokens.
- kNN imputation is a string heuristic (shared prefix of >= 5 characters, else same character shape). It is not a semantic neighbour. Its result is only about that heuristic.
- `prior` is a constant shift, so its AUC/AP equal `orig` and are not listed. The prior shift 0.7159 was applied to all models as asked, including the ones where the published round 1 did not use it.
- `thr_only` and the `+thr` variants use a grid of 801 shifts between the 1 % and 99 % margin quantiles plus 0 (ties go to the smallest shift). `lstsq_head` standardizes features with fit-half statistics, uses a +-1 target, and picks ridge from {1e-3..1e4} by inner-split MSE (the chosen counts per model are in splits_M{n}.json). For the transformers the features are the input of classifier.4 (after the second LayerNorm, 384-d); M0: fc4 input (64-d); M2: fc2 input (150-d). The refit is on about 3.8k samples, much less than the original training set, so it is a weak baseline for the full-capacity classifier.
- Placebo effect in the split tables is the average over the 20 seeds of the per-seed metrics; per-seed values are in splits_perseed_M{n}.npz (split x seed arrays) and per-seed means over splits are in splits_M{n}.json.
- Holm: the p-values come from a percentile bootstrap share (not a null-distribution p-value), converted to two-sided by 2*min(p,1-p); ties and the 1/B floor are handled as described above. The Holm family is all 24 published rows, including M6, M7, M14, M15 that are outside my experiment scope.
- M0 margins are very large (max about 69) because its LSTM message branch is unbounded, so M0 recenter shifts and AP changes are on a different scale than the transformers.

## Files

- /home/claude/work/ext/out/REPORT.md
- /home/claude/work/ext/out/splits.json (all models; also splits_M{n}.json per model, splits_perseed_M{n}.npz per-seed arrays)
- /home/claude/work/ext/out/holm.json
- /home/claude/work/ext/out/logits_M{n}.npz for n in 0,2,3,4,5,8,12,13 (y plus margins for orig, prior, untrained_zero, rare_zero, placebo_untrained_0..19, placebo_freqmatch_0..19, untrained_impute_mean, untrained_impute_knn)
- /home/claude/work/ext/out/feat_M{n}.npz (head-input features of the last linear layer, with its W and b)
- /home/claude/work/ext/out/meta_M{n}.json (verification, counts, placebo matching totals); placebo_rows_M{n}.npz (row ids of every placebo); mean_vec_norm.json
- /home/claude/work/ext/common.py, step1.py, splits.py, holm.py, mean_norm.py, make_report.py; logs in /home/claude/work/ext/logs/; temp sessions in /home/claude/work/ext/tmp/
