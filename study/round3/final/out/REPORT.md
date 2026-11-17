# Round 3 final evaluation on the sealed holdout (3,804 samples)

Frozen recipe and decisions: see FROZEN.md. All numbers below are read from round3_holdout.json. F1 in percent, change in F1 points, 95 % paired bootstrap interval (10,000 resamples, seed 12345). p = two-sided p (2*min(p_up, 1-p_up); no changed prediction: 1). Holm36 = Holm over the 24 published + 12 new F1 comparisons, Holm12 = over the 12 new alone, flag * = significant at 0.05.

| Model | Decision | F1 orig | F1 round 2 (pub.) | F1 round 3 | Change vs orig [95 % interval] | fixed/broken | p (F1) | McNemar p | Holm36 p | Holm12 p | Round 3 - round 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 LSTM baseline (PatchRNN) | recipe | 73.748 | 74.195 | 73.754 | +0.006 [-1.07, +1.07] | 95/124 | 0.9932 | 0.05823 | 1.0000 | 1.0000 | -0.441 |
| M2 Conv2D-E | recipe | 73.319 | 73.882 | 74.307 | +0.988 [-0.06, +2.07] | 82/80 | 0.0644 | 0.9374 | 1.0000 | 0.5796 | +0.425 |
| M3 Trans-1H | recipe | 72.671 | 74.487 | 75.000 | +2.329 [+1.35, +3.30] | 127/50 | 0.0002 | 0 | 0.0072* | 0.0024* | +0.513 |
| M4 Trans-2H | recipe | 71.619 | 73.114 | 73.117 | +1.498 [+0.25, +2.70] | 185/77 | 0.0156 | 0 | 0.4836 | 0.1716 | +0.003 |
| M5 Trans-8H | recipe | 74.041 | 74.398 | 74.398 | +0.357 [-0.45, +1.15] | 89/65 | 0.3870 | 0.06348 | 1.0000 | 1.0000 | +0.000 |
| M6 Trans-8H+B | original (refused) | 65.130 | 65.129 | 65.130 | +0.000 [+0.00, +0.00] | 0/0 | 1.0000 | 1 | 1.0000 | 1.0000 | +0.001 |
| M7 Trans-8H+R | recipe | 68.388 | 68.442 | 68.613 | +0.225 [-0.25, +0.71] | 23/12 | 0.3662 | 0.08953 | 1.0000 | 1.0000 | +0.171 [-0.29, +0.65] |
| M8 Trans-16H | recipe | 71.754 | 72.537 | 72.107 | +0.353 [-0.67, +1.37] | 103/66 | 0.4966 | 0.005461 | 1.0000 | 1.0000 | -0.430 |
| M12 Trans-8H-2E | recipe | 74.542 | 74.269 | 75.390 | +0.848 [-0.03, +1.75] | 104/70 | 0.0556 | 0.01214 | 1.0000 | 0.5560 | +1.121 |
| M13 Trans-1H-2E | recipe | 72.831 | 73.507 | 73.477 | +0.646 [-0.31, +1.61] | 105/79 | 0.1832 | 0.06503 | 1.0000 | 1.0000 | -0.030 |
| M14 ParCombCtl (hybrid) | recipe | 76.577 | 76.691 | 76.710 | +0.133 [-0.78, +1.09] | 62/79 | 0.7806 | 0.1776 | 1.0000 | 1.0000 | +0.019 |
| M15 CodeBERT-SC | original (n/a) | 85.234 | 85.158 | 85.234 | +0.000 [+0.00, +0.00] | 0/0 | 1.0000 | 1 | 1.0000 | 1.0000 | +0.076 |

## Checks

- Original holdout F1 equals the published round 1 original F1 for all 12 models: max abs gap 0.0 (other metrics: {'acc': 0.0, 'auc': 0.0, 'mcc': 0.0, 'prec': 0.0, 'rec': 0.0}).
- Round 3 minus round 2 is a difference of published F1 values. A paired interval is given only where round 2 predictions were rebuilt exactly (F1, fixed, broken, FP, FN all equal): models M7. For the others the rebuild from "untrained rows to zero" did not reproduce the published round 2 numbers (see round3_holdout.json, vs_round2.rebuild), or round 2 used an F1 threshold, so no interval is available.
- McNemar p = 0 after rounding to 6 decimals is entered as 1e-6 in Holm, as in ext/holm.py.
- Published rows whose Holm F1 flag changes when the family grows from 24 to 36: none.
- Significant F1 changes after Holm over 36: M3. Over the 12 new alone: M3.
- Holdout margins were computed once per model (step2a_forward.py, saved in holdout_margins_M*.npz). No crash, no method change.
