# FROZEN: Round 3 final evaluation

Written 2026-11-16T17:15:12 (UTC) before any holdout sample was forwarded.

## Recipe
Edit E3, k = 3, no recenter. Every message-embedding row is multiplied by c/(c+3), c = training document frequency of the token (untrained rows become 0). The pad row is unchanged. For M14 the edit is applied to all three message tables (cnn, trans, lstm). Prediction = margin (logit1 - logit0) > 0. Code path: r3lib.new_table with edit E3 and fwd.edited_margins (copy of r3/compute_margins.py logic).

sha256 recipe.json: `dec231a0f68ec3ce00a9fc413cf07ec356b60313e806ea854c34e62a7820ef8e`

sha256 decisions.json: `f8574104b952d45829051bf549a9621bcac7b292e560f8df794ebe55bddedff7`

## Decisions (12)

| model | decision | guard F1 original | guard F1 recipe |
|---|---|---|---|
| M0 | E3_k3|raw | 73.214 | 73.435 |
| M2 | E3_k3|raw | 71.654 | 73.738 |
| M3 | E3_k3|raw | 73.148 | 74.187 |
| M4 | E3_k3|raw | 72.763 | 73.872 |
| M5 | E3_k3|raw | 72.317 | 73.080 |
| M8 | E3_k3|raw | 72.444 | 73.054 |
| M12 | E3_k3|raw | 73.569 | 73.616 |
| M13 | E3_k3|raw | 73.346 | 74.458 |
| M6 | original | 64.023 | 63.886 |
| M7 | E3_k3|raw | 67.122 | 67.257 |
| M14 | E3_k3|raw | 75.697 | 76.190 |
| M15 | original (not applicable) | 83.909 | n/a |

## Step 1 check (diag and guard, before holdout)
Re-ran the recipe on diag+guard for the 10 applied models: max abs gap of F1 against decisions.json (recipe and original, diag and guard) = 0.0; max abs gap of margins against the saved r3 margins = 0.0. M6 and M15 (original kept): F1 from the cached head equals decisions.json (gap 0.0). Details: out/step1_verify.json.

## Rules fixed before the holdout run
- Holdout = bundle part 2 (3,804 samples; M6, M7, M15 use their own bundles, M14 the default bundle, as r3 did). Margins of each model are computed once and saved, metrics are computed from the saved file.
- Bootstrap and McNemar: scalpel/report.py (compare, paired_bootstrap: 10,000 resamples, seed 12345), the same code as rounds 1 and 2. One-sided share p_f1_up = share of resamples with F1 difference > 0. Two-sided p = 2*min(p_up, 1-p_up); p_up = 0 or 1 -> 2/10000; no changed prediction (fixed = broken = 0) -> p = 1 (as in ext/holm.py). McNemar p = 0.0 after rounding -> 1e-6.
- Holm: ext/holm.py function, over 36 (24 from ext/out/holm.json + 12 new) and over the 12 new alone; alpha 0.05; F1 two-sided p is the main test, McNemar is reported next to it.
- Round 2 rebuild: only where the published round 2 edit is plain "untrained rows to zero" (M4, M6, M7, M8, M12, M13; M14 = CNN table only). The rebuild counts as exact only if F1, fixed, broken, FP and FN all equal the published round 2 values. Otherwise only the F1 difference is given, with no interval.
- No change of method after seeing holdout numbers except to fix a crash.
