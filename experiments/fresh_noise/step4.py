"""Step 4: build out/REPORT.md tables from the saved json files only."""
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
import json
import numpy as np
O = os.path.join(_HERE, 'out')
MODELS = [3, 4, 5, 8, 12, 13]
A = {n: json.load(open('%s/t1t2_M%d.json' % (O, n))) for n in MODELS}
B = {n: json.load(open('%s/t3t4_M%d.json' % (O, n))) for n in MODELS}
L = []
w = L.append
def table(head, rows):
    w('| ' + ' | '.join(head) + ' |'); w('|' + '---|' * len(head))
    for r in rows: w('| ' + ' | '.join(r) + ' |')
    w('')
f = lambda x, k=2: ('%.' + str(k) + 'f') % x

w('# Test of hypothesis H on six word-level Transformer security-patch classifiers\n')
w('All numbers are read from `out/t1t2_M*.json` and `out/t3t4_M*.json`, written by `step1.py` and `step3.py`. F1 and AUC in percent. No gradient training. Test set: all 7608 samples; holdout = part 2.\n')
w('## 0. Setup and checks\n')
table(['Model', 'heads', 'layers', 'untrained rows', 'samples with untrained token', 'max abs gap orig', 'max abs gap zero', 'bias shift (zero)', 'F1 orig', 'F1 zero', 'gain F1 (all)', 'gain F1 (holdout)'],
      [[f'M{n}', str(A[n]['n_heads_registry']), str(A[n]['n_layers_checkpoint']), str(A[n]['n_untrained_rows']), f"{A[n]['n_hit_samples']} ({100*A[n]['T2']['frac_hit']:.0f}%)",
        '%.1e' % A[n]['gap_orig_vs_cache'], '%.1e' % A[n]['gap_zero_vs_cache'], f(A[n]['bias_shift_zero'], 3), f(A[n]['orig']['f1']), f(A[n]['zero']['f1']), f(A[n]['zero_gain_f1']), f(A[n]['zero_gain_f1_ho'])] for n in MODELS])
w('Layers are counted from the checkpoint keys. The head count is NOT stored in the checkpoint (the attention weight shapes are the same for any head count); it comes from the loader registry (`scalpel/models.py`), so it is taken from the project, not verified from the weights. The recomputed orig and zero margins equal the cached ones (gap 0). The recentered zero edit reproduces the stated gains (M3 +1.14, M4 +1.03, M8 +0.64, M13 +0.58, M5 +0.25, M12 +0.27). Recenter used here: subtract the median over all samples of (new margin - orig margin).\n')

w('## T1. Fresh noise in the untrained rows\n')
w('Replacement rows: independent N(0, scale^2) draws, seeds 0..K-1, bias recentered as for the zero edit. Gap = F1(zero) - mean F1(fresh). Percentile = position of the original model F1 among the fresh-noise F1 values (ties count half). sd is over draws.\n')
for sc, title in (('1.0', 'scale 1.0, K=30'), ('0.5', 'scale 0.5, K=10'), ('2.0', 'scale 2.0, K=10')):
    w('### ' + title + ' (all 7608)\n')
    table(['Model', 'F1 zero', 'F1 orig', 'F1 fresh mean', 'F1 fresh sd', 'gap zero-fresh', 'orig - fresh mean', 'orig percentile', 'AUC zero', 'AUC fresh mean', 'AUC gap'],
          [[f'M{n}', f(A[n]['zero']['f1']), f(A[n]['orig']['f1']), f(A[n]['T1'][sc]['f1_mean']), f(A[n]['T1'][sc]['f1_sd']), f(A[n]['T1'][sc]['gap_f1_zero_minus_fresh']),
            f(A[n]['T1'][sc]['orig_minus_fresh_f1']), f(A[n]['T1'][sc]['orig_f1_percentile'], 0), f(A[n]['zero']['auc'], 3), f(A[n]['T1'][sc]['auc_mean'], 3), f(A[n]['T1'][sc]['gap_auc_zero_minus_fresh'], 3)] for n in MODELS])
w('### Holdout (part 2), scale 1.0\n')
table(['Model', 'F1 zero', 'F1 fresh mean', 'F1 fresh sd', 'gap', 'F1 orig', 'orig percentile'],
      [[f'M{n}', f(A[n]['zero']['f1_ho']), f(A[n]['T1']['1.0']['f1_ho_mean']), f(A[n]['T1']['1.0']['f1_ho_sd']), f(A[n]['T1']['1.0']['gap_f1_ho']), f(A[n]['orig']['f1_ho']), f(A[n]['T1']['1.0']['orig_f1_ho_percentile'], 0)] for n in MODELS])
w('Dose-response: gap F1(zero) - F1(fresh) by scale.\n')
table(['Model', '0.5', '1.0', '2.0'], [[f'M{n}'] + [f(A[n]['T1'][sc]['gap_f1_zero_minus_fresh']) for sc in ('0.5', '1.0', '2.0')] for n in MODELS])

w('## T2. Clean-margin density and noise-to-signal (scale 1.0, K=30)\n')
w('Clean = zeroed model (recentered). sd_c = sd of the clean margin over the 7608 samples. Noise sd = per-sample sd of the margin over the 30 fresh draws. rms = root mean square over samples. Flip rate = share of predictions that differ between zero and a fresh draw (mean over draws).\n')
table(['Model', 'sd_c', '|m|<0.25 sd_c', '|m|<0.5 sd_c', '|m|<1.0 sd_c', 'noise sd / sd_c (rms, all)', '(rms, hit samples)', 'flip rate all (%)', 'flip rate hit (%)', 'sd(orig-zero)', 'sd(fresh-zero)'],
      [[f'M{n}', f(A[n]['T2']['clean_margin_sd']), f(100*A[n]['T2']['share_abs_clean_below_0.25sd'], 1) + '%', f(100*A[n]['T2']['share_abs_clean_below_0.50sd'], 1) + '%', f(100*A[n]['T2']['share_abs_clean_below_1.00sd'], 1) + '%',
        f(A[n]['T2']['noise_sd_over_clean_sd_rms_all'], 3), f(A[n]['T2']['noise_sd_over_clean_sd_rms_hit'], 3), f(100*A[n]['T2']['flip_rate_zero_vs_fresh_all'], 2), f(100*A[n]['T2']['flip_rate_zero_vs_fresh_hit'], 2),
        f(A[n]['T2']['change_sd_orig_minus_zero'], 3), f(A[n]['T2']['change_sd_fresh_minus_zero'], 3)] for n in MODELS])

w('## T3. Attention of the first encoder layer on untrained tokens (original models)\n')
w('Samples with at least one untrained token. The manual layer-0 forward reproduces the cached margins (max abs gap in the table). Mass = attention weight on untrained key positions. The CLS query is the one that feeds the feature in a 1-layer model. Two denominators: share of all 200 positions (padding is not masked in these models, so padding takes part of the mass) and share of real (non-pad) positions, where ratio = (mass on untrained / mass on real positions) / (untrained share of real positions). Ratio > 1 = attends more than chance.\n')
table(['Model', 'forward gap', 'untrained share all pos', 'untrained share real pos', 'CLS mass on untrained', 'CLS ratio (all pos)', 'CLS ratio (real only)', 'all-query ratio (all pos)', 'all-query ratio (real only)'],
      [[f'M{n}', '%.1e' % B[n]['manual_forward_max_abs_gap_all'], f(B[n]['T3']['share_U_of_all_positions'], 4), f(B[n]['T3']['share_U_of_real_positions'], 3), f(B[n]['T3']['cls_query']['mass_U_mean_over_heads'], 4), f(B[n]['T3']['cls_query']['ratio_to_share_all_positions'], 2), f(B[n]['T3']['cls_query']['ratio_real_only'], 2),
        f(B[n]['T3']['all_query']['ratio_to_share_all_positions'], 2), f(B[n]['T3']['all_query']['ratio_real_only'], 2)] for n in MODELS])
w('Per head, CLS query, ratio to share of all positions:\n')
for n in MODELS:
    w(f'- M{n}: ' + ', '.join(f(x, 2) for x in B[n]['T3']['cls_query']['ratio_per_head']))
w('')

w('## T4. Averaging across heads (multi-head models)\n')
w('Change of each head output at the CLS attention output of layer 0 (the only place where the per-head split is exact, because out_proj is linear in the concatenated heads). It is before residual, LayerNorm and FF, and in M12 also before layer 1, so it is NOT the pooled representation. Reference for independent equal-norm vectors: cosine 0, norm ratio 1/sqrt(H). Parallel vectors: cosine 1, ratio 1. "orig" = original rows minus zero rows; "fresh" = mean over fresh draws 0..4 minus zero rows.\n')
rows = []
for n in MODELS:
    if 'T4' not in B[n]: continue
    t = B[n]['T4']
    rows.append([f'M{n}', str(B[n]['n_heads']), f(t['orig_vs_zero']['mean_pairwise_cosine'], 3), f(t['orig_vs_zero']['ratio_norm_sum_over_sum_norms'], 3), f(t['fresh_vs_zero_5_draws_mean']['mean_pairwise_cosine'], 3), f(t['fresh_vs_zero_5_draws_mean']['ratio_norm_sum_over_sum_norms'], 3), f(t['orig_vs_zero']['ratio_reference_orthogonal_1_over_sqrt_H'], 3), f(t['orig_vs_zero']['mean_head_change_norm'], 3), f(t['orig_vs_zero']['norm_of_summed_change'], 3)])
table(['Model', 'heads', 'cos (orig)', 'ratio (orig)', 'cos (fresh)', 'ratio (fresh)', '1/sqrt(H)', 'mean head change norm (orig)', 'norm of summed change (orig)'], rows)
open('%s/REPORT.md' % O, 'w').write('\n'.join(L) + '\n')

# ---- summary numbers computed from the json files (rank correlations over the n = 6 models)
from scipy.stats import spearmanr
gap = [A[n]['T1']['1.0']['gap_f1_zero_minus_fresh'] for n in MODELS]
gain = [A[n]['zero_gain_f1'] for n in MODELS]
nsr = [A[n]['T2']['noise_sd_over_clean_sd_rms_all'] for n in MODELS]
dens = [A[n]['T2']['share_abs_clean_below_0.50sd'] for n in MODELS]
heads = [A[n]['n_heads_registry'] for n in MODELS]
lay = [A[n]['n_layers_checkpoint'] for n in MODELS]
S = {}
for name, a, b in (('gap_vs_heads', gap, heads), ('gap_vs_layers', gap, lay), ('nsr_vs_heads', nsr, heads), ('gain_vs_gap', gain, gap),
                   ('gap_vs_density05', gap, dens), ('nsr_vs_gap', nsr, gap)):
    r = spearmanr(a, b)
    S[name] = {'rho': float(r[0]), 'p_two_sided_n6': float(r[1])}
S['orig_f1_minus_fresh_mean_scale1'] = {f'M{n}': A[n]['T1']['1.0']['orig_minus_fresh_f1'] for n in MODELS}
json.dump(S, open('%s/summary.json' % O, 'w'), indent=1)
with open('%s/REPORT.md' % O, 'a') as fh:
    fh.write('\n## Rank correlations over the six models (from out/summary.json)\n\n| pair | Spearman rho | p (n=6, not corrected) |\n|---|---|---|\n')
    for k, v in S.items():
        if 'rho' in v: fh.write('| %s | %.2f | %.2f |\n' % (k, v['rho'], v['p_two_sided_n6']))
    fh.write('\nNegative rho for gap_vs_heads means: more heads, smaller gap (H direction). With n = 6 these are descriptive only.\n')
