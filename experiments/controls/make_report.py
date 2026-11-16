"""Build out/REPORT.md from the computed files. Every number comes from out/*.json."""
import json
import numpy as np

OUT = '/home/claude/work/ext/out'
MODELS = [0, 2, 3, 4, 5, 8, 12, 13]
R = json.load(open(OUT + '/splits.json'))['models']
H = json.load(open(OUT + '/holm.json'))
MN = json.load(open(OUT + '/mean_vec_norm.json'))
META = {n: json.load(open('%s/meta_M%d.json' % (OUT, n))) for n in MODELS}
lines = []
P = lines.append


def sm(d):
    return '%+.2f [%+.2f, %+.2f]' % (d['mean'], d['p2.5'], d['p97.5'])


def pair(a, b, n, key='per_split_f1_diff'):
    """Paired per-split difference of two methods (F1)."""
    d = np.array(R[str(n)][key][a]) - np.array(R[str(n)][key][b])
    return '%+.2f [%+.2f, %+.2f] / %.2f' % (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean())


def table(head, rows):
    P('| ' + ' | '.join(head) + ' |')
    P('|' + '|'.join(['---'] * len(head)) + '|')
    for r in rows:
        P('| ' + ' | '.join(str(x) for x in r) + ' |')
    P('')


M = lambda n, k: R[str(n)]['methods'][k]
PS = lambda n, tag: R[str(n)]['placebo_per_seed_mean_over_splits'][tag]


def rng_(v):
    return '%+.2f [%+.2f, %+.2f]' % (np.mean(v), min(v), max(v))


def rng4(v):
    return '%+.4f [%+.4f, %+.4f]' % (np.mean(v), min(v), max(v))


P('# Scalpel control experiments (word-level models M0, M2, M3, M4, M5, M8, M12, M13)')
P('')
P('All numbers below were computed by the scripts in `/home/claude/work/ext/` and read from `/home/claude/work/ext/out/`. '
  'Units: F1, accuracy, precision, recall in percent points (pp). "Effect" = method minus `orig` at threshold 0, on the eval half. '
  '`UZ` = `untrained_zero`. "rc" = label-free recenter (fit-half mean margin set equal to the fit-half mean of `orig`). '
  'Format "mean [p2.5, p97.5]" is over 50 random stratified 50/50 splits (seeds 0..49). '
  'The splits overlap, so these percentiles show split-to-split variation; they are NOT independent-sample confidence intervals. '
  'The only real intervals are the paired bootstrap ones (2,000 resamples of the 7,608 samples).')
P('')
P('## 0. Verification of `orig`')
P('')
rows = []
for n in MODELS:
    m = META[n]
    rows.append([n, '%d/7608' % m['orig_pred_equals_cached_base_pred'], '%d/7608' % m['orig_pred_equals_archive_predicted_txt'],
                 '%.3f' % m['accuracy_full_test_pct'], '%.3f / %.3f' % (m['holdout_f1_recomputed'], m['holdout_f1_summary']),
                 '%.3f / %.3f' % (m['holdout_acc_recomputed'], m['holdout_acc_summary']), m['holdout_idx_matches_bundle_part']])
table(['model', 'pred = cached base_pred', 'pred = archive predicted.txt', 'acc full test %', 'holdout F1 (mine / summary_all)', 'holdout acc (mine / summary_all)', 'holdout idx from DEV_SEED = bundle part'], rows)
P('One mismatch: M0, sample 7513, margin +0.0015 (my prediction and the cached one are 1 and `predicted.txt` has 0). This is a float-noise case at the decision boundary. All other models match `predicted.txt` 7608/7608. '
  'Holdout F1 and accuracy of `orig` equal the summary_all.json values for all 8 models (to the 3 printed decimals). '
  'Message-stage recompute vs cached features: max abs margin gap 0.0 for all models.')
P('')
P('## 1. Does zeroing untrained rows beat the placebos?')
P('')
P('Placebos: both zero the same NUMBER of message rows as `untrained_zero` (about 14.8k rows), chosen from rows that were trained, 20 seeds each. '
  '`uniform` = uniformly at random among trained rows. `freqmatch` = each untrained row gets a distinct random trained row with the same number of occurrences in the test messages '
  '(if that frequency bucket is used up: the nearest available frequency). Achieved total test occurrences vs the untrained rows: '
  + '; '.join('M%d %d vs %d' % (n, round(np.mean(META[n]['placebo_total_test_occurrences']['freqmatch'])), META[n]['test_occurrences_untrained_rows']) for n in MODELS[:1])
  + ' (same for the others up to 1 %; freqmatch is about 8 % below the target; uniform placebos have about 2.7x the occurrences). '
  '**freqmatch is the fairer control**: untrained rows are rare in the test set (about 1 occurrence each), while a uniform trained row is often a frequent token, so uniform placebos remove far more input signal and are '
  'damaged by exposure, not by the kind of row. freqmatch equalizes exposure. It is still not perfect: a trained row carries a small learned signal that an untrained row does not.')
P('')
P('F1 effect (pp), rc. Placebo cells: mean over seeds [min, max over the 20 seeds], each seed averaged over the 50 splits.')
P('')
rows = []
for n in MODELS:
    pf, pu = PS(n, 'placebo_freqmatch')['f1'], PS(n, 'placebo_untrained')['f1']
    c = R[str(n)]['untrained_vs_placebo']['placebo_freqmatch']['f1']
    b = R[str(n)]['bootstrap_full_test']
    rows.append([n, '%.2f' % M(n, 'orig')['abs_mean']['f1'], sm(M(n, 'untrained_zero')['f1']), rng_(pf), rng_(pu),
                 '%d/20' % round(20 * c['frac_seeds_placebo_mean_effect_ge_untrained']),
                 sm(c['untrained_minus_placebo_seedmean']), '[%+.2f, %+.2f]' % tuple(b['untrained_zero_minus_placebo_freqmatch']['ci95'])])
table(['model', 'orig F1 (eval-half mean)', 'UZ effect', 'freqmatch placebo', 'uniform placebo', 'freqmatch seeds with effect >= UZ', 'UZ - freqmatch placebo mean (per split)', 'bootstrap 95 % full set, UZ - freqmatch'], rows)
P('Threshold-free effects (AUC and average precision, no recenter needed; mean over 50 splits; placebo as above):')
P('')
rows = []
for n in MODELS:
    rows.append([n, '%.4f' % M(n, 'orig')['auc']['abs_mean'], '%+.4f' % M(n, 'untrained_zero')['auc']['mean'], rng4(PS(n, 'placebo_freqmatch')['auc']), rng4(PS(n, 'placebo_untrained')['auc']),
                 '%.4f' % M(n, 'orig')['ap']['abs_mean'], '%+.4f' % M(n, 'untrained_zero')['ap']['mean'], rng4(PS(n, 'placebo_freqmatch')['ap']), rng4(PS(n, 'placebo_untrained')['ap'])])
table(['model', 'orig AUC', 'UZ dAUC', 'freqmatch dAUC', 'uniform dAUC', 'orig AP', 'UZ dAP', 'freqmatch dAP', 'uniform dAP'], rows)
P('Without the recenter the F1 effects are (UZ / freqmatch mean / uniform mean): ' + '; '.join(
    'M%d %+.2f / %+.2f / %+.2f' % (n, M(n, 'untrained_zero__norc')['f1']['mean'], M(n, 'placebo_freqmatch__norc')['f1']['mean'], M(n, 'placebo_untrained__norc')['f1']['mean']) for n in MODELS) + '.')
P('')
P('## 2. Does it beat `thr_only` and `lstsq_head`?')
P('')
P('Label-using baselines are fitted on the fit half only. `lstsq_head`: ridge least squares of a +-1 target on standardized features of the input of the last linear layer (ridge by inner 80/20 split, MSE), then best-F1 threshold on the fit half (`+thr`). '
  '`lstsq_head_argmax` = threshold 0 (a one-hot two-output fit with argmax is the same linear rule). Paired columns: UZ(rc) minus the baseline, per split, "mean [p2.5, p97.5] / fraction of splits with UZ better".')
P('')
rows = []
for n in MODELS:
    rows.append([n, '%+.2f' % M(n, 'untrained_zero')['f1']['mean'], '%+.2f' % M(n, 'thr_only')['f1']['mean'], '%+.2f' % M(n, 'lstsq_head+thr')['f1']['mean'],
                 '%+.2f' % M(n, 'lstsq_head_argmax')['f1']['mean'], '%+.2f' % M(n, 'untrained_zero+thr')['f1']['mean'],
                 pair('untrained_zero', 'thr_only', n), pair('untrained_zero', 'lstsq_head+thr', n),
                 '%+.4f' % (M(n, 'lstsq_head')['auc']['mean'])])
table(['model', 'UZ (rc)', 'thr_only', 'lstsq_head+thr', 'lstsq_head argmax', 'UZ+thr', 'UZ - thr_only', 'UZ - lstsq_head+thr', 'lstsq_head dAUC'], rows)
P('(F1 effects are means over 50 splits. `lstsq_head` dAUC is the AUC change of the refitted head; UZ dAUC is in table 1.)')
P('')
P('## 3. Does imputation beat zero?')
P('')
P('Mean imputation = mean of all trained message rows (pad row excluded). kNN imputation = mean of trained rows sharing the longest common prefix (>= 5 characters); if none, mean of trained rows with the same character shape (hex / number / snake_case / word). '
  'This is a string heuristic, not a semantic neighbour (untrained rows are random vectors, so embedding-space neighbours are not available). Share of untrained rows by source for M3: '
  + ', '.join('%s %d' % kv for kv in META[3]['knn_neighbour_source'].items()) + '. Paired columns: zero minus imputation (per split), "mean [p2.5, p97.5] / fraction where zero is better".')
P('')
rows = []
for n in MODELS:
    rows.append([n, '%+.2f' % M(n, 'untrained_zero')['f1']['mean'], '%+.2f' % M(n, 'untrained_impute_mean')['f1']['mean'], '%+.2f' % M(n, 'untrained_impute_knn')['f1']['mean'],
                 pair('untrained_zero', 'untrained_impute_mean', n), pair('untrained_zero', 'untrained_impute_knn', n),
                 '%+.4f / %+.4f / %+.4f' % (M(n, 'untrained_zero')['auc']['mean'], M(n, 'untrained_impute_mean')['auc']['mean'], M(n, 'untrained_impute_knn')['auc']['mean']),
                 '%+.4f / %+.4f / %+.4f' % (M(n, 'untrained_zero')['ap']['mean'], M(n, 'untrained_impute_mean')['ap']['mean'], M(n, 'untrained_impute_knn')['ap']['mean']),
                 '%.4f' % MN[str(n)]['ratio']])
table(['model', 'zero F1', 'mean F1', 'kNN F1', 'zero - mean', 'zero - kNN', 'dAUC zero / mean / kNN', 'dAP zero / mean / kNN', '|mean trained row| / mean row norm'], rows)
P('The last column explains the first result: trained message rows have the same norm as untrained ones (about 11.3, the size of a random N(0,1) 128-vector), so training moved them little, and the mean of the trained rows is nearly the zero vector. Zero and mean imputation are almost the same edit.')
P('')
P('## 4. Stability across the 50 splits (F1 effect, pp; mean [p2.5, p97.5] / fraction of splits with a positive effect)')
P('')
cols = [('untrained_zero', 'UZ (rc)'), ('untrained_zero__norc', 'UZ (no rc)'), ('rare_zero', 'rare_zero (rc)'), ('prior', 'prior'), ('thr_only', 'thr_only'), ('untrained_zero+thr', 'UZ+thr'), ('lstsq_head+thr', 'lstsq_head+thr')]
rows = []
for n in MODELS:
    rows.append([n] + ['%s / %.2f' % (sm(M(n, k)['f1']), M(n, k)['f1']['frac_pos']) for k, _ in cols])
table(['model'] + [c for _, c in cols], rows)
P('Accuracy / precision / recall effect of UZ (rc), mean over splits:')
P('')
rows = [[n] + ['%+.2f' % M(n, 'untrained_zero')[m]['mean'] for m in ('acc', 'precision', 'recall')] + ['%+.2f' % M(n, 'prior')[m]['mean'] for m in ('acc', 'precision', 'recall')] for n in MODELS]
table(['model', 'UZ acc', 'UZ prec', 'UZ rec', 'prior acc', 'prior prec', 'prior rec'], rows)
P('Paired bootstrap on all 7,608 samples (2,000 resamples, seed 777; F1 effect in pp, 95 % percentile interval; rc = mean margin on all samples set to the `orig` mean):')
P('')
rows = []
for n in MODELS:
    b = R[str(n)]['bootstrap_full_test']
    f = lambda k: '%+.2f [%+.2f, %+.2f]' % (b[k]['mean'], b[k]['ci95'][0], b[k]['ci95'][1])
    rows.append([n, '%.2f' % b['point_f1_orig'], f('untrained_zero'), f('untrained_zero__norc'), f('rare_zero'), f('untrained_impute_mean'), f('untrained_impute_knn'), f('placebo_freqmatch'), f('placebo_untrained')])
table(['model', 'orig F1 (full test)', 'UZ (rc)', 'UZ (no rc)', 'rare_zero (rc)', 'impute mean (rc)', 'impute kNN (rc)', 'freqmatch placebo mean (rc)', 'uniform placebo mean (rc)'], rows)
P('## 5. Holm correction of the 24 published comparisons (12 models x 2 rounds)')
P('')
P('Source: `/home/claude/work/results/summary_all.json` (holdout, 3,804 samples). `p_f1_up` is one-sided (share of bootstrap resamples with a positive F1 change); converted to two-sided as 2*min(p, 1-p). '
  'Bootstrap resolution: B = 10,000 (`N_BOOT` in scalpel/report.py), so a reported 0 means p < 1/B; its two-sided value is bounded here by 2/B = 0.0002 (used in Holm, so adjusted values for those are lower bounds on the true level, conservative). '
  'M15 round 1 changed no prediction (d_f1 = 0, p_f1_up = 0 is a tie artefact); I set its p to 1. Resamples with a zero difference count as "not positive", which inflates the one-sided p for rows with many ties (e.g. M7 round 2). '
  'McNemar (exact, two-sided, accuracy-type test on fixed/broken pairs) is corrected in a second family of 24; a reported 0.000000 is replaced by 1e-6.')
P('')
rows = []
for it in H['results']:
    rows.append([it['round'], it['model'], '%+.3f' % it['d_f1'], '%.4f' % it['p_f1_two_sided'], '%.4f' % it['holm_p_f1'], 'yes' if it['sig_f1_holm'] else 'no',
                 '%.6f' % it['mcnemar_p'], '%.4f' % it['holm_p_mcnemar'], 'yes' if it['sig_mcnemar_holm'] else 'no'])
table(['round', 'model', 'd_f1 pp', 'p two-sided (F1)', 'Holm p (F1)', 'sig at 0.05 (F1)', 'McNemar p', 'Holm p (McNemar)', 'sig at 0.05 (McNemar)'], rows)
sigf = ['R%d M%d (%+.2f pp)' % (i['round'], i['model'], i['d_f1']) for i in H['results'] if i['sig_f1_holm']]
sigm = ['R%d M%d' % (i['round'], i['model']) for i in H['results'] if i['sig_mcnemar_holm']]
rawf = ['R%d M%d' % (i['round'], i['model']) for i in H['results'] if i['sig_f1_raw']]
P('Significant after Holm, F1 change: ' + ', '.join(sigf) + '. Significant before correction (F1): ' + ', '.join(rawf) + '.')
P('Significant after Holm, McNemar: ' + ', '.join(sigm) + '. McNemar tests whether fixed != broken, in either direction and about accuracy; it can be significant when F1 does not move (e.g. prior-correction rounds trade precision for recall).')
P('Note M14 round 1 and round 2 are the same edit with the same numbers, counted twice as asked.')
P('')
P('## What failed, what I could not verify, deviations')
P('')
for t in open(OUT + '/caveats.txt').read().strip().split('\n'):
    P('- ' + t)
P('')
P('## Files')
P('')
for t in open(OUT + '/files.txt').read().strip().split('\n'):
    P('- ' + t)
open(OUT + '/REPORT.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
