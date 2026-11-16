"""Write out/REPORT.md and out/per_model_full.md from the saved json files only."""
import json
HERE = '/home/claude/work/r3'
O = HERE + '/out'
M = json.load(open(O + '/metrics.json'))
S = json.load(open(O + '/summary.json'))
R = json.load(open(O + '/recipe.json'))
D = json.load(open(O + '/decisions.json'))
C = json.load(open(O + '/choice_diag.json'))
ORDER = ['0', '2', '3', '4', '5', '8', '12', '13', '6', '7', '14']
LABEL = {'0': 'M0 LSTM', '2': 'M2 CNN', '3': 'M3', '4': 'M4', '5': 'M5', '8': 'M8', '12': 'M12', '13': 'M13', '6': 'M6 (BERT)', '7': 'M7 (RoBERTa)', '14': 'M14 (hybrid)'}
meta = {n: json.load(open('%s/meta_M%s.json' % (O, n))) for n in ORDER}
f = lambda x, p=2: ('%+.' + str(p) + 'f') % x
g = lambda x, p=2: ('%.' + str(p) + 'f') % x
ci = lambda c: '[%+.2f, %+.2f]' % tuple(c)
rcfg = R['config']
L = []
w = L.append

w('# Round 3: weight edits of the message embedding table (no gradient training)\n')
w('All numbers are read from `metrics.json`, `summary.json`, `choice_diag.json`, `recipe.json`, `decisions.json`, `meta_M*.json` in this folder. F1, AUC, AP in percent. Change = edit minus reference, in points. Intervals: paired bootstrap, 2,000 resamples of the samples of the part (seed 2026 diag, 2027 guard); for group means the same resample is used for all models of the group. Holdout (part 2) was never forwarded and no metric was computed on it (`check_idx` in `r3lib.py` raises on a holdout index; holdout cells of the saved margin files are NaN).\n')
w('Rule (fixed before any guard number was read, see `RULE.md`): choose by the best mean diag F1 change against E1 over the six word-level Transformers (M3, M4, M5, M8, M12, M13), 22 candidates (E2, E3 and E4 with k in {1,3,10,30,100}, each with and without recenter), ties within 0.05 pp go to E2 over E4 over E3; a noise gate (bootstrap interval of the mean must exclude 0) else E1.\n')

w('## 1. Recipe\n')
w('**Recipe: %s, k = %s, recenter = %s.** Row := a(c) * row with a(c) = c/(c+3), c = training count (document frequency in training samples) of the token. Untrained rows become 0. The pad row is never changed.\n' % (R['edit'], R['k'], 'yes' if R['recenter'] else 'no'))
d, gd = R['diag']['transformers6'], R['guard']['transformers6']
w('| set | mean F1 change vs E1 (6 Transformers) | 95 % interval | models improved vs E1 | mean change vs E0 | interval | improved vs E0 |')
w('|---|---|---|---|---|---|---|')
for nm, x in (('diag (chosen on)', d), ('guard (confirmation)', gd)):
    w('| %s | %s | %s | %d/6 | %s | %s | %d/6 |' % (nm, f(x['mean_d_f1_E1']), ci(x['ci95_mean_d_f1_E1']), x['n_improve_E1'], f(x['mean_d_f1_E0']), ci(x['ci95_mean_d_f1_E0']), x['n_improve_E0']))
w('')
w('**Guard check: %s.** Against E1 the guard gain is smaller than the diag gain (%s vs %s) and its interval includes 0. Against the original (E0) the guard gain is clear (%s, interval %s, %d/6 improved). Per model change vs E1 (F1, pp):\n' % (R['guard_confirms_diag'], f(gd['mean_d_f1_E1']), f(d['mean_d_f1_E1']), f(gd['mean_d_f1_E0']), ci(gd['ci95_mean_d_f1_E0']), gd['n_improve_E0']))
w('| model | ' + ' | '.join('M' + m for m in R['diag']['per_model_d_f1_vs_E1']) + ' |')
w('|---|' + '---|' * 6)
for nm in ('diag', 'guard'):
    w('| %s | ' % nm + ' | '.join(f(v) for v in R[nm]['per_model_d_f1_vs_E1'].values()) + ' |')
w('')
w('Top 8 candidates on diag (score = mean diag F1 change vs E1 over six Transformers):\n')
w('| candidate | score | 95 % interval |')
w('|---|---|---|')
for r in sorted(C['rows'], key=lambda r: -r['score'])[:8]:
    w('| %s | %s | %s |' % (r['cfg'].replace('|', ' / '), f(r['score'], 3), ci(r['ci95'])))
w('')
w('Winner after the tie rule: %s (tied set: %s); noise gate passed: %s.\n' % (C['winner_before_gate'], ', '.join(C['tied']), C['gate_passed']))

w('## 2. All candidates, summary over models\n')
w('Mean F1 change over the group of models, with the number of models that improve. E1 = Round 2 edit with the same recenter flag as the candidate. "raw" = no recenter, "rc" = recenter.\n')
cfgs = [c for c in S['diag']['transformers6'] if not c.startswith('E1|') and c != 'E0']
for pn in ('diag', 'guard'):
    w('### %s\n' % pn)
    w('| candidate | T6 vs E1 | T6 interval | T6 improved | T6 vs E0 | all-11 vs E1 | improved | all-11 vs E0 | improved |')
    w('|---|---|---|---|---|---|---|---|---|')
    # include E1 rows vs E0 for reference
    for c in ['E1|raw', 'E1|rc'] + cfgs:
        a, b = S[pn]['transformers6'][c], S[pn]['all_evaluated'][c]
        w('| %s | %s | %s | %d/6 | %s | %s | %d/11 | %s | %d/11 |' % (c.replace('|', ' / '), f(a['mean_d_f1_E1']), ci(a['ci95_mean_d_f1_E1']), a['n_improve_E1'], f(a['mean_d_f1_E0']), f(b['mean_d_f1_E1']), b['n_improve_E1'], f(b['mean_d_f1_E0']), b['n_improve_E0']))
    w('')

SHOW = ['E0', 'E1|raw', 'E1|rc', 'E2|raw', 'E2|rc', 'E3_k3|raw', 'E3_k3|rc', 'E4_k100|raw', 'E4_k100|rc']
def model_table(pn, cfg_list, no):
    m = M[pn][no]
    out = ['| config | F1 | AUC | AP | dF1 vs E0 | dF1 vs E1 | 95 % interval dF1 vs E1 | dAUC vs E0 | dAP vs E0 |', '|---|---|---|---|---|---|---|---|---|']
    for c in cfg_list:
        r = m[c]
        out.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (c.replace('|', ' / '), g(r['f1']), g(r['auc']), g(r['ap'], 2), f(r['d_f1_E0']), f(r['d_f1_E1']), ci(r['d_f1_E1_ci95']), f(r['d_auc_E0'], 3), f(r['d_ap_E0'], 3)))
    return out

w('## 3. Per model: selected configurations\n')
w('Full tables (all 45 configurations) are in `per_model_full.md`. Rows: original, Round 2 baseline E1 (raw and recentered), E2, the recipe E3 k=3, E3 k=3 with recenter, E4 k=100. dF1 vs E1 compares with the E1 row of the same recenter flag (E0 row: raw E1).\n')
for pn in ('diag', 'guard'):
    w('### %s\n' % pn)
    for no in ORDER:
        w('**%s**\n' % LABEL[no])
        w('\n'.join(model_table(pn, SHOW, no)))
        w('')
w('## 4. Per-model decisions (recipe accepted only if guard F1 does not fall against the original)\n')
w('| model | decision | diag F1 before | diag F1 after | guard F1 before | guard F1 after | recipe alone: diag / guard |')
w('|---|---|---|---|---|---|---|')
for no in ['0', '2', '3', '4', '5', '8', '12', '13', '6', '7', '14', '15']:
    e = D[no]
    alone = '%s / %s' % (g(e['recipe_if_applied_diag_f1']), g(e['recipe_if_applied_guard_f1'])) if 'recipe_if_applied_diag_f1' in e else 'n/a'
    w('| %s | %s | %s | %s | %s | %s | %s |' % (no, e['applied'].replace('|', ' / '), g(e['diag_f1_before']), g(e['diag_f1_after']), g(e['guard_f1_before']), g(e['guard_f1_after']), alone))
w('')
for no in ('6', '15'):
    w('- M%s: %s' % (no, D[no]['reason']))
w('')

w('## 5. Checks\n')
w('| model | pad id | pad = fill value of msg | pad row training count | E0 vs cached head (max abs margin gap) | E0 vs ext orig | E1 vs ext untrained_zero | rows changed by E1 |')
w('|---|---|---|---|---|---|---|---|')
for no in ORDER:
    m = meta[no]
    w('| %s | %d | %s | %s | %s | %s | %s | %s |' % (LABEL[no], m['pad_id'], m['pad_equals_fill_value'], m['pad_row_training_count'], m['E0_vs_cached_head_max_abs_margin_gap'], m.get('E0_vs_ext_orig_max_abs_gap', 'n/a'), m.get('E1_vs_ext_untrained_zero_max_abs_gap', 'n/a'), m['n_rows_changed_E1']))
w('')
w('Gaps are on diag+guard samples only. For M6, M7 and M14 there is no ext file. Their E1 with recenter reproduces the Round 2 numbers of `results/summary_all.json` (dev_f1): M6 diag %s / guard %s (Round 2: 66.071 / 64.394), M7 %s / %s (68.826 / 67.251), M14 with the CNN table only %s / %s (78.249 / 76.028).\n' % tuple(g(M[p][n][c]['f1'], 3) for n, c in (('6', 'E1|rc'), ('7', 'E1|rc'), ('14', 'E1_cnnonly|rc')) for p in ('diag', 'guard')))
w('E0 diag/guard F1 as recomputed here: M6 %s / %s, M7 %s / %s, M14 %s / %s (Round 2 before-values 65.986 / 64.023, 68.826 / 67.122, 78.112 / 75.697).\n' % tuple(g(M[p][n]['E0']['f1'], 3) for n in ('6', '7', '14') for p in ('diag', 'guard')))

open(O + '/REPORT_head.md', 'w').write('\n'.join(L))

# full per-model tables
F = ['# Full per-model tables (all configurations)\n']
allc = list(M['diag']['3'].keys())
allc = [c for c in allc if c != '_shifts']
for pn in ('diag', 'guard'):
    F.append('## %s\n' % pn)
    for no in ORDER:
        cl = [c for c in allc if not c.startswith('E1_cnnonly') or no == '14']
        F.append('### %s\n' % LABEL[no])
        F.append('\n'.join(model_table(pn, cl, no)))
        F.append('')
open(O + '/per_model_full.md', 'w').write('\n'.join(F))
print('ok')
