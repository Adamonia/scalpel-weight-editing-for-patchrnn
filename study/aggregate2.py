"""Summary of all final sessions, F1 first. Round 1: runs/M<no> (MCC rule). Round 2: runs/R2_M<no> (F1 rule)."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import json, os
R = RUNS + os.sep; OUT = os.path.join(_HERE, '..', 'results') + os.sep
MODELS = [0, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]
def short(e):
    e = e.lower()
    if 'sub-model' in e: return 'untrained rows to zero (CNN sub-model)'
    if 'recentering' in e: return 'recentering'
    if e.startswith('rare'): return 'rare rows to zero'
    for key, name in (('prior', 'prior correction'), ('threshold not at', 'F1 threshold'), ('untrained', 'untrained rows to zero'), ('inert code', 'code branches removed')):
        if key in e: return name
    return e[:40]
rows, out = [], {}
for rnd, pfx in ((1, 'M'), (2, 'R2_M')):
    for no in MODELS:
        f = R + '%s%d/FINAL.json' % (pfx, no)
        if not os.path.exists(f): continue
        d = json.load(open(f)); h = d['parts']['holdout']; b, a, bs = h['before'], h['after'], h['bootstrap']
        rec = {'round': rnd, 'model': no, 'label': d['label'], 'commits': [short(e['error_type']) for e in d['error_types']],
               'f1': [b['f1'], a['f1']], 'd_f1': h['delta']['f1'], 'ci_f1': bs['f1']['ci95'], 'p_f1_up': bs['f1']['p_improve'],
               'acc': [b['acc'], a['acc']], 'd_acc': h['delta']['acc'], 'ci_acc': bs['acc']['ci95'],
               'prec': [b['precision'], a['precision']], 'rec': [b['recall'], a['recall']], 'mcc': [b['mcc'], a['mcc']],
               'auc': [h['auc']['before'], h['auc']['after']], 'fixed': h['n_fixed'], 'broken': h['n_broken'], 'mcnemar_p': h['mcnemar_p'],
               'fp': [b['fp'], a['fp']], 'fn': [b['fn'], a['fn']],
               'dev_f1': {p: [d['parts'][p]['before']['f1'], d['parts'][p]['after']['f1']] for p in ('diag', 'guard')}}
        rows.append(rec)
json.dump(rows, open(OUT + 'summary_all.json', 'w'), indent=1)
L = []
for rnd, title in ((1, 'Round 1 (MCC rule; recipes: prior correction, rare or untrained rows, code branches)'), (2, 'Round 2 (F1 rule; recipes: untrained rows, rare rows, F1 threshold)')):
    L += ['## ' + title, '', '| Model | Accepted edits | F1 orig | F1 edit | d F1 [95% CI] | Acc orig | Acc edit | d Acc [95% CI] | Prec | Rec | AUC | fixed/broken | McNemar p |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['round'] != rnd: continue
        L.append('| M%d %s | %s | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %.1f -> %.1f | %.1f -> %.1f | %.4f -> %.4f | %d/%d | %.2g |' % (
            r['model'], r['label'], '; '.join(r['commits']) or 'none', r['f1'][0], r['f1'][1], r['d_f1'], r['ci_f1'][0], r['ci_f1'][1], r['acc'][0], r['acc'][1], r['d_acc'], r['ci_acc'][0], r['ci_acc'][1],
            r['prec'][0], r['prec'][1], r['rec'][0], r['rec'][1], r['auc'][0], r['auc'][1], r['fixed'], r['broken'], r['mcnemar_p']))
    L.append('')
open(OUT + 'summary_all.md', 'w').write('\n'.join(L))
print('\n'.join(L))
