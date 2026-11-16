"""Step 2b: metrics, paired bootstrap, McNemar, Holm from the saved holdout margins. Uses scalpel/report.py as rounds 1 and 2."""
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
import sys, json, datetime
import numpy as np
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scalpel.report import compare, paired_bootstrap, roc_auc, mcnemar_exact, N_BOOT
from scalpel.engine import confusion_metrics

OUTD = os.path.join(_HERE, '..', 'out') + '/'
ALL = [0, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]
dec = json.load(open(ROOT + '/experiments/round3/out/decisions.json'))
pub = json.load(open(ROOT + '/results/summary_all.json'))
r1 = {r['model']: r for r in pub if r['round'] == 1}
r2 = {r['model']: r for r in pub if r['round'] == 2}
holm_pub = json.load(open(ROOT + '/experiments/controls/out/holm.json'))['results']


def holm(ps):                                     # same function as ext/holm.py
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def p_conv(r):                                    # same convention as ext/holm.py
    p_up = r['p_f1_up']
    if r['fixed'] == 0 and r['broken'] == 0:
        return 1.0, 'no prediction changed: p = 1'
    if p_up == 0.0 or p_up == 1.0:
        return 2.0 / N_BOOT, 'bootstrap share at the 1/B limit: 2/B'
    return min(1.0, 2 * min(p_up, 1 - p_up)), 'two-sided = 2*min(p_up, 1-p_up)'


out, rows, orig_gaps = {}, [], {}
for no in ALL:
    z = np.load(OUTD + 'holdout_margins_M%d.npz' % no)
    y = z['y'].astype(np.int64)
    assert len(y) == 3804
    p0, p3 = z['E0'] > 0, z['R'] > 0
    c = compare(y, p0, p3)
    auc0, auc3 = round(roc_auc(y, z['E0']), 5), round(roc_auc(y, z['R']), 5)
    b, a, bs = c['before'], c['after'], c['bootstrap']
    pub1 = r1[no]
    orig_gaps[no] = {'f1': abs(b['f1'] - pub1['f1'][0]), 'acc': abs(b['acc'] - pub1['acc'][0]), 'auc': abs(auc0 - pub1['auc'][0]),
                     'mcc': abs(b['mcc'] - pub1['mcc'][0]), 'prec': abs(b['precision'] - pub1['prec'][0]), 'rec': abs(b['recall'] - pub1['rec'][0])}
    d = dec[str(no)]
    applied = d['applied'] != 'original'
    row = {'round': 3, 'model': no, 'label': pub1['label'], 'commits': ['message embedding scaling c/(c+3)'] if applied else [],
           'f1': [b['f1'], a['f1']], 'd_f1': c['delta']['f1'], 'ci_f1': bs['f1']['ci95'], 'p_f1_up': bs['f1']['p_improve'],
           'acc': [b['acc'], a['acc']], 'd_acc': c['delta']['acc'], 'ci_acc': bs['acc']['ci95'],
           'prec': [b['precision'], a['precision']], 'rec': [b['recall'], a['recall']], 'mcc': [b['mcc'], a['mcc']],
           'auc': [auc0, auc3], 'fixed': c['n_fixed'], 'broken': c['n_broken'], 'mcnemar_p': c['mcnemar_p'],
           'fp': [b['fp'], a['fp']], 'fn': [b['fn'], a['fn']],
           'dev_f1': {'diag': [round(d['diag_f1_before'], 3), round(d['diag_f1_after'], 3)], 'guard': [round(d['guard_f1_before'], 3), round(d['guard_f1_after'], 3)]}}
    rows.append(row)
    # round 2 comparison
    p2 = r2[no]
    cmp2 = {'f1_round2_published': p2['f1'][1], 'd_f1_r3_minus_r2': round(a['f1'] - p2['f1'][1], 5), 'paired_interval': None, 'rebuild': 'not possible (round 2 edit was not plain untrained-rows-to-zero)'}
    if 'E1' in z.files:
        pe = z['E1'] > 0
        ce = compare(y, p0, pe, bootstrap=False)
        ok = (ce['after']['f1'] == p2['f1'][1] and ce['n_fixed'] == p2['fixed'] and ce['n_broken'] == p2['broken']
              and ce['after']['fp'] == p2['fp'][1] and ce['after']['fn'] == p2['fn'][1])
        cmp2['rebuild'] = {'exact': bool(ok), 'rebuilt_f1': ce['after']['f1'], 'rebuilt_fixed': ce['n_fixed'], 'rebuilt_broken': ce['n_broken'],
                           'published_f1': p2['f1'][1], 'published_fixed': p2['fixed'], 'published_broken': p2['broken']}
        if ok:
            c32 = compare(y, pe, p3)
            bs2 = c32['bootstrap']
            cmp2['paired_interval'] = {'d_f1': c32['delta']['f1'], 'ci95': bs2['f1']['ci95'], 'p_up': bs2['f1']['p_improve'],
                                       'fixed': c32['n_fixed'], 'broken': c32['n_broken'], 'mcnemar_p': c32['mcnemar_p']}
    out[no] = {'model': no, 'label': pub1['label'], 'decision': d['applied'], 'applicable': d.get('applicable', True),
               'f1_original': b['f1'], 'f1_round2_published': p2['f1'][1], 'f1_round3': a['f1'], 'd_f1_vs_original': c['delta']['f1'],
               'ci95': bs['f1']['ci95'], 'p_up_one_sided': bs['f1']['p_improve'], 'mcnemar_p': c['mcnemar_p'],
               'fixed': c['n_fixed'], 'broken': c['n_broken'],
               'metrics_original': {k: b[k] for k in ('f1', 'acc', 'precision', 'recall', 'mcc')} | {'auc': auc0},
               'metrics_round3': {k: a[k] for k in ('f1', 'acc', 'precision', 'recall', 'mcc')} | {'auc': auc3},
               'vs_round2': cmp2, 'original_gap_to_published_round1': orig_gaps[no]}

# ---- Holm
new = []
for r in rows:
    p2s, note = p_conv(r)
    mc = r['mcnemar_p'] if r['mcnemar_p'] != 0.0 else 1e-6
    new.append({'p2': p2s, 'note': note, 'mc': mc})
pf = [h['p_f1_two_sided'] for h in holm_pub] + [n['p2'] for n in new]
pm = [h['mcnemar_p_used'] for h in holm_pub] + [n['mc'] for n in new]
h36f, h36m = holm(pf), holm(pm)
h12f, h12m = holm([n['p2'] for n in new]), holm([n['mc'] for n in new])
for i, r in enumerate(rows):
    o = out[r['model']]
    j = 24 + i
    o.update({'p_f1_two_sided': new[i]['p2'], 'p_note': new[i]['note'], 'mcnemar_p_used': new[i]['mc'],
              'holm36_p_f1': h36f[j], 'holm36_sig_f1': h36f[j] < 0.05, 'holm36_p_mcnemar': h36m[j], 'holm36_sig_mcnemar': h36m[j] < 0.05,
              'holm12_p_f1': h12f[i], 'holm12_sig_f1': h12f[i] < 0.05, 'holm12_p_mcnemar': h12m[i], 'holm12_sig_mcnemar': h12m[i] < 0.05})
pub_changes = []
for k, h in enumerate(holm_pub):
    was = h['sig_f1_holm']
    now = h36f[k] < 0.05
    if was != now:
        pub_changes.append({'round': h['round'], 'model': h['model'], 'sig_f1_holm_24': was, 'sig_f1_holm_36': now})
res = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
       'family': {'published': 24, 'new': 12, 'total': 36, 'alpha': 0.05, 'B_bootstrap': N_BOOT, 'bootstrap_seed': 12345},
       'original_vs_published_round1': {'max_abs_gap_f1': max(g['f1'] for g in orig_gaps.values()),
                                       'max_abs_gap_other_metrics': {k: max(g[k] for g in orig_gaps.values()) for k in ('acc', 'auc', 'mcc', 'prec', 'rec')},
                                       'models_with_f1_gap': [m for m, g in orig_gaps.items() if g['f1'] > 0]},
       'published_rows_whose_holm_f1_flag_changes_in_36_family': pub_changes,
       'models': {str(k): v for k, v in out.items()}}
json.dump(res, open(OUTD + 'round3_holdout.json', 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
json.dump(pub + rows, open(OUTD + 'summary_all_with_r3.json', 'w'), indent=1)
print(json.dumps(res['original_vs_published_round1']))
for k, v in out.items():
    print('M%-2d %-9s o %.3f r2 %.3f r3 %.3f d %+.3f ci %s fx/br %d/%d p2 %.4f mc %.4g h36 %.4f %s h12 %.4f %s | r2 rebuild %s' % (
        k, v['decision'][:9], v['f1_original'], v['f1_round2_published'], v['f1_round3'], v['d_f1_vs_original'], v['ci95'], v['fixed'], v['broken'],
        v['p_f1_two_sided'], v['mcnemar_p'], v['holm36_p_f1'], v['holm36_sig_f1'], v['holm12_p_f1'], v['holm12_sig_f1'],
        v['vs_round2']['rebuild'] if isinstance(v['vs_round2']['rebuild'], str) else v['vs_round2']['rebuild']['exact']))
print(pub_changes)
