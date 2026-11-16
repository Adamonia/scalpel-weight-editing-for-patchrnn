"""Step 3: Holm correction over the 24 published comparisons (12 models x 2 rounds)."""
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
import json

B = 10000      # N_BOOT in scalpel/report.py; the smallest nonzero bootstrap fraction is 1/B
rows = json.load(open(os.path.join(ROOT, 'results', 'summary_all.json')))


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


items = []
for r in rows:
    p_up = r['p_f1_up']
    degenerate = r['fixed'] == 0 and r['broken'] == 0          # no prediction changed: F1 diff is exactly 0
    if degenerate:
        p2, note = 1.0, 'no prediction changed; p_f1_up=0 is a tie artefact, set p=1'
    elif p_up == 0.0:
        p2, note = 2.0 / B, 'bootstrap p_up=0 means < 1/B; two-sided bound 2/B used (true p is smaller)'
    elif p_up == 1.0:
        p2, note = 2.0 / B, 'bootstrap p_up=1 means 1-p < 1/B; two-sided bound 2/B used'
    else:
        p2, note = min(1.0, 2 * min(p_up, 1 - p_up)), 'two-sided = 2*min(p_up, 1-p_up)'
    mc = r['mcnemar_p']
    mc_note = ''
    if mc == 0.0:
        mc, mc_note = 1e-6, 'reported as 0 after rounding to 6 decimals; 1e-6 used (true p smaller)'
    items.append({'round': r['round'], 'model': r['model'], 'commits': r['commits'], 'd_f1': r['d_f1'],
                  'ci_f1': r['ci_f1'], 'p_f1_up': p_up, 'p_f1_two_sided': p2, 'p_f1_note': note,
                  'mcnemar_p': r['mcnemar_p'], 'mcnemar_p_used': mc, 'mcnemar_note': mc_note,
                  'fixed': r['fixed'], 'broken': r['broken']})
a1 = holm([i['p_f1_two_sided'] for i in items])
a2 = holm([i['mcnemar_p_used'] for i in items])
for it, x, z in zip(items, a1, a2):
    it['holm_p_f1'] = x
    it['holm_p_mcnemar'] = z
    it['sig_f1_raw'] = it['p_f1_two_sided'] < 0.05
    it['sig_f1_holm'] = x < 0.05
    it['sig_mcnemar_raw'] = it['mcnemar_p_used'] < 0.05
    it['sig_mcnemar_holm'] = z < 0.05
out = {'m': len(items), 'alpha': 0.05, 'B_bootstrap': B, 'bootstrap_resolution': 1.0 / B,
       'notes': ['p_f1_up in summary_all.json is the share of paired-bootstrap resamples with a positive F1 difference (one-sided); converted to two-sided as 2*min(p,1-p).',
                 'Resamples with an exactly zero difference count as not positive, so ties inflate the one-sided p.',
                 'M14 round 1 and round 2 are the same edit with identical numbers (counted as two comparisons, as asked).',
                 'The F1 p-values come from the 3,804 holdout samples; McNemar uses the same holdout pairs.'],
       'results': items}
json.dump(out, open(os.path.join(_HERE, 'out', 'holm.json'), 'w'), indent=1)
print('%-3s %-3s %-8s %-8s %-9s %-9s %-9s %-9s' % ('rd', 'M', 'd_f1', 'p2(F1)', 'holm(F1)', 'mcn', 'holm(mcn)', ''))
for it in items:
    print('%-3d %-3d %-8.3f %-8.4f %-9.4f %-9.6f %-9.4f %s%s' % (it['round'], it['model'], it['d_f1'], it['p_f1_two_sided'], it['holm_p_f1'], it['mcnemar_p'], it['holm_p_mcnemar'],
          'F1* ' if it['sig_f1_holm'] else '', 'MCN* ' if it['sig_mcnemar_holm'] else ''))
