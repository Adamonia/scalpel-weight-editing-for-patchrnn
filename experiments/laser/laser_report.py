"""Read out/laser_rows.csv (and the margins files) and write out/REPORT.md and out/summary.json.

Two ways to choose an edit, both on diag only:
  per model : the row with the largest diag F1 gain over the original whose diag MCC does not fall.
              It is used only if guard F1 and guard MCC do not fall either; else the model keeps its weights.
  one recipe: (type, branch, layer position, rho) with the largest mean diag F1 gain over the models that have it
              (at least --min-models), mean diag MCC change >= 0.
Holdout numbers are reported for both; they are not used for any choice.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BUNDLE_OF = {6: 'bundle_bert', 7: 'bundle_roberta', 15: 'bundle_codebert'}
LABEL = {0: 'M0 LSTM', 2: 'M2 CNN', 3: 'M3 Trans-1H', 4: 'M4 Trans-2H', 5: 'M5 Trans-8H', 6: 'M6 Trans-8H+BERT',
         7: 'M7 Trans-8H+RoBERTa', 8: 'M8 Trans-16H', 12: 'M12 Trans-8H-2E', 13: 'M13 Trans-1H-2E', 14: 'M14 Hybrid',
         15: 'M15 CodeBERT'}
P = ('diag', 'guard', 'holdout')


def boot_ci(y, a, b, idx, n=5000, seed=0):
    """95 % interval of F1(b) - F1(a) on idx, paired bootstrap of samples."""
    rng = np.random.default_rng(seed)
    t = y[idx].astype(bool)
    pa, pb = a[idx] > 0, b[idx] > 0
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, len(idx), len(idx))
        ts = t[s]
        fa = 200.0 * np.sum(pa[s] & ts) / max(pa[s].sum() + ts.sum(), 1)
        fb = 200.0 * np.sum(pb[s] & ts) / max(pb[s].sum() + ts.sum(), 1)
        d[i] = fb - fa
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--bundles', default=ROOT)
    ap.add_argument('--min-models', type=int, default=6)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(args.out, 'laser_rows.csv'))))
    for r in rows:
        for p in P:
            r[p + '_f1'], r[p + '_mcc'] = float(r[p + '_f1']), float(r[p + '_mcc'])
        r['model'] = int(r['model'])
    by = defaultdict(list)
    for r in rows:
        by[r['model']].append(r)
    orig = {m: next(r for r in rs if r['param'] == 'original') for m, rs in by.items()}

    def delta(r):
        o = orig[r['model']]
        return {p: (r[p + '_f1'] - o[p + '_f1'], 100 * (r[p + '_mcc'] - o[p + '_mcc'])) for p in P}

    # ---- per model
    per = {}
    for m, rs in sorted(by.items()):
        cand = [r for r in rs if r['param'] != 'original' and delta(r)['diag'][1] >= 0 and delta(r)['diag'][0] > 0]
        best = max(cand, key=lambda r: delta(r)['diag'][0]) if cand else None
        res = {'model': m, 'original': {p: [orig[m][p + '_f1'], orig[m][p + '_mcc']] for p in P}}
        if best is None:
            res.update(choice='original', reason='no diag gain')
        else:
            d = delta(best)
            ok = d['guard'][0] >= 0 and d['guard'][1] >= 0
            res.update(best_on_diag='%s|%s' % (best['param'], best['rho']), type=best['type'],
                       d_diag=d['diag'], d_guard=d['guard'], d_holdout=d['holdout'],
                       choice='%s|%s' % (best['param'], best['rho']) if ok else 'original',
                       reason='guard ok' if ok else 'guard falls')
            mf = os.path.join(args.out, 'margins_M%d.npz' % m)
            if ok and os.path.exists(mf):
                z = np.load(mf)
                b = np.load(os.path.join(args.bundles, BUNDLE_OF.get(m, 'bundle'), 'bundle.npz'))
                h = np.where(b['part'] == 2)[0]
                res['holdout_f1_ci95'] = boot_ci(b['y'].astype(int), z['original'].astype(np.float64),
                                                 z[res['choice']].astype(np.float64), h)
        per[m] = res

    # ---- one recipe for all models
    grp = defaultdict(list)
    for r in rows:
        if r['param'] != 'original':
            grp[(r['type'], r['branch'], r['pos'], r['rho'])].append(r)
    recipes = []
    for k, rs in grp.items():
        ms = {}
        for r in rs:            # several matrices of one key in one model (M14): take the mean
            ms.setdefault(r['model'], []).append(delta(r))
        if len(ms) < args.min_models:
            continue
        mean = {p: tuple(np.mean([[d[p][0], d[p][1]] for v in ms.values() for d in v], axis=0)) for p in P}
        recipes.append({'key': list(k), 'n_models': len(ms), **{'d_' + p: [round(x, 3) for x in mean[p]] for p in P}})
    recipes.sort(key=lambda x: -x['d_diag'][0])
    good = [x for x in recipes if x['d_diag'][1] >= 0]

    json.dump({'per_model': per, 'recipes_top20': recipes[:20], 'recipe_choice': good[0] if good else None},
              open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)

    # ---- report
    L = ['# LASER sweep (low-rank copy of one weight matrix)', '',
         'Choice on diag only, guard confirms, holdout is reported. F1 change in points, MCC change x 100.', '',
         '## Per model', '',
         '| Model | Best on diag | Diag dF1 / dMCC | Guard dF1 / dMCC | Used | Holdout F1 orig -> new | Holdout dF1 [95 %] |',
         '|---|---|---|---|---|---|---|']
    for m, r in per.items():
        o = r['original']['holdout'][0]
        if 'best_on_diag' not in r:
            L.append('| %s | none | | | original | %.2f | |' % (LABEL[m], o))
            continue
        used = r['choice'] != 'original'
        new = o + r['d_holdout'][0] if used else o
        ci = r.get('holdout_f1_ci95')
        L.append('| %s | `%s` | %+.2f / %+.2f | %+.2f / %+.2f | %s | %.2f -> %.2f | %s |' % (
            LABEL[m], r['best_on_diag'], *r['d_diag'], *r['d_guard'], 'yes' if used else 'no', o, new,
            '%+.2f [%+.2f, %+.2f]' % (new - o, *ci) if ci else ''))
    L += ['', '## One recipe for all models (top 10 by mean diag F1 change, at least %d models)' % args.min_models, '',
          '| Type | Branch | Layer | rho | Models | Diag dF1 / dMCC | Guard dF1 / dMCC | Holdout dF1 / dMCC |',
          '|---|---|---|---|---|---|---|---|']
    for x in recipes[:10]:
        L.append('| %s | %s | %s | %s | %d | %+.2f / %+.2f | %+.2f / %+.2f | %+.2f / %+.2f |' % (
            *x['key'], x['n_models'], *x['d_diag'], *x['d_guard'], *x['d_holdout']))
    L += ['', 'Chosen recipe (best diag F1 with diag MCC not falling): %s' % (good[0]['key'] if good else 'none')]
    open(os.path.join(args.out, 'REPORT.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
