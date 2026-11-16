# Metrics for every variant on all 7,608 and on the 3,804 holdout, plus paired bootstrap of the F1 difference.
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for common.py
from common import *
VAR = [('orig', 'none'), ('orig', 'twin'), ('space', 'none'), ('space', 'twin'), ('space', 'twin+mean'),
       ('pertok', 'none'), ('pertok', 'twin')]
res = {}
for m in (6, 7):
    b = load_bundle(m); y = b['y'].astype(int); hold = b['part'] == 2
    L = {}
    for fmt, ed in VAR:
        p = '%s/logits_M%d_%s_%s.npy' % (CACHE, m, fmt, ed.replace('+', '_'))
        if os.path.exists(p): L[(fmt, ed)] = np.load(p)
    base = L[('orig', 'none')]
    res[m] = {'n_all': len(y), 'n_hold': int(hold.sum()), 'rows': []}
    for k, lg in L.items():
        row = {'format': k[0], 'edit': k[1]}
        for name, sel in (('all', np.ones(len(y), bool)), ('hold', hold)):
            row[name] = metrics(y[sel], lg[sel])
            row[name]['dF1'], row[name]['lo'], row[name]['hi'], row[name]['p'] = boot_f1_diff(y[sel], base[sel], lg[sel], 2000, seed=1)
            row[name]['flips'] = int((lg[sel].argmax(1) != base[sel].argmax(1)).sum())
            row[name]['pos_rate'] = 100 * float((lg[sel].argmax(1) == 1).mean())
        res[m]['rows'].append(row)
json.dump(res, open(OUT + '/results.json', 'w'), indent=1)
for m in res:
    print('M%d' % m)
    for r in res[m]['rows']:
        for part in ('all', 'hold'):
            x = r[part]
            print('  %-7s %-10s %-4s n=%d F1 %.2f acc %.2f P %.2f R %.2f AUC %.2f | dF1 %+.2f [%+.2f,%+.2f] p=%.3f flips %d posrate %.1f' % (
                r['format'], r['edit'], part, x['n'], x['f1'], x['acc'], x['prec'], x['rec'], x['auc'], x['dF1'], x['lo'], x['hi'], x['p'], x['flips'], x['pos_rate']))
