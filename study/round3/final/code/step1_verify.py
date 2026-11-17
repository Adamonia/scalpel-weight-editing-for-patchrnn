"""Step 1: re-run the diag+guard numbers of the frozen recipe and compare with decisions.json.
Only rows with part < 2 are forwarded. Holdout is not touched here."""
import hashlib, time
from fwd import *

FIN = os.path.dirname(HERE)   # final/
dec = json.load(open(ROOT + '/study/round3/out/decisions.json'))
res = {}
for no in ALL_MODELS:
    d = dec[str(no)]
    if d['applied'] == 'original':
        res[str(no)] = {'applied': 'original', 'note': 'kept original: no edit to re-run'}
        continue
    t0 = time.time()
    s0 = get_session(no)
    dev = np.where(s0.part < HOLDOUT)[0]
    check_idx(s0.part, dev)
    o = edited_margins(no, dev, ('R',))
    part, y = o['part'], o['y']
    saved = np.load(ROOT + '/study/round3/out/margins_M%d.npz' % no)
    gap_margin = float(np.abs(o['R'] - saved['E3_k3'][dev]).max())
    gap_e0 = float(np.abs(o['E0'] - saved['E0'][dev]).max())
    r = {'applied': d['applied'], 'seconds': round(time.time() - t0, 1), 'n_hit': o['hit']['R'],
         'max_abs_margin_gap_R_vs_saved_r3': gap_margin, 'max_abs_margin_gap_E0_vs_saved_r3': gap_e0}
    for pn, key in (('diag', 'diag_f1_after'), ('guard', 'guard_f1_after')):
        idx = np.where(part == {'diag': 0, 'guard': 1}[pn])[0]
        full_idx = dev[idx]
        f_new = f1_pct(part, idx, y, o['R'])
        f_old = f1_pct(part, idx, y, o['E0'])
        r[pn] = {'f1_recipe': f_new, 'f1_orig': f_old,
                 'gap_vs_decisions_recipe': abs(f_new - d['recipe_if_applied_%s_f1' % pn]),
                 'gap_vs_decisions_original': abs(f_old - d['%s_f1_before' % pn])}
    res[str(no)] = r
    print(no, json.dumps(r), flush=True)
gaps = [v[p][g] for v in res.values() if 'diag' in v for p in ('diag', 'guard') for g in ('gap_vs_decisions_recipe', 'gap_vs_decisions_original')]
# M6 and M15 keep original: compare their stored F1 before with the cached head
res['_summary'] = {'max_abs_f1_gap_vs_decisions': max(gaps), 'max_abs_margin_gap_vs_saved': max(v['max_abs_margin_gap_R_vs_saved_r3'] for v in res.values() if 'diag' in v)}
json.dump(res, open(FIN + '/out/step1_verify.json', 'w'), indent=1)
print(res['_summary'])
