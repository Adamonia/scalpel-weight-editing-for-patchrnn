"""Step 2a: forward the sealed holdout (part == 2) once per model and save the margins. No metrics here."""
import time
from fwd import *

OUTD = os.path.dirname(HERE) + '/out/'
dec = json.load(open(ROOT + '/experiments/round3/out/decisions.json'))
R2_PLAIN = {4: 'E1', 6: 'E1', 7: 'E1', 8: 'E1', 12: 'E1', 13: 'E1', 14: 'E1c'}   # round 2 edit that can be rebuilt
for no in ALL_MODELS:
    t0 = time.time()
    if no == 15:                                       # not applicable: original kept, cached head output
        b = np.load(BUNDLE + '_codebert/bundle.npz')
        h = np.load(BUNDLE + '_codebert/base_feats_M15.npz')['head']
        rows = np.where(b['part'] == HOLDOUT)[0]
        m0 = margin_of(h[rows])
        np.savez(OUTD + 'holdout_margins_M15.npz', y=b['y'][rows], E0=m0, R=m0, rows=rows)
        print(no, 'cached', len(rows), flush=True)
        continue
    applied = dec[str(no)]['applied'] != 'original'
    want = (('R',) if applied else ()) + ((R2_PLAIN[no],) if no in R2_PLAIN else ())
    s = get_session(no)
    rows = np.where(s.part == HOLDOUT)[0]
    o = edited_margins(no, rows, want)
    assert len(rows) == 3804
    save = {'y': o['y'], 'E0': o['E0'], 'R': o['R'] if applied else o['E0'], 'rows': rows}
    if no in R2_PLAIN:
        save['E1'] = o[R2_PLAIN[no]]
    np.savez(OUTD + 'holdout_margins_M%d.npz' % no, **save)
    print(no, 'applied' if applied else 'kept', 'hit', o['hit'], 'cache gap', o['cached_head_gap'], '%.0fs' % (time.time() - t0), flush=True)
