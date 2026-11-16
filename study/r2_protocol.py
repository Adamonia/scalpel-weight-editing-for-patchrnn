"""Round 2: F1 as the acceptance rule. Three recipes, each one committed only if the F1 of diag and guard does not decrease.

Usage: r2_protocol.py MODEL steps   (steps: untrained,rare,thr)
  untrained : message rows that training never changed -> zero, then recenter (label-free)
  rare      : message rows with a training count of 10 or less -> zero, then recenter (label-free)
  thr       : shift of the output bias to the value with the best F1 on the diag part (grid of 0.1)
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
import numpy as np
from scalpel.engine import Session, confusion_metrics
from scalpel.tools import dispatch
no = int(sys.argv[1]); steps = sys.argv[2].split(',')
s = Session(os.path.join(RUNS, 'R2_M%d' % no)); A = 'claude-fable-5-2'
assert s.limits['guard_metric'] == 'f1'
call = lambda tool, args=None: dispatch(s, tool, args or {}, actor=A)
def attempt(tag, rationale, error_type):
    ev = call('evaluate')
    txt = ' | '.join('%s %d/%d f1 %.2f->%.2f acc %.2f->%.2f' % (p, ev[p]['n_fixed'], ev[p]['n_broken'], ev[p]['before']['f1'], ev[p]['after']['f1'], ev[p]['before']['acc'], ev[p]['after']['acc']) for p in ('diag', 'guard'))
    r = call('commit', {'rationale': rationale, 'error_type': error_type})
    ok = 'error' not in r
    print('  %-10s %s  %s' % (tag, 'ACCEPT' if ok else 'refuse', txt))
    if not ok: call('rollback')
    return ok
print('== R2 Model%d %s' % (no, s.label))
for step, kw, name in (('untrained', {'only_untrained': True}, 'untrained message embedding rows (random vectors act as noise)'),
                       ('rare', {'only_untrained': False, 'max_train_count': 10}, 'rare-token noise (message rows with a training count of 10 or less)')):
    if step in steps:
        call('edit_embedding_rows', {'table': 'msg', 'targets': dict(regex='.', **kw), 'recipe': {'type': 'zero'}})
        rc = call('edit_recenter')
        attempt(step, 'Round 2 (F1 rule). The %s message rows are set to zero. The output bias is then shifted by %+.3f so that the mean margin of the diag part stays the same (no label is used).' % (step, rc['bias_shift']), name)
if 'thr' in steps:
    grid = [round(x, 1) for x in np.arange(-1.5, 1.51, 0.1)]
    scan = call('bias_scan', {'deltas': grid})['table']
    best = max(scan, key=lambda t: (t['f1'], -abs(t['delta'])))
    base = [t for t in scan if abs(t['delta']) < 1e-9][0]
    print('  thr scan: best F1 on diag at shift %+.1f (f1 %.2f, at 0: %.2f)' % (best['delta'], best['f1'], base['f1']))
    if abs(best['delta']) > 1e-9:
        call('edit_tensor', {'name': s.model.OUT_BIAS, 'op': 'add', 'value': best['delta'], 'index': [1]})
        attempt('thr %+.1f' % best['delta'], 'Round 2 (F1 rule). The decision threshold is not at the F1 optimum. A scan of the output bias on the diag part (grid of 0.1) gives the best F1 at a shift of %+.1f. The edit adds this value to the security entry of the output bias.' % best['delta'],
                'decision threshold not at the F1 optimum')
for p in ('diag', 'guard'):
    m = call('metrics', {'part': p}); print('  %s: f1 %.2f -> %.2f, acc %.2f -> %.2f' % (p, m['baseline']['f1'], m['current']['f1'], m['baseline']['acc'], m['current']['acc']))
s.save()
