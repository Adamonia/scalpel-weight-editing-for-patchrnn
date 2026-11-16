import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
import numpy as np
from scalpel.engine import Session
from scalpel.tools import dispatch
s = Session(os.path.join(RUNS, 'R2_M14')); A = 'claude-fable-5-2'
call = lambda tool, args=None: dispatch(s, tool, args or {}, actor=A)
count = s.df['msg'][0] + s.df['msg'][1]
untr = [int(i) for i in np.where(count == 0)[0]]
call('edit_tensor', {'name': 'cnn.embedMsg.weight', 'op': 'scale', 'value': 0.0, 'index': [untr]})
rc = call('edit_recenter')
ev = call('evaluate')
print(' | '.join('%s %d/%d f1 %.2f->%.2f' % (p, ev[p]['n_fixed'], ev[p]['n_broken'], ev[p]['before']['f1'], ev[p]['after']['f1']) for p in ('diag', 'guard')))
r = call('commit', {'rationale': 'Round 2 (F1 rule). In the hybrid only the CNN sub-model reads its untrained message rows as noise (max-pooling takes single large values). The %d untrained rows of cnn.embedMsg are set to zero; the output bias is shifted by %+.3f to keep the mean margin of the diag part.' % (len(untr), rc['bias_shift']), 'error_type': 'untrained message embedding rows in the CNN sub-model of the hybrid'})
print('ACCEPT' if 'error' not in r else 'refuse')
if 'error' in r: call('rollback')
s.save()
