"""Model14: test of the round-1 recipes of each sub-model inside the hybrid (one sub-model table at a time)."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
import numpy as np
from scalpel.engine import Session
from scalpel.tools import dispatch
s = Session(os.path.join(RUNS, 'M14'))
A = 'claude-fable-5-2'
call = lambda tool, args=None: dispatch(s, tool, args or {}, actor=A)
def attempt(tag, rationale, error_type):
    ev = call('evaluate')
    print('  %s eval: diag %d/%d mcc %.4f->%.4f | guard %d/%d mcc %.4f->%.4f' % (tag, ev['diag']['n_fixed'], ev['diag']['n_broken'], ev['diag']['before']['mcc'], ev['diag']['after']['mcc'], ev['guard']['n_fixed'], ev['guard']['n_broken'], ev['guard']['before']['mcc'], ev['guard']['after']['mcc']))
    r = call('commit', {'rationale': rationale, 'error_type': error_type})
    if 'error' in r:
        print('   refused'); call('rollback'); return False
    print('   COMMIT', r['commit']); return True
count = s.df['msg'][0] + s.df['msg'][1]
untr = [int(i) for i in np.where(count == 0)[0]]
rare = [int(i) for i in np.where(count <= 10)[0]]
for sub, ids, name in (('trans', untr, 'untrained'), ('cnn', rare, 'rare<=10'), ('lstm', untr, 'untrained'), ('cnn', untr, 'untrained'), ('trans', rare, 'rare<=10')):
    call('edit_tensor', {'name': sub + '.embedMsg.weight', 'op': 'scale', 'value': 0.0, 'index': [ids]})
    rc = call('edit_recenter')
    attempt('%s %s rows -> 0 (recenter %+.3f)' % (sub, name, rc['bias_shift']),
            'Test of a round-1 recipe inside the hybrid: the %s message embedding rows of the sub-model %s are set to zero (%d rows). The output bias is shifted to keep the mean margin of the diag part.' % (name, sub, len(ids)),
            'rare-token noise in one sub-model of the hybrid')
for sub_layer in (('lstm.fc1',),):
    call('edit_feature_gain', {'stage': 'twin_b', 'units': list(range(428, 492)), 'factor': 0.0})
    call('edit_feature_gain', {'stage': 'twin_a', 'units': list(range(428, 492)), 'factor': 0.0})
    attempt('lstm code features x0', 'Test of the Model0 recipe inside the hybrid: the gain of the LSTM code features is set to zero (mean-preserving).', 'inert code branches of the LSTM sub-model')
for lo, hi, nm in ((0, 300, 'cnn'), (300, 428, 'trans')):
    call('edit_feature_gain', {'stage': 'twin_b', 'units': list(range(lo, hi)), 'factor': 0.0})
    call('edit_feature_gain', {'stage': 'twin_a', 'units': list(range(lo, hi)), 'factor': 0.0})
    attempt('%s code features x0' % nm, 'Ablation: the gain of the %s code features is set to zero (mean-preserving).' % nm, 'inert code branches of one sub-model')
s.save()
