"""Re-test of the three edit recipes on one model. Each recipe is evaluated and committed only if the guard rule accepts it.

Usage: protocol.py MODEL steps   (steps: bias,rare,code)
The recipes were found by the agent in the sessions of Model0, Model2 and Model3.
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))

import sys, json, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
from scalpel.engine import Session
from scalpel.tools import dispatch
no = int(sys.argv[1]); steps = sys.argv[2].split(',')
s = Session(os.path.join(RUNS, 'M%d' % no))
A = 'claude-fable-5-2'
def call(tool, args=None):
    r = dispatch(s, tool, args or {}, actor=A)
    return r
def show(tag, r):
    if 'error' in r: print('  %s: %s' % (tag, r['error'][:150])); return False
    print('  %s COMMIT %d: diag %d fixed / %d broken (acc %.2f->%.2f, mcc %.4f->%.4f); guard %d fixed / %d broken (acc %.2f->%.2f, mcc %.4f->%.4f)' % (
        tag, r['commit'], r['diag']['n_fixed'], r['diag']['n_broken'], r['diag']['before']['acc'], r['diag']['after']['acc'], r['diag']['before']['mcc'], r['diag']['after']['mcc'],
        r['guard']['n_fixed'], r['guard']['n_broken'], r['guard']['before']['acc'], r['guard']['after']['acc'], r['guard']['before']['mcc'], r['guard']['after']['mcc']))
    return True
def attempt(tag, rationale, error_type):
    ev = call('evaluate')
    print('  %s eval: diag %d/%d mcc %.4f->%.4f | guard %d/%d mcc %.4f->%.4f' % (tag, ev['diag']['n_fixed'], ev['diag']['n_broken'], ev['diag']['before']['mcc'], ev['diag']['after']['mcc'], ev['guard']['n_fixed'], ev['guard']['n_broken'], ev['guard']['before']['mcc'], ev['guard']['after']['mcc']))
    r = call('commit', {'rationale': rationale, 'error_type': error_type})
    ok = show(tag, r)
    if not ok: call('rollback')
    return ok
print('== Model%d %s' % (no, s.label))
if 'bias' in steps:
    d = -math.log(float(s.b['n_train_neg']) / float(s.b['n_train_pos']))
    m = call('metrics', {'part': 'diag'})['current']
    scan = call('bias_scan', {'deltas': [round(d * k, 3) for k in (2.0, 1.5, 1.0, 0.5, 0.0, -0.5)]})
    print('  bias scan: ' + ' | '.join('d=%+.2f mcc %.4f' % (t['delta'], t['mcc']) for t in scan['table']))
    call('edit_tensor', {'name': s.model.OUT_BIAS, 'op': 'add', 'value': round(d, 4), 'index': [1]})
    attempt('bias', 'Re-test of the class-weight prior hypothesis. Training used a class-weighted cross-entropy with weights 0.3283 and 0.6717, which shifts the margins by ln(0.6717/0.3283) = +0.716. On the diag part the model has %d FP and %d FN at margin 0. The edit subtracts 0.7159 from the security entry of the output bias. The value comes from the training class counts.' % (m['fp'], m['fn']),
            'class-weight prior shift left in the output bias (decision threshold too permissive)')
if 'rare' in steps:
    fr = call('effect_by_frequency', {'table': 'msg', 'per_bucket': 300})
    b = {x['train_count']: x for x in fr['buckets']}
    txt = ', '.join('%s: %.3f (SE %.3f)' % (k, b[k]['mean_help'], b[k]['help_stderr']) for k in ('0', '1', '2-3', '4-10', '11-100', '101-1000', '>1000') if k in b)
    print('  help by train count: ' + txt)
    call('edit_embedding_rows', {'table': 'msg', 'targets': {'regex': '.', 'only_untrained': False, 'max_train_count': 10}, 'recipe': {'type': 'zero'}})
    rc = call('edit_recenter'); print('  recenter shift %+.3f' % rc['bias_shift'])
    ok = attempt('rare<=10', 'Re-test of the rare-token noise hypothesis. Mean help of one message token for the correct class by training count: %s. Rows with a training count of 10 or less are set to zero. The output bias is then shifted so that the mean margin of the diag part stays the same (no label is used).' % txt,
                 'rare-token noise (memorized or untrained message embedding rows)')
    if not ok:
        call('edit_embedding_rows', {'table': 'msg', 'targets': {'regex': '.', 'only_untrained': True}, 'recipe': {'type': 'zero'}})
        rc = call('edit_recenter'); print('  recenter shift %+.3f' % rc['bias_shift'])
        attempt('untrained', 'Re-test of the untrained-row hypothesis after the wider rule (training count of 10 or less) was refused. Mean help by training count: %s. The rows that training never changed keep random start values; the edit sets them to zero. The output bias is then shifted so that the mean margin of the diag part stays the same (no label is used).' % txt,
                'untrained message embedding rows (random vectors act as noise)')
if 'code' in steps:
    ba = call('branch_attribution')
    g = ba['groups']
    txt = '; '.join('%s: twin_b %+.2f, twin_a %+.2f, msg %+.2f' % (k, v['twin_b'], v['twin_a'], v['msg']) for k, v in g.items())
    for st in ('twin_b', 'twin_a'): call('edit_branch_gain', {'stage': st, 'factor': 0.0})
    attempt('code x0', 'Re-test of the inert-code-branch hypothesis. Mean branch contributions to the margin: %s. The edit sets the gain of both code branches to zero (mean-preserving).' % txt,
            'inert code branches (the head does not use the code features)')
for p in ('diag', 'guard'):
    m = call('metrics', {'part': p}); print('  %s now: acc %.2f f1 %.2f mcc %.4f (baseline acc %.2f f1 %.2f mcc %.4f)' % (p, m['current']['acc'], m['current']['f1'], m['current']['mcc'], m['baseline']['acc'], m['baseline']['f1'], m['baseline']['mcc']))
s.save()
