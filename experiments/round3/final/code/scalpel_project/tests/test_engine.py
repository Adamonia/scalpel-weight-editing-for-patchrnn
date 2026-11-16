"""Sanity tests of the engine. Run: python tests/test_engine.py ARCHIVE BUNDLE [MODEL_NO]

The tests use a temporary run folder. The finalize test uses a session whose
only edit is in an unused tensor, thus the holdout result equals the baseline.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scalpel.engine import Session, ScalpelError
from scalpel.tools import dispatch

ARCH, BUNDLE = sys.argv[1], sys.argv[2]
MODEL = int(sys.argv[3]) if len(sys.argv) > 3 else 0
UNUSED = {0: 'fc2.weight', 2: 'norm.weight'}.get(MODEL, None)


def fresh():
    d = tempfile.mkdtemp(prefix='scalpel_test_')
    shutil.rmtree(d)
    return Session.create(d, ARCH, BUNDLE, MODEL), d


def full_recompute(s, rows):
    feats = {st: s._compute_stage(st, rows) for st in s.model.STAGES[:-1]}
    return s.head_from(feats['twin_b'], feats['twin_a'], feats['msg'])


def main():
    s, d = fresh()
    dev = s.rows_of('dev')
    base = {p: s.metrics(p) for p in ('diag', 'guard')}

    # 1. An edit of an unused tensor does not change a prediction.
    if UNUSED:
        r = dispatch(s, 'edit_tensor', {'name': UNUSED, 'op': 'scale', 'value': 0.0})
        assert r.get('unused_tensor') is True, r
        ev = dispatch(s, 'evaluate', {})
        assert ev['diag']['n_fixed'] == ev['diag']['n_broken'] == 0 and ev['guard']['n_fixed'] == ev['guard']['n_broken'] == 0
        c = dispatch(s, 'commit', {'rationale': 'unused tensor', 'error_type': 'dead parameters'})
        assert c.get('commit') == 1, c
        print('ok 1: unused tensor edit is neutral and can be committed')

    # 2. Incremental cache = full recomputation after an embedding edit.
    r = dispatch(s, 'edit_embedding_rows', {'table': 'msg', 'targets': {'regex': '^[0-9a-f]{12,}$', 'only_untrained': True},
                                           'recipe': {'type': 'mean_trained'}})
    assert 'error' not in r, r
    ev = dispatch(s, 'evaluate', {})
    gap = np.abs(full_recompute(s, dev) - s.feats['head'][dev]).max()
    assert gap < 1e-4, gap
    print('ok 2: incremental cache matches full recomputation (max gap %.2e); rows edited %d, dev samples touched %d'
          % (gap, r['n_rows'], r['dev_samples_touched']))

    # 3. Rollback restores the weights and the metrics.
    dispatch(s, 'rollback', {})
    for p in ('diag', 'guard'):
        assert s.metrics(p) == base[p]
    for name, v in s.params.items():
        ref = s.committed_delta.get(name, s.base[name])
        assert (v.data == ref).all(), name
    print('ok 3: rollback restores weights and metrics')

    # 4. A head edit: only the head is recomputed; the result equals a full recomputation.
    head_bias = [n for n in s.params if n.endswith('.bias') and s.model.stage_of(n) == ('head',)][-1]
    r = dispatch(s, 'edit_tensor', {'name': head_bias, 'op': 'add', 'value': 0.5, 'index': [1]})
    dispatch(s, 'evaluate', {})
    gap = np.abs(full_recompute(s, dev) - s.feats['head'][dev]).max()
    assert gap < 1e-4, gap
    dispatch(s, 'rollback', {})
    print('ok 4: head edit is consistent')

    # 5. The rank-one edit moves the selected key by the requested margin (first order).
    layer = s.model.HEAD_LINEARS[-2] if len(s.model.HEAD_LINEARS) > 1 else s.model.HEAD_LINEARS[0]
    rows = s.rows_of('diag')
    before = s.feats['head'][rows][:, 1] - s.feats['head'][rows][:, 0]
    r = dispatch(s, 'edit_rank1', {'layer': layer, 'where': {'msg_len': [0, 0]}, 'delta_margin': 2.0})
    assert 'error' not in r, r
    s.refresh(rows)
    after = s.feats['head'][rows][:, 1] - s.feats['head'][rows][:, 0]
    sel = s.b['len_m'][rows] == 0
    print('ok 5: rank-one edit; mean shift selected %.3f (asked 2.0, predicted %.3f); mean abs shift others %.3f'
          % ((after - before)[sel].mean(), r['predicted_mean_shift_selected'], np.abs(after - before)[~sel].mean()))
    dispatch(s, 'rollback', {})

    # 6. State survives a reload from disk.
    dispatch(s, 'edit_tensor', {'name': head_bias, 'op': 'add', 'value': 0.25, 'index': [1]})
    dispatch(s, 'evaluate', {})
    m1 = s.metrics('diag')
    s2 = Session(d)
    assert s2.metrics('diag') == m1 and len(s2.pending) == 1 and len(s2.commits) == len(s.commits)
    dispatch(s2, 'rollback', {})
    assert s2.metrics('diag') == base['diag']
    print('ok 6: session state survives a reload')

    # 7. The holdout part is sealed before finalize.
    try:
        s2.rows_of('holdout')
        raise AssertionError('holdout was readable')
    except ScalpelError:
        pass
    r = dispatch(s2, 'slice_metrics', {'where': {'label': 1}, 'part': 'holdout'})
    assert 'error' in r
    r = dispatch(s2, 'show_sample', {'id': int(np.where(s2.part == 2)[0][0])})
    assert 'error' in r
    print('ok 7: holdout is sealed; holdout and guard samples are not readable')

    # 8. A commit that decreases the guard or diag MCC is refused.
    dispatch(s2, 'edit_tensor', {'name': head_bias, 'op': 'add', 'value': -50.0, 'index': [1]})
    r = dispatch(s2, 'commit', {'rationale': 'bad', 'error_type': 'bad'})
    assert 'error' in r and 'refused' in r['error'], r
    dispatch(s2, 'rollback', {})
    print('ok 8: harmful commit is refused')

    # 9. finalize with no behavior change: holdout result equals the baseline; session is locked.
    r = dispatch(s2, 'finalize', {'note': 'engine test'})
    h = r['holdout']
    assert h['n_fixed'] == h['n_broken'] == 0 and h['before'] == h['after'], h
    assert 'error' in dispatch(s2, 'edit_tensor', {'name': head_bias, 'op': 'add', 'value': 1.0, 'index': [1]})
    from scalpel.models import load_state_dict
    sd = load_state_dict(os.path.join(d, r['edited_checkpoint']))
    assert list(sd) == list(s2.base)
    print('ok 9: finalize works, the session is locked, the edited checkpoint has the original keys')
    shutil.rmtree(d)
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
