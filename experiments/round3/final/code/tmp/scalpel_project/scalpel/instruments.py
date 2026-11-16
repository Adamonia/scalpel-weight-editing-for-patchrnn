"""Read-only instruments. Each function returns a small JSON-serializable dict.

Sign convention: margin = logit(security) - logit(non-security).
A positive contribution pushes the prediction to "security patch".
"""
import re

import numpy as np
import torch

from .engine import PARTS, STAGE_INPUTS, ScalpelError, confusion_metrics

BRANCHES = ('twin_b', 'twin_a', 'msg')


def _margin(logits):
    return logits[:, 1] - logits[:, 0]


def _r(x, nd=3):
    return round(float(x), nd)


def _compile(regex):
    try:
        return re.compile(regex)
    except re.error as exc:
        raise ScalpelError('Bad regular expression: %s' % exc)


def match_ids(s, table, regex=None, tokens=None, fullmatch=False):
    """Token ids of a vocabulary that match a regex or a token list."""
    if tokens is not None:
        return np.array([s.token_id(table, t) for t in tokens], dtype=np.int64)
    rx = _compile(regex)
    fn = rx.fullmatch if fullmatch else rx.search
    return np.array([i for i, t in enumerate(s.vocab[table]) if i != s.pad_id and fn(t)], dtype=np.int64)


def sample_stats(s):
    """Per-sample counts that several instruments use. Cached in the session object."""
    if getattr(s, '_stats', None) is None:
        msg = s.b['msg'][:, :s.win_msg]
        tb = s.b['tok_b'][:, :s.win_code]
        ta = s.b['tok_a'][:, :s.win_code]
        untr_m = (~s.trained['msg'][msg]) & (msg != s.pad_id)
        untr_c = ((~s.trained['code'][tb]) & (tb != s.pad_id)).sum(1) + ((~s.trained['code'][ta]) & (ta != s.pad_id)).sum(1)
        s._stats = {'untr_msg': untr_m.sum(1), 'untr_code': untr_c,
                    'len_m': s.b['len_m'], 'len_b': s.b['len_b'], 'len_a': s.b['len_a']}
    return s._stats


def _select(s, part, where):
    """Rows of a part that satisfy the 'where' filter."""
    rows = s.rows_of(part)
    if not where:
        return rows
    st = sample_stats(s)
    keep = np.ones(len(rows), dtype=bool)
    known = {'msg_len', 'len_b', 'len_a', 'code_len', 'untr_msg_min', 'untr_code_min', 'label', 'pred',
             'msg_has', 'msg_regex', 'code_has', 'code_regex', 'msg_lacks_regex', 'ids'}
    bad = set(where) - known
    if bad:
        raise ScalpelError('Unknown filter keys %s. Known keys: %s' % (sorted(bad), sorted(known)))
    for key, arr in (('msg_len', st['len_m']), ('len_b', st['len_b']), ('len_a', st['len_a']),
                     ('code_len', np.maximum(st['len_b'], st['len_a']))):
        if key in where:
            lo, hi = where[key]
            keep &= (arr[rows] >= lo) & (arr[rows] <= hi)
    if 'untr_msg_min' in where:
        keep &= st['untr_msg'][rows] >= where['untr_msg_min']
    if 'untr_code_min' in where:
        keep &= st['untr_code'][rows] >= where['untr_code_min']
    if 'label' in where:
        keep &= s.y[rows] == int(where['label'])
    if 'ids' in where:
        keep &= np.isin(rows, np.asarray(where['ids'], dtype=np.int64))
    if 'pred' in where:
        s.refresh(rows)
        keep &= s.feats['head'][rows].argmax(1) == int(where['pred'])
    msg = s.b['msg'][rows][:, :s.win_msg]
    if 'msg_has' in where:
        keep &= np.isin(msg, match_ids(s, 'msg', tokens=where['msg_has'])).any(1)
    if 'msg_regex' in where:
        keep &= np.isin(msg, match_ids(s, 'msg', regex=where['msg_regex'])).any(1)
    if 'msg_lacks_regex' in where:
        keep &= ~np.isin(msg, match_ids(s, 'msg', regex=where['msg_lacks_regex'])).any(1)
    if 'code_has' in where or 'code_regex' in where:
        ids = (match_ids(s, 'code', tokens=where['code_has']) if 'code_has' in where
               else match_ids(s, 'code', regex=where['code_regex']))
        code = np.concatenate((s.b['tok_b'][rows][:, :s.win_code], s.b['tok_a'][rows][:, :s.win_code]), 1)
        keep &= np.isin(code, ids).any(1)
    return rows[keep]


# ------------------------------------------------------------------ overview
def overview(s):
    """Model structure, tensors, split sizes and current metrics."""
    tensors = []
    for name, p in s.params.items():
        st = s.model.stage_of(name)
        changed = name in s.committed_delta or name in s.working_touched
        tensors.append('%s %s -> %s%s' % (name, list(p.shape), '+'.join(st) if st else 'UNUSED',
                                          ' [edited]' if changed else ''))
    stats = sample_stats(s)
    return {
        'model': 'Model%d %s (%s)' % (s.model_no, s.label, type(s.model).__name__),
        'architecture': s.model.DESCRIPTION,
        'tensors': tensors,
        'input_window': {'code_tokens': s.win_code, 'msg_tokens': s.win_msg},
        'parts': {k: int((s.part == v).sum()) for k, v in PARTS.items()},
        'metrics': {'diag': s.metrics('diag'), 'guard': s.metrics('guard')},
        'vocab': {t: {'rows': len(s.vocab[t]), 'untrained_rows': int((~s.trained[t]).sum())} for t in ('msg', 'code')},
        'empty_message_samples_diag': int((stats['len_m'][s.rows_of('diag')] == 0).sum()),
        'state': {'commits': len(s.commits), 'pending_edits': len(s.pending),
                  'evaluations_used': s.n_evaluations, 'limits': s.limits},
    }


def metrics(s, part='diag'):
    return {'part': part, 'current': s.metrics(part), 'committed': s.metrics(part, committed=True),
            'baseline': confusion_metrics(s.y[s.rows_of(part)], s.base_pred[s.rows_of(part)])}


def _bucket_table(s, rows, values, edges, labels):
    pred = s.feats['head'][rows].argmax(1)
    y = s.y[rows]
    out = []
    for (lo, hi), lab in zip(edges, labels):
        m = (values >= lo) & (values <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        fn = int(((pred == 0) & (y == 1) & m).sum())
        fp = int(((pred == 1) & (y == 0) & m).sum())
        out.append({'bucket': lab, 'n': n, 'n_pos': int((y[m] == 1).sum()), 'FN': fn, 'FP': fp,
                    'err_pct': _r(100.0 * (fn + fp) / n, 1)})
    return out


def error_census(s, part='diag'):
    """Error rates of the current weights by simple input properties."""
    rows = s.rows_of(part)
    s.refresh(rows)
    st = sample_stats(s)
    big = 10 ** 9
    code_len = np.maximum(st['len_b'], st['len_a'])[rows]
    w = s.win_code
    out = {
        'part': part, 'overall': confusion_metrics(s.y[rows], s.feats['head'][rows].argmax(1)),
        'by_message_length': _bucket_table(s, rows, st['len_m'][rows],
                                           [(0, 0), (1, 5), (6, 20), (21, 60), (61, s.win_msg), (s.win_msg + 1, big)],
                                           ['0 (empty)', '1-5', '6-20', '21-60', '61-%d' % s.win_msg, '>%d (cut)' % s.win_msg]),
        'by_code_length': _bucket_table(s, rows, code_len, [(0, 50), (51, 200), (201, w), (w + 1, big)],
                                        ['<=50', '51-200', '201-%d' % w, '>%d (cut)' % w]),
        'by_before_version': _bucket_table(s, rows, st['len_b'][rows], [(0, 0), (1, big)], ['empty', 'not empty']),
        'by_untrained_msg_tokens': _bucket_table(s, rows, st['untr_msg'][rows], [(0, 0), (1, 1), (2, 3), (4, big)],
                                                 ['0', '1', '2-3', '>=4']),
        'by_untrained_code_tokens': _bucket_table(s, rows, st['untr_code'][rows], [(0, 0), (1, 2), (3, big)],
                                                  ['0', '1-2', '>=3']),
    }
    return out


def slice_metrics(s, where, part='diag'):
    """Metrics of the samples that satisfy a filter, and of the other samples."""
    rows_all = s.rows_of(part)
    s.refresh(rows_all)
    rows = _select(s, part, where)
    rest = np.setdiff1d(rows_all, rows)
    out = {'part': part, 'where': where}
    for name, r in (('slice', rows), ('rest', rest)):
        m = confusion_metrics(s.y[r], s.feats['head'][r].argmax(1))
        m['err_pct'] = _r(100.0 - m['acc'], 2) if m['n'] else None
        m['mean_margin'] = _r(_margin(s.feats['head'][r]).mean()) if m['n'] else None
        out[name] = m
    return out


def _msg_tokens(s, row, mark=True):
    ids = s.b['msg'][row][:s.win_msg]
    ids = ids[ids != s.pad_id]
    return [s.vocab['msg'][i] + ('' if (not mark or s.trained['msg'][i]) else '(?)') for i in ids]


def list_errors(s, kind='all', sort='confident', limit=15, offset=0, where=None):
    """Misclassified diag samples. kind: FN, FP or all. sort: confident or borderline."""
    rows = _select(s, 'diag', where)
    s.refresh(rows)
    margin = _margin(s.feats['head'][rows])
    pred = (margin > 0).astype(int)
    y = s.y[rows]
    err = pred != y
    if kind == 'FN':
        err &= y == 1
    elif kind == 'FP':
        err &= y == 0
    elif kind != 'all':
        raise ScalpelError('kind must be FN, FP or all.')
    idx = np.where(err)[0]
    order = np.argsort(-np.abs(margin[idx])) if sort == 'confident' else np.argsort(np.abs(margin[idx]))
    idx = idx[order]
    st = sample_stats(s)
    items = []
    for i in idx[offset:offset + limit]:
        r = int(rows[i])
        items.append({'id': r, 'label': int(y[i]), 'margin': _r(margin[i], 2),
                      'len_m': int(st['len_m'][r]), 'len_b': int(st['len_b'][r]), 'len_a': int(st['len_a'][r]),
                      'untr_m': int(st['untr_msg'][r]), 'untr_c': int(st['untr_code'][r]),
                      'msg': ' '.join(_msg_tokens(s, r)[:14])})
    return {'kind': kind, 'n_errors': int(len(idx)), 'n_selected': int(len(rows)), 'offset': offset, 'items': items}


def _branch_contrib(s, rows, ref_rows=None):
    """Mean-ablation contribution of each branch to the margin of each row."""
    ref_rows = s.rows_of('diag') if ref_rows is None else ref_rows
    s.refresh(ref_rows)
    s.refresh(rows)
    f = {b: s.feats[b][rows] for b in BRANCHES}
    mean = {b: s.feats[b][ref_rows].mean(0, keepdims=True) for b in BRANCHES}
    full = _margin(s.head_from(f['twin_b'], f['twin_a'], f['msg']))
    out = {}
    for b in BRANCHES:
        g = dict(f)
        g[b] = np.repeat(mean[b], len(rows), axis=0)
        out[b] = full - _margin(s.head_from(g['twin_b'], g['twin_a'], g['msg']))
    base = _margin(s.head_from(mean['twin_b'], mean['twin_a'], mean['msg']))[0]
    return full, out, float(base)


def branch_attribution(s, part='diag', where=None):
    """Mean contribution of each branch to the margin, for each outcome group."""
    rows = _select(s, part, where)
    full, contrib, base = _branch_contrib(s, rows)
    y = s.y[rows]
    pred = (full > 0).astype(int)
    groups = {'TP': (pred == 1) & (y == 1), 'TN': (pred == 0) & (y == 0),
              'FP': (pred == 1) & (y == 0), 'FN': (pred == 0) & (y == 1)}
    out = {'note': 'contribution = margin - margin with this branch set to its diag mean feature',
           'margin_at_mean_features': _r(base), 'groups': {}}
    for name, m in groups.items():
        if m.sum() == 0:
            continue
        out['groups'][name] = dict({'n': int(m.sum()), 'mean_margin': _r(full[m].mean())},
                                   **{b: _r(contrib[b][m].mean()) for b in BRANCHES})
    return out


def show_sample(s, id, max_msg_chars=500, max_diff_chars=1200, code_tokens=40):
    """Raw text, model input and prediction of one diag sample."""
    row = s.visible(id)
    raw = s.raw(row)
    rows = np.array([row])
    full, contrib, _ = _branch_contrib(s, rows)
    st = sample_stats(s)

    def code(key):
        ids = s.b[key][row][:s.win_code]
        ids = ids[ids != s.pad_id][:code_tokens]
        return ' '.join(s.vocab['code'][i] + ('' if s.trained['code'][i] else '(?)') for i in ids)
    return {
        'id': row, 'label': int(s.y[row]), 'pred': int(full[0] > 0), 'margin': _r(full[0], 2),
        'p_security': _r(1.0 / (1.0 + np.exp(-full[0])), 3),
        'branch_contribution': {b: _r(contrib[b][0], 2) for b in BRANCHES},
        'lengths': {'msg': int(st['len_m'][row]), 'before': int(st['len_b'][row]), 'after': int(st['len_a'][row]),
                    'diff_chars': raw['diff_chars'], 'hunks': raw['n_hunks']},
        'untrained_tokens': {'msg': int(st['untr_msg'][row]), 'code': int(st['untr_code'][row])},
        'raw_message': raw['msg'][:max_msg_chars],
        'message_tokens': ' '.join(_msg_tokens(s, row)[:80]),
        'raw_diff': raw['diff'][:max_diff_chars],
        'before_tokens': code('tok_b'), 'after_tokens': code('tok_a'),
        'note': '(?) marks a token whose embedding row was never trained',
    }


def token_attribution(s, id, branch='msg', top=10, max_positions=256):
    """Occlusion: replace one input position with padding and read the margin change."""
    row = s.visible(id)
    if branch not in BRANCHES:
        raise ScalpelError('branch must be one of %s.' % (BRANCHES,))
    rows = np.array([row])
    s.refresh(rows)
    keys = STAGE_INPUTS[branch]
    tok_key = keys[0]
    win = s.win_msg if branch == 'msg' else s.win_code
    tok = s.b[tok_key][row]
    pos = np.where(tok[:win] != s.pad_id)[0][:max_positions]
    if len(pos) == 0:
        return {'id': row, 'branch': branch, 'note': 'This branch input is empty.'}
    override = {k: np.repeat(s.b[k][row][None, :], len(pos), axis=0).copy() for k in keys}
    for j, p in enumerate(pos):
        for k in keys:
            override[k][j, p] = s.pad_id if k == tok_key else 0
    feat = s._compute_stage(branch, None, override=override, n_rows=len(pos))
    f = {b: np.repeat(s.feats[b][rows], len(pos), axis=0) for b in BRANCHES}
    f[branch] = feat
    orig = _margin(s.feats['head'][rows])[0]
    delta = orig - _margin(s.head_from(f['twin_b'], f['twin_a'], f['msg']))
    table = 'msg' if branch == 'msg' else 'code'
    names = [s.vocab[table][tok[p]] + ('' if s.trained[table][tok[p]] else '(?)') for p in pos]
    order = np.argsort(-delta)
    def item(j):
        return {'pos': int(pos[j]), 'token': names[j], 'effect': _r(delta[j], 3)}
    return {'id': row, 'branch': branch, 'label': int(s.y[row]), 'margin': _r(orig, 2),
            'positions_tested': int(len(pos)), 'sum_of_effects': _r(delta.sum(), 2),
            'pushes_to_security': [item(j) for j in order[:top] if delta[j] > 0],
            'pushes_to_non_security': [item(j) for j in order[::-1][:top] if delta[j] < 0]}


def _diag_doc_count(s, table, ids):
    rows = s.rows_of('diag')
    if table == 'msg':
        arr = s.b['msg'][rows][:, :s.win_msg]
    else:
        arr = np.concatenate((s.b['tok_b'][rows][:, :s.win_code], s.b['tok_a'][rows][:, :s.win_code]), 1)
    out = np.zeros(len(ids), dtype=np.int64)
    present = np.isin(ids, np.unique(arr))
    for j in np.where(present)[0]:
        out[j] = int((arr == ids[j]).any(1).sum())
    return out


def vocab_search(s, table, regex, limit=30, only_untrained=False, fullmatch=False, sort='train'):
    """Vocabulary rows whose token matches a regex, with training counts."""
    ids = match_ids(s, table, regex=regex, fullmatch=fullmatch)
    if only_untrained:
        ids = ids[~s.trained[table][ids]]
    pos, neg = s.df[table]
    n_match = len(ids)
    n_untr = int((~s.trained[table][ids]).sum())
    if sort == 'train':
        ids = ids[np.argsort(-(pos[ids] + neg[ids]), kind='stable')]
    ids = ids[:limit]
    w = s.tensor(s.table_param[table])
    dcount = _diag_doc_count(s, table, ids)
    items = [{'token': s.vocab[table][i], 'train_pos': int(pos[i]), 'train_neg': int(neg[i]),
              'diag_docs': int(dcount[j]), 'row_norm': _r(w[i].norm().item(), 2)} for j, i in enumerate(ids)]
    tot_pos, tot_neg = int(pos[match_ids(s, table, regex=regex, fullmatch=fullmatch)].sum()), \
        int(neg[match_ids(s, table, regex=regex, fullmatch=fullmatch)].sum())
    return {'table': table, 'regex': regex, 'n_match': n_match, 'n_untrained_match': n_untr,
            'train_docs_sum': {'pos': tot_pos, 'neg': tot_neg,
                               'train_set': {'pos': int(s.b['n_train_pos']), 'neg': int(s.b['n_train_neg'])}},
            'items': items}


_SHAPES = {
    'msg': [('cve_id', r'^cve-?\d'), ('number', r'^[\d.,:x-]*\d[\d.,:x-]*$'), ('hex_hash', r'^(?=.*\d)(?=.*[a-f])[0-9a-f]{7,}$'),
            ('path_or_dotted', r'[/\\.]'), ('has_digit', r'\d'), ('snake_or_dash', r'[_-]'), ('alpha', r'^[a-z]+$')],
    'code': [('VARn', r'^VAR\d+$'), ('FUNCn', r'^FUNC\d+$'), ('number', r'^\d+$')],
}


def _shape(table, token):
    for name, rx in _SHAPES[table]:
        if re.search(rx, token):
            return name
    return 'other'


def untrained_summary(s, table='msg', examples=8):
    """Embedding rows that training never changed, and where they occur in the diag part."""
    rows = s.rows_of('diag')
    s.refresh(rows)
    st = sample_stats(s)
    if table == 'msg':
        arr = s.b['msg'][rows][:, :s.win_msg]
        per_sample = st['untr_msg'][rows]
    else:
        arr = np.concatenate((s.b['tok_b'][rows][:, :s.win_code], s.b['tok_a'][rows][:, :s.win_code]), 1)
        per_sample = st['untr_code'][rows]
    ids, counts = np.unique(arr, return_counts=True)
    untr = (~s.trained[table][ids]) & (ids != s.pad_id)
    ids_u, cnt_u = ids[untr], counts[untr]
    shapes = {}
    for i, c in zip(ids_u, cnt_u):
        e = shapes.setdefault(_shape(table, s.vocab[table][i]), {'rows': 0, 'occurrences': 0, 'examples': []})
        e['rows'] += 1
        e['occurrences'] += int(c)
        if len(e['examples']) < examples:
            e['examples'].append(s.vocab[table][i][:40])
    pred = s.feats['head'][rows].argmax(1)
    y = s.y[rows]
    return {
        'table': table, 'rows_total': len(s.vocab[table]), 'rows_untrained': int((~s.trained[table]).sum()),
        'untrained_rows_in_diag_inputs': int(len(ids_u)),
        'token_occurrences_in_diag': {'all': int(counts[ids != s.pad_id].sum()), 'untrained': int(cnt_u.sum())},
        'diag_samples_with_untrained': int((per_sample > 0).sum()), 'diag_samples': int(len(rows)),
        'err_pct_with': _r(100.0 * (pred != y)[per_sample > 0].mean(), 2) if (per_sample > 0).any() else None,
        'err_pct_without': _r(100.0 * (pred != y)[per_sample == 0].mean(), 2),
        'shape_classes': shapes,
        'note': 'Untrained rows keep the random N(0,1) start values: the token was never in a training input.',
    }


def token_effect(s, table, tokens=None, regex=None, part='diag', max_tokens=15, max_rows=300):
    """In-context effect of a token: margin change when all its occurrences become padding."""
    ids = match_ids(s, table, regex=regex, tokens=tokens)
    rows_all = s.rows_of(part)
    s.refresh(rows_all)
    out = []
    branches = ('msg',) if table == 'msg' else ('twin_b', 'twin_a')
    rng = np.random.RandomState(0)
    scored = []
    for i in ids:
        hits = s.rows_with_tokens(table, [i])
        m = np.zeros(s.n, dtype=bool)
        for b in branches:
            m |= hits[b]
        rows = rows_all[m[rows_all]]
        if len(rows):
            scored.append((len(rows), int(i), rows))
    scored.sort(key=lambda t: -t[0])
    for n_rows, i, rows in scored[:max_tokens]:
        if len(rows) > max_rows:
            rows = np.sort(rng.choice(rows, max_rows, replace=False))
        f = {b: s.feats[b][rows].copy() for b in BRANCHES}
        for b in branches:
            keys = STAGE_INPUTS[b]
            override = {k: s.b[k][rows].copy() for k in keys}
            mask = override[keys[0]] == i
            for k in keys:
                override[k][mask] = s.pad_id if k == keys[0] else 0
            f[b] = s._compute_stage(b, None, override=override, n_rows=len(rows))
        delta = _margin(s.feats['head'][rows]) - _margin(s.head_from(f['twin_b'], f['twin_a'], f['msg']))
        y = s.y[rows]
        pos, neg = s.df[table]
        out.append({'token': s.vocab[table][i], 'n_docs': int(n_rows), 'n_used': int(len(rows)),
                    'mean_effect': _r(delta.mean()),
                    'mean_effect_on_pos': _r(delta[y == 1].mean()) if (y == 1).any() else None,
                    'mean_effect_on_neg': _r(delta[y == 0].mean()) if (y == 0).any() else None,
                    'part_pos_share': _r((y == 1).mean(), 2),
                    'train_pos': int(pos[i]), 'train_neg': int(neg[i])})
    return {'table': table, 'part': part, 'n_matching_tokens': int(len(ids)), 'tokens_present': len(scored),
            'note': 'effect > 0: the token pushes to security. Compare with the label share.', 'items': out}


FREQ_BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 10), (11, 100), (101, 1000), (1001, 10 ** 9)]


def effect_by_frequency(s, table='msg', per_bucket=200, part='diag', seed=0):
    """Occlusion effect of single token positions, grouped by the training count of the token.

    help = effect * (+1 for security, -1 for non-security). A bucket with a large mean |effect|
    and a mean help near zero adds noise to the decision.
    """
    rows = s.rows_of(part)
    s.refresh(rows)
    pos, neg = s.df[table]
    count = pos + neg
    rng = np.random.RandomState(seed)
    branches = ('msg',) if table == 'msg' else ('twin_b', 'twin_a')
    win = s.win_msg if table == 'msg' else s.win_code
    out = []
    occ = {}
    for b in branches:
        tok = s.b[STAGE_INPUTS[b][0]][rows][:, :win]
        r_idx, p_idx = np.nonzero(tok != s.pad_id)
        occ[b] = (r_idx, p_idx, count[tok[r_idx, p_idx]])
    total = sum(len(v[0]) for v in occ.values())
    for lo, hi in FREQ_BUCKETS:
        cand = []
        for b in branches:
            r_idx, p_idx, c = occ[b]
            m = (c >= lo) & (c <= hi)
            cand.extend((b, int(r), int(p)) for r, p in zip(r_idx[m], p_idx[m]))
        n_occ = len(cand)
        if n_occ == 0:
            continue
        if n_occ > per_bucket:
            cand = [cand[i] for i in rng.choice(n_occ, per_bucket, replace=False)]
        effects, signs = [], []
        for b in branches:
            sel = [(r, p) for bb, r, p in cand if bb == b]
            if not sel:
                continue
            rr = rows[[r for r, _ in sel]]
            keys = STAGE_INPUTS[b]
            override = {k: s.b[k][rr].copy() for k in keys}
            for j, (_, p) in enumerate(sel):
                for k in keys:
                    override[k][j, p] = s.pad_id if k == keys[0] else 0
            f = {x: s.feats[x][rr] for x in BRANCHES}
            f[b] = s._compute_stage(b, None, override=override, n_rows=len(rr))
            effects.append(_margin(s.feats['head'][rr]) - _margin(s.head_from(f['twin_b'], f['twin_a'], f['msg'])))
            signs.append(np.where(s.y[rr] == 1, 1.0, -1.0))
        e = np.concatenate(effects)
        h = e * np.concatenate(signs)
        label = '%d' % lo if lo == hi else ('%d-%d' % (lo, hi) if hi < 10 ** 9 else '>%d' % (lo - 1))
        out.append({'train_count': label, 'occurrences': n_occ, 'share_pct': _r(100.0 * n_occ / total, 2),
                    'sampled': int(len(e)), 'mean_abs_effect': _r(np.abs(e).mean()), 'mean_help': _r(h.mean()),
                    'share_helpful_pct': _r(100.0 * (h > 0).mean(), 1),
                    'help_stderr': _r(h.std() / max(1, len(h)) ** 0.5)})
    return {'table': table, 'part': part, 'occlusion': 'one position becomes padding',
            'note': 'mean_help > 0: the tokens of the bucket move the margin to the correct class on average',
            'buckets': out}


def solo_probe(s, tokens=None, regex=None, limit=25):
    """Margin of a message that has only one token, minus the margin of an empty message.

    The code branches use their mean diag features.
    """
    ids = match_ids(s, 'msg', regex=regex, tokens=tokens)[:limit]
    if len(ids) == 0:
        return {'items': []}
    rows = s.rows_of('diag')
    s.refresh(rows)
    mb = s.feats['twin_b'][rows].mean(0, keepdims=True)
    ma = s.feats['twin_a'][rows].mean(0, keepdims=True)
    msg = np.full((len(ids) + 1, s.b['msg'].shape[1]), s.pad_id, dtype=s.b['msg'].dtype)
    msg[1:, 0] = ids
    fm = s._compute_stage('msg', None, override={'msg': msg}, n_rows=len(msg))
    marg = _margin(s.head_from(np.repeat(mb, len(msg), 0), np.repeat(ma, len(msg), 0), fm))
    pos, neg = s.df['msg']
    items = [{'token': s.vocab['msg'][i], 'solo_effect': _r(marg[j + 1] - marg[0]),
              'train_pos': int(pos[i]), 'train_neg': int(neg[i])} for j, i in enumerate(ids)]
    return {'empty_message_margin': _r(marg[0]), 'items': items}


def row_neighbors(s, table, token, k=8, trained_only=True):
    """Nearest embedding rows by cosine similarity."""
    i = s.token_id(table, token)
    w = s.tensor(s.table_param[table])
    if s.is_tf and table == 'code':
        w = w[:, :-5]
    wn = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = (wn @ wn[i]).numpy()
    if trained_only:
        sim[~s.trained[table]] = -2
    sim[i] = -2
    order = np.argsort(-sim)[:k]
    pos, neg = s.df[table]
    return {'token': token, 'trained': bool(s.trained[table][i]),
            'neighbors': [{'token': s.vocab[table][j], 'cos': _r(sim[j]), 'train_pos': int(pos[j]),
                           'train_neg': int(neg[j])} for j in order]}


def unit_report(s, stage, top=10, part='diag'):
    """First-order contribution of each feature unit of a branch to the correct class."""
    if stage not in BRANCHES:
        raise ScalpelError('stage must be one of %s.' % (BRANCHES,))
    rows = s.rows_of(part)
    s.refresh(rows)
    acts = {b: torch.from_numpy(s.feats[b][rows].copy()).requires_grad_(b == stage) for b in BRANCHES}
    with torch.enable_grad():
        logits = s.model.run_stage('head', None, acts)
        (logits[:, 1] - logits[:, 0]).sum().backward()
    g = acts[stage].grad.numpy()
    f = s.feats[stage][rows]
    y = s.y[rows]
    sign = np.where(y == 1, 1.0, -1.0)[:, None]
    contrib = g * (f - f.mean(0, keepdims=True)) * sign        # > 0 helps the correct class.
    pred = s.feats['head'][rows].argmax(1)
    err = pred != y
    helps = contrib.mean(0)
    on_err = contrib[err].mean(0)
    std = f.std(0)
    def item(u):
        return {'unit': int(u), 'mean_help': _r(helps[u], 4), 'mean_help_on_errors': _r(on_err[u], 4),
                'act_mean_pos': _r(f[y == 1, u].mean()), 'act_mean_neg': _r(f[y == 0, u].mean()),
                'act_std': _r(std[u]), 'mean_abs_grad': _r(np.abs(g[:, u]).mean(), 4)}
    order = np.argsort(helps)
    return {'stage': stage, 'n_units': int(f.shape[1]), 'n_errors': int(err.sum()),
            'note': 'help = d(margin)/d(unit) * (activation - mean) * (+1 for security, -1 for non-security)',
            'sum_help_all_units': _r(helps.sum()), 'n_units_with_negative_help': int((helps < 0).sum()),
            'near_constant_units': [int(u) for u in np.where(std < 1e-4)[0][:20]],
            'act_std_quantiles_0_10_50_90_100': [_r(q, 4) for q in np.percentile(std, [0, 10, 50, 90, 100])],
            'units_with_std_below_0.05': int((std < 0.05).sum()),
            'most_harmful': [item(u) for u in order[:top]],
            'most_helpful': [item(u) for u in order[::-1][:top]]}


def bias_scan(s, deltas=None):
    """Metrics of the diag part for shifts of the output bias (no weight change).

    A shift d adds d to the margin of each sample. It is the same as an edit of the
    positive-class entry of the output bias (tensor name in 'output_bias').
    """
    rows = s.rows_of('diag')
    s.refresh(rows)
    margin = _margin(s.feats['head'][rows])
    y = s.y[rows]
    if deltas is None:
        sd = float(margin.std())
        deltas = [round(k * sd, 2) for k in (-0.6, -0.4, -0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25, 0.4, 0.6)]
    table = []
    for d in deltas:
        m = confusion_metrics(y, (margin + d) > 0)
        table.append({'delta': d, 'acc': m['acc'], 'f1': m['f1'], 'mcc': m['mcc'], 'fn': m['fn'], 'fp': m['fp']})
    base = s.base_feats['head'][rows]
    comm = s.c_feats['head'][rows]
    means = {'current': _r(margin.mean(), 4), 'committed': _r(_margin(comm).mean(), 4),
             'baseline': _r(_margin(base).mean(), 4)}
    return {'output_bias': s.model.OUT_BIAS, 'index_of_security_class': 1, 'margin_std': _r(margin.std()),
            'mean_margin': means,
            'margin_quantiles_5_25_50_75_95': [_r(q, 2) for q in np.percentile(margin, [5, 25, 50, 75, 95])],
            'part': 'diag', 'table': table}


def parse_index(index, ndim):
    """Index spec: list with one entry for each dimension. int, [start, stop], list of ints, or null."""
    if index is None:
        return tuple([slice(None)] * ndim)
    if not isinstance(index, list) or len(index) > ndim:
        raise ScalpelError('index must be a list with at most %d entries.' % ndim)
    out = []
    for item in index:
        if item is None:
            out.append(slice(None))
        elif isinstance(item, int):
            out.append(item)
        elif isinstance(item, dict) and 'range' in item:
            out.append(slice(int(item['range'][0]), int(item['range'][1])))
        elif isinstance(item, list) and all(isinstance(v, int) for v in item):
            out.append(item)
        else:
            raise ScalpelError('Bad index entry %r. Use an int, a list of ints, {"range":[a,b]} or null.' % (item,))
    return tuple(out)


def weight_read(s, name, index=None, max_values=40):
    """Statistics of one tensor and the values of a small slice."""
    w = s.tensor(name)
    base = s.base[name]
    out = {'name': name, 'shape': list(w.shape), 'stages': list(s.model.stage_of(name)) or 'UNUSED',
           'mean': _r(w.mean().item(), 5), 'std': _r(w.std().item(), 5) if w.numel() > 1 else 0.0,
           'min': _r(w.min().item(), 4), 'max': _r(w.max().item(), 4), 'l2': _r(w.norm().item(), 3),
           'elements_changed_vs_original': int((w != base).sum().item()),
           'l2_change_vs_original': _r((w - base).norm().item(), 4)}
    if index is not None:
        sl = w[parse_index(index, w.dim())]
        flat = sl.reshape(-1)
        out['slice_shape'] = list(sl.shape)
        out['slice_values'] = [_r(v, 4) for v in flat[:max_values].tolist()]
        if flat.numel() > max_values:
            out['slice_note'] = 'first %d of %d values' % (max_values, flat.numel())
    return out
