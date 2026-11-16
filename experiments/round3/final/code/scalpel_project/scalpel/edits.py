"""Edit primitives. Each primitive writes values directly into weight tensors.

No primitive uses gradient descent on the weights. The edits stay pending
until commit() accepts them or rollback() removes them.
"""
import re

import numpy as np
import torch
import torch.nn as nn

from .engine import ScalpelError
from .instruments import BRANCHES, _margin, _r, _select, match_ids, parse_index


def _target_ids(s, table, targets):
    if 'tokens' in targets:
        ids = match_ids(s, table, tokens=targets['tokens'])
    elif 'regex' in targets:
        ids = match_ids(s, table, regex=targets['regex'], fullmatch=bool(targets.get('fullmatch', False)))
        pos, neg = s.df[table]
        if targets.get('only_untrained', True):
            ids = ids[~s.trained[table][ids]]
        if targets.get('max_train_count') is not None:
            ids = ids[(pos[ids] + neg[ids]) <= int(targets['max_train_count'])]
    else:
        raise ScalpelError('targets needs "tokens" or "regex".')
    if len(ids) == 0:
        raise ScalpelError('No vocabulary row matches the targets.')
    return ids


def _recipe_vector(s, table, w, recipe):
    """The vector that a recipe writes into the target rows. None for in-place recipes."""
    kind = recipe.get('type')
    pos, neg = s.df[table]
    if kind == 'blend':
        srcs = recipe['sources']
        ids = [s.token_id(table, x['token']) for x in srcs]
        wt = torch.tensor([float(x.get('weight', 1.0)) for x in srcs])
        if recipe.get('normalize', True):
            wt = wt / wt.sum()
        return (w[ids] * wt[:, None]).sum(0)
    if kind == 'mean_of':
        ids = match_ids(s, table, regex=recipe['regex'], fullmatch=bool(recipe.get('fullmatch', False)))
        ids = ids[(pos[ids] + neg[ids]) >= int(recipe.get('min_train_count', 3))]
        if recipe.get('label') == 'pos':
            ids = ids[pos[ids] > neg[ids]]
        elif recipe.get('label') == 'neg':
            ids = ids[neg[ids] > pos[ids]]
        if len(ids) == 0:
            raise ScalpelError('No trained source row matches the recipe regex.')
        if recipe.get('weight_by_count', False):
            wt = torch.tensor((pos[ids] + neg[ids]).astype(np.float32))
            return (w[ids] * (wt / wt.sum())[:, None]).sum(0)
        return w[ids].mean(0)
    if kind == 'mean_trained':
        return w[torch.from_numpy(np.where(s.trained[table])[0])].mean(0)
    if kind == 'zero':
        return torch.zeros(w.shape[1])
    if kind == 'pad':
        return w[s.pad_id].clone()
    if kind == 'scale':
        return None
    raise ScalpelError('Unknown recipe type. Use blend, mean_of, mean_trained, zero, pad, scale, '
                       'compose or nearest_numeric.')


_STEMMER = None


def _stem(word):
    """Porter stem, as in the message preprocessing. No change if NLTK is not installed."""
    global _STEMMER
    if _STEMMER is None:
        try:
            from nltk.stem import PorterStemmer
            _STEMMER = PorterStemmer().stem
        except ImportError:
            _STEMMER = lambda w: w
    return _STEMMER(word)


def _per_target_vectors(s, table, w, ids, recipe):
    """Recipes that give a different vector to each target row.

    compose         : split the token text, stem the parts, average the rows of the trained parts.
    nearest_numeric : token = prefix + number. Copy the trained row with the same prefix and the nearest number.
    Returns (list of kept ids, tensor of vectors, example list).
    """
    kind = recipe['type']
    pos, neg = s.df[table]
    count = pos + neg
    keep, vecs, examples = [], [], []
    if kind == 'compose':
        splitter = re.compile(recipe.get('split', r'[_\-./:+=]+'))
        min_count = int(recipe.get('min_train_count', 5))
        min_parts = int(recipe.get('min_parts', 1))
        index = s._tok_index.get(table) or {t: i for i, t in enumerate(s.vocab[table])}
        s._tok_index[table] = index
        for i in ids:
            parts = [p for p in splitter.split(s.vocab[table][i]) if p]
            src = []
            for part in parts:
                for cand in (part, _stem(part)):
                    j = index.get(cand)
                    if j is not None and j != i and count[j] >= min_count:
                        src.append(j)
                        break
            if len(src) >= min_parts and len(parts) > 1:
                keep.append(int(i))
                vecs.append(w[src].mean(0))
                if len(examples) < 10:
                    examples.append('%s <- %s' % (s.vocab[table][i], '+'.join(s.vocab[table][j] for j in src)))
    elif kind == 'nearest_numeric':
        rx = re.compile(r'^(\D*)(\d+)$')
        k = int(recipe.get('k', 1))
        families = {}
        for j, tok in enumerate(s.vocab[table]):
            m = rx.match(tok)
            if m and s.trained[table][j] and count[j] >= int(recipe.get('min_train_count', 1)):
                families.setdefault(m.group(1), []).append((int(m.group(2)), j))
        fam = {pfx: (np.array([n for n, _ in v]), np.array([j for _, j in v])) for pfx, v in families.items()}
        for i in ids:
            m = rx.match(s.vocab[table][i])
            if not m or m.group(1) not in fam:
                continue
            nums, rows = fam[m.group(1)]
            order = np.argsort(np.abs(nums - int(m.group(2))), kind='stable')[:k]
            keep.append(int(i))
            vecs.append(w[rows[order]].mean(0))
            if len(examples) < 10:
                examples.append('%s <- %s' % (s.vocab[table][i], '+'.join(s.vocab[table][j] for j in rows[order])))
    if not keep:
        raise ScalpelError('The recipe found no source rows for the targets.')
    return np.array(keep, dtype=np.int64), torch.stack(vecs), examples


def edit_embedding_rows(s, table, targets, recipe, mix=1.0):
    """Write a recipe vector into embedding rows.

    table   : 'msg' or 'code'
    targets : {"tokens": [...]} or {"regex": "...", "only_untrained": true, "max_train_count": k}
    recipe  : {"type": "blend", "sources": [{"token": t, "weight": w}, ...]}
              {"type": "mean_of", "regex": "...", "min_train_count": 3, "label": "pos"|"neg"|null}
              {"type": "mean_trained"} | {"type": "zero"} | {"type": "pad"} | {"type": "scale", "factor": f}
              {"type": "compose", "split": regex, "min_train_count": 5}   (a vector for each target)
              {"type": "nearest_numeric", "k": 1}                         (a vector for each target)
    mix     : new_row = (1 - mix) * old_row + mix * recipe_vector
    """
    s.check_open()
    if not s.table_tensors.get(table):
        raise ScalpelError("table must be 'msg' or 'code', and the model must have this table.")
    if not 0.0 <= float(mix) <= 1.0:
        raise ScalpelError('mix must be in [0, 1].')
    names = s.table_tensors[table]
    ids = _target_ids(s, table, targets)
    n_targets = int(len(ids))
    examples = None
    l2 = 0.0
    for name in names:                      # One tensor, or one for each sub-model of a hybrid model.
        w = s.tensor(name)
        if recipe.get('type') in ('compose', 'nearest_numeric'):
            ids_k, vecs, examples = _per_target_vectors(s, table, w, ids, recipe)
            idx = torch.from_numpy(ids_k)
            old = w[idx].clone()
            w[idx] = (1.0 - float(mix)) * old + float(mix) * vecs
        else:
            ids_k = ids
            idx = torch.from_numpy(ids_k)
            old = w[idx].clone()
            vec = _recipe_vector(s, table, w, recipe)
            if vec is None:
                w[idx] = old * float(recipe['factor'])
            else:
                w[idx] = (1.0 - float(mix)) * old + float(mix) * vec[None, :]
        l2 += float(((w[idx] - old) ** 2).sum().item())
        s.mark_stale(name, token_ids=ids_k)
    ids = ids_k
    hits = s.rows_with_tokens(table, ids)
    dev = s.part < 2
    n_dev = int(np.any([h for h in hits.values()], axis=0)[dev].sum())
    rec = {'op': 'embedding_rows', 'tensor': names[0], 'table': table, 'targets': targets, 'recipe': recipe,
           'mix': float(mix), 'n_targets_matched': n_targets, 'n_rows': int(len(ids)),
           'examples': examples or [s.vocab[table][i] for i in ids[:10]],
           'l2_change': _r(l2 ** 0.5, 3), 'dev_samples_touched': n_dev}
    if len(names) > 1:
        rec['tensors'] = list(names)
    return s.record_edit(rec)


def edit_tensor(s, name, op, value, index=None):
    """Direct write into a tensor slice. op: set, add or scale.

    index: list with one entry for each dimension: int, list of ints, {"range": [a, b]} or null.
    """
    s.check_open()
    w = s.tensor(name)
    sl = parse_index(index, w.dim())
    old = w[sl].clone()
    val = torch.as_tensor(value, dtype=w.dtype)
    try:
        if op == 'set':
            w[sl] = val
        elif op == 'add':
            w[sl] = old + val
        elif op == 'scale':
            w[sl] = old * val
        else:
            raise ScalpelError('op must be set, add or scale.')
    except RuntimeError as exc:
        raise ScalpelError('The value does not fit the slice %s: %s' % (list(old.shape), exc))
    token_ids = None
    if any(name in names for names in s.table_tensors.values()) and index:
        first = index[0]
        if isinstance(first, int):
            token_ids = [first]
        elif isinstance(first, list):
            token_ids = first
    s.mark_stale(name, token_ids=token_ids)
    new = w[sl]
    rec = {'op': 'tensor', 'tensor': name, 'index': index, 'mode': op,
           'value': value if np.size(value) <= 16 else 'array%s' % (list(np.shape(value)),),
           'n_elements': int(new.numel()), 'l2_change': _r((new - old).norm().item(), 4),
           'unused_tensor': len(s.model.stage_of(name)) == 0}
    return s.record_edit(rec)


_GAIN_MAP = {
    'TwinRNN': {'twin_b': ('fc1', 0), 'twin_a': ('fc1', 64), 'msg': ('fc3', 64)},
    'TwinCNN': {'twin_b': ('fcB', 0), 'twin_a': ('fcA', 0), 'msg': ('fcM', 0)},
}


def _readers(s, stage):
    """Linear layers that read the features of one branch: (layer, first column, first unit, number of units)."""
    cls = type(s.model).__name__
    if cls in _GAIN_MAP:
        layer, offset = _GAIN_MAP[cls][stage]
        return ((layer, offset, 0, s.feats[stage].shape[1]),)
    return getattr(s.model, 'READERS', {}).get(stage)


def edit_feature_gain(s, stage, units, factor):
    """Scale the effect of branch feature units. The diag mean of each unit is kept.

    The edit changes the weight columns that read the units, and the bias.
    Available if Linear layers read the features (not for TwinTransformer: a LayerNorm reads them).
    """
    s.check_open()
    if stage not in BRANCHES:
        raise ScalpelError('stage must be one of %s.' % (BRANCHES,))
    readers = _readers(s, stage)
    if not readers:
        raise ScalpelError('Not available for the %s branch of %s: no Linear layer reads the features. '
                           'Use edit_branch_gain, or edit_tensor on the first head layer.' % (stage, type(s.model).__name__))
    rows = s.rows_of('diag')
    s.refresh(rows)
    mean = torch.from_numpy(s.feats[stage][rows].mean(0))
    l2 = 0.0
    names = []
    for layer, col0, unit0, n in readers:
        sel = [int(u) for u in units if unit0 <= int(u) < unit0 + n]
        if not sel:
            continue
        w = s.tensor(layer + '.weight')
        b = s.tensor(layer + '.bias')
        old_w, old_b = w.clone(), b.clone()
        for u in sel:
            col = col0 + u - unit0
            b += (1.0 - float(factor)) * w[:, col] * mean[u]
            w[:, col] *= float(factor)
        s.mark_stale(layer + '.weight')
        s.mark_stale(layer + '.bias')
        l2 += ((w - old_w).norm() + (b - old_b).norm()).item()
        names.append(layer + '.weight')
    rec = {'op': 'feature_gain', 'stage': stage, 'units': [int(u) for u in units] if len(units) <= 64 else 'all %d' % len(units),
           'factor': float(factor), 'tensor': names[0], 'l2_change': _r(l2, 4)}
    if len(names) > 1:
        rec['tensors'] = names
    return s.record_edit(rec)


def edit_branch_gain(s, stage, factor):
    """Scale the deviation of a whole branch feature vector from its diag mean.

    factor 0 removes the branch (its feature becomes the constant mean), factor 1 changes nothing.
    Models with Linear readers: the columns that read the branch are scaled (as edit_feature_gain).
    TwinTransformer : the last LayerNorm of the branch is rewritten: w' = factor*w,
                      b' = factor*b + (1-factor)*mean. This is exact.
    """
    s.check_open()
    if stage not in BRANCHES:
        raise ScalpelError('stage must be one of %s.' % (BRANCHES,))
    if _readers(s, stage):
        return edit_feature_gain(s, stage, list(range(s.feats[stage].shape[1])), factor)
    if not hasattr(s.model, 'last_norm'):
        raise ScalpelError('The %s branch of %s has no weights to scale.' % (stage, type(s.model).__name__))
    rows = s.rows_of('diag')
    s.refresh(rows)
    mean = torch.from_numpy(s.feats[stage][rows].mean(0))
    norm = s.model.last_norm(stage)
    w = s.tensor(norm + '.weight')
    b = s.tensor(norm + '.bias')
    old_w, old_b = w.clone(), b.clone()
    b.copy_(float(factor) * old_b + (1.0 - float(factor)) * mean)
    w.copy_(float(factor) * old_w)
    s.mark_stale(norm + '.weight')
    s.mark_stale(norm + '.bias')
    rec = {'op': 'branch_gain', 'stage': stage, 'factor': float(factor), 'tensor': norm + '.weight',
           'l2_change': _r(((w - old_w).norm() + (b - old_b).norm()).item(), 4)}
    return s.record_edit(rec)


def edit_recenter(s, target_mean=None):
    """Shift the output bias so that the mean margin of the diag part has a target value.

    Default target: the mean margin of the committed weights. The edit then removes the
    mean shift of the pending edits and keeps their other effects. No label is used.
    """
    s.check_open()
    rows = s.rows_of('diag')
    s.refresh(rows)
    current = float(_margin(s.feats['head'][rows]).mean())
    if target_mean is None:
        target_mean = float(_margin(s.c_feats['head'][rows]).mean())
    delta = float(target_mean) - current
    name = s.model.OUT_BIAS
    s.tensor(name)[1] += delta
    s.mark_stale(name)
    rec = {'op': 'recenter', 'tensor': name, 'index': [1], 'mean_margin_before': _r(current, 4),
           'mean_margin_target': _r(target_mean, 4), 'bias_shift': _r(delta, 4), 'l2_change': _r(abs(delta), 4)}
    return s.record_edit(rec)


def _layer_io(s, layer, rows):
    """Input of a head Linear layer and d(margin)/d(output), for the given rows."""
    mods = dict(s.model.named_modules())
    if layer not in s.model.HEAD_LINEARS or not isinstance(mods.get(layer), nn.Linear):
        raise ScalpelError('layer must be one of %s.' % (s.model.HEAD_LINEARS,))
    store = {}

    def hook(_mod, inp, out):
        store['key'] = inp[0].detach()
        out.retain_grad()
        store['out'] = out
    handle = mods[layer].register_forward_hook(hook)
    try:
        acts = {b: torch.from_numpy(s.feats[b][rows].copy()).requires_grad_(True) for b in BRANCHES}
        with torch.enable_grad():
            logits = s.model.run_stage('head', None, acts)
            (logits[:, 1] - logits[:, 0]).sum().backward()
    finally:
        handle.remove()
    return store['key'].numpy().astype(np.float64), store['out'].grad.numpy().astype(np.float64)


def edit_rank1(s, layer, where, delta_margin, ridge=0.05):
    """Closed-form rank-one edit of a head Linear layer (ROME-style associative edit).

    The mean layer input of the selected diag samples (the key k) gets a new output:
    W' k = W k + dv, where dv moves the margin of that key by delta_margin (first order).
    The update is W' = W + dv (C^-1 k)^T / (k^T C^-1 k). C is the second-moment matrix of the
    layer inputs on the diag part plus a ridge term. Thus the outputs for other keys change little.
    """
    s.check_open()
    rows_all = s.rows_of('diag')
    s.refresh(rows_all)
    rows = _select(s, 'diag', where)
    if len(rows) < 5:
        raise ScalpelError('The filter selects %d diag samples. A minimum of 5 is necessary.' % len(rows))
    keys, grads = _layer_io(s, layer, rows_all)
    sel = np.isin(rows_all, rows)
    k = keys[sel].mean(0)
    g = grads[sel].mean(0)
    c = keys.T @ keys / len(keys)
    c += float(ridge) * np.trace(c) / len(c) * np.eye(len(c))
    u = np.linalg.solve(c, k)
    dv = float(delta_margin) * g / (g @ g)
    dw = np.outer(dv, u) / (k @ u)
    w = s.tensor(layer + '.weight')
    old = w.clone()
    w += torch.from_numpy(dw).to(w.dtype)
    s.mark_stale(layer + '.weight')
    shift = (keys @ u) / (k @ u) * float(delta_margin)        # First-order margin shift of each diag row.
    rec = {'op': 'rank1', 'tensor': layer + '.weight', 'where': where, 'delta_margin': float(delta_margin),
           'ridge': float(ridge), 'n_key_samples': int(len(rows)),
           'predicted_mean_shift_selected': _r(shift[sel].mean(), 3),
           'predicted_mean_abs_shift_others': _r(np.abs(shift[~sel]).mean(), 3) if (~sel).any() else 0.0,
           'l2_change': _r((w - old).norm().item(), 4)}
    return s.record_edit(rec)
