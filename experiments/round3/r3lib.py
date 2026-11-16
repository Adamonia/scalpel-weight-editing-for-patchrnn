"""Helpers for Round 3. Holdout (part 2) is sealed: no metric is ever computed on it."""
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/scalpel_project')
import numpy as np
import torch

torch.set_num_threads(2)
from scalpel.engine import Session

W = '/home/claude/work'
ARCHIVE = '/mnt/user-data/uploads/patchrnn-archive/model-archive'
OUT = HERE + '/out'
BUNDLES = {6: W + '/bundle_bert', 7: W + '/bundle_roberta', 15: W + '/bundle_codebert'}
ALL_MODELS = [0, 2, 3, 4, 5, 8, 12, 13, 6, 7, 14, 15]
TRANSFORMERS6 = [3, 4, 5, 8, 12, 13]
KS = [1, 3, 10, 30, 100]
HOLDOUT = 2


# ---------------------------------------------------------------- holdout guard
def check_idx(part, idx):
    """Raise if any index belongs to the sealed holdout part."""
    if np.any(np.asarray(part)[np.asarray(idx)] == HOLDOUT):
        raise RuntimeError('holdout index used for a metric')


def part_idx(part, name):
    if name not in ('diag', 'guard'):
        raise RuntimeError('only diag and guard may be used, got %r' % name)
    return np.where(part == {'diag': 0, 'guard': 1}[name])[0]


# ---------------------------------------------------------------- metrics (all go through check_idx)
def f1_pct(part, idx, y, margin):
    check_idx(part, idx)
    p = margin[idx] > 0
    t = y[idx].astype(bool)
    tp = np.sum(p & t)
    return 100.0 * 2 * tp / max(2 * tp + np.sum(p & ~t) + np.sum(~p & t), 1)


def auc_pct(part, idx, y, score):
    check_idx(part, idx)
    s, t = score[idx], y[idx].astype(bool)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = t.sum()
    return 100.0 * (ranks[t].sum() - n1 * (n1 + 1) / 2.0) / (n1 * (len(t) - n1))


def ap_pct(part, idx, y, score):
    """Average precision = mean of precision at each positive, ties broken by order (stable)."""
    check_idx(part, idx)
    s, t = score[idx], y[idx].astype(bool)
    o = np.argsort(-s, kind='stable')
    t = t[o]
    prec = np.cumsum(t) / (np.arange(len(t)) + 1.0)
    return 100.0 * prec[t].mean()


def boot_weights(n, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.ones(n) / n, size=n_boot).astype(np.float64)


def conf_cols(part, idx, y, margin):
    check_idx(part, idx)
    p = margin[idx] > 0
    t = y[idx].astype(bool)
    return np.stack([p & t, p & ~t, ~p & t]).astype(np.float64)       # tp, fp, fn


def boot_f1(w, c):
    """F1 (percent) of every bootstrap resample. w: (B, n) weights, c: (3, n) tp/fp/fn."""
    t = w @ c.T
    return 200.0 * t[:, 0] / np.maximum(2 * t[:, 0] + t[:, 1] + t[:, 2], 1)


# ---------------------------------------------------------------- model access
def get_session(no):
    d = '%s/tmp/S_M%d' % (HERE, no)
    os.makedirs(d, exist_ok=True)
    cfg = d + '/config.json'
    if not os.path.exists(cfg):
        json.dump({'archive_dir': ARCHIVE, 'bundle_dir': BUNDLES.get(no, W + '/bundle'), 'model_no': no,
                   'batch_size': 16, 'limits': {}, 'created': 0}, open(cfg, 'w'))
    return Session(d)


def table_specs(s, no):
    """List of (parameter name, window, training count per row) of the message embedding tables."""
    b = s.b
    df = (b['df_msg_pos'] + b['df_msg_neg']).astype(np.int64)
    if no in (0, 2):
        return [('embedMsg.weight', 200, df)]
    if no == 14:
        dtf = (b['df_msg_tf_pos'] + b['df_msg_tf_neg']).astype(np.int64)
        return [('cnn.embedMsg.weight', 200, df), ('trans.embedMsg.weight', 199, dtf), ('lstm.embedMsg.weight', 200, df)]
    if no == 15:
        return []
    dtf = (b['df_msg_tf_pos'] + b['df_msg_tf_neg']).astype(np.int64)
    return [('embedMsg.weight', 199, dtf)]


def new_table(W0, count, pad, edit, k=None):
    """New weight matrix (torch, float32). The pad row is never changed.

    E1: rows with count 0 := 0.   E2: rows with count 0 := pad row.
    E3: row := a*row, a = c/(c+k).   E4: row := a*row + (1-a)*pad row.
    """
    c = torch.from_numpy(count).double()
    pad_row = W0[pad].clone()
    W1 = W0.clone()
    if edit in ('E1', 'E2'):
        m = torch.from_numpy(count == 0)
        m[pad] = False
        W1[m] = 0.0 if edit == 'E1' else pad_row
    else:
        a = (c / (c + k)).float().unsqueeze(1)
        a[pad] = 1.0
        W1 = a * W0 if edit == 'E3' else a * W0 + (1 - a) * pad_row
    W1[pad] = pad_row
    return W1


def msg_features(model, msg, rows, bs):
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            out.append(model.run_stage('msg', {'msg': torch.from_numpy(np.ascontiguousarray(msg[r]).astype(np.int64))}, None).numpy().copy())
    return np.concatenate(out) if out else None


def margin_of(logits):
    return (logits[:, 1] - logits[:, 0]).astype(np.float64)


def recenter_shift(part, orig, new):
    """Label-free shift: mean orig margin on diag minus mean new margin on diag."""
    d = part_idx(part, 'diag')
    check_idx(part, d)
    return float(orig[d].mean() - new[d].mean())


def variant_name(edit, k=None):
    return edit if k is None else '%s_k%d' % (edit, k)


def all_variants():
    return [('E1', None), ('E2', None)] + [(e, k) for e in ('E3', 'E4') for k in KS]
