"""Helpers shared by train_seeds.py and analyse_seeds.py."""
import json
from pathlib import Path

import numpy as np
import torch

VAL_FRACTION = 0.10
VAL_SPLIT_SEED = 12345       # Fixed. All runs use the same validation rows.


def pick_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def load_data(path):
    z = np.load(Path(path), allow_pickle=False)
    return {k: z[k] for k in z.files}


def split_rows(train_idx, limit=0):
    """Return (fit_rows, val_rows). limit > 0 keeps only the first rows (smoke test)."""
    rows = np.asarray(train_idx)
    if limit:
        rows = rows[:limit]
    perm = np.random.RandomState(VAL_SPLIT_SEED).permutation(len(rows))
    n_val = int(round(VAL_FRACTION * len(rows)))
    # Keep the order of train_idx. The dataset itself is ordered by class,
    # so sorted rows would give batches of one class when shuffling is off.
    return rows[np.sort(perm[n_val:])], rows[np.sort(perm[:n_val])]


def fit_val_rows(data, select_on='val', limit=0):
    """Rows to fit on and rows for model selection.

    val : fit on 90 % of the training set, select on the other 10 % (fixed split seed).
    test: fit on the whole training set, select on the test set (replicates the original code, biased).
    """
    if select_on == 'test':
        test = data['test_idx'][:limit] if limit else data['test_idx']
        train = data['train_idx'][:limit] if limit else data['train_idx']
        return np.asarray(train), np.asarray(test)
    return split_rows(data['train_idx'], limit)


def make_batch(data, rows, device):
    """Tensors for the given sample rows."""
    out = {}
    for key in ('tok_b', 'typ_b', 'tok_a', 'typ_a', 'msg'):
        out[key] = torch.from_numpy(np.ascontiguousarray(data[key][rows]).astype(np.int64)).to(device)
    return out


def forward_batch(model, b):
    return model(b['tok_b'], b['typ_b'], b['tok_a'], b['typ_a'], b['msg'])


def f1_score(y, margin):
    """F1 in percent. Prediction is margin > 0. Positive class is 1."""
    p = margin > 0
    y = np.asarray(y).astype(bool)
    tp = np.sum(p & y)
    fp = np.sum(p & ~y)
    fn = np.sum(~p & y)
    return 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)


def auc_score(y, score):
    """AUC in percent by the rank statistic. Ties get the mean rank."""
    y = np.asarray(y).astype(bool)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt)
    ranks = (ends - (cnt - 1) / 2.0)[inv]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return 100.0 * (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def read_json(path):
    with open(path, encoding='utf-8') as fp:
        return json.load(fp)
