import sys; sys.dont_write_bytecode = True
"""Shared helpers for the control experiments (word-level models only).

Nothing here writes to the archive, the bundles or runs/. A temporary Session
directory in ext/tmp is used only to load the model, vocabulary and counts.
"""
import copy
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/work/scalpel_project')
import numpy as np
import torch

from scalpel.engine import Session

ARCHIVE = '/mnt/user-data/uploads/patchrnn-archive/model-archive'
BUNDLE = '/home/claude/work/bundle'
EXT = '/home/claude/work/exth2'
OUT = EXT + '/out'
CACHE = '/home/claude/work/ext/out'   # read only: cached margins
MODELS = [0, 2, 3, 4, 5, 8, 12, 13]
PRIOR_SHIFT = 0.7159          # ln(20442/9991), subtracted from the security entry of the output bias
N_SEEDS = 20


def get_session(no):
    """Load a model through the package. The temp Session only reads data; nothing is edited."""
    d = '%s/tmp/S_M%d' % (EXT, no)
    if os.path.exists(d + '/config.json'):
        return Session(d)
    return Session.create(d, ARCHIVE, BUNDLE, no)


def margin(logits):
    return (logits[:, 1] - logits[:, 0]).astype(np.float64)


def msg_window(s):
    return s.b['msg'][:, :s.win_msg]


def run_msg_stage(model, s, rows, bs=128):
    """Message-branch features of the given test rows for a model object (a copy)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            batch = {'msg': torch.from_numpy(np.ascontiguousarray(s.b['msg'][r]))}
            out.append(model.run_stage('msg', batch, None).numpy().copy())
    return np.concatenate(out)


def variant_margin(s, model, base_msg_feat, new_rows_ids, base_logits=None):
    """Margin of a model copy whose embedMsg rows were edited.

    Only samples that contain an edited token in the model window are recomputed;
    the other samples keep the cached message feature (they cannot change).
    """
    w = msg_window(s)
    hit = np.where(np.isin(w, np.asarray(new_rows_ids)).any(1))[0]
    fm = base_msg_feat.copy()
    if len(hit):
        fm[hit] = run_msg_stage(model, s, hit)
    logits = s.head_from(s.base_feats['twin_b'], s.base_feats['twin_a'], fm)
    return margin(logits), len(hit)


def edited_copy(s, ids, vectors):
    """Deep copy of the model with embedMsg rows 'ids' set to 'vectors' (1 vector or one per row)."""
    m = copy.deepcopy(s.model)
    w = m.embedMsg.weight
    with torch.no_grad():
        w[torch.from_numpy(np.asarray(ids))] = torch.as_tensor(vectors, dtype=w.dtype)
    return m


def untrained_ids(s, max_count=None):
    """Same selection as edit_embedding_rows(regex='.'): vocab rows without the pad row.

    max_count None: rows never changed by training. Otherwise rows with train count <= max_count.
    """
    pos, neg = s.df['msg']
    cnt = pos + neg
    ids = np.array([i for i, t in enumerate(s.vocab['msg']) if i != s.pad_id and re.search('.', t)])
    if max_count is None:
        return ids[~s.trained['msg'][ids]]
    return ids[cnt[ids] <= max_count]


def trained_pool(s):
    ids = np.where(s.trained['msg'])[0]
    return ids[ids != s.pad_id]


def test_freq(s):
    """Occurrences of each message token id in the test messages (inside the model window)."""
    return np.bincount(msg_window(s).ravel(), minlength=len(s.vocab['msg']))


def head_input_features(s, fm):
    """Input of the LAST linear layer of the head, for the given message features."""
    name = s.model.HEAD_LINEARS[-1]
    mod = dict(s.model.named_modules())[name]
    store = []
    h = mod.register_forward_pre_hook(lambda m, inp: store.append(inp[0].detach().numpy().copy()))
    try:
        s.head_from(s.base_feats['twin_b'], s.base_feats['twin_a'], fm)
    finally:
        h.remove()
    return np.concatenate(store), mod.weight.detach().numpy().copy(), mod.bias.detach().numpy().copy()


# ---------------------------------------------------------------- helpers of this study (exth)
from sklearn.metrics import roc_auc_score

MODELS_H = [3, 4, 5, 8, 12, 13]


def f1_pct(y, m):
    """F1 (percent) of the prediction margin > 0. Positive class = 1."""
    p = m > 0
    y = y.astype(bool)
    tp = np.sum(p & y); fp = np.sum(p & ~y); fn = np.sum(~p & y)
    return 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)


def hit_rows(s, U):
    """Samples whose message window holds at least one untrained token."""
    return np.where(np.isin(msg_window(s), U).any(1))[0]


class Noiser:
    """Sets rows of the message embedding of a model, runs the message stage on the hit samples, restores."""

    def __init__(self, s):
        self.s = s
        self.U = untrained_ids(s)
        self.hit = hit_rows(s, self.U)
        self.W = s.model.embedMsg.weight
        self.orig_rows = self.W.data[torch.from_numpy(self.U)].clone()
        self.base_fm = s.base_feats['msg']
        self.orig_margin = margin(s.base_feats['head'])

    def margin_with(self, rows_value):
        """rows_value: tensor (len(U), d). Returns the raw margin of all samples (not recentered)."""
        ids = torch.from_numpy(self.U)
        with torch.no_grad():
            self.W.data[ids] = rows_value
            try:
                fm = self.base_fm.copy()
                fm[self.hit] = run_msg_stage(self.s.model, self.s, self.hit)
            finally:
                self.W.data[ids] = self.orig_rows
        return margin(self.s.head_from(self.s.base_feats['twin_b'], self.s.base_feats['twin_a'], fm))

    def noise(self, seed, scale=1.0):
        d = self.orig_rows.shape[1]
        r = np.random.RandomState(seed).randn(len(self.U), d) * scale
        return torch.from_numpy(r.astype(np.float32))

    def recenter(self, m):
        """Shift the margin so that the median change (new - orig) over all samples is zero."""
        return m - np.median(m - self.orig_margin)
