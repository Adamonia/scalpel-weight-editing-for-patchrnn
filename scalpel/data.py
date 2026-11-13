"""Data pipeline.

This module rebuilds the PatchRNN twin data from the public arrays
(data.npy, props.npy, msgs.npy) and the fixed split of the model archive.
The steps follow SecurityPatchIdentificationRNN.py (demoTwin).
"""
import json
import os
import random

import numpy as np

from .safe_npy import load_object_npy

TWIN_MAX_LEN = 800
MSG_MAX_LEN = 200
SPLIT_SEED = 10          # This seed reproduces reals.txt of the archive.
TEST_RATE = 0.2
DEV_SEED = 20261111      # Seed of the diagnose/guard/holdout division of the test set.
DIAG_RATE = 0.30
GUARD_RATE = 0.20
PAD = '<pad>'


def _abstract(tokens, types):
    """Replace identifiers with VARn/FUNCn and non-digit literals with LITERAL."""
    n_tok = len(types)
    mark_var = [False] * n_tok
    mark_fun = [False] * n_tok
    for n in range(n_tok):
        t = types[n]
        if t == 5:
            tokens[n] = 'COMMENT'
        elif t == 3:
            if not tokens[n].isdigit():
                tokens[n] = 'LITERAL'
        elif t == 2:
            if n < n_tok - 1 and tokens[n + 1] == '(':
                mark_fun[n] = True
            else:
                mark_var[n] = True
    var_dict = {}
    fun_dict = {}
    for n in range(n_tok):
        if mark_var[n]:
            var_dict.setdefault(tokens[n], 'VAR' + str(len(var_dict)))
        elif mark_fun[n]:
            fun_dict.setdefault(tokens[n], 'FUNC' + str(len(fun_dict)))
    for n in range(n_tok):
        if mark_var[n]:
            tokens[n] = var_dict[tokens[n]]
        elif mark_fun[n]:
            tokens[n] = fun_dict[tokens[n]]
    return tokens


def _pad(seq, length, pad_value=0):
    out = np.full(length, pad_value, dtype=np.int32)
    k = min(len(seq), length)
    out[:k] = seq[:k]
    return out


def build_dataset(pub_dir, cache_file):
    """Build all arrays. The result is a dict of numpy arrays and lists.

    pub_dir: folder with props.npy and msgs.npy from the public PatchRNN repo.
    cache_file: .npz path. The function loads this file if it exists.
    """
    vocab_file = cache_file + '.vocab.json'
    if os.path.exists(cache_file) and os.path.exists(vocab_file):
        z = np.load(cache_file, allow_pickle=False)
        out = {k: z[k] for k in z.files}
        with open(vocab_file, encoding='utf-8') as fp:
            out.update(json.load(fp))
        return out

    props = load_object_npy(os.path.join(pub_dir, 'props.npy'))
    msgs = load_object_npy(os.path.join(pub_dir, 'msgs.npy'))
    n_all = len(props)
    assert len(msgs) == n_all

    diff_dict = {}
    tok_b = np.zeros((n_all, TWIN_MAX_LEN), dtype=np.int32)
    tok_a = np.zeros((n_all, TWIN_MAX_LEN), dtype=np.int32)
    typ_b = np.zeros((n_all, TWIN_MAX_LEN), dtype=np.int8)
    typ_a = np.zeros((n_all, TWIN_MAX_LEN), dtype=np.int8)
    len_b = np.zeros(n_all, dtype=np.int32)
    len_a = np.zeros(n_all, dtype=np.int32)
    labels = np.zeros(n_all, dtype=np.int8)

    for i in range(n_all):
        tokens, types, diffs, label = props[i]
        # Keep the comments (_CTYP_ = 0 in the Colab notebook). Each comment
        # becomes the token COMMENT. Copy the lists before the changes.
        tokens = list(tokens)
        types = list(types)
        diffs = list(diffs)
        tokens = _abstract(tokens, types)
        ids = []
        for tk in tokens:
            idx = diff_dict.get(tk)
            if idx is None:
                idx = len(diff_dict) + 1
                diff_dict[tk] = idx
            ids.append(idx)
        ids_b = [ids[n] for n in range(len(ids)) if diffs[n] <= 0]
        ids_a = [ids[n] for n in range(len(ids)) if diffs[n] >= 0]
        ty_b = [types[n] for n in range(len(ids)) if diffs[n] <= 0]
        ty_a = [types[n] for n in range(len(ids)) if diffs[n] >= 0]
        tok_b[i] = _pad(ids_b, TWIN_MAX_LEN)
        tok_a[i] = _pad(ids_a, TWIN_MAX_LEN)
        typ_b[i] = _pad(ty_b, TWIN_MAX_LEN)
        typ_a[i] = _pad(ty_a, TWIN_MAX_LEN)
        len_b[i] = len(ids_b)
        len_a[i] = len(ids_a)
        labels[i] = label
        props[i] = None

    msg_dict = {}
    msg = np.zeros((n_all, MSG_MAX_LEN), dtype=np.int32)
    len_m = np.zeros(n_all, dtype=np.int32)
    for i in range(n_all):
        tokens, label = msgs[i]
        assert label == labels[i]
        ids = []
        for tk in tokens:
            idx = msg_dict.get(tk)
            if idx is None:
                idx = len(msg_dict) + 1
                msg_dict[tk] = idx
            ids.append(idx)
        msg[i] = _pad(ids, MSG_MAX_LEN)
        len_m[i] = len(ids)

    diff_vocab = [PAD] + list(diff_dict.keys())
    msg_vocab = [PAD] + list(msg_dict.keys())

    # Fixed split of the archive: random.seed(10), shuffle, first 20 % is the test set.
    order = list(range(n_all))
    random.Random(SPLIT_SEED).shuffle(order)
    n_test = int(np.floor(n_all * TEST_RATE))
    test_idx = np.array(order[:n_test], dtype=np.int64)
    train_idx = np.array(order[n_test:], dtype=np.int64)

    # Division of the test set into three parts:
    # 0 = diagnose (the agent reads samples), 1 = guard (aggregate metrics only),
    # 2 = holdout (sealed until the final report).
    rng = np.random.RandomState(DEV_SEED)
    perm = rng.permutation(n_test)
    n_diag = int(round(n_test * DIAG_RATE))
    n_guard = int(round(n_test * GUARD_RATE))
    part = np.full(n_test, 2, dtype=np.int8)
    part[perm[:n_diag]] = 0
    part[perm[n_diag:n_diag + n_guard]] = 1

    out = dict(tok_b=tok_b, tok_a=tok_a, typ_b=typ_b, typ_a=typ_a, msg=msg,
               len_b=len_b, len_a=len_a, len_m=len_m, labels=labels,
               test_idx=test_idx, train_idx=train_idx, part=part)
    np.savez(cache_file, **out)
    vocab = dict(diff_vocab=diff_vocab, msg_vocab=msg_vocab)
    with open(vocab_file, 'w', encoding='utf-8') as fp:
        json.dump(vocab, fp, ensure_ascii=False)
    out.update(vocab)
    return out
