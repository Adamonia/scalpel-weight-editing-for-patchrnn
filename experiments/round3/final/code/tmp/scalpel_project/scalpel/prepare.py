"""Build the data bundle that the tool uses.

Inputs : props.npy, msgs.npy, data.npy of the public PatchRNN repository.
Outputs: bundle.npz   test arrays, split parts, token statistics of the training set
         vocab.json   the two vocabularies
         raw_test.json  raw commit message and diff of each test sample

Usage: python -m scalpel.prepare <PatchRNN/temp folder> <bundle folder>
"""
import json
import os
import sys

import numpy as np

from .data import MSG_MAX_LEN, TWIN_MAX_LEN, build_dataset
from .safe_npy import load_object_npy


def _doc_freq(arr, vocab_size):
    """Number of rows that contain each token id."""
    n_rows = arr.shape[0]
    keys = (np.arange(n_rows, dtype=np.int64)[:, None] * vocab_size + arr).ravel()
    keys = np.unique(keys)
    return np.bincount((keys % vocab_size).astype(np.int64), minlength=vocab_size).astype(np.int32)


def build_bundle(pub_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d = build_dataset(pub_dir, os.path.join(out_dir, 'dataset.npz'))
    test_idx, train_idx = d['test_idx'], d['train_idx']
    labels = d['labels']
    v_code, v_msg = len(d['diff_vocab']), len(d['msg_vocab'])
    out = {'y': labels[test_idx].astype(np.int8), 'part': d['part'], 'test_idx': test_idx}
    for key in ('tok_b', 'typ_b', 'tok_a', 'typ_a', 'msg', 'len_b', 'len_a', 'len_m'):
        out[key] = d[key][test_idx]

    # Token statistics of the training set. A token is "trained" if it occurs
    # in the input window of one training sample or more.
    y_tr = labels[train_idx]
    for cls, name in ((1, 'pos'), (0, 'neg')):
        rows = train_idx[y_tr == cls]
        code = np.concatenate((d['tok_b'][rows], d['tok_a'][rows]), axis=1)
        out['df_code_' + name] = _doc_freq(code, v_code)
        out['df_msg_' + name] = _doc_freq(d['msg'][rows], v_msg)
        # Transformer models read one token less (the class token takes one position).
        code_tf = np.concatenate((d['tok_b'][rows][:, :TWIN_MAX_LEN - 1],
                                  d['tok_a'][rows][:, :TWIN_MAX_LEN - 1]), axis=1)
        out['df_code_tf_' + name] = _doc_freq(code_tf, v_code)
        out['df_msg_tf_' + name] = _doc_freq(d['msg'][rows][:, :MSG_MAX_LEN - 1], v_msg)
    out['n_train_pos'] = np.int64((y_tr == 1).sum())
    out['n_train_neg'] = np.int64((y_tr == 0).sum())
    np.savez(os.path.join(out_dir, 'bundle.npz'), **out)

    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump({'code': list(d['diff_vocab']), 'msg': list(d['msg_vocab'])}, fp, ensure_ascii=False)

    raw_path = os.path.join(pub_dir, 'data.npy')
    if os.path.exists(raw_path):
        raw = load_object_npy(raw_path)
        texts = []
        for i in test_idx:
            msg_lines, hunks, label = raw[i]
            assert label == labels[i]
            diff = '\n@@\n'.join(''.join(h) for h in hunks)
            texts.append({'msg': ''.join(msg_lines)[:4000], 'diff': diff[:12000],
                          'diff_chars': len(diff), 'n_hunks': len(hunks)})
        with open(os.path.join(out_dir, 'raw_test.json'), 'w', encoding='utf-8') as fp:
            json.dump(texts, fp, ensure_ascii=False)
    return out_dir


if __name__ == '__main__':
    build_bundle(sys.argv[1], sys.argv[2])
    print('bundle written to', sys.argv[2])
