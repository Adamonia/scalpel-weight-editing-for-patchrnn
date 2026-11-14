"""Bundle for the CodeBERT sequence classifier (Model15).

The notebook builds one text for each patch:
    f"COMMIT: {commit} CODE: {code}"
'commit' is the list of raw message lines and 'code' is the list of diff hunks (lists of raw lines).
Thus the text holds the Python list notation (brackets, quotes, the two characters backslash and n).
The tokenizer is the CodeBERT (RoBERTa) tokenizer: 512 ids with <s> and </s>, padding id 1.

Usage: python -m scalpel.textbundle PUB_DIR BASE_BUNDLE OUT_BUNDLE VOCAB_JSON MERGES_TXT
"""
import json
import os
import shutil
import sys

import numpy as np

from .prepare import _doc_freq
from .safe_npy import load_object_npy

LEN = 512
CUT = 12000          # Characters. A text that is longer is cut before the exact (slow) tokenizer reads it.


def patch_text(msg_lines, hunks):
    return 'COMMIT: %s CODE: %s' % (list(msg_lines), [list(h) for h in hunks])


def encode_exact(tok, text):
    """Ids as in the notebook. A long text is cut first; the cut is behind the 512-token window."""
    short = text[:CUT]
    ids = tok(short, padding='max_length', truncation=True, max_length=LEN)['input_ids']
    if len(text) > CUT and ids[-1] == tok.pad_token_id:
        ids = tok(text, padding='max_length', truncation=True, max_length=LEN)['input_ids']
    return ids


def build(pub_dir, base_dir, out_dir, vocab_json, merges_txt, log=sys.stderr):
    import transformers
    os.makedirs(out_dir, exist_ok=True)
    slow = transformers.RobertaTokenizer(vocab_json, merges_txt)
    fast = transformers.RobertaTokenizerFast(vocab_file=vocab_json, merges_file=merges_txt)
    for t in (slow, fast):
        t.model_max_length = 10 ** 9
    z = np.load(os.path.join(base_dir, 'dataset.npz'))
    labels, test_idx, train_idx, part = z['labels'], z['test_idx'], z['train_idx'], z['part']
    raw = load_object_npy(os.path.join(pub_dir, 'data.npy'))
    pad = slow.pad_token_id
    n = len(test_idx)
    ids = np.full((n, LEN), pad, dtype=np.int32)
    full_len = np.zeros(n, dtype=np.int32)       # Tokens of the complete text (fast tokenizer).
    msg_tokens = np.zeros(n, dtype=np.int32)     # Tokens before the word CODE (message part of the window).
    n_diff = 0
    for j, i in enumerate(test_idx):
        msg_lines, hunks, label = raw[i]
        assert label == labels[i]
        text = patch_text(msg_lines, hunks)
        ids[j] = encode_exact(slow, text)
        f_ids = fast(text)['input_ids']
        full_len[j] = len(f_ids) - 2
        head = fast('COMMIT: %s' % (list(msg_lines),))['input_ids']
        msg_tokens[j] = len(head) - 2
        n_diff += int(list(ids[j][ids[j] != pad]) != (f_ids[:LEN - 1] + [f_ids[-1]] if len(f_ids) > LEN else f_ids))
        if j % 1000 == 0:
            print('test', j, 'fast/slow differences', n_diff, flush=True, file=log)
    print('fast tokenizer differs from the exact tokenizer in %d of %d test texts' % (n_diff, n), flush=True, file=log)
    one = np.zeros((n, 1), dtype=np.int32)
    out = {'y': labels[test_idx].astype(np.int8), 'part': part, 'test_idx': test_idx, 'msg': ids,
           'tok_b': one, 'tok_a': one, 'typ_b': one.astype(np.int8), 'typ_a': one.astype(np.int8),
           'len_b': one[:, 0], 'len_a': one[:, 0], 'len_m': full_len, 'msg_tokens': msg_tokens,
           'pad_id': np.int64(pad)}
    np.savez(os.path.join(out_dir, 'bundle_test_only.npz'), **out)

    # Training document counts of the sub-word tokens (fast tokenizer; statistics only).
    v = slow.vocab_size
    y_tr = labels[train_idx]
    tr = np.full((len(train_idx), LEN), pad, dtype=np.int32)
    for j, i in enumerate(train_idx):
        msg_lines, hunks, _ = raw[i]
        f_ids = fast(patch_text(msg_lines, hunks)[:4 * CUT], truncation=True, max_length=LEN)['input_ids']
        tr[j, :len(f_ids)] = f_ids
        if j % 5000 == 0:
            print('train', j, flush=True, file=log)
    for cls, name in ((1, 'pos'), (0, 'neg')):
        df = _doc_freq(tr[y_tr == cls], v)
        out['df_msg_' + name] = df
        out['df_code_' + name] = np.zeros_like(df)
    out['n_train_pos'] = np.int64((y_tr == 1).sum())
    out['n_train_neg'] = np.int64((y_tr == 0).sum())
    np.savez(os.path.join(out_dir, 'bundle.npz'), **out)
    os.remove(os.path.join(out_dir, 'bundle_test_only.npz'))
    names = [t for t, _ in sorted(slow.get_vocab().items(), key=lambda kv: kv[1])][:v]
    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump({'code': names, 'msg': names}, fp, ensure_ascii=False)
    raw_json = os.path.join(base_dir, 'raw_test.json')
    if os.path.exists(raw_json):
        shutil.copy(raw_json, os.path.join(out_dir, 'raw_test.json'))


if __name__ == '__main__':
    build(*sys.argv[1:6])
    print('bundle written to', sys.argv[3])
