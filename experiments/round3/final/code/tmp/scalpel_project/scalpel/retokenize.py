"""Bundle for the models that read sub-word tokens (Model6: BERT WordPiece, Model7: RoBERTa BPE).

The notebooks decode the abstracted tokens of each input window, join them without a separator,
and encode the text again with a pre-trained tokenizer (800 ids for each input, no special tokens).
An empty text gives 800 zeros. The token-type columns are zero after this step.

Usage: python -m scalpel.retokenize BASE_BUNDLE OUT_BUNDLE bert VOCAB_TXT
       python -m scalpel.retokenize BASE_BUNDLE OUT_BUNDLE roberta VOCAB_JSON MERGES_TXT
BASE_BUNDLE must hold dataset.npz and vocab.json (made by scalpel.prepare).
"""
import json
import os
import shutil
import sys

import numpy as np

from .data import MSG_MAX_LEN, TWIN_MAX_LEN
from .prepare import _doc_freq

LEN = 800


def load_tokenizer(kind, files):
    import transformers
    if kind == 'bert':
        tok = transformers.BertTokenizer(files[0], do_lower_case=True)
    else:
        tok = transformers.RobertaTokenizer(files[0], files[1])
    tok.model_max_length = 10 ** 9
    return tok


def encode(tok, text):
    ids = tok.encode(text=text, add_special_tokens=False, padding='max_length', max_length=LEN,
                     truncation='longest_first')
    if len(ids) == 0:
        return np.zeros(LEN, dtype=np.int32)
    return np.asarray(ids, dtype=np.int32)


def retokenize_rows(tok, d, vocab, rows, log=None):
    """Sub-word ids of the three inputs for the given dataset rows."""
    code_v, msg_v = vocab['code'], vocab['msg']
    out = {k: np.zeros((len(rows), LEN), dtype=np.int32) for k in ('tok_b', 'tok_a', 'msg')}
    for j, i in enumerate(rows):
        for key, table in (('tok_b', code_v), ('tok_a', code_v), ('msg', msg_v)):
            ids = d[key][i]
            out[key][j] = encode(tok, ''.join(table[t] for t in ids if t != 0))
        if log and j % 2000 == 0:
            print('retokenized', j, 'of', len(rows), flush=True, file=log)
    return out


def build(base_dir, out_dir, kind, files, log=sys.stderr):
    os.makedirs(out_dir, exist_ok=True)
    tok = load_tokenizer(kind, files)
    pad = tok.pad_token_id
    z = np.load(os.path.join(base_dir, 'dataset.npz'))
    d = {k: z[k] for k in ('tok_b', 'tok_a', 'msg', 'labels', 'test_idx', 'train_idx', 'part')}
    with open(os.path.join(base_dir, 'vocab.json'), encoding='utf-8') as fp:
        vocab = json.load(fp)
    labels = d['labels']
    test = retokenize_rows(tok, d, vocab, d['test_idx'], log)
    n_test = len(d['test_idx'])
    out = {'y': labels[d['test_idx']].astype(np.int8), 'part': d['part'], 'test_idx': d['test_idx'],
           'tok_b': test['tok_b'], 'tok_a': test['tok_a'], 'msg': test['msg'][:, :MSG_MAX_LEN],
           'typ_b': np.zeros((n_test, LEN), dtype=np.int8), 'typ_a': np.zeros((n_test, LEN), dtype=np.int8),
           'pad_id': np.int64(pad)}
    win_c, win_m = TWIN_MAX_LEN - 1, MSG_MAX_LEN - 1
    # Lengths = number of sub-word tokens in the model window. Position 0 can hold id 0 for an empty text.
    def real(arr, win):
        return ((arr[:, :win] != pad) & ~((arr[:, :win] == 0).all(1))[:, None]).sum(1).astype(np.int32)
    out['len_b'], out['len_a'], out['len_m'] = real(test['tok_b'], win_c), real(test['tok_a'], win_c), real(test['msg'], win_m)
    np.savez(os.path.join(out_dir, 'bundle_test_only.npz'), **out)

    train = retokenize_rows(tok, d, vocab, d['train_idx'], log)
    y_tr = labels[d['train_idx']]
    v = tok.vocab_size
    for cls, name in ((1, 'pos'), (0, 'neg')):
        m = y_tr == cls
        code = np.concatenate((train['tok_b'][m][:, :win_c], train['tok_a'][m][:, :win_c]), axis=1)
        df_code = _doc_freq(code, v)
        df_msg = _doc_freq(train['msg'][m][:, :win_m], v)
        for sfx in ('', '_tf'):
            out['df_code%s_%s' % (sfx, name)] = df_code
            out['df_msg%s_%s' % (sfx, name)] = df_msg
    out['n_train_pos'] = np.int64((y_tr == 1).sum())
    out['n_train_neg'] = np.int64((y_tr == 0).sum())
    np.savez(os.path.join(out_dir, 'bundle.npz'), **out)
    os.remove(os.path.join(out_dir, 'bundle_test_only.npz'))
    inv = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    names = [t for t, _ in inv][:v]
    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump({'code': names, 'msg': names}, fp, ensure_ascii=False)
    raw = os.path.join(base_dir, 'raw_test.json')
    if os.path.exists(raw):
        shutil.copy(raw, os.path.join(out_dir, 'raw_test.json'))


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
    print('bundle written to', sys.argv[2])
