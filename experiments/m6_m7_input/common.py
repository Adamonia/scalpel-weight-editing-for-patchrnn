"""Shared helpers for the M6/M7 input-format study. Read-only use of the existing project files."""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, '/home/claude/work/scalpel_project')
from scalpel import retokenize as R
from scalpel.models import build_model

ARCH = '/mnt/user-data/uploads/patchrnn-archive/model-archive'
W = '/home/claude/work'
OUT = W + '/ext67/out'
CACHE = W + '/ext67/cache'
os.makedirs(CACHE, exist_ok=True)
CFG = {
    6: dict(kind='bert', bundle=W + '/bundle_bert', files=[W + '/tokenizers/bert/vocab.txt']),
    7: dict(kind='roberta', bundle=W + '/bundle_roberta',
            files=[W + '/tokenizers/roberta/vocab.json', W + '/tokenizers/roberta/merges.txt']),
}
LEN = R.LEN


def load_base():
    z = np.load(W + '/cache/dataset.npz')
    d = {k: z[k] for k in ('tok_b', 'tok_a', 'msg', 'labels', 'test_idx', 'part')}
    v = json.load(open(W + '/cache/dataset.npz.vocab.json'))
    return d, v['diff_vocab'], v['msg_vocab']


def load_bundle(m):
    z = np.load(CFG[m]['bundle'] + '/bundle.npz')
    return {k: z[k] for k in z.files}


def texts(d, vocab_code, vocab_msg, rows, sep):
    """Text of each input for the given dataset rows. sep='' is the training format."""
    out = {'tok_b': [], 'tok_a': [], 'msg': []}
    for i in rows:
        for key, tab in (('tok_b', vocab_code), ('tok_a', vocab_code), ('msg', vocab_msg)):
            out[key].append(sep.join(tab[t] for t in d[key][i] if t != 0))
    return out


def encode_variant(m, variant, cache=True):
    """Return dict tok_b, tok_a, msg (int32 arrays) for the test set in one of the formats:
    orig (join, no separator), space (join with one space), pertok (each abstract token encoded alone)."""
    path = '%s/ids_M%d_%s.npz' % (CACHE, m, variant)
    if cache and os.path.exists(path):
        z = np.load(path); return {k: z[k] for k in z.files}
    d, vc, vm = load_base()
    tok = R.load_tokenizer(CFG[m]['kind'], CFG[m]['files'])
    rows = d['test_idx']
    if variant in ('orig', 'space'):
        tx = texts(d, vc, vm, rows, '' if variant == 'orig' else ' ')
        out = {k: np.stack([R.encode(tok, t) for t in tx[k]]) for k in tx}
    elif variant == 'pertok':
        memo = {}
        def one(t):
            if t not in memo:
                memo[t] = tok.encode(t, add_special_tokens=False)
            return memo[t]
        out = {}
        for key, tab in (('tok_b', vc), ('tok_a', vc), ('msg', vm)):
            arr = np.full((len(rows), LEN), tok.pad_token_id, dtype=np.int32)
            for j, i in enumerate(rows):
                ids = [x for t in d[key][i] if t != 0 for x in one(tab[t])][:LEN]
                if len(ids) == 0: arr[j] = 0
                else: arr[j, :len(ids)] = ids
            out[key] = arr
    else:
        raise ValueError(variant)
    out['msg'] = out['msg'][:, :200]
    if cache: np.savez(path, **out)
    return out


def twin_id(tokstr, kind, vocab):
    """Boundary twin of a token string: Ġ variant (RoBERTa) or ## variant (BERT). None if absent."""
    if kind == 'roberta':
        t = tokstr[1:] if tokstr.startswith('Ġ') else 'Ġ' + tokstr
    else:
        t = tokstr[2:] if tokstr.startswith('##') else '##' + tokstr
    return vocab.get(t)


def trained_masks(b):
    """Rows seen in training: code table (embedTwin) and message table (embedMsg)."""
    code = (b['df_code_pos'] + b['df_code_neg']) > 0
    msg = (b['df_msg_pos'] + b['df_msg_neg']) > 0
    return code, msg


def run_model(model, ids, typ_shape_n, bs=32, log=None):
    """Logits for the whole set. Types are zero (as in the training-time bundle)."""
    n = len(ids['tok_b']); out = []
    zt = torch.zeros((bs, LEN), dtype=torch.int8)
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(n, s + bs)
            b = {'tok_b': torch.from_numpy(ids['tok_b'][s:e]), 'tok_a': torch.from_numpy(ids['tok_a'][s:e]),
                 'msg': torch.from_numpy(ids['msg'][s:e]), 'typ_b': zt[:e - s], 'typ_a': zt[:e - s]}
            acts = {}
            for st in model.STAGES:
                acts[st] = model.run_stage(st, b, acts)
            out.append(acts['head'].numpy())
            if log and (s // bs) % 40 == 0: print('  batch', s, 'of', n, time.strftime('%X'), flush=True)
    return np.concatenate(out)


def metrics(y, logits):
    pred = logits.argmax(1); s = logits[:, 1] - logits[:, 0]
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # AUC by rank (Mann-Whitney), ties get average rank
    from scipy.stats import rankdata
    r = rankdata(s); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return dict(n=int(len(y)), acc=100 * (tp + tn) / len(y), prec=100 * prec, rec=100 * rec, f1=100 * f1,
                auc=100 * float(auc), tp=tp, fp=fp, fn=fn, tn=tn)


def boot_f1_diff(y, la, lb, n_boot=2000, seed=0):
    """Paired bootstrap of F1(b) - F1(a) in points. Returns (diff, lo, hi, p_two_sided)."""
    pa, pb = la.argmax(1), lb.argmax(1)
    def comp(p):
        return np.stack([(p == 1) & (y == 1), (p == 1) & (y == 0), (p == 0) & (y == 1)]).astype(np.float64)
    ca, cb = comp(pa), comp(pb)
    rng = np.random.default_rng(seed); n = len(y)
    w = rng.multinomial(n, np.ones(n) / n, size=n_boot).astype(np.float64)
    def f1(c):
        t = w @ c.T; return 200 * t[:, 0] / np.maximum(2 * t[:, 0] + t[:, 1] + t[:, 2], 1)
    dd = f1(cb) - f1(ca)
    f = lambda c: 200 * c[0].sum() / (2 * c[0].sum() + c[1].sum() + c[2].sum())
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    return float(f(cb) - f(ca)), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)), float(min(p, 1))
