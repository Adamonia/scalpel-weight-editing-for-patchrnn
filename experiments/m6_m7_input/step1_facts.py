# Facts: training-format vs normal (space-separated) tokenization, untrained rows, twins.
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for common.py
from common import *
from collections import Counter
res = {}
for m in (6, 7):
    kind = CFG[m]['kind']; b = load_bundle(m); pad = int(b['pad_id'])
    tok = R.load_tokenizer(kind, CFG[m]['files']); vocab = tok.get_vocab(); inv = {i: t for t, i in vocab.items()}
    sd = torch.load('%s/Model%d/content/temp/model_TwinTransformer.pth' % (ARCH, m), map_location='cpu', weights_only=True)
    mk_code, mk_msg = trained_masks(b)
    r = {'vocab': len(vocab)}
    # Rows never seen in training and their weight statistics (init is N(0,1), norm ~ sqrt(123))
    for name, mk, key in (('code', mk_code, 'embedTwin.weight'), ('msg', mk_msg, 'embedMsg.weight')):
        E = sd[key][:, :-5].numpy() if name == 'code' else sd[key].numpy()
        nrm = np.linalg.norm(E, axis=1)
        r['rows_' + name] = dict(trained=int(mk.sum()), untrained=int((~mk).sum()), frac_untrained=float((~mk).mean()),
                                 norm_trained_mean=float(nrm[mk].mean()), norm_untrained_mean=float(nrm[~mk].mean()),
                                 norm_untrained_std=float(nrm[~mk].std()), std_untrained=float(E[~mk].std()), std_trained=float(E[mk].std()))
    # Normal format ids vs training format ids
    orig = encode_variant(m, 'orig'); sp = encode_variant(m, 'space')
    for key, win, mk in (('tok_b', 799, mk_code), ('tok_a', 799, mk_code), ('msg', 199, mk_msg)):
        A, B = orig[key][:, :win], sp[key][:, :win]
        realA = (A != pad) & ~((A == 0).all(1))[:, None]; realB = (B != pad) & ~((B == 0).all(1))[:, None]
        same_seq = float((A == B).all(1).mean())
        # share of positions of the normal format that use an untrained row
        unt = float((~mk[B] & realB).sum() / realB.sum())
        # same for the training format on the test set (rows unseen in training)
        unt_o = float((~mk[A] & realA).sum() / realA.sum())
        # multiset mismatch: tokens of the normal format with no partner in the training format, per sample
        mism = 0; tot = 0
        for a, bb, ra, rb in zip(A, B, realA, realB):
            ca, cb = Counter(a[ra].tolist()), Counter(bb[rb].tolist())
            mism += sum((cb - ca).values()); tot += rb.sum()
        # border markers in the normal format
        isb = np.zeros(len(vocab) + 10, bool)
        for t, i in vocab.items():
            if (kind == 'roberta' and t.startswith('Ġ')) or (kind == 'bert' and t.startswith('##')): isb[i] = True
        r['pos_' + key] = dict(len_orig=float(realA.sum(1).mean()), len_space=float(realB.sum(1).mean()),
                               frac_samples_identical=same_seq, frac_pos_untrained_row_space=unt, frac_pos_untrained_row_orig=unt_o,
                               frac_multiset_mismatch_space=mism / tot,
                               frac_border_marker_orig=float((isb[A] & realA).sum() / realA.sum()),
                               frac_border_marker_space=float((isb[B] & realB).sum() / realB.sum()))
    # twins of untrained rows
    for name, mk in (('code', mk_code), ('msg', mk_msg)):
        unt = np.where(~mk)[0]; tw = [twin_id(inv[i], kind, vocab) for i in unt]
        has = np.array([t is not None for t in tw]); tr = np.array([t is not None and mk[t] for t in tw])
        r['twins_' + name] = dict(untrained=int(len(unt)), has_twin=int(has.sum()), twin_trained=int(tr.sum()))
    # Frequent examples of the difference
    d, vc, vm = load_base(); i0 = 5
    ids = [t for t in d['tok_a'][d['test_idx'][i0]] if t]
    s = [vc[t] for t in ids[:14]]
    r['example'] = dict(abstract=s, orig=tok.convert_ids_to_tokens(tok.encode(''.join(s), add_special_tokens=False)),
                        space=tok.convert_ids_to_tokens(tok.encode(' '.join(s), add_special_tokens=False)))
    # top unseen-in-training tokens met in the normal format
    c = Counter(); B = sp['tok_a'][:, :799]
    for row in B:
        for t in row[(row != pad)]:
            if not mk_code[t]: c[int(t)] += 1
    r['top_untrained_in_space_tok_a'] = [(inv[t], n) for t, n in c.most_common(12)]
    res[m] = r
json.dump(res, open(OUT + '/facts.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
