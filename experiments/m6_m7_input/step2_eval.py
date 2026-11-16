# Inference-time variants for M6/M7: input format x label-free embedding edit. Logits are cached per variant.
import sys; sys.path.insert(0, '/home/claude/work/ext67')
from common import *

def edit_rows(model, b, kind, vocab, mode):
    """mode 'twin': copy the trained boundary twin into an untrained row.
       mode 'twin+mean': also set rows without a trained twin to the mean of the trained rows."""
    inv = {i: t for t, i in vocab.items()}
    mk_code, mk_msg = trained_masks(b)
    stats = {}
    for name, emb, mk in (('code', model.embedTwin, mk_code), ('msg', model.embedMsg, mk_msg)):
        W_ = emb.weight.data; src = W_.clone(); n_copy = n_mean = 0
        mean = src[torch.from_numpy(mk)].mean(0)
        for u in np.where(~mk)[0]:
            t = twin_id(inv[int(u)], kind, vocab)
            if t is not None and mk[t]:
                W_[u] = src[t]; n_copy += 1
            elif mode == 'twin+mean':
                W_[u] = mean; n_mean += 1
        stats[name] = dict(copied=n_copy, mean_filled=n_mean)
    return stats

# ('orig','none') is already computed by step0_check.py (logits_M*_orig.npy, 7608/7608 equal to predicted.txt).
VARIANTS = [('orig', 'twin'), ('space', 'none'), ('space', 'twin'), ('space', 'twin+mean'),
            ('pertok', 'none'), ('pertok', 'twin')]
which = [int(x) for x in sys.argv[1:]] or [6, 7]
for m in which:
    kind = CFG[m]['kind']; b = load_bundle(m)
    vocab = R.load_tokenizer(kind, CFG[m]['files']).get_vocab()
    for fmt, ed in VARIANTS:
        path = '%s/logits_M%d_%s_%s.npy' % (CACHE, m, fmt, ed.replace('+', '_'))
        if os.path.exists(path): continue
        ids = encode_variant(m, fmt)
        model = build_model(ARCH, m)
        st = edit_rows(model, b, kind, vocab, ed) if ed != 'none' else {}
        t = time.time(); print('M%d' % m, fmt, ed, st, flush=True)
        lg = run_model(model, ids, None)
        np.save(path, lg); print('  done %.0fs' % (time.time() - t), flush=True)
