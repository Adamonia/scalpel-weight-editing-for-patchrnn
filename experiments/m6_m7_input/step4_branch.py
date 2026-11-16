# Which input branch causes the loss with normal tokenization? Mix branch features from the two formats.
import sys; sys.path.insert(0, '/home/claude/work/ext67')
from common import *
def feats(model, ids, n_bs=32):
    out = {s: [] for s in ('twin_b', 'twin_a', 'msg')}; n = len(ids['tok_b']); zt = torch.zeros((n_bs, LEN), dtype=torch.int8)
    with torch.no_grad():
        for s in range(0, n, n_bs):
            e = min(n, s + n_bs)
            b = {'tok_b': torch.from_numpy(ids['tok_b'][s:e]), 'tok_a': torch.from_numpy(ids['tok_a'][s:e]),
                 'msg': torch.from_numpy(ids['msg'][s:e]), 'typ_b': zt[:e - s], 'typ_a': zt[:e - s]}
            for st in out: out[st].append(model.run_stage(st, b, {}).clone())  # clone: the output is a view of a big tensor
    return {k: torch.cat(v) for k, v in out.items()}
res = {}
for m in (6, 7):
    model = build_model(ARCH, m); b = load_bundle(m); y = b['y'].astype(int); hold = b['part'] == 2
    F = {}
    for fmt in ('orig', 'space'):
        p = '%s/feats_M%d_%s.pt' % (CACHE, m, fmt)
        if os.path.exists(p): F[fmt] = torch.load(p); continue
        t = time.time(); F[fmt] = feats(model, encode_variant(m, fmt)); torch.save(F[fmt], p); print('M%d feats %s %.0fs' % (m, fmt, time.time() - t), flush=True)
    def head(code, msg):
        with torch.no_grad():
            return model.run_stage('head', None, {'twin_b': code['twin_b'], 'twin_a': code['twin_a'], 'msg': msg['msg']}).numpy()
    combos = {'all orig (check)': head(F['orig'], F['orig']), 'code orig + msg normal': head(F['orig'], F['space']),
              'code normal + msg orig': head(F['space'], F['orig']), 'all normal (check)': head(F['space'], F['space'])}
    base = np.load('%s/logits_M%d_orig_none.npy' % (CACHE, m)); res[m] = {}
    for k, lg in combos.items():
        r = {}
        for name, sel in (('all', np.ones(len(y), bool)), ('hold', hold)):
            mm = metrics(y[sel], lg[sel]); mm['dF1'], mm['lo'], mm['hi'], _ = boot_f1_diff(y[sel], base[sel], lg[sel], 2000, seed=1); r[name] = mm
        res[m][k] = r
        print('M%d %-24s all F1 %.2f (d %+.2f [%+.2f,%+.2f]) hold F1 %.2f (d %+.2f [%+.2f,%+.2f])' % (m, k, r['all']['f1'], r['all']['dF1'], r['all']['lo'], r['all']['hi'], r['hold']['f1'], r['hold']['dF1'], r['hold']['lo'], r['hold']['hi']), flush=True)
json.dump(res, open(OUT + '/branch_ablation.json', 'w'), indent=1)
