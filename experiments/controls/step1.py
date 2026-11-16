"""Step 1: margins of all variants on the full test set (7,608 samples). Usage: step1.py MODEL_NO [n_seeds]"""
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
import bisect
import json
import sys
import time

from common import *

no = int(sys.argv[1])
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else N_SEEDS
t0 = time.time()
s = get_session(no)
y = s.y
V = len(s.vocab['msg'])
base_fm = s.base_feats['msg']
res = {}
meta = {'model': no, 'is_tf': bool(s.is_tf), 'win_msg': int(s.win_msg)}

# --- orig: recompute the message feature with an untouched deep copy, and check against the cache.
m0 = copy.deepcopy(s.model)
res['orig'], _ = variant_margin(s, m0, base_fm, [], None)       # no edit: uses cached features
logit_cache = margin(s.base_feats['head'])
fm_re = run_msg_stage(m0, s, np.arange(s.n))
logit_re = margin(s.head_from(s.base_feats['twin_b'], s.base_feats['twin_a'], fm_re))
meta['orig_vs_cache_max_abs_margin_gap'] = float(np.abs(res['orig'] - logit_cache).max())
meta['orig_full_recompute_msg_stage_gap'] = float(np.abs(logit_re - logit_cache).max())
res['prior'] = res['orig'] - PRIOR_SHIFT

# --- head input features of orig (input of last linear layer)
feat, W, b = head_input_features(s, base_fm)
meta['head_layer'] = s.model.HEAD_LINEARS[-1]
meta['head_feat_dim'] = int(feat.shape[1])
meta['head_feat_check_max_gap'] = float(np.abs((feat.astype(np.float64) @ W.astype(np.float64).T + b)[:, 1:] - (feat.astype(np.float64) @ W.astype(np.float64).T + b)[:, :1] - res['orig'][:, None]).max())
np.savez_compressed('%s/feat_M%d.npz' % (OUT, no), y=y, feat=feat.astype(np.float32), W=W, b=b)

# --- zero variants
U = untrained_ids(s)
R = untrained_ids(s, max_count=10)
meta.update(n_untrained=int(len(U)), n_rare=int(len(R)))
for name, ids in (('untrained_zero', U), ('rare_zero', R)):
    res[name], nh = variant_margin(s, edited_copy(s, ids, np.zeros(s.model.embedMsg.weight.shape[1])), base_fm, ids)
    meta['n_samples_hit_' + name] = nh

# --- placebos
pool = trained_pool(s)
f = test_freq(s)
T_u = int(f[U].sum())
meta['test_occurrences_untrained_rows'] = T_u
meta['n_pool_trained'] = int(len(pool))
placebo_rows = {}
achieved = {'untrained': [], 'freqmatch': []}
by_f = {}
for i in pool:
    by_f.setdefault(int(f[i]), []).append(int(i))
f_u = f[U]
for k in range(n_seeds):
    rng = np.random.RandomState(k)
    # (a) uniform among trained rows, same number of rows
    ids_a = rng.choice(pool, size=len(U), replace=False)
    # (b) frequency matched: each untrained row gets a distinct trained row with the same test
    #     frequency; if that frequency bucket is used up, the nearest available frequency is taken.
    buckets = {key: list(v) for key, v in by_f.items()}
    for key in buckets:
        rng.shuffle(buckets[key])
    keys = sorted(buckets)
    ids_b = []
    for fu in rng.permutation(f_u):
        fu = int(fu)
        cand = [kk for kk in keys if buckets[kk]]
        kk = min(cand, key=lambda q: (abs(q - fu), rng.rand()))
        ids_b.append(buckets[kk].pop())
    ids_b = np.array(ids_b)
    for tag, ids in (('untrained', ids_a), ('freqmatch', ids_b)):
        achieved[tag].append(int(f[ids].sum()))
        placebo_rows['%s_%d' % (tag, k)] = ids
        res['placebo_%s_%d' % (tag, k)], _ = variant_margin(s, edited_copy(s, ids, np.zeros(s.model.embedMsg.weight.shape[1])), base_fm, ids)
meta['placebo_total_test_occurrences'] = achieved
meta['placebo_n_rows'] = int(len(U))
np.savez_compressed('%s/placebo_rows_M%d.npz' % (OUT, no), **placebo_rows)

# --- imputations of untrained rows
W0 = s.model.embedMsg.weight.detach()
mean_vec = W0[torch.from_numpy(pool)].mean(0)
res['untrained_impute_mean'], _ = variant_margin(s, edited_copy(s, U, mean_vec), base_fm, U)

# kNN: label-free neighbour = trained tokens that share the longest common prefix (>= 5 chars);
# otherwise the trained tokens of the same character shape; otherwise the global trained mean.
vocab = s.vocab['msg']
sorted_tr = sorted((vocab[i], int(i)) for i in pool)
strs = [t for t, _ in sorted_tr]


def lcp(a, b):
    n = 0
    for x, z in zip(a, b):
        if x != z:
            break
        n += 1
    return n


def shape(t):
    if re.fullmatch(r'<?(0x)?[0-9a-f]{6,}>?', t):
        return 'hex'
    if re.fullmatch(r'\d+', t):
        return 'num'
    if '_' in t:
        return 'snake'
    return 'word'


shape_pool = {}
for i in pool:
    shape_pool.setdefault(shape(vocab[i]), []).append(int(i))
shape_mean = {k: W0[torch.tensor(v)].mean(0) for k, v in shape_pool.items()}
vecs, how = [], {'prefix': 0, 'shape': 0, 'global': 0}
for u in U:
    t = vocab[u]
    p = bisect.bisect_left(strs, t)
    best, lo = 0, p
    for j in range(max(0, p - 3), min(len(strs), p + 3)):
        l = lcp(t, strs[j])
        if l > best:
            best = l
    if best >= 5:
        pref = t[:best]
        a = bisect.bisect_left(strs, pref)
        bnd = bisect.bisect_left(strs, pref[:-1] + chr(ord(pref[-1]) + 1))
        ids = [sorted_tr[j][1] for j in range(a, bnd)]
        vecs.append(W0[torch.tensor(ids)].mean(0))
        how['prefix'] += 1
    elif shape(t) in shape_mean:
        vecs.append(shape_mean[shape(t)])
        how['shape'] += 1
    else:
        vecs.append(mean_vec)
        how['global'] += 1
res['untrained_impute_knn'], _ = variant_margin(s, edited_copy(s, U, torch.stack(vecs)), base_fm, U)
meta['knn_neighbour_source'] = how

# --- verification against archive predictions and summary_all.json
pred = res['orig'] > 0
arch = np.loadtxt('%s/Model%d/content/predicted.txt' % (ARCHIVE, no)).astype(int)
meta['orig_pred_equals_archive_predicted_txt'] = int((pred.astype(int) == arch).sum())
meta['orig_pred_equals_cached_base_pred'] = int((pred.astype(int) == s.base_pred).sum())
meta['accuracy_full_test_pct'] = float(100 * (pred == y).mean())
rng_ = np.random.RandomState(20261111).permutation(s.n)          # DEV_SEED split, as in scalpel/data.py
part = np.full(s.n, 2); part[rng_[:int(round(s.n * .3))]] = 0; part[rng_[int(round(s.n * .3)):int(round(s.n * .3)) + int(round(s.n * .2))]] = 1
meta['holdout_idx_matches_bundle_part'] = bool((part == s.part).all())
ho = np.where(part == 2)[0]
from scalpel.engine import confusion_metrics
m = confusion_metrics(y[ho], pred[ho])
sm = [r for r in json.load(open(os.path.join(ROOT, 'results', 'summary_all.json'))) if r['model'] == no and r['round'] == 1][0]
meta['holdout_f1_recomputed'] = m['f1']; meta['holdout_f1_summary'] = sm['f1'][0]
meta['holdout_acc_recomputed'] = m['acc']; meta['holdout_acc_summary'] = sm['acc'][0]
meta['seconds'] = round(time.time() - t0, 1)
np.savez('%s/logits_M%d.npz' % (OUT, no), y=y, **{k: v.astype(np.float32) for k, v in res.items()})
json.dump(meta, open('%s/meta_M%d.json' % (OUT, no), 'w'), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != 'placebo_total_test_occurrences'}, indent=1))
print('placebo occ untrained mean', np.mean(achieved['untrained']), 'freqmatch mean', np.mean(achieved['freqmatch']), 'target', T_u)
