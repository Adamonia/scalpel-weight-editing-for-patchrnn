"""Error overlap of the 8 base models on the diag part (labels of this part are open)."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))

import numpy as np, json
z = np.load(os.path.join(BUNDLE, 'bundle.npz')); y = z['y']; part = z['part']
nums = [0, 2, 3, 4, 5, 8, 12, 13]
wrong = np.stack([np.load(os.path.join(BUNDLE, 'base_feats_M%d.npz' % n))['head'].argmax(1) != y for n in nums])
d = part == 0
k = wrong[:, d].sum(0)
print('diag samples', int(d.sum()), '| wrong in k of 8 models:', {int(i): int((k == i).sum()) for i in range(9)})
yd = y[d]
for lab, name in ((1, 'security (FN)'), (0, 'non-security (FP)')):
    m = yd == lab
    print('  %s: n=%d, all 8 wrong: %d, >=6 wrong: %d, any wrong: %d' % (name, int(m.sum()), int((k[m] == 8).sum()), int((k[m] >= 6).sum()), int((k[m] >= 1).sum())))
print('  per-model diag errors:', {n: int(w) for n, w in zip(nums, wrong[:, d].sum(1))})
# oracle and majority on diag
print('  oracle (any model right) acc %.2f; majority vote acc %.2f' % (100 * (k < 8).mean(), 100 * (k <= 3).mean()))
rows = np.where(d)[0]
hard = rows[k == 8]
rng = np.random.RandomState(7)
pick_fn = rng.choice(hard[y[hard] == 1], 16, replace=False); pick_fp = rng.choice(hard[y[hard] == 0], 14, replace=False)
json.dump({'fn': sorted(int(i) for i in pick_fn), 'fp': sorted(int(i) for i in pick_fp), 'n_hard': int(len(hard))}, open(os.path.join(_HERE, 'hard_sample.json'), 'w'))
print('  hard-core sample ids FN:', sorted(int(i) for i in pick_fn)); print('  hard-core sample ids FP:', sorted(int(i) for i in pick_fp))
