"""Step 5: bootstrap over test samples for the noise-to-signal (NSR) of each model and for the A-vs-B group contrast.
Reads tmp/arrays_M*.npz. Writes out/bootstrap.json. 1000 resamples of the 7608 test samples (same resample for all models).
This covers test-sample uncertainty only. It does not cover the choice of the 10 noise draws or the training run."""
import json
import numpy as np
O = '/home/claude/work/exth2/out'
MODELS = [3, 4, 5, 8, 12, 13]
A = {n: np.load('/home/claude/work/exth2/tmp/arrays_M%d.npz' % n) for n in MODELS}
N = 7608
rng = np.random.RandomState(0)
idx = rng.randint(0, N, size=(1000, N))
nsr = {n: np.zeros(1000) for n in MODELS}
for n in MODELS:
    v = A[n]['per_sd'] ** 2          # per-sample variance over draws, all 7608 samples (0 outside hit)
    full = v
    m = A[n]['m_clean_all']
    for b in range(1000):
        i = idx[b]
        nsr[n][b] = np.sqrt(full[i].mean()) / m[i].std()
ga = np.mean([nsr[n] for n in (3, 4, 13)], axis=0); gb = np.mean([nsr[n] for n in (5, 8, 12)], axis=0)
ratio = ga / gb
res = {'n_boot': 1000, 'nsr_ci95': {'M%d' % n: [float(np.percentile(nsr[n], 2.5)), float(np.percentile(nsr[n], 97.5))] for n in MODELS},
       'group_ratio_A_over_B_mean_nsr': {'median': float(np.median(ratio)), 'ci95': [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]},
       'min_A_minus_max_B': {'median': float(np.median(np.min([nsr[n] for n in (3, 4, 13)], 0) - np.max([nsr[n] for n in (5, 8, 12)], 0))),
                             'share_positive': float(np.mean(np.min([nsr[n] for n in (3, 4, 13)], 0) > np.max([nsr[n] for n in (5, 8, 12)], 0)))}}
json.dump(res, open(O + '/bootstrap.json', 'w'), indent=1)
print(res)
