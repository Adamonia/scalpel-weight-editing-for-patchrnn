"""Step 1: reproduction check, T1 (fresh noise, dose-response) and T2 (clean-margin density).
Usage: step1.py MODEL_NO. Writes out/t1t2_M{n}.json and out/margins_fresh_M{n}.npz.
No gradient training. Only embedMsg rows of untrained tokens are replaced (in memory)."""
import json, sys, time
from common import *

no = int(sys.argv[1])
t0 = time.time()
s = get_session(no)
y = s.y.astype(bool)
ho = np.where(s.part == 2)[0]
nz = Noiser(s)
cache = np.load('%s/logits_M%d.npz' % (CACHE, no))
d = nz.orig_rows.shape[1]

# layers and heads: layers are counted from the checkpoint keys, heads come from the loader registry
# (the checkpoint stores no head count: in_proj_weight has the same shape for any head count).
sd = torch.load('%s/Model%d/content/temp/model_TwinTransformer.pth' % (ARCHIVE, no), map_location='cpu')
sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
n_layers = len({k.split('.')[2] for k in sd if k.startswith('transformerMsg.layers.')})
info = {'model': no, 'n_layers_checkpoint': n_layers, 'n_heads_registry': int(s.model.nhead),
        'd_model': int(d), 'n_untrained_rows': int(len(nz.U)), 'n_hit_samples': int(len(nz.hit))}

# --- reproduction: orig (rows unchanged, msg stage recomputed on hit rows) and untrained_zero
m_orig_re = nz.margin_with(nz.orig_rows)
m_zero = nz.margin_with(torch.zeros_like(nz.orig_rows))
info['gap_orig_vs_cache'] = float(np.abs(m_orig_re - cache['orig']).max())
info['gap_orig_base_vs_cache'] = float(np.abs(nz.orig_margin - cache['orig']).max())
info['gap_zero_vs_cache'] = float(np.abs(m_zero - cache['untrained_zero']).max())
m_zero_c = nz.recenter(m_zero)
info['bias_shift_zero'] = float(np.median(m_zero - nz.orig_margin))


def summ(m):
    return {'f1': f1_pct(y, m), 'auc': 100 * roc_auc_score(y, m),
            'f1_ho': f1_pct(y[ho], m[ho]), 'auc_ho': 100 * roc_auc_score(y[ho], m[ho])}


orig = summ(nz.orig_margin)
zero = summ(m_zero_c)
info['orig'] = orig
info['zero'] = zero
info['zero_gain_f1'] = zero['f1'] - orig['f1']
info['zero_gain_f1_ho'] = zero['f1_ho'] - orig['f1_ho']


def run(scale, K):
    ms, raw = [], []
    for k in range(K):
        m = nz.margin_with(nz.noise(k, scale))
        raw.append(m)
        ms.append(nz.recenter(m))
    return np.array(ms), np.array(raw)


def pct(v, x):
    """Percentile (0..100) of x inside sample v; ties count half."""
    v = np.asarray(v)
    return 100 * (np.mean(v < x) + 0.5 * np.mean(v == x))


T1 = {}
for scale, K in ((1.0, 30), (0.5, 10), (2.0, 10)):
    M, raw = run(scale, K)
    rs = [summ(m) for m in M]
    f1 = np.array([r['f1'] for r in rs]); auc = np.array([r['auc'] for r in rs])
    f1h = np.array([r['f1_ho'] for r in rs]); auch = np.array([r['auc_ho'] for r in rs])
    f1_raw = np.array([f1_pct(y, m) for m in raw])
    T1[str(scale)] = {
        'K': K, 'f1_mean': f1.mean(), 'f1_sd': f1.std(ddof=1), 'f1_min': f1.min(), 'f1_max': f1.max(),
        'auc_mean': auc.mean(), 'auc_sd': auc.std(ddof=1),
        'gap_f1_zero_minus_fresh': zero['f1'] - f1.mean(), 'gap_auc_zero_minus_fresh': zero['auc'] - auc.mean(),
        'orig_f1_percentile': pct(f1, orig['f1']), 'orig_auc_percentile': pct(auc, orig['auc']),
        'orig_minus_fresh_f1': orig['f1'] - f1.mean(),
        'f1_ho_mean': f1h.mean(), 'f1_ho_sd': f1h.std(ddof=1), 'gap_f1_ho': zero['f1_ho'] - f1h.mean(),
        'auc_ho_mean': auch.mean(), 'orig_f1_ho_percentile': pct(f1h, orig['f1_ho']),
        'f1_mean_without_recenter': f1_raw.mean(),
        'f1_fresh_all': f1.tolist(), 'auc_fresh_all': auc.tolist(),
    }
    if scale == 1.0:
        np.save('%s/margins_fresh_M%d.npy' % (OUT, no), M.astype(np.float32))
        Mfresh = M
info['T1'] = T1

# --- T2: clean-margin density and noise-to-signal (scale 1, K=30), clean = zeroed model (recentered)
sd_clean = float(m_zero_c.std())
T2 = {'clean_margin_sd': sd_clean, 'clean_margin_mean': float(m_zero_c.mean())}
for c in (0.25, 0.5, 1.0):
    T2['share_abs_clean_below_%.2fsd' % c] = float(np.mean(np.abs(m_zero_c) < c * sd_clean))
    T2['share_hit_abs_clean_below_%.2fsd' % c] = float(np.mean(np.abs(m_zero_c[nz.hit]) < c * sd_clean))
per_sd = Mfresh.std(axis=0, ddof=1)
T2['noise_sd_over_clean_sd_rms_all'] = float(np.sqrt((per_sd ** 2).mean()) / sd_clean)
T2['noise_sd_over_clean_sd_rms_hit'] = float(np.sqrt((per_sd[nz.hit] ** 2).mean()) / sd_clean)
T2['noise_sd_over_clean_sd_median_hit'] = float(np.median(per_sd[nz.hit]) / sd_clean)
# share of samples whose predicted class changes between zero and a fresh draw, mean over draws
flip = (Mfresh > 0) != (m_zero_c > 0)[None]
T2['flip_rate_zero_vs_fresh_all'] = float(flip.mean())
T2['flip_rate_zero_vs_fresh_hit'] = float(flip[:, nz.hit].mean())
# margin change size: sd over samples of (fresh - zero), mean over draws
T2['change_sd_fresh_minus_zero'] = float((Mfresh - m_zero_c[None]).std(axis=1).mean())
T2['change_sd_orig_minus_zero'] = float((nz.orig_margin - m_zero_c).std())
T2['frac_hit'] = float(len(nz.hit) / s.n)
info['T2'] = T2
info['seconds'] = round(time.time() - t0, 1)
json.dump(info, open('%s/t1t2_M%d.json' % (OUT, no), 'w'), indent=1)
print(no, {k: info[k] for k in ('n_layers_checkpoint', 'n_heads_registry', 'gap_orig_vs_cache', 'gap_zero_vs_cache', 'zero_gain_f1', 'seconds')})
