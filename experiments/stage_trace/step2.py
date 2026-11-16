"""Step 2 (S2): readout gain at the pooled message vector.
Usage: step2.py MODEL_NO (after step1.py). Reads tmp/arrays_M{n}.npz, writes out/readout_M{n}.json.

For each hit sample i: g_i = d margin / d pooled_i at the clean pooled vector (twin features fixed).
This is an input gradient for analysis only; no weights are trained or changed.
Change dp_ik = pooled(noise draw k) - pooled(clean). Split dp into the part along g_i and the rest.
Margin sd is the per-sample sd over the K draws, then root-mean-square over all 7608 samples
(samples without untrained tokens contribute zero), as in the earlier reports.
"""
import json, sys
from common import *

no = int(sys.argv[1])
s = get_session(no)
model = s.model.eval()
A = np.load('%s/tmp/arrays_M%d.npz' % (EXT, no))
dP, hit, P, mc, Mn = A['dP'], A['hit'], A['clean_p_all'], A['m_clean_all'], A['M_noise_all']
K, nh, d = dP.shape
N = len(mc)
fb, fa = s.base_feats['twin_b'][hit], s.base_feats['twin_a'][hit]
cls = model.classifier

# gradient of the margin w.r.t. the pooled vector, in batches
g = np.zeros((nh, d), np.float32)
for i in range(0, nh, 512):
    p = torch.from_numpy(P[hit[i:i + 512]].copy()).requires_grad_(True)
    o = cls(torch.cat((torch.from_numpy(fb[i:i + 512]), torch.from_numpy(fa[i:i + 512]), p), 1))
    (o[:, 1] - o[:, 0]).sum().backward()
    g[i:i + 512] = p.grad.numpy()

gn = np.linalg.norm(g, axis=1)
dpn = np.linalg.norm(dP, axis=2)                                  # (K,nh)
along = np.einsum('knd,nd->kn', dP, g) / gn[None]                 # signed length along the gradient
ortho = np.sqrt(np.maximum(dpn ** 2 - along ** 2, 0))
lin = along * gn[None]                                            # linearised margin change
act = Mn[:, hit] - mc[hit][None]                                  # actual margin change


def rms_all(v_k_n):
    """sd over draws per sample, then rms over all N samples (zeros outside hit)."""
    return float(np.sqrt((v_k_n.std(0, ddof=1) ** 2).sum() / N))


sd_c = float(mc.std())
pc = P - P.mean(0, keepdims=True)
rms_p_signal = float(np.sqrt((np.linalg.norm(pc, axis=1) ** 2).mean()))
rms_dp_all = float(np.sqrt((dpn ** 2).sum(1).mean() / N))         # rms over all samples and draws of |dp|
noise_margin_rms = rms_all(Mn[:, hit])
res = {'model': no,
       'grad_norm_mean': float(gn.mean()), 'grad_norm_rms': float(np.sqrt((gn ** 2).mean())),
       'mean_dp_norm': float(dpn.mean()), 'mean_clean_pooled_norm': float(np.linalg.norm(P[hit], axis=1).mean()),
       'mean_abs_along_over_mean_dp_norm': float(np.abs(along).mean() / dpn.mean()),
       'mean_per_sample_abs_cos_dp_grad': float((np.abs(along) / np.maximum(dpn, 1e-12)).mean()),
       'rms_cos_dp_grad': float(np.sqrt((along ** 2).sum() / (dpn ** 2).sum())),
       'random_direction_mean_abs_cos': float(np.sqrt(2 / (np.pi * d))), 'random_direction_rms_cos': float(1 / np.sqrt(d)),
       'mean_abs_along': float(np.abs(along).mean()), 'mean_ortho': float(ortho.mean()),
       'margin_sd_actual_rms_all': noise_margin_rms,
       'margin_sd_from_along_gradient_rms_all': rms_all(lin),
       'corr_linear_vs_actual_change': float(np.corrcoef(lin.ravel(), act.ravel())[0, 1]),
       'rms_actual_change_hit': float(np.sqrt((act ** 2).mean())), 'rms_linear_change_hit': float(np.sqrt((lin ** 2).mean())),
       'clean_margin_sd': sd_c,
       # decomposition of noise-to-signal: NSR = (rms dp / rms signal p) * (gain on noise / gain on signal)
       'rms_dp_all': rms_dp_all, 'rms_clean_pooled_centered': rms_p_signal,
       'relative_change_at_pooled': rms_dp_all / rms_p_signal,
       'gain_noise_margin_per_unit_dp': noise_margin_rms / rms_dp_all,
       'gain_signal_margin_per_unit_p': sd_c / rms_p_signal,
       'gain_ratio_noise_over_signal': (noise_margin_rms / rms_dp_all) / (sd_c / rms_p_signal),
       'nsr_all': noise_margin_rms / sd_c}
# shape of the change and of the clean pooled vectors: participation ratio and share of the top PCs
def spectrum(X):
    X = X - X.mean(0, keepdims=True)
    ev = np.linalg.svd(X, compute_uv=False) ** 2
    return {'participation_ratio': float(ev.sum() ** 2 / (ev ** 2).sum()), 'top1_share': float(ev[0] / ev.sum()), 'top5_share': float(ev[:5].sum() / ev.sum())}
res['dp_spectrum'] = spectrum(dP.reshape(-1, d)[::3])
res['clean_pooled_spectrum_all_samples'] = spectrum(P)
# gradient alignment with the signal: cos between g_i and the centered clean pooled vector
pch = pc[hit]
res['mean_cos_grad_vs_centered_clean_pooled'] = float((np.einsum('nd,nd->n', g, pch) / (gn * np.linalg.norm(pch, axis=1))).mean())
# mean change vector over draws: systematic part vs draw-to-draw part
res['norm_mean_dp_over_draws'] = float(np.linalg.norm(dP.mean(0), axis=1).mean())
res['norm_dp_minus_mean_over_draws'] = float(np.linalg.norm(dP - dP.mean(0)[None], axis=2).mean())
json.dump(res, open('%s/readout_M%d.json' % (OUT, no), 'w'), indent=1)
print(no, {k: round(v, 4) for k, v in res.items() if isinstance(v, float) and k in ('nsr_all', 'relative_change_at_pooled', 'gain_ratio_noise_over_signal', 'mean_abs_along_over_mean_dp_norm')})
