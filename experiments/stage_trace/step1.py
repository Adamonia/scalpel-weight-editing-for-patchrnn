"""Step 1: stage trace of the message branch (S1), attention shape (S3), clean signal (S5),
sample composition (S6) and the check of the zeroed model against the cached margins.
Usage: step1.py MODEL_NO.  Writes out/trace_M{n}.json and tmp/arrays_M{n}.npz (for step2).

No gradient training and no autograd here (torch.no_grad()). The message branch is run by hand
(own attention code, so the head count can be changed for the head-count check):
  x0 = pos + [CLS ; embedMsg(tokens)]
  per layer (post-norm): a = out_proj(attention(x)); n1 = norm1(x + a); ff = lin2(gelu(lin1(n1))); n2 = norm2(n1 + ff)
  pooled = final_norm(n2 at position 0)
  classifier on cat(twin_b, twin_a, pooled): LN(384) -> Linear -> LN(384) -> Linear(2); margin = logit1 - logit0.
Clean model = untrained rows set to 0. Noise = untrained rows replaced by N(0,1) draws, seeds 0..9
(same draw rule as exth: RandomState(seed).randn(n_rows, d)).
Only samples with at least one untrained token ("hit", 4328) can change; the others keep clean = noisy.
"""
import json, sys, time
import torch.nn.functional as F
from common import *

no = int(sys.argv[1])
K = 10
BS = 32
t0 = time.time()
s = get_session(no)
model = s.model.eval()
nz = Noiser(s)
enc = model.transformerMsg
layers = list(enc.layers)
NL = len(layers)
H_REG = int(model.nhead)
cache = np.load('%s/logits_M%d.npz' % (CACHE, no))
y = s.y.astype(bool)
N = s.n
hit = nz.hit
U = nz.U
win = msg_window(s)
L = win.shape[1] + 1
d = nz.orig_rows.shape[1]
W0 = model.embedMsg.weight.data
isU_all = np.isin(win, U)                      # (N, L-1)
cnt_all = isU_all.sum(1)
base = {k: s.base_feats[k] for k in ('twin_b', 'twin_a', 'msg')}


# ---------------------------------------------------------------- manual forward
def attention(layer, x, H, cls_only):
    """Multi-head self-attention by hand. Returns out_proj output a, weights w (B,H,Lq,L), values v (B,H,L,dh)."""
    sa = layer.self_attn
    Wi, bi = sa.in_proj_weight, sa.in_proj_bias
    B, Lk, _ = x.shape
    dh = d // H
    xq = x[:, :1] if cls_only else x
    q = F.linear(xq, Wi[:d], bi[:d]).view(B, xq.shape[1], H, dh).transpose(1, 2)
    k = F.linear(x, Wi[d:2 * d], bi[d:2 * d]).view(B, Lk, H, dh).transpose(1, 2)
    v = F.linear(x, Wi[2 * d:], bi[2 * d:]).view(B, Lk, H, dh).transpose(1, 2)
    w = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5, dim=-1)
    z = (w @ v).transpose(1, 2).reshape(B, xq.shape[1], d)
    return sa.out_proj(z), w, v


def msg_forward(tok, rows, H=H_REG):
    """tok (B, L-1) long. rows: value of the untrained embedding rows (len(U), d).
    Returns dict stage -> tensor (B, Lq, d) and dict of CLS attention rows per layer (B,H,L), layer-0 w and v."""
    Wc = W0.clone()
    Wc[torch.from_numpy(U)] = rows
    e = F.embedding(tok, Wc)
    x = model.embedPositionMsg + torch.cat((model.embedClassM.expand(e.shape[0], -1, -1), e), dim=1)
    st = {'emb': x}
    rowsw = {}
    extra = {}
    for li, layer in enumerate(layers):
        last = li == NL - 1 and li > 0            # last layer of a deep model: only the CLS output is needed
        a, w, v = attention(layer, x, H, last)
        xr = x[:, :1] if last else x
        n1 = layer.norm1(xr + a)
        ff = layer.linear2(layer.activation(layer.linear1(n1)))
        n2 = layer.norm2(n1 + ff)
        st['L%d.a' % li], st['L%d.n1' % li], st['L%d.ff' % li], st['L%d.n2' % li] = a, n1, ff, n2
        rowsw[li] = w[:, :, 0, :]
        if li == 0:
            extra['w'] = w[:, :, 0:1, :]; extra['v'] = v
        x = n2
    pooled = enc.norm(x[:, 0]) if enc.norm is not None else x[:, 0]
    st['pool'] = pooled[:, None, :]
    return st, rowsw, extra


def head_forward(fb, fa, p):
    """Classifier by hand. Returns stage dict (c0, c2, c3 as (B,384)) and the margin (B,)."""
    c = model.classifier
    z0 = c[0](torch.cat((fb, fa, p), 1))
    z2 = c[2](z0)
    z3 = c[3](z2)
    o = c[4](z3)
    return {'c0': z0, 'c2': z2, 'c3': z3}, o[:, 1] - o[:, 0]


def tt(a):
    return torch.from_numpy(np.ascontiguousarray(a))


# ---------------------------------------------------------------- check 1: clean margins vs cache (all samples)
zero_rows = torch.zeros_like(nz.orig_rows)
clean_p_all = base['msg'].copy()
with torch.no_grad():
    for i in range(0, len(hit), BS):
        r = hit[i:i + BS]
        st, _, _ = msg_forward(tt(s.b['msg'][r][:, :L - 1]).long(), zero_rows)
        clean_p_all[r] = st['pool'][:, 0].numpy()
    m_clean_all = np.concatenate([head_forward(tt(base['twin_b'][i:i + 512]), tt(base['twin_a'][i:i + 512]),
                                               tt(clean_p_all[i:i + 512]))[1].numpy() for i in range(0, N, 512)]).astype(np.float64)
gap_zero = float(np.abs(m_clean_all - cache['untrained_zero'].astype(np.float64)).max())
res = {'model': no, 'n_layers': NL, 'n_heads_registry': H_REG, 'n_hit': int(len(hit)), 'n_untrained_rows': int(len(U)),
       'gap_zero_vs_cache_all_samples': gap_zero}

# ---------------------------------------------------------------- check 2: head count by function (clean model, 256 hit samples)
sub = hit[:256]
hc = {}
with torch.no_grad():
    for Hc in (1, 2, 4, 8, 16):
        pp = []
        for i in range(0, len(sub), BS):
            st, _, _ = msg_forward(tt(s.b['msg'][sub[i:i + BS]][:, :L - 1]).long(), zero_rows, H=Hc)
            pp.append(st['pool'][:, 0])
        pp = torch.cat(pp)
        _, mm = head_forward(tt(base['twin_b'][sub]), tt(base['twin_a'][sub]), pp)
        hc[str(Hc)] = float(np.abs(mm.numpy() - m_clean_all[sub]).max())
res['head_count_check_max_abs_margin_gap'] = hc

# ---------------------------------------------------------------- main pass over hit samples: clean + K draws
nh = len(hit)
draws = [nz.noise(k) for k in range(K)]
STG = (['emb'] + sum([['L%d.a' % l, 'L%d.n1' % l, 'L%d.ff' % l, 'L%d.n2' % l] for l in range(NL)], []) + ['pool', 'c0', 'c2', 'c3', 'margin'])
sdn = {k: np.zeros(nh) for k in STG}             # per sample: norm of the sd vector over the K draws (noise around its own mean)
dn = {k: np.zeros((K, nh)) for k in STG}            # norm of change, CLS position (or scalar for margin)
cn = {k: [] for k in STG}                           # clean CLS vectors
dseq = {k: np.zeros((K, nh)) for k in STG if k == 'emb' or k.startswith('L0.')}   # sequence-level Frobenius norms
cseq = {k: np.zeros(nh) for k in dseq}
dP = np.zeros((K, nh, d), np.float32)
m_noise = np.zeros((K, nh))
m_clean_hit = np.zeros(nh)
# attention rows: stats per (condition, layer): entropy, max weight, mass on U, max single U weight   shape (n,H)
ATT = {c: {li: {q: [] for q in ('ent', 'maxw', 'massU', 'maxU')} for li in range(NL)} for c in range(K + 1)}   # c=0 clean, 1..K draws
# layer-0 CLS attention: value path vs weight path
PATH = {'dA': np.zeros((K, nh)), 'val': np.zeros((K, nh)), 'wt': np.zeros((K, nh)), 'cos': np.zeros((K, nh))}
Wo0 = layers[0].self_attn.out_proj.weight


def att_stats(w, isU):
    """w (B,H,L) CLS attention row; isU (B,L) bool. Returns dict of (B,H) arrays."""
    ent = -(w * torch.log(w.clamp_min(1e-30))).sum(-1)
    mx = w.max(-1).values
    f = isU[:, None, :].float()
    return {'ent': ent.numpy(), 'maxw': mx.numpy(), 'massU': (w * f).sum(-1).numpy(), 'maxU': (w * f).max(-1).values.numpy()}


with torch.no_grad():
    for i in range(0, nh, BS):
        r = hit[i:i + BS]
        sl = slice(i, i + len(r))
        tok = tt(s.b['msg'][r][:, :L - 1]).long()
        isU = torch.cat((torch.zeros(len(r), 1, dtype=torch.bool), tt(isU_all[r])), 1)
        fb, fa = tt(base['twin_b'][r]), tt(base['twin_a'][r])
        stc, rw, ex0 = msg_forward(tok, zero_rows)
        hc_, mc = head_forward(fb, fa, stc['pool'][:, 0])
        stc.update({k: v[:, None, :] for k, v in hc_.items()})
        stc['margin'] = mc[:, None, None]
        m_clean_hit[sl] = mc.numpy()
        for k in STG:
            cn[k].append(stc[k][:, 0].numpy().reshape(len(r), -1).copy())   # copy: a view would keep the whole batch tensor alive
        for k in cseq:
            cseq[k][sl] = stc[k].flatten(1).norm(dim=1).numpy()
        for li in range(NL):
            for q, v in att_stats(rw[li], isU).items():
                ATT[0][li][q].append(v)
        bucket = {q: [] for q in STG}
        for k in range(K):
            stn, rwn, exn = msg_forward(tok, draws[k])
            hn, mn = head_forward(fb, fa, stn['pool'][:, 0])
            stn.update({q: v[:, None, :] for q, v in hn.items()})
            stn['margin'] = mn[:, None, None]
            m_noise[k, sl] = mn.numpy()
            for q in STG:
                bucket[q].append(stn[q][:, 0].reshape(len(r), -1))
                dn[q][k, sl] = (stn[q][:, 0] - stc[q][:, 0]).reshape(len(r), -1).norm(dim=1).numpy()
            for q in dseq:
                dseq[q][k, sl] = (stn[q] - stc[q]).flatten(1).norm(dim=1).numpy()
            dP[k, sl] = (stn['pool'][:, 0] - stc['pool'][:, 0]).numpy()
            for li in range(NL):
                for q, v in att_stats(rwn[li], isU).items():
                    ATT[k + 1][li][q].append(v)
            # layer-0 CLS attention output: delta = out_proj(w' (v'-v)) + out_proj((w'-w) v), exact split
            zv = (exn['w'] @ (exn['v'] - ex0['v'])).transpose(1, 2).reshape(len(r), d)
            zw = ((exn['w'] - ex0['w']) @ ex0['v']).transpose(1, 2).reshape(len(r), d)
            pv, pw = zv @ Wo0.T, zw @ Wo0.T
            PATH['dA'][k, sl] = (stn['L0.a'][:, 0] - stc['L0.a'][:, 0]).norm(dim=1).numpy()
            PATH['val'][k, sl] = pv.norm(dim=1).numpy()
            PATH['wt'][k, sl] = pw.norm(dim=1).numpy()
            PATH['cos'][k, sl] = F.cosine_similarity(pv, pw, dim=1).numpy()
        for q in STG:   # sd over draws of each coordinate, then norm of that sd vector
            sdn[q][sl] = torch.stack(bucket[q]).var(dim=0, unbiased=True).sum(1).sqrt().numpy()
cn = {k: np.concatenate(v) for k, v in cn.items()}

# margins of all samples: non-hit samples have noisy = clean
M_noise_all = np.tile(m_clean_all[None], (K, 1))
M_noise_all[:, hit] = m_noise
assert np.abs(m_clean_hit - m_clean_all[hit]).max() < 1e-4

# ---------------------------------------------------------------- S1 stage trace
trace = []
prev = None
for k in STG:
    cv = cn[k]
    raw_norm = float(np.linalg.norm(cv, axis=1).mean())
    cen_norm = float(np.linalg.norm(cv - cv.mean(0, keepdims=True), axis=1).mean())
    dnorm = float(dn[k].mean())
    row = {'stage': k, 'clean_norm_mean': raw_norm, 'clean_centered_norm_mean': cen_norm, 'delta_norm_mean': dnorm,
           'ratio_raw': dnorm / raw_norm, 'ratio_centered': dnorm / cen_norm}
    row['sd_norm_mean'] = float(sdn[k].mean())
    row['nsr_sd_mean_centered'] = float(sdn[k].mean() / cen_norm)
    row['nsr_sd_rms_centered'] = float(np.sqrt((sdn[k] ** 2).mean()) / np.sqrt((np.linalg.norm(cv - cv.mean(0, keepdims=True), axis=1) ** 2).mean()))
    if k in dseq:
        row['seq_clean_norm_mean'] = float(cseq[k].mean()); row['seq_delta_norm_mean'] = float(dseq[k].mean())
        row['seq_ratio'] = row['seq_delta_norm_mean'] / row['seq_clean_norm_mean']
    if prev is not None and prev['ratio_centered'] > 0 and dnorm > 0:
        row['amp_raw_vs_prev'] = row['ratio_raw'] / prev['ratio_raw']
        row['amp_centered_vs_prev'] = row['ratio_centered'] / prev['ratio_centered']
        row['amp_nsr_sd_rms_vs_prev'] = row['nsr_sd_rms_centered'] / prev['nsr_sd_rms_centered']
    if dnorm > 0:
        prev = row
    trace.append(row)
res['S1_trace_hit_samples'] = trace

# layer-0 CLS attention output: which path carries the change
res['S1_layer0_cls_attention_paths'] = {
    'mean_norm_delta_a': float(PATH['dA'].mean()), 'mean_norm_value_path': float(PATH['val'].mean()),
    'mean_norm_weight_path': float(PATH['wt'].mean()), 'mean_cos_value_weight_path': float(PATH['cos'].mean()),
    'value_path_over_delta': float(PATH['val'].mean() / PATH['dA'].mean()), 'weight_path_over_delta': float(PATH['wt'].mean() / PATH['dA'].mean())}

# branch contribution to margin variance (all samples, clean model). The head has LayerNorm: shares need not add up.
with torch.no_grad():
    fb_all, fa_all, p_all = tt(base['twin_b']), tt(base['twin_a']), tt(clean_p_all)

    def hm(b, a, p):
        return np.concatenate([head_forward(b[i:i + 1024], a[i:i + 1024], p[i:i + 1024])[1].numpy() for i in range(0, N, 1024)]).astype(np.float64)
    ones = torch.ones(N, 1)
    m_msg_only = hm(ones * fb_all.mean(0), ones * fa_all.mean(0), p_all)
    m_code_only = hm(fb_all, fa_all, ones * p_all.mean(0))
    m_twinb_only = hm(fb_all, ones * fa_all.mean(0), ones * p_all.mean(0))
    m_twina_only = hm(ones * fb_all.mean(0), fa_all, ones * p_all.mean(0))
vm = m_clean_all.var()
res['S1_branch_variance'] = {'var_margin_clean': float(vm), 'var_msg_varies_only': float(m_msg_only.var()),
                             'var_code_varies_only': float(m_code_only.var()), 'var_twin_b_only': float(m_twinb_only.var()),
                             'var_twin_a_only': float(m_twina_only.var()),
                             'share_msg': float(m_msg_only.var() / vm), 'share_code': float(m_code_only.var() / vm),
                             'corr2_full_vs_msg_only': float(np.corrcoef(m_clean_all, m_msg_only)[0, 1] ** 2),
                             'corr2_full_vs_code_only': float(np.corrcoef(m_clean_all, m_code_only)[0, 1] ** 2),
                             'sd_msg_only_clean': float(m_msg_only.std())}

# ---------------------------------------------------------------- S5 clean signal, noise sd
sd_c = float(m_clean_all.std())
per_sd = M_noise_all.std(axis=0, ddof=1)                 # per-sample sd over the K draws
noise_rms = float(np.sqrt((per_sd ** 2).mean()))
dm = M_noise_all - m_clean_all[None]
res['S5'] = {'clean_margin_sd': sd_c, 'clean_margin_mean': float(m_clean_all.mean()),
             'f1_clean': f1_pct(y, m_clean_all), 'auc_clean': 100 * roc_auc_score(y, m_clean_all),
             'noise_sd_rms_all': noise_rms, 'noise_sd_rms_hit': float(np.sqrt((per_sd[hit] ** 2).mean())),
             'nsr_all': noise_rms / sd_c, 'nsr_hit_over_sd_clean_all': float(np.sqrt((per_sd[hit] ** 2).mean()) / sd_c),
             'rms_change_all': float(np.sqrt((dm ** 2).mean())), 'mean_change_over_draws_rms_all': float(np.sqrt((dm.mean(0) ** 2).mean())),
             'sd_clean_hit_samples': float(m_clean_all[hit].std()),
             'nsr_over_msg_only_sd': noise_rms / float(m_msg_only.std()),
             'f1_fresh_mean': float(np.mean([f1_pct(y, m) for m in M_noise_all])),
             'f1_fresh_sd': float(np.std([f1_pct(y, m) for m in M_noise_all], ddof=1))}
res['S5']['f1_gap_clean_minus_fresh'] = res['S5']['f1_clean'] - res['S5']['f1_fresh_mean']

# ---------------------------------------------------------------- S6 sample composition
S6 = {}
for name, sel in (('1', cnt_all == 1), ('2', cnt_all == 2), ('3+', cnt_all >= 3)):
    sel_h = sel[hit]
    ps = per_sd[sel]
    S6[name] = {'n': int(sel.sum()), 'noise_sd_rms': float(np.sqrt((ps ** 2).mean())), 'noise_sd_median': float(np.median(ps)),
                'nsr_over_sd_clean_all': float(np.sqrt((ps ** 2).mean()) / sd_c),
                'sd_clean_in_group': float(m_clean_all[sel].std()),
                'nsr_over_sd_clean_in_group': float(np.sqrt((ps ** 2).mean()) / m_clean_all[sel].std()),
                'delta_pool_norm_mean': float(np.linalg.norm(dP[:, sel_h], axis=2).mean()),
                'mean_untrained_tokens': float(cnt_all[sel].mean())}
res['S6'] = S6
res['S6_counts'] = {'max': int(cnt_all.max()), 'mean_hit': float(cnt_all[hit].mean())}

# ---------------------------------------------------------------- S3 attention shape (CLS query row, per layer and head)
def dist(v):
    v = np.asarray(v).ravel()
    return {'mean': float(v.mean()), 'sd': float(v.std()), 'p90': float(np.percentile(v, 90)), 'p99': float(np.percentile(v, 99))}


S3 = {'n_keys': L, 'max_entropy_nats': float(np.log(L)), 'untrained_share_of_keys_mean': float(isU_all[hit].sum(1).mean() / L),
      'chance_mass_on_U_mean': float(cnt_all[hit].mean() / L)}
for li in range(NL):
    per = {}
    for cond, ids in (('clean', [0]), ('fresh', list(range(1, K + 1)))):
        A = {q: np.concatenate([np.concatenate(ATT[c][li][q]) for c in ids]) for q in ('ent', 'maxw', 'massU', 'maxU')}   # (n*len(ids), H)
        per[cond] = {'entropy_mean': float(A['ent'].mean()), 'entropy_per_head': A['ent'].mean(0).tolist(),
                     'maxw_mean': float(A['maxw'].mean()), 'maxw_per_head': A['maxw'].mean(0).tolist(),
                     'massU_dist_over_samples_and_heads': dist(A['massU']),
                     'massU_per_head_mean': A['massU'].mean(0).tolist(),
                     'massU_per_sample_mean_over_heads_dist': dist(A['massU'].mean(1)),
                     'max_single_U_weight_dist_over_samples_and_heads': dist(A['maxU']),
                     'max_single_U_weight_per_sample_max_over_heads_dist': dist(A['maxU'].max(1))}
    S3['layer%d' % li] = per
res['S3'] = S3
res['seconds'] = round(time.time() - t0, 1)
json.dump(res, open('%s/trace_M%d.json' % (OUT, no), 'w'), indent=1)
np.savez('%s/tmp/arrays_M%d.npz' % (EXT, no), dP=dP, clean_p_all=clean_p_all, m_clean_all=m_clean_all, M_noise_all=M_noise_all,
         hit=hit, cnt=cnt_all, m_msg_only=m_msg_only, cn_pool=cn['pool'], dn_pool=dn['pool'], per_sd=per_sd)
print(no, 'gap_zero', gap_zero, 'headcheck', hc, 'sec', res['seconds'])
