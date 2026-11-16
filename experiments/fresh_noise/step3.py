"""Step 3: T3 (attention to untrained tokens, layer 0 of the message encoder) and T4 (per-head change).
Usage: step3.py MODEL_NO. Writes out/t3t4_M{n}.json.

Message stage: x = [CLS ; embedMsg(tokens)] + position; transformerMsg (post-norm layers, final LayerNorm);
feature = output at position 0. Layer 0 is computed by hand:
  a = self_attn(x,x,x)         (need_weights=True, per head weights)
  x1 = norm1(x + a);  out = norm2(x1 + ff(x1))
The hand forward is checked against the cached margins before anything is reported.
T4: attention output at the CLS query = out_proj(concat_h z_h) = sum_h Wo_h z_h + bo, so a per-head
split of this vector is exact (linear). After norm1/FF/layer 1 it is not linear: no split there."""
import json, sys, time
import torch.nn.functional as F
from common import *

no = int(sys.argv[1])
s = get_session(no)
nz = Noiser(s)
model = s.model
enc = model.transformerMsg
layer0 = enc.layers[0]
H = model.nhead
d = nz.orig_rows.shape[1]
dh = d // H
cache = np.load('%s/logits_M%d.npz' % (CACHE, no))
hit = nz.hit
pad = s.pad_id
win = msg_window(s)
L = win.shape[1] + 1
BS = 32


def embed_in(rows):
    """Input of layer 0 for the given samples."""
    tok = torch.from_numpy(np.ascontiguousarray(s.b['msg'][rows][:, :L - 1])).long()
    x = model.embedMsg(tok)
    x = torch.cat((model.embedClassM.expand(x.shape[0], -1, -1), x), dim=1)
    return tok, model.embedPositionMsg + x


def layer0_manual(x):
    """Returns (layer-0 output, attention weights (B,H,L,L), per-head CLS contributions (B,H,d))."""
    a, w = layer0.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
    x1 = layer0.norm1(x + a)
    ff = layer0.linear2(layer0.activation(layer0.linear1(x1)))
    out = layer0.norm2(x1 + ff)
    # per-head split of the CLS attention output (before out_proj bias): z_h = w_h @ V_h
    Wi = layer0.self_attn.in_proj_weight; bi = layer0.self_attn.in_proj_bias
    v = F.linear(x, Wi[2 * d:], bi[2 * d:]).view(x.shape[0], L, H, dh)          # (B,L,H,dh)
    z = torch.einsum('bhl,blhe->bhe', w[:, :, 0, :], v)                            # CLS query only
    Wo = layer0.self_attn.out_proj.weight.view(d, H, dh)                           # (d,H,dh)
    contrib = torch.einsum('dhe,bhe->bhd', Wo, z)                                  # (B,H,d)
    return out, w, contrib, a[:, 0]


def full_feature(rows):
    """Manual layer 0 + standard remaining layers + final norm -> message feature, and extras."""
    feats, contribs, attn_cls = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), BS):
            r = rows[i:i + BS]
            tok, x = embed_in(r)
            out, w, contrib, a0 = layer0_manual(x)
            h = out
            for lay in enc.layers[1:]:
                h = lay(h)
            if enc.norm is not None:
                h = enc.norm(h)
            feats.append(h[:, 0].numpy().copy())
            contribs.append(contrib.numpy().copy())
            attn_cls.append(a0.numpy().copy())
            att_stats.append(attention_stats(w, tok))
    return np.concatenate(feats), np.concatenate(contribs), np.concatenate(attn_cls)


def attention_stats(w, tok):
    """Per sample: attention mass on untrained positions, for CLS query and for all queries.
    mass is summed over key positions; shapes (B,H)."""
    isU = torch.from_numpy(np.isin(tok.numpy(), nz.U))                               # (B,L-1)
    isU = torch.cat((torch.zeros(len(tok), 1, dtype=torch.bool), isU), dim=1)         # CLS is key 0
    real = torch.cat((torch.ones(len(tok), 1, dtype=torch.bool), tok != pad), dim=1)  # CLS + non-pad
    wc = w[:, :, 0, :]                                                                # CLS query (B,H,L)
    wa = w.mean(2)                                                                    # mean over queries
    f = isU.float()[:, None, :]
    r = real.float()[:, None, :]
    return {'cls_mass_U': (wc * f).sum(-1).numpy(), 'all_mass_U': (wa * f).sum(-1).numpy(),
            'cls_mass_real': (wc * r).sum(-1).numpy(), 'all_mass_real': (wa * r).sum(-1).numpy(),
            'share_U_all': isU.float().mean(1).numpy(), 'share_U_real': ((isU & real).float().sum(1) / real.float().sum(1)).numpy()}


att_stats = []
t0 = time.time()
feat, contrib_orig, attn_orig = full_feature(hit)
fm = nz.base_fm.copy(); fm[hit] = feat
m_manual = margin(s.head_from(s.base_feats['twin_b'], s.base_feats['twin_a'], fm))
gap = float(np.abs(m_manual - cache['orig']).max())
gap_hit = float(np.abs(m_manual[hit] - cache['orig'][hit]).max())
res = {'model': no, 'n_heads': H, 'n_layers': len(enc.layers), 'n_hit': int(len(hit)),
       'manual_forward_max_abs_gap_all': gap, 'manual_forward_max_abs_gap_hit': gap_hit}
ok = gap < 1e-3
res['manual_forward_ok'] = bool(ok)

if ok:
    S = {k: np.concatenate([a[k] for a in att_stats]) for k in att_stats[0]}
    t3 = {}
    sh_all, sh_real = S['share_U_all'].mean(), S['share_U_real'].mean()
    t3['share_U_of_all_positions'] = float(sh_all)
    t3['share_U_of_real_positions'] = float(sh_real)
    for q in ('cls', 'all'):
        mU = S['%s_mass_U' % q]                      # (n,H)
        mReal = S['%s_mass_real' % q]
        t3['%s_query' % q] = {
            'mass_U_mean_over_heads': float(mU.mean()),
            'mass_U_per_head': mU.mean(0).tolist(),
            'ratio_to_share_all_positions': float(mU.mean() / sh_all),
            'ratio_per_head': (mU.mean(0) / sh_all).tolist(),
            'mass_on_real_positions': float(mReal.mean()),
            # among real (non-pad) positions only: mass on U relative to mass on real positions vs their share
            'ratio_real_only': float((mU.sum() / mReal.sum()) / sh_real),
        }
    res['T3'] = t3

    # --- T4: per-head change of the CLS attention output, orig vs zero rows, and fresh draws vs zero
    def contribs_for(rows_value):
        ids = torch.from_numpy(nz.U)
        with torch.no_grad():
            nz.W.data[ids] = rows_value
            try:
                global att_stats
                keep = att_stats; att_stats = []
                _, c, _ = full_feature(hit)
                att_stats = keep
            finally:
                nz.W.data[ids] = nz.orig_rows
        return c

    def t4_stats(delta):
        """delta (n,H,d). Mean pairwise cosine between heads, norm(sum)/sum(norms)."""
        n = delta.shape[0]
        nrm = np.linalg.norm(delta, axis=2)                                            # (n,H)
        u = delta / np.maximum(nrm[:, :, None], 1e-12)
        G = np.einsum('nhd,ngd->nhg', u, u)
        iu = np.triu_indices(H, 1)
        cos = G[:, iu[0], iu[1]]                                                       # (n,H(H-1)/2)
        ratio = np.linalg.norm(delta.sum(1), axis=1) / np.maximum(nrm.sum(1), 1e-12)
        ok_n = nrm.sum(1) > 1e-9
        out = {'mean_pairwise_cosine': float(cos[ok_n].mean()), 'ratio_norm_sum_over_sum_norms': float(ratio[ok_n].mean()),
               'ratio_reference_orthogonal_1_over_sqrt_H': float(1 / np.sqrt(H)),
               'mean_head_change_norm': float(nrm.mean()), 'norm_of_summed_change': float(np.linalg.norm(delta.sum(1), axis=1).mean())}
        return out

    if H >= 2:
        c_zero = contribs_for(torch.zeros_like(nz.orig_rows))
        t4 = {'orig_vs_zero': t4_stats(contrib_orig - c_zero)}
        fr = [t4_stats(contribs_for(nz.noise(k)) - c_zero) for k in range(5)]
        t4['fresh_vs_zero_5_draws_mean'] = {k: float(np.mean([f[k] for f in fr])) for k in fr[0]}
        t4['caveat'] = ('Exact split only for the CLS attention output of layer 0, before residual, LayerNorm and FF. '
                        + ('The model has 2 layers: layer 1 follows, so this is not the pooled representation.' if len(enc.layers) > 1 else 'One layer: LayerNorm and FF still follow this vector.'))
        res['T4'] = t4
res['seconds'] = round(time.time() - t0, 1)
json.dump(res, open('%s/t3t4_M%d.json' % (OUT, no), 'w'), indent=1)
print(no, {k: res[k] for k in ('manual_forward_max_abs_gap_all', 'manual_forward_ok', 'seconds')})
