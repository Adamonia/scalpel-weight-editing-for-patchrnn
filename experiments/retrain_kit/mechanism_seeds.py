"""Mechanism analysis of the message branch for every finished run. No training.

For each run (test set, clean model = untrained message rows set to 0):
  3a  analytic noise prediction at the class token (layer 0) and the measured noise (K fresh N(0,1) draws)
  3b  mediators: attention entropy, attention mass on untrained tokens, value norms, heads, layers
  3c  causal tests at inference: attention temperature, head ablation, uniform attention (layer 0)
Writes mechanism.csv (one row per run and setting) and mechanism_summary.json.
The settings of 3c are outside what the models were trained on. They show what the mechanism does
when it is changed by hand. They do not describe a trained model of that kind.
Example: python mechanism_seeds.py --data dataset.npz --runs runs --results results.csv --out-dir .
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyse_seeds import untrained_ids
from common import f1_score, fit_val_rows, load_data, pick_device, read_json
from model import ARCHS, MSG_MAX_LEN, build_arch

K_DRAWS = 5
TEMPS = [0.5, 1.0, 2.0, 4.0]
N_SUBSETS = 3
CHECK_N = 64
# Limits for the self-check. GPU float32 gives gaps near 1e-3; a real fault gives gaps above 0.1.
CHECK_TOL = 1e-2
REC_TOL = 1e-2
WINDOW = MSG_MAX_LEN - 1


# ----------------------------------------------------------------- manual forward of the message branch
def make_context(model, data, fit_rows, device, k_draws):
    """Tables and noise. Clean table = untrained rows 0. Noise draws are the same as in analyse_seeds.py."""
    w_orig = model.embedMsg.weight.data
    u = untrained_ids(data['msg'], fit_rows, w_orig.shape[0])
    uidx = torch.full((w_orig.shape[0],), -1, dtype=torch.long)
    uidx[torch.from_numpy(u)] = torch.arange(len(u))
    w_clean = w_orig.clone()
    w_clean[torch.from_numpy(u).to(device)] = 0.0
    d = w_orig.shape[1]
    noise = np.stack([np.random.RandomState(k).randn(len(u), d).astype(np.float32) for k in range(k_draws)])
    return {'w_orig': w_orig, 'w_clean': w_clean, 'uidx': uidx.to(device), 'noise': torch.from_numpy(noise).to(device),
            'n_untrained': int(len(u))}


def attention(layer, x, n_heads, all_queries, temp=1.0, head_scale=None, uniform=False):
    """Self-attention by hand. Returns out_proj output (B,Lq,d), weights (B,H,Lq,L)."""
    sa = layer.self_attn
    w_in, b_in = sa.in_proj_weight, sa.in_proj_bias
    b, length, d = x.shape
    dh = d // n_heads
    xq = x if all_queries else x[:, :1]
    q = F.linear(xq, w_in[:d], b_in[:d]).view(b, xq.shape[1], n_heads, dh).transpose(1, 2)
    k = F.linear(x, w_in[d:2 * d], b_in[d:2 * d]).view(b, length, n_heads, dh).transpose(1, 2)
    v = F.linear(x, w_in[2 * d:], b_in[2 * d:]).view(b, length, n_heads, dh).transpose(1, 2)
    w = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5 * temp, dim=-1)
    if uniform:
        w = torch.full_like(w, 1.0 / length)
    z = w @ v
    if head_scale is not None:
        z = z * head_scale.view(1, n_heads, 1, 1)
    z = z.transpose(1, 2).reshape(b, xq.shape[1], d)
    return sa.out_proj(z), w


def msg_forward(model, ctx, tok, mode, n_heads, cfg=None, want_layer0=False):
    """Message branch by hand. tok: (B, WINDOW) long.
    mode: 'orig' (trained table), 'clean' (untrained rows 0) or k (clean table plus draw k on untrained rows).
    cfg: dict with temp, head_scale, uniform. They act on layer 0 only.
    Returns dict: a0 (layer-0 attention output at the class token), pooled, and with want_layer0 also w0 (B,H,L), x0 (B,L,d)."""
    cfg = cfg or {}
    enc = model.transformerMsg
    n_layers = len(enc.layers)
    e = F.embedding(tok, ctx['w_orig'] if mode == 'orig' else ctx['w_clean'])
    if isinstance(mode, int):
        ui = ctx['uidx'][tok]
        e = e + ctx['noise'][mode][ui.clamp(min=0)] * (ui >= 0).unsqueeze(-1).to(e.dtype)
    x = model.embedPositionMsg + torch.cat((model.embedClassM.expand(e.shape[0], -1, -1), e), dim=1)
    out = {}
    if want_layer0:
        out['x0'] = x
    for li, layer in enumerate(enc.layers):
        last = li == n_layers - 1
        kw = {}
        if li == 0:
            kw = {'temp': cfg.get('temp', 1.0), 'head_scale': cfg.get('head_scale'), 'uniform': cfg.get('uniform', False)}
        a, w = attention(layer, x, n_heads, all_queries=not last, **kw)
        xq = x if not last else x[:, :1]
        n1 = layer.norm1(xq + a)
        n2 = layer.norm2(n1 + layer.linear2(layer.activation(layer.linear1(n1))))
        if li == 0:
            out['a0'] = a[:, 0]
            if want_layer0:
                out['w0'] = w[:, :, 0]
        x = n2
    out['pooled'] = enc.norm(x[:, 0]) if enc.norm is not None else x[:, 0]
    return out


def margin_from(model, fb, fa, pooled):
    lg = model.head(fb, fa, pooled).float()
    return lg[:, 1] - lg[:, 0]


def to_t(arr, rows, device):
    return torch.from_numpy(np.ascontiguousarray(arr[rows]).astype(np.int64)).to(device)


# ----------------------------------------------------------------- weights
def value_maps(model, n_heads):
    """M_h = W_out[:, block_h] @ W_value[block_h, :] for every head. Returns M (H,d,d), the Gram matrix G (H,H),
    the spectral norm of each value block (H,), and the biases needed for the check."""
    layer = model.transformerMsg.layers[0]
    sa = layer.self_attn
    d = sa.in_proj_weight.shape[1]
    dh = d // n_heads
    w_value = sa.in_proj_weight.detach()[2 * d:]
    w_out = sa.out_proj.weight.detach()
    m = torch.stack([w_out[:, h * dh:(h + 1) * dh] @ w_value[h * dh:(h + 1) * dh] for h in range(n_heads)])
    flat = m.reshape(n_heads, -1).double()
    gram = (flat @ flat.T).float()
    vnorm = torch.stack([torch.linalg.matrix_norm(w_value[h * dh:(h + 1) * dh], ord=2) for h in range(n_heads)])
    const = sa.out_proj.bias.detach() + w_out @ sa.in_proj_bias.detach()[2 * d:]
    return m, gram, vnorm, const


# ----------------------------------------------------------------- checks
def check_manual_forward(model, ctx, data, rows, fb, fa, n_heads, device):
    """The manual forward must equal the model forward (margin). Also checks the M_h slicing on layer 0."""
    n = min(CHECK_N, len(rows))
    sub = np.arange(n)
    with torch.no_grad():
        b = {k: to_t(data[k], rows[sub], device) for k in ('tok_b', 'typ_b', 'tok_a', 'typ_a', 'msg')}
        ref = model(b['tok_b'], b['typ_b'], b['tok_a'], b['typ_a'], b['msg']).float()
        ref = (ref[:, 1] - ref[:, 0]).cpu().numpy()
        gap = 0.0
        gap_a = 0.0
        m, _, _, const = value_maps(model, n_heads)
        step = 8
        for s in range(0, n, step):
            tok = b['msg'][s:s + step, :WINDOW]
            o = msg_forward(model, ctx, tok, 'orig', n_heads, want_layer0=True)
            mine = margin_from(model, fb[s:s + step], fa[s:s + step], o['pooled']).cpu().numpy()
            gap = max(gap, float(np.abs(mine - ref[s:s + step]).max()))
            xbar = torch.einsum('bhj,bjc->bhc', o['w0'], o['x0'])
            rec = torch.einsum('hdc,bhc->bd', m, xbar) + const
            gap_a = max(gap_a, float((rec - o['a0']).abs().max()))
    print('self-check: manual forward gap %.2e, attention sum gap %.2e' % (gap, gap_a), flush=True)
    if gap >= CHECK_TOL:
        raise SystemExit('Manual forward differs from the model forward: max abs gap %.2e (limit %.0e).' % (gap, CHECK_TOL))
    if gap_a >= REC_TOL:
        raise SystemExit('Layer-0 attention output differs from the sum of M_h terms: gap %.2e (limit %.0e). '
                         'Check the slicing of the value block.' % (gap_a, REC_TOL))
    return gap, gap_a


# ----------------------------------------------------------------- one setting
def run_setting(model, ctx, tok, fb, fa, y, n_heads, cfg, batch, base=False, extra=None):
    """Clean F1, noise ratios and margin noise for one setting. With base=True also the analytic part and mediators.
    tok: (N, WINDOW) tensor of test messages."""
    n = tok.shape[0]
    d = fb.shape[1]
    uidx = ctx['uidx']
    is_u = uidx[tok] >= 0
    hit = torch.where(is_u.any(1))[0]
    nh = len(hit)
    a_clean = torch.zeros(n, d, device=tok.device)
    p_clean = torch.zeros(n, d, device=tok.device)
    acc = {'ent': [], 'mass': [], 'share': [], 'P': [], 'same': [], 'Pg': [], 'vp': [], 'sum_a2': 0.0, 'n_pos': 0}
    if base:
        m, gram, _, _ = value_maps(model, n_heads)
        diag = torch.diagonal(gram)
    for s in range(0, n, batch):
        sl = slice(s, s + batch)
        o = msg_forward(model, ctx, tok[sl], 'clean', n_heads, cfg, want_layer0=base)
        a_clean[sl], p_clean[sl] = o['a0'], o['pooled']
        if base:
            w = o['w0']                                              # (B,H,L)
            acc['ent'].append((-(w * torch.log(w.clamp_min(1e-30))).sum(-1)).mean(1).cpu())     # (B,) mean over heads
            u = F.pad(is_u[sl], (1, 0))                              # (B,L), the class position is never untrained
            uf = u.to(w.dtype)
            has = u.any(1)
            acc['mass'].append(((w * uf[:, None, :]).sum(-1).mean(1))[has].cpu())
            acc['share'].append(uf.mean(1)[has].cpu())
            # analytic power per position: a_j^T G a_j. same-head part: sum_h a_h^2 G_hh.
            qf = torch.einsum('bhj,hg,bgk->bjk', w, gram, w)
            pos = torch.diagonal(qf, dim1=1, dim2=2)                 # (B,L)
            same = ((w ** 2) * diag.view(1, -1, 1)).sum(1)
            acc['P'].append((pos * uf).sum(1)[has].cpu())
            acc['same'].append((same * uf).sum(1)[has].cpu())
            tk = F.pad(tok[sl], (1, 0))
            pair = (tk[:, :, None] == tk[:, None, :]) & u[:, :, None] & u[:, None, :]
            acc['Pg'].append((qf * pair.to(qf.dtype)).sum((1, 2))[has].cpu())
            # Measured value path: attention weights held fixed at their clean values, real noise draws.
            ui = uidx[tok[sl]]
            z = F.pad(ctx['noise'][:, ui.clamp(min=0)] * (ui >= 0).unsqueeze(0).unsqueeze(-1).to(w.dtype), (0, 0, 1, 0))
            delta = torch.einsum('hdc,kbhc->kbd', m, torch.einsum('bhj,kbjc->kbhc', w, z))
            acc['vp'].append(delta.double().var(dim=0, unbiased=True).sum(-1)[has].cpu())
            acc['sum_a2'] += float(((w ** 2).sum(1) * uf).sum())
            acc['n_pos'] += int(uf.sum())
    margin_clean = torch.cat([margin_from(model, fb[s:s + batch], fa[s:s + batch], p_clean[s:s + batch])
                              for s in range(0, n, batch)]).cpu().numpy().astype(np.float64)
    # noise draws on hit samples only. The other samples cannot change.
    a_k = torch.zeros(K_DRAWS, nh, d, device=tok.device)
    p_k = torch.zeros(K_DRAWS, nh, d, device=tok.device)
    for s in range(0, nh, batch):
        idx = hit[s:s + batch]
        for k in range(K_DRAWS):
            o = msg_forward(model, ctx, tok[idx], k, n_heads, cfg)
            a_k[k, s:s + len(idx)], p_k[k, s:s + len(idx)] = o['a0'], o['pooled']
    m_k = torch.stack([torch.cat([margin_from(model, fb[hit[s:s + batch]], fa[hit[s:s + batch]], p_k[k, s:s + batch])
                                  for s in range(0, nh, batch)]) for k in range(K_DRAWS)]).cpu().numpy().astype(np.float64)
    hit_np = hit.cpu().numpy()
    full = np.tile(margin_clean[None], (K_DRAWS, 1))
    full[:, hit_np] = m_k
    sd_m = margin_clean.std(ddof=1)
    var_k = full.var(axis=0, ddof=1)
    a_c, p_c = a_clean.double(), p_clean.double()
    s_a = float(a_c.var(dim=0, unbiased=True).sum().sqrt())
    s_p = float(p_c.var(dim=0, unbiased=True).sum().sqrt())
    noise_a = float(a_k.double().var(dim=0, unbiased=True).sum(-1).mean().sqrt())
    noise_p = float(p_k.double().var(dim=0, unbiased=True).sum(-1).mean().sqrt())
    row = {'f1_clean': f1_score(y, margin_clean), 'f1_noise_mean': float(np.mean([f1_score(y, m) for m in full])),
           'margin_sd_clean': float(sd_m), 'nsr_margin': float(np.sqrt(var_k.mean()) / sd_m),
           'nsr_margin_hit': float(np.sqrt(var_k[hit_np].mean()) / sd_m),
           'ratio_attn_hit': noise_a / s_a, 'ratio_pool_hit': noise_p / s_p, 'signal_attn': s_a, 'signal_pool': s_p}
    if not base:
        return row
    # recentered margin noise, same rule as analyse_seeds.py (needs the original model on the hit samples)
    p_orig = torch.cat([msg_forward(model, ctx, tok[hit[s:s + batch]], 'orig', n_heads, cfg)['pooled']
                        for s in range(0, nh, batch)])
    m_orig = margin_clean.copy()
    m_orig[hit_np] = torch.cat([margin_from(model, fb[hit[s:s + batch]], fa[hit[s:s + batch]], p_orig[s:s + batch])
                                for s in range(0, nh, batch)]).cpu().numpy()
    rc = full - np.median(full - m_orig[None], axis=1, keepdims=True)
    row['nsr_margin_rc'] = float(np.sqrt(rc.var(axis=0, ddof=1).mean()) / sd_m)
    _, gram, vnorm, _ = value_maps(model, n_heads)
    p_pred = float(torch.cat(acc['P']).double().mean())
    p_same = float(torch.cat(acc['same']).double().mean())
    row.update({
        'pred_power': p_pred, 'pred_power_same_head': p_same, 'pred_power_cross_head': p_pred - p_same,
        'cross_head_share': (p_pred - p_same) / p_pred,
        'pred_power_token_grouped': float(torch.cat(acc['Pg']).double().mean()),
        'mean_sum_a2': acc['sum_a2'] / max(acc['n_pos'], 1),
        'mean_mnorm_F2': float(torch.diagonal(gram).mean()),
        'pred_ratio': float(np.sqrt(p_pred) / s_a),
        'ratio_valuepath_hit': float(torch.cat(acc['vp']).mean().sqrt() / s_a),
        'pred_ratio_token_grouped': float(np.sqrt(float(torch.cat(acc['Pg']).double().mean())) / s_a),
        'entropy_mean': float(torch.cat(acc['ent']).mean()),
        'mass_ratio': float(torch.cat(acc['mass']).mean() / torch.cat(acc['share']).mean()),
        'vnorm_head_max': float(vnorm.max()), 'mnorm_F_mean': float(torch.diagonal(gram).sqrt().mean()),
        'n_hit': int(nh), 'n_untrained': ctx['n_untrained']})
    return row


# ----------------------------------------------------------------- settings
def make_settings(n_heads, device):
    """List of dicts: name, group, kind, temp, k, subset, rescale, cfg."""
    out = []
    for t in TEMPS:
        g = 'temp_%g' % t
        out.append({'name': g, 'group': g, 'kind': 'temperature', 'temp': t, 'k': '', 'subset': '', 'rescale': '',
                    'cfg': {'temp': t}})
    k = 1
    while k < n_heads:
        combos = list(itertools.combinations(range(n_heads), k))
        rng = np.random.RandomState(1000 * n_heads + k)       # same subsets for every run with the same head count
        pick = rng.permutation(len(combos))[:N_SUBSETS]
        for rescale in (False, True):
            g = 'ablate_k%d_%s' % (k, 'rescale' if rescale else 'plain')
            for si, ci in enumerate(pick):
                scale = torch.zeros(n_heads, device=device)
                scale[list(combos[ci])] = float(n_heads) / k if rescale else 1.0
                out.append({'name': '%s_s%d' % (g, si), 'group': g, 'kind': 'head_ablation', 'temp': '', 'k': k,
                            'subset': ','.join(map(str, combos[ci])), 'rescale': int(rescale), 'cfg': {'head_scale': scale}})
        k *= 2
    out.append({'name': 'uniform', 'group': 'uniform', 'kind': 'uniform_attention', 'temp': '', 'k': '', 'subset': '',
                'rescale': '', 'cfg': {'uniform': True}})
    return out


# ----------------------------------------------------------------- statistics
def corr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def demean(v, labels):
    v = np.asarray(v, dtype=np.float64).copy()
    for a in np.unique(labels):
        v[labels == a] -= v[labels == a].mean()
    return v


def between_ss(v, labels):
    g = v.mean()
    return float(sum((labels == a).sum() * (v[labels == a].mean() - g) ** 2 for a in np.unique(labels)))


def remaining_share(y, x, labels):
    """Share of the between-architecture sum of squares of y that is left after a linear fit of y on x."""
    ok = np.isfinite(x) & np.isfinite(y)
    y, x, labels = y[ok], x[ok], labels[ok]
    if len(y) < 3 or x.std() < 1e-12 or between_ss(y, labels) < 1e-12:
        return None
    X = np.column_stack([np.ones(len(x)), x])
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return between_ss(res, labels) / between_ss(y, labels)


def summarise(rows, results):
    base = [r for r in rows if r['setting'] == 'temp_1']
    labels = np.array([r['arch'] for r in base])
    out = {'n_runs': len(base), 'k_draws': K_DRAWS,
           'note_layers': 'For two-layer models the analytic part (3a) and the mediators (3b) use layer 0 only. '
                          'The measured ratios are given at the layer-0 attention output, the final pooled vector and the margin.',
           'note_interventions': 'Temperature, head ablation and uniform attention are outside the training of the models. '
                                 'They change layer 0 only.'}
    pred = np.array([r['pred_ratio'] for r in base])
    meas = np.array([r['ratio_attn_hit'] for r in base])
    out['pred_vs_measured'] = {
        'corr_all': corr(pred, meas), 'corr_within_arch': corr(demean(pred, labels), demean(meas, labels)),
        'corr_pred_vs_nsr_margin_all': corr(pred, [r['nsr_margin'] for r in base]),
        'corr_pred_vs_nsr_margin_within_arch': corr(demean(pred, labels), demean([r['nsr_margin'] for r in base], labels)),
        'corr_pred_vs_valuepath_all': corr(pred, [r['ratio_valuepath_hit'] for r in base]),
        'mean_pred_over_valuepath': float(np.mean(pred / np.array([r['ratio_valuepath_hit'] for r in base]))),
        'corr_token_grouped_vs_measured_all': corr([r['pred_ratio_token_grouped'] for r in base], meas),
        'mean_pred_over_measured': float(np.mean(pred / meas)),
        'per_arch_mean': {a: {'pred_ratio': float(pred[labels == a].mean()), 'measured_ratio_attn': float(meas[labels == a].mean()),
                              'measured_valuepath_fixed_weights': float(np.mean([r['ratio_valuepath_hit'] for r in base if r['arch'] == a])),
                              'measured_nsr_margin': float(np.mean([r['nsr_margin'] for r in base if r['arch'] == a])),
                              'cross_head_share': float(np.mean([r['cross_head_share'] for r in base if r['arch'] == a]))}
                          for a in sorted(set(labels), key=list(ARCHS).index)}}
    # Mediators against gain_f1 and nsr of results.csv.
    if results:
        key = {(r['arch'], int(r['seed'])): r for r in results}
        have = [r for r in base if (r['arch'], r['seed']) in key]
        lab = np.array([r['arch'] for r in have])
        gain = np.array([float(key[(r['arch'], r['seed'])]['gain_f1']) for r in have])
        nsr = np.array([float(key[(r['arch'], r['seed'])]['nsr']) for r in have])
        meds = {'entropy_mean': 'entropy_mean', 'mass_ratio': 'mass_ratio', 'vnorm_head_max': 'vnorm_head_max',
                'mnorm_F_mean': 'mnorm_F_mean', 'log2_heads': None, 'layers': 'layers', 'pred_ratio': 'pred_ratio',
                'cross_head_share': 'cross_head_share', 'mean_sum_a2': 'mean_sum_a2'}
        med = {}
        for name, col in meds.items():
            x = np.log2([r['heads'] for r in have]) if col is None else np.array([r[col] for r in have], dtype=np.float64)
            med[name] = {'corr_gain_f1_all': corr(x, gain), 'corr_gain_f1_within_arch': corr(demean(x, lab), demean(gain, lab)),
                         'corr_nsr_all': corr(x, nsr), 'corr_nsr_within_arch': corr(demean(x, lab), demean(nsr, lab)),
                         'share_between_arch_nsr_remaining': remaining_share(nsr, x, lab)}
        out['mediators'] = {'n_runs_joined': len(have), 'per_mediator': med,
                            'corr_mech_nsr_margin_rc_vs_results_nsr': corr([r['nsr_margin_rc'] for r in have], nsr)}
    else:
        out['mediators'] = 'results.csv not found: not computed'
    # Mean nsr and F1 per architecture and setting group.
    groups = []
    for r in rows:
        if r['group'] not in groups:
            groups.append(r['group'])
    inter = {}
    for g in groups:
        inter[g] = {}
        for a in sorted(set(labels), key=list(ARCHS).index):
            sel = [r for r in rows if r['group'] == g and r['arch'] == a]
            if sel:
                inter[g][a] = {'nsr_margin': float(np.mean([r['nsr_margin'] for r in sel])),
                               'f1_clean': float(np.mean([r['f1_clean'] for r in sel])),
                               'f1_noise_mean': float(np.mean([r['f1_noise_mean'] for r in sel])), 'n_rows': len(sel)}
    out['interventions'] = inter
    return out


def fmt(v):
    return '  n/a' if v is None else '%+.2f' % v


def print_tables(s):
    pm = s['pred_vs_measured']
    print('\nANALYTIC PREDICTION (layer 0 attention output at the class token), mean per architecture')
    print('pred = first order, weights fixed. valpath = measured with weights fixed (should match pred). measured = full model.')
    print('%-7s %8s %8s %9s %11s %12s' % ('arch', 'pred', 'valpath', 'measured', 'nsr_margin', 'cross share'))
    for a, v in pm['per_arch_mean'].items():
        print('%-7s %8.3f %8.3f %9.3f %11.3f %12.3f' % (a, v['pred_ratio'], v['measured_valuepath_fixed_weights'], v['measured_ratio_attn'],
                                                       v['measured_nsr_margin'], v['cross_head_share']))
    print('corr pred vs measured: all %s, within architecture %s; mean pred/measured %.2f' % (
        fmt(pm['corr_all']), fmt(pm['corr_within_arch']), pm['mean_pred_over_measured']))
    print('corr pred vs valpath: all %s; mean pred/valpath %.2f' % (fmt(pm['corr_pred_vs_valuepath_all']), pm['mean_pred_over_valuepath']))
    if isinstance(s['mediators'], dict):
        print('\nMEDIATORS vs results.csv (n=%d runs): corr with gain_f1 / nsr (all, within arch), remaining between-arch share of nsr' % s['mediators']['n_runs_joined'])
        for m, v in s['mediators']['per_mediator'].items():
            sh = v['share_between_arch_nsr_remaining']
            print('  %-17s gain %s %s | nsr %s %s | remaining %s' % (m, fmt(v['corr_gain_f1_all']), fmt(v['corr_gain_f1_within_arch']),
                  fmt(v['corr_nsr_all']), fmt(v['corr_nsr_within_arch']), 'n/a' if sh is None else '%.2f' % sh))
    print('\nINTERVENTIONS (outside training). cell = nsr_margin / clean F1')
    archs = list(pm['per_arch_mean'])
    print('%-20s' % 'setting' + ''.join('%14s' % a for a in archs))
    for g, per in s['interventions'].items():
        print('%-20s' % g + ''.join('%14s' % ('%.3f/%.1f' % (per[a]['nsr_margin'], per[a]['f1_clean']) if a in per else '-') for a in archs))


# ----------------------------------------------------------------- main
def read_results(path):
    if not path or not Path(path).exists():
        return None
    with open(path, newline='', encoding='utf-8') as fp:
        return list(csv.DictReader(fp))


CSV_COLS = ['arch', 'heads', 'layers', 'seed', 'setting', 'group', 'kind', 'temp', 'k', 'subset', 'rescale',
            'f1_clean', 'f1_noise_mean', 'margin_sd_clean', 'nsr_margin', 'nsr_margin_hit', 'ratio_attn_hit', 'ratio_pool_hit',
            'signal_attn', 'signal_pool', 'nsr_margin_rc', 'pred_power', 'pred_power_same_head', 'pred_power_cross_head',
            'cross_head_share', 'pred_power_token_grouped', 'ratio_valuepath_hit', 'mean_sum_a2', 'mean_mnorm_F2', 'pred_ratio', 'pred_ratio_token_grouped',
            'entropy_mean', 'mass_ratio', 'vnorm_head_max', 'mnorm_F_mean', 'n_hit', 'n_untrained', 'manual_gap']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset.npz')
    p.add_argument('--runs', default='runs')
    p.add_argument('--results', default='results.csv')
    p.add_argument('--out-dir', default='.')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--batch', type=int, default=32, help='samples per forward pass (use 4 on a small CPU)')
    p.add_argument('--limit-runs', type=int, default=0, help='use only the first runs (smoke test)')
    p.add_argument('--test-limit', type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    device = pick_device(args.device)
    data = load_data(args.data)
    test_rows = data['test_idx']
    if args.test_limit:
        test_rows = test_rows[:args.test_limit]
    y = data['labels'][test_rows]
    tok = to_t(data['msg'], test_rows, device)[:, :WINDOW]
    logs = sorted(Path(args.runs).glob('*/log.json'))
    if args.limit_runs:
        logs = logs[:args.limit_runs]
    if not logs:
        raise SystemExit('No finished runs found in %s' % args.runs)

    rows, gaps = [], {}
    with torch.no_grad():
        # Code features do not depend on the message branch. They are computed once per run.
        for log_path in logs:
            log = read_json(log_path)
            arch, seed = log['arch'], log['seed']
            n_heads = ARCHS[arch][0]
            fit_rows, _ = fit_val_rows(data, log['args'].get('select_on', 'val'), log['args'].get('limit', 0))
            model = build_arch(arch, int(data['n_code_vocab']), int(data['n_msg_vocab']), log['args']['dropout']).to(device)
            model.load_state_dict(torch.load(log_path.parent / 'model.pth', map_location=device, weights_only=True), strict=True)
            model.eval()
            ctx = make_context(model, data, fit_rows, device, K_DRAWS)
            fb, fa = [], []
            for s in range(0, len(test_rows), args.batch):
                r = test_rows[s:s + args.batch]
                fb.append(model.code_feature(to_t(data['tok_b'], r, device), to_t(data['typ_b'], r, device), 'B'))
                fa.append(model.code_feature(to_t(data['tok_a'], r, device), to_t(data['typ_a'], r, device), 'A'))
            fb, fa = torch.cat(fb), torch.cat(fa)
            gap, gap_a = check_manual_forward(model, ctx, data, test_rows, fb, fa, n_heads, device)
            gaps['%s_seed%d' % (arch, seed)] = {'margin_gap': gap, 'layer0_slicing_gap': gap_a}
            first = None
            for st in make_settings(n_heads, device):
                base = st['name'] == 'temp_1'
                res = run_setting(model, ctx, tok, fb, fa, y, n_heads, st['cfg'], args.batch, base=base)
                row = {'arch': arch, 'heads': n_heads, 'layers': ARCHS[arch][1], 'seed': seed, 'manual_gap': gap}
                row.update({k: st[k] for k in ('kind', 'temp', 'k', 'subset', 'rescale', 'group')})
                row['setting'] = st['name']
                row.update(res)
                rows.append(row)
                if base:
                    first = row
            print('%s seed %d: manual gap %.1e | pred %.3f measured %.3f | nsr_margin %.3f (rc %.3f) | cross-head share %.2f | f1 %.2f' % (
                arch, seed, gap, first['pred_ratio'], first['ratio_attn_hit'], first['nsr_margin'], first['nsr_margin_rc'],
                first['cross_head_share'], first['f1_clean']), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'mechanism.csv', 'w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in CSV_COLS})
    summary = summarise(rows, read_results(args.results))
    summary['manual_forward_gaps'] = gaps
    with open(out_dir / 'mechanism_summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=1)
    print_tables(summary)
    print('\nWrote', out_dir / 'mechanism.csv', 'and', out_dir / 'mechanism_summary.json')


if __name__ == '__main__':
    main()
