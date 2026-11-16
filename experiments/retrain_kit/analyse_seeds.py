"""Measure, for every finished run, how much the zero edit of untrained message rows helps.

Writes results.csv and summary.json. No weights are written.
Example: python analyse_seeds.py --data dataset.npz --runs runs
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from common import (auc_score, f1_score, fit_val_rows, load_data, pick_device, read_json)
from model import ARCHS, MSG_MAX_LEN, build_arch

K_DRAWS = 10
N_PERM = 10000
PERM_SEED = 2024
GROUP_A = ['M3', 'M4', 'M13']
GROUP_B = ['M5', 'M8', 'M12']
METRICS = ['f1', 'auc', 'zero_f1', 'zero_auc', 'gain_f1', 'gain_auc', 'nsr', 'vnorm']


# ----------------------------------------------------------------- one run
def untrained_ids(msg, fit_rows, vocab_size):
    """Message token ids that never occur in the input window of a training sample. Pad id 0 is excluded."""
    seen = np.zeros(vocab_size, dtype=bool)
    seen[np.unique(msg[fit_rows][:, :MSG_MAX_LEN - 1])] = True
    seen[0] = True
    return np.where(~seen)[0]


def run_in_batches(fn, n, batch):
    out = []
    with torch.no_grad():
        for s in range(0, n, batch):
            out.append(fn(slice(s, min(s + batch, n))).float().cpu().numpy())
    return np.concatenate(out)


def analyse_model(model, data, test_rows, fit_rows, device, k_draws=K_DRAWS, batch=64):
    """Return a dict of metrics and a dict of margin arrays for one model."""
    model.eval()
    y = data['labels'][test_rows]
    msg = data['msg'][test_rows].astype(np.int64)
    tb = lambda key, sl: torch.from_numpy(data[key][test_rows[sl]].astype(np.int64)).to(device)

    # Features of the three branches for the unedited model.
    fb = run_in_batches(lambda sl: model.code_feature(tb('tok_b', sl), tb('typ_b', sl), 'B'), len(test_rows), batch)
    fa = run_in_batches(lambda sl: model.code_feature(tb('tok_a', sl), tb('typ_a', sl), 'A'), len(test_rows), batch)
    fm = run_in_batches(lambda sl: model.msg_feature(torch.from_numpy(msg[sl]).to(device)), len(test_rows), batch)

    def margin_of(fm_now):
        with torch.no_grad():
            lg = model.head(torch.from_numpy(fb).to(device), torch.from_numpy(fa).to(device),
                            torch.from_numpy(fm_now).to(device)).float().cpu().numpy()
        return (lg[:, 1] - lg[:, 0]).astype(np.float64)

    orig = margin_of(fm)
    U = untrained_ids(data['msg'], fit_rows, model.embedMsg.weight.shape[0])
    hit = np.where(np.isin(msg[:, :MSG_MAX_LEN - 1], U).any(1))[0]
    W = model.embedMsg.weight
    U_t = torch.from_numpy(U).to(device)
    saved = W.data[U_t].clone()

    def margin_with(rows_value):
        """Margin (not recentered) when the untrained rows hold rows_value."""
        fm_new = fm.copy()
        try:
            W.data[U_t] = rows_value
            fm_new[hit] = run_in_batches(
                lambda sl: model.msg_feature(torch.from_numpy(msg[hit[sl]]).to(device)), len(hit), batch)
        finally:
            W.data[U_t] = saved
        return margin_of(fm_new)

    def recenter(m):
        """Shift the margin so that the median change (new - orig) is zero."""
        return m - np.median(m - orig)

    zero_raw = margin_with(torch.zeros_like(saved))
    zero = recenter(zero_raw)
    d = saved.shape[1]
    draws = []
    for k in range(k_draws):
        noise = np.random.RandomState(k).randn(len(U), d).astype(np.float32)
        draws.append(recenter(margin_with(torch.from_numpy(noise).to(device))))
    draws = np.array(draws)
    nsr = float(np.sqrt((draws.std(axis=0, ddof=1) ** 2).mean()) / zero.std())

    w = model.transformerMsg.layers[0].self_attn.in_proj_weight.detach().float().cpu()
    d_model = w.shape[1]
    vnorm = float(torch.linalg.matrix_norm(w[2 * d_model:], ord=2))

    res = {'f1': f1_score(y, orig), 'auc': auc_score(y, orig),
           'zero_f1': f1_score(y, zero), 'zero_auc': auc_score(y, zero),
           'zero_f1_raw': f1_score(y, zero_raw), 'nsr': nsr, 'vnorm': vnorm,
           'n_untrained': int(len(U)), 'n_hit': int(len(hit))}
    res['gain_f1'] = res['zero_f1'] - res['f1']
    res['gain_auc'] = res['zero_auc'] - res['auc']
    return res, {'orig': orig, 'zero_raw': zero_raw, 'zero': zero}


# ----------------------------------------------------------------- statistics
def f_statistic(values, labels):
    """One-way ANOVA F and the share of variance between groups (SS_between / SS_total)."""
    groups = np.unique(labels)
    grand = values.mean()
    ss_t = ((values - grand) ** 2).sum()
    ss_b = sum(((labels == g).sum()) * (values[labels == g].mean() - grand) ** 2 for g in groups)
    ss_w = ss_t - ss_b
    k, n = len(groups), len(values)
    if ss_w <= 1e-12 or n <= k:
        return float('nan'), float(ss_b / ss_t) if ss_t > 0 else float('nan')
    return float((ss_b / (k - 1)) / (ss_w / (n - k))), float(ss_b / ss_t)


def permutation_p(values, labels, rng, n_perm=N_PERM):
    obs, share = f_statistic(values, labels)
    if np.isnan(obs):
        return obs, share, float('nan')
    count = 0
    for _ in range(n_perm):
        f, _ = f_statistic(values, rng.permutation(labels))
        count += f >= obs - 1e-12
    return obs, share, (count + 1) / (n_perm + 1)


def arch_level_p(means, a_names, b_names):
    """Exact test over all ways to split the architectures into two groups of the same sizes."""
    names = a_names + b_names
    if any(n not in means for n in names):
        return float('nan')
    obs = abs(np.mean([means[n] for n in a_names]) - np.mean([means[n] for n in b_names]))
    stats = []
    for combo in itertools.combinations(names, len(a_names)):
        rest = [n for n in names if n not in combo]
        stats.append(abs(np.mean([means[n] for n in combo]) - np.mean([means[n] for n in rest])))
    return float(np.mean(np.array(stats) >= obs - 1e-12))


def summarise(rows):
    rng = np.random.RandomState(PERM_SEED)
    archs = [a for a in ARCHS if any(r['arch'] == a for r in rows)]
    labels = np.array([r['arch'] for r in rows])
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in rows], dtype=np.float64)
        per = {a: {'mean': float(v[labels == a].mean()), 'sd': float(v[labels == a].std(ddof=1)) if (labels == a).sum() > 1 else None,
                   'n': int((labels == a).sum())} for a in archs}
        entry = {'per_arch': per}
        if len(archs) > 1:
            f, share, p = permutation_p(v, labels, rng)
            entry['between_arch'] = {'F': f, 'p_perm': p, 'share_var_between': share,
                                     'share_expected_if_no_effect': (len(archs) - 1) / (len(v) - 1)}
        grp = np.array(['A' if a in GROUP_A else 'B' if a in GROUP_B else '' for a in labels])
        keep = grp != ''
        if len(set(grp[keep])) == 2:
            f, share, p = permutation_p(v[keep], grp[keep], rng)
            entry['contrast_M3_M4_M13_vs_M5_M8_M12'] = {
                'mean_A': float(v[grp == 'A'].mean()), 'mean_B': float(v[grp == 'B'].mean()),
                'F': f, 'p_perm_runs': p, 'share_var_between': share,
                'p_exact_architectures': arch_level_p({a: per[a]['mean'] for a in archs}, GROUP_A, GROUP_B)}
        out[m] = entry
    return out


def print_table(summary):
    archs = list(summary['f1']['per_arch'])
    print('\n%-8s' % 'metric' + ''.join('%16s' % a for a in archs) + '   F p_perm  share')
    for m in METRICS:
        e = summary[m]
        line = '%-8s' % m
        for a in archs:
            s = e['per_arch'][a]
            line += '%16s' % ('%.3f+-%.3f' % (s['mean'], s['sd']) if s['sd'] is not None else '%.3f' % s['mean'])
        b = e.get('between_arch')
        if b:
            line += '   %.2f %.4f  %.2f' % (b['F'], b['p_perm'], b['share_var_between'])
        print(line)


# ----------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset.npz')
    p.add_argument('--runs', default='runs')
    p.add_argument('--out-dir', default='.')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--test-limit', type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    device = pick_device(args.device)
    data = load_data(args.data)
    test_rows = data['test_idx']
    if args.test_limit:
        test_rows = test_rows[:args.test_limit]

    rows = []
    for log_path in sorted(Path(args.runs).glob('*/log.json')):
        log = read_json(log_path)
        arch, seed = log['arch'], log['seed']
        fit_rows, _ = fit_val_rows(data, log['args'].get('select_on', 'val'), log['args'].get('limit', 0))
        model = build_arch(arch, int(data['n_code_vocab']), int(data['n_msg_vocab']), log['args']['dropout']).to(device)
        model.load_state_dict(torch.load(log_path.parent / 'model.pth', map_location=device, weights_only=True), strict=True)
        res, _ = analyse_model(model, data, test_rows, fit_rows, device)
        res.update({'arch': arch, 'seed': seed, 'best_epoch': log['best_epoch'], 'best_val_f1': log['best_val_f1'], 'best_val_acc': log.get('best_val_acc')})
        rows.append(res)
        print('%s seed %d: f1 %.2f zero_f1 %.2f gain %+.2f nsr %.3f vnorm %.3f' % (
            arch, seed, res['f1'], res['zero_f1'], res['gain_f1'], res['nsr'], res['vnorm']))
    if not rows:
        raise SystemExit('No finished runs found in %s' % args.runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ['arch', 'seed', 'best_epoch', 'best_val_f1', 'best_val_acc'] + METRICS + ['zero_f1_raw', 'n_untrained', 'n_hit']
    with open(out_dir / 'results.csv', 'w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    summary = summarise(rows)
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=1)
    print_table(summary)
    print('\nWrote', out_dir / 'results.csv', 'and', out_dir / 'summary.json')


if __name__ == '__main__':
    main()
