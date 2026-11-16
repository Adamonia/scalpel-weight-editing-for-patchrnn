"""Round 3: apply E0, E1 and the fixed recipe (recipe.json) to every finished run of the retraining kit.

Put this file, recipe.json and the kit files (model.py, common.py) in one folder.
Example: python apply_round3.py --data dataset.npz --runs runs --out-dir round3
Writes round3_results.csv and round3_summary.json. No weights are written.

Edits (message embedding table only, the pad row is never changed):
  E0 original.  E1 rows never seen in the fit rows := 0.  E2 same rows := pad row.
  E3 row := a*row, a = c/(c+k).  E4 row := a*row + (1-a)*pad row.   c = training count of the token
  (number of fit samples that contain the token in the message window).
Recenter (label-free): all margins are shifted so that their mean on the evaluated rows equals the mean margin
of the original model on the same rows. It is used only if recipe.json says "recenter": true.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from common import auc_score, f1_score, fit_val_rows, load_data, pick_device, read_json
from model import MSG_MAX_LEN, build_arch

N_BOOT = 10000
N_FLIPS = 100000
SEED = 2025
PAD_ID = 0


# ----------------------------------------------------------------- edits
def token_counts(msg, rows, vocab_size):
    """Number of samples (of the given rows) that contain each token in the model window."""
    win = msg[rows][:, :MSG_MAX_LEN - 1].astype(np.int64)
    keys = np.unique(np.arange(len(win))[:, None] * vocab_size + win)
    return np.bincount(keys % vocab_size, minlength=vocab_size)


def new_table(weight, count, pad_id, edit, k):
    """New message table for one edit. weight: (V, d) float tensor. The pad row is copied unchanged."""
    pad_row = weight[pad_id].clone()
    out = weight.clone()
    if edit in ('E1', 'E2'):
        unseen = torch.from_numpy(count == 0)
        unseen[pad_id] = False
        out[unseen] = 0.0 if edit == 'E1' else pad_row
    elif edit in ('E3', 'E4'):
        c = torch.from_numpy(count).double()
        a = (c / (c + k)).float().unsqueeze(1)
        a[pad_id] = 1.0
        out = a * weight if edit == 'E3' else a * weight + (1 - a) * pad_row
    elif edit != 'E0':
        raise ValueError(edit)
    out[pad_id] = pad_row
    return out


def run_in_batches(fn, n, batch):
    out = []
    with torch.no_grad():
        for s in range(0, n, batch):
            out.append(fn(slice(s, min(s + batch, n))).float().cpu().numpy())
    return np.concatenate(out)


def evaluate_run(model, data, test_rows, fit_rows, recipe, device, batch=64):
    """Margins of E0, E1 (raw and recentered) and the recipe on the test rows."""
    model.eval()
    msg = data['msg'][test_rows].astype(np.int64)
    tb = lambda key, sl: torch.from_numpy(data[key][test_rows[sl]].astype(np.int64)).to(device)
    n = len(test_rows)
    fb = run_in_batches(lambda sl: model.code_feature(tb('tok_b', sl), tb('typ_b', sl), 'B'), n, batch)
    fa = run_in_batches(lambda sl: model.code_feature(tb('tok_a', sl), tb('typ_a', sl), 'A'), n, batch)
    fm0 = run_in_batches(lambda sl: model.msg_feature(torch.from_numpy(msg[sl]).to(device)), n, batch)

    def margin_of(fm):
        with torch.no_grad():
            lg = model.head(torch.from_numpy(fb).to(device), torch.from_numpy(fa).to(device),
                            torch.from_numpy(fm).to(device)).float().cpu().numpy()
        return (lg[:, 1] - lg[:, 0]).astype(np.float64)

    W = model.embedMsg.weight
    V = W.shape[0]
    saved = W.data.clone()
    count = token_counts(data['msg'], fit_rows, V)
    # the pad id must be the fill value of the message array and must have been seen in training
    fill = np.bincount(data['msg'][:, -1].astype(np.int64)).argmax()
    if fill != PAD_ID or count[PAD_ID] == 0:
        raise RuntimeError('pad id %d is not the fill value of msg (%d)' % (PAD_ID, fill))

    def margin_with(edit, k):
        W1 = new_table(saved, count, PAD_ID, edit, k).to(device)
        changed = (W1 != saved.to(device)).any(1).cpu().numpy()
        hit = np.where(changed[msg[:, :MSG_MAX_LEN - 1]].any(1))[0]
        fm = fm0.copy()
        try:
            W.data.copy_(W1)
            if len(hit):
                fm[hit] = run_in_batches(lambda sl: model.msg_feature(torch.from_numpy(msg[hit[sl]]).to(device)), len(hit), batch)
        finally:
            W.data.copy_(saved)
        return margin_of(fm), int(changed.sum())

    m0 = margin_of(fm0)
    m1, n_changed_e1 = margin_with('E1', None)
    mr, n_changed_r = margin_with(recipe['edit'], recipe['k'])
    shift = lambda m: m + (m0.mean() - m.mean())
    use_rc = bool(recipe['recenter'])
    out = {'E0': m0, 'E1_raw': m1, 'E1_rc': shift(m1), 'R_raw': mr, 'R_rc': shift(mr)}
    out['E1'] = out['E1_rc'] if use_rc else out['E1_raw']
    out['R'] = out['R_rc'] if use_rc else out['R_raw']
    return out, {'n_untrained': int((count == 0).sum() - 0), 'n_rows_changed_E1': n_changed_e1, 'n_rows_changed_recipe': n_changed_r}


# ----------------------------------------------------------------- statistics over runs
def boot_ci(d, rng):
    d = np.asarray(d, dtype=np.float64)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def signflip_p(d, rng):
    """Two-sided paired sign-flip permutation p for mean(d) = 0 (exact for up to 16 runs)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.float64)
    else:
        signs = rng.choice([-1.0, 1.0], size=(N_FLIPS, n))
    m = np.abs((signs * d).mean(1))
    return float((np.sum(m >= obs - 1e-12) + (0 if n <= 16 else 1)) / (len(m) + (0 if n <= 16 else 1)))


def describe(d, rng):
    d = np.asarray(d, dtype=np.float64)
    return {'n_runs': int(len(d)), 'mean': float(d.mean()), 'ci95_bootstrap_over_runs': boot_ci(d, rng),
            'p_signflip': signflip_p(d, rng), 'n_runs_improved': int((d > 0).sum())}


def summarise(rows):
    rng = np.random.default_rng(SEED)
    out = {}
    groups = {a: [r for r in rows if r['arch'] == a] for a in sorted({r['arch'] for r in rows})}
    groups['overall'] = rows
    for name, rs in groups.items():
        out[name] = {'recipe_vs_E0_f1': describe([r['d_f1_R_vs_E0'] for r in rs], rng),
                     'recipe_vs_E1_f1': describe([r['d_f1_R_vs_E1'] for r in rs], rng),
                     'E1_vs_E0_f1': describe([r['d_f1_E1_vs_E0'] for r in rs], rng),
                     'mean_f1_E0': float(np.mean([r['f1_E0'] for r in rs])),
                     'mean_f1_E1': float(np.mean([r['f1_E1'] for r in rs])),
                     'mean_f1_recipe': float(np.mean([r['f1_R'] for r in rs]))}
    return out


# ----------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset.npz')
    p.add_argument('--runs', default='runs')
    p.add_argument('--recipe', default=str(here / 'recipe.json'))
    p.add_argument('--out-dir', default='.')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--subset', default=None, help='optional .npy with positions inside test_idx to evaluate (smoke tests)')
    args = p.parse_args()
    device = pick_device(args.device)
    recipe = read_json(args.recipe)
    recipe = {'edit': recipe['edit'], 'k': recipe.get('k'), 'recenter': bool(recipe['recenter'])}
    data = load_data(args.data)
    test_rows = data['test_idx']
    if args.subset:
        test_rows = test_rows[np.load(args.subset)]
    y = data['labels'][test_rows]

    rows = []
    for log_path in sorted(Path(args.runs).glob('*/log.json')):
        log = read_json(log_path)
        arch, seed = log['arch'], log['seed']
        fit_rows, _ = fit_val_rows(data, log['args'].get('select_on', 'val'), log['args'].get('limit', 0))
        model = build_arch(arch, int(data['n_code_vocab']), int(data['n_msg_vocab']), log['args']['dropout']).to(device)
        model.load_state_dict(torch.load(log_path.parent / 'model.pth', map_location=device, weights_only=True), strict=True)
        m, info = evaluate_run(model, data, test_rows, fit_rows, recipe, device, args.batch)
        r = {'arch': arch, 'seed': seed, 'n_test': len(test_rows), **info}
        for key in ('E0', 'E1_raw', 'E1', 'R_raw', 'R'):
            r['f1_' + key] = f1_score(y, m[key])
            r['auc_' + key] = auc_score(y, m[key])
        r['d_f1_R_vs_E0'] = r['f1_R'] - r['f1_E0']
        r['d_f1_R_vs_E1'] = r['f1_R'] - r['f1_E1']
        r['d_f1_E1_vs_E0'] = r['f1_E1'] - r['f1_E0']
        r['d_auc_R_vs_E0'] = r['auc_R'] - r['auc_E0']
        r['d_auc_R_vs_E1'] = r['auc_R'] - r['auc_E1']
        rows.append(r)
        print('%s seed %s: f1 E0 %.2f E1 %.2f recipe %.2f (vs E0 %+.2f, vs E1 %+.2f)' % (
            arch, seed, r['f1_E0'], r['f1_E1'], r['f1_R'], r['d_f1_R_vs_E0'], r['d_f1_R_vs_E1']))
    if not rows:
        raise SystemExit('No finished runs found in %s' % args.runs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'round3_results.csv', 'w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {'recipe': recipe, 'note': 'E1 and the recipe use the recenter flag of the recipe; E1_raw/R_raw are without recenter',
               'groups': summarise(rows)}
    with open(out / 'round3_summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=1)
    for name, g in summary['groups'].items():
        a, b = g['recipe_vs_E0_f1'], g['recipe_vs_E1_f1']
        print('%-8s n=%d  vs E0 %+.3f [%+.3f, %+.3f] p=%.4f | vs E1 %+.3f [%+.3f, %+.3f] p=%.4f' % (
            name, a['n_runs'], a['mean'], *a['ci95_bootstrap_over_runs'], a['p_signflip'],
            b['mean'], *b['ci95_bootstrap_over_runs'], b['p_signflip']))
    print('Wrote', out / 'round3_results.csv', 'and', out / 'round3_summary.json')


if __name__ == '__main__':
    main()
