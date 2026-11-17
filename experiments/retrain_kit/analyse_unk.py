"""Compare UNK-trained runs with the first 50 runs (base, base + zero edit, base + round 3 recipe).

Reads: the UNK runs folder, the base runs folder (only if a number is missing in the csv files), dataset.npz.
Base numbers come from round3_results.csv (columns f1_E0, f1_E1_raw, f1_R_raw) if the pair is in it.
Otherwise they are computed from the saved base model with the code of apply_round3.py.
Writes unk_results.csv and unk_summary.json. The rule is in unk_rule.md. Do not change it after seeing results.
Example: python analyse_unk.py --data dataset.npz --runs RUNS --unk-runs UNKRUNS --out-dir out
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from analyse_seeds import sign_test_p, trend
from apply_round3 import evaluate_run
from common import f1_score, fit_val_rows, load_data, pick_device, read_json
from model import ARCHS, build_arch
from train_seeds import evaluate
from unk_tools import apply_map, build_unk_map

N_BOOT = 10000
BOOT_SEED = 12345
SAME_ARGS = ['epochs', 'lr', 'batch', 'dropout', 'weight_decay', 'shuffle', 'select_on', 'amp', 'limit']
HERE = Path(__file__).resolve().parent


def find_file(name, dirs):
    for d in dirs:
        if (Path(d) / name).exists():
            return Path(d) / name
    return None


def read_base_csv(path):
    """(arch, seed) -> (f1 base, f1 zero edit raw, f1 round 3 raw)."""
    out = {}
    if path is None:
        return out
    with open(path, encoding='utf-8') as fp:
        for r in csv.DictReader(fp):
            out[(r['arch'], int(r['seed']))] = (float(r['f1_E0']), float(r['f1_E1_raw']), float(r['f1_R_raw']))
    return out


def unk_f1(log, run_dir, data, device, batch):
    """F1 of one UNK run on the test rows. No edit."""
    fit_rows, _ = fit_val_rows(data, log['args'].get('select_on', 'val'), log['args'].get('limit', 0))
    v_msg = int(data['n_msg_vocab'])
    table, unk_id, _ = build_unk_map(data['msg'], fit_rows, v_msg, log['args']['unk'])
    if unk_id != log['unk_id']:
        raise RuntimeError('UNK id differs from the log in %s' % run_dir)
    mapped = apply_map(data, table)
    model = build_arch(log['arch'], int(data['n_code_vocab']), v_msg + 1, log['args']['dropout']).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pth', map_location=device, weights_only=True), strict=True)
    f1, _, _ = evaluate(model, mapped, mapped['test_idx'], device, False, batch)
    return f1


def base_f1s(arch, seed, base_runs, data, recipe, device, batch):
    """Compute (base, zero edit raw, round 3 raw) from the saved base model."""
    run_dir = Path(base_runs) / ('%s_seed%d' % (arch, seed))
    log = read_json(run_dir / 'log.json')
    fit_rows, _ = fit_val_rows(data, log['args'].get('select_on', 'val'), log['args'].get('limit', 0))
    model = build_arch(arch, int(data['n_code_vocab']), int(data['n_msg_vocab']), log['args']['dropout']).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pth', map_location=device, weights_only=True), strict=True)
    m, _ = evaluate_run(model, data, data['test_idx'], fit_rows, recipe, device, batch)
    y = data['labels'][data['test_idx']]
    return f1_score(y, m['E0']), f1_score(y, m['E1_raw']), f1_score(y, m['R_raw'])


# ----------------------------------------------------------------- statistics
def paired(d):
    """Mean, 95 % paired bootstrap interval (10,000 resamples, seed 12345) and exact sign test."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    means = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    return {'n_pairs': int(len(d)), 'mean': float(d.mean()),
            'ci95_bootstrap': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            'n_positive': n_pos, 'n_negative': n_neg, 'n_zero': int(len(d) - n_pos - n_neg),
            'p_sign_two_sided': sign_test_p(n_pos, n_pos + n_neg)}


def slope_info(rows, key):
    t = trend(rows, key, np.random.default_rng(BOOT_SEED))
    fit = t['run_level_fit']
    return {'run_level_slope_per_log2_heads': fit['coef'].get('log2_heads') if fit else None,
            'run_level_layers2': fit['coef'].get('layers2') if fit else None,
            'arch_level_slope_per_log2_heads': t['heads_slope_arch_level']['slope_per_log2_heads_mean_of_strata'],
            'arch_level_p_perm': t['heads_slope_arch_level']['p_perm_two_sided'],
            'missing_cells': t['missing_cells']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset.npz')
    p.add_argument('--runs', default='runs', help='folder of the first 50 runs (scalpel_runs_v2)')
    p.add_argument('--unk-runs', default='runs_unk', help='folder of the UNK runs (scalpel_runs_unk)')
    p.add_argument('--round3', default=None, help='round3_results.csv (default: out-dir, then kit folder, then kit/out)')
    p.add_argument('--recipe', default=str(HERE / 'recipe.json'))
    p.add_argument('--out-dir', default='out')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--test-limit', type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    if args.test_limit:      # smoke test only: numbers of the csv files do not fit, so they are not used
        data['test_idx'] = data['test_idx'][:args.test_limit]
    r3 = None if args.test_limit else (Path(args.round3) if args.round3 else
                                       find_file('round3_results.csv', [out_dir, HERE, HERE / 'out']))
    base_csv = read_base_csv(r3)
    print('base numbers from:', r3 if base_csv else 'saved base models')
    recipe = read_json(args.recipe)
    recipe = {'edit': recipe['edit'], 'k': recipe.get('k'), 'recenter': bool(recipe['recenter'])}

    rows, warnings = [], []
    for log_path in sorted(Path(args.unk_runs).glob('*/log.json')):
        log = read_json(log_path)
        arch, seed = log['arch'], log['seed']
        if log['args'].get('unk') is None:
            raise SystemExit('%s is not an UNK run' % log_path)
        f1_unk = unk_f1(log, log_path.parent, data, device, args.batch)
        if (arch, seed) in base_csv:
            f1_base, f1_zero, f1_r3 = base_csv[(arch, seed)]
            source = 'csv'
        else:
            f1_base, f1_zero, f1_r3 = base_f1s(arch, seed, args.runs, data, recipe, device, args.batch)
            source = 'model'
        base_log = Path(args.runs) / ('%s_seed%d' % (arch, seed)) / 'log.json'
        if base_log.exists():
            bargs = read_json(base_log)['args']
            diff = [k for k in SAME_ARGS if bargs.get(k) != log['args'].get(k)]
            if diff:
                warnings.append('%s seed %d: base and UNK runs differ in %s' % (arch, seed, ','.join(diff)))
        r = {'arch': arch, 'heads': ARCHS[arch][0], 'layers': ARCHS[arch][1], 'seed': seed,
             'f1_base': f1_base, 'f1_zero': f1_zero, 'f1_round3': f1_r3, 'f1_unk': f1_unk,
             'd_unk_vs_zero': f1_unk - f1_zero, 'd_unk_vs_base': f1_unk - f1_base, 'd_unk_vs_round3': f1_unk - f1_r3,
             'd_zero_vs_base': f1_zero - f1_base, 'd_round3_vs_base': f1_r3 - f1_base,
             'unk_k': log['args']['unk'], 'unk_id': log['unk_id'], 'n_ids_mapped': log['n_ids_mapped'],
             'unk_best_epoch': log['best_epoch'], 'unk_row_max_change': log['unk_row_max_change'],
             'base_source': source}
        rows.append(r)
        print('%s seed %d: base %.2f zero %.2f round3 %.2f UNK %.2f (UNK - zero %+.2f)' % (
            arch, seed, f1_base, f1_zero, f1_r3, f1_unk, r['d_unk_vs_zero']))
    if not rows:
        raise SystemExit('No finished UNK runs found in %s' % args.unk_runs)
    for w in warnings:
        print('WARNING:', w)

    with open(out_dir / 'unk_results.csv', 'w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_expected = len(ARCHS) * 5
    keys = {'unk_minus_zero': 'd_unk_vs_zero', 'unk_minus_base': 'd_unk_vs_base', 'unk_minus_round3': 'd_unk_vs_round3'}
    summary = {'rule': 'unk_rule.md', 'n_pairs': len(rows), 'n_expected': n_expected, 'complete': len(rows) == n_expected,
               'unk_k': sorted({r['unk_k'] for r in rows}), 'warnings': warnings,
               'mean_f1': {k: float(np.mean([r[k] for r in rows])) for k in ('f1_base', 'f1_zero', 'f1_round3', 'f1_unk')},
               'primary_unk_minus_zero': paired([r['d_unk_vs_zero'] for r in rows]),
               'secondary': {name: paired([r[key] for r in rows]) for name, key in keys.items() if name != 'unk_minus_zero'},
               'slope_over_log2_heads': {name: slope_info(rows, key) for name, key in keys.items()},
               'per_arch_mean_d_unk_vs_zero': {a: float(np.mean([r['d_unk_vs_zero'] for r in rows if r['arch'] == a]))
                                               for a in ARCHS if any(r['arch'] == a for r in rows)}}
    with open(out_dir / 'unk_summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=1)
    pr = summary['primary_unk_minus_zero']
    print('\nPRIMARY UNK - (base + zero edit): n=%d mean %+.3f CI [%+.3f, %+.3f] sign p=%.4f (%d up, %d down)' % (
        pr['n_pairs'], pr['mean'], *pr['ci95_bootstrap'], pr['p_sign_two_sided'], pr['n_positive'], pr['n_negative']))
    for name, v in summary['secondary'].items():
        print('%-18s mean %+.3f CI [%+.3f, %+.3f] sign p=%.4f' % (name, v['mean'], *v['ci95_bootstrap'], v['p_sign_two_sided']))
    if not summary['complete']:
        print('NOTE: only %d of %d runs found.' % (len(rows), n_expected))
    print('Wrote', out_dir / 'unk_results.csv', 'and', out_dir / 'unk_summary.json')


if __name__ == '__main__':
    main()
