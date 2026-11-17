"""Choose the Round 3 recipe from DIAG numbers only (rule in RULE.md). Writes out/choice_diag.json."""
from analysis import *

part_name = 'diag'
cands = [('%s|%s' % (variant_name(e, k), rc), e, k, rc) for e, k in all_variants() if e != 'E1' for rc in ('raw', 'rc')]
conf, n, _ = {}, None, None
for no in TRANSFORMERS6:
    conf[no], n, yv = margins_and_conf(no, part_name)
w = boot_weights(n, 2000, SEEDS[part_name])
F = {no: {cfg: boot_f1(w, c) for cfg, c in conf[no].items()} for no in TRANSFORMERS6}
point = {no: {cfg: 200.0 * c[0].sum() / (2 * c[0].sum() + c[1].sum() + c[2].sum()) for cfg, c in conf[no].items()} for no in TRANSFORMERS6}
rows = []
for cfg, e, k, rc in cands:
    name, flag = cfg.split('|')
    base = 'E1|' + flag
    d = [point[no][cfg] - point[no][base] for no in TRANSFORMERS6]
    bd = np.mean([F[no][cfg] - F[no][base] for no in TRANSFORMERS6], axis=0)
    rows.append({'cfg': cfg, 'edit': e, 'k': k, 'rc': flag == 'rc', 'score': float(np.mean(d)),
                 'per_model': dict(zip(map(str, TRANSFORMERS6), map(float, d))),
                 'ci95': [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]})
best = max(r['score'] for r in rows)
tied = [r for r in rows if r['score'] >= best - 0.05]
order = {'E2': 0, 'E4': 1, 'E3': 2}
tied.sort(key=lambda r: (order[r['edit']], r['rc'], r['k'] or 0))
win = tied[0]
gate = win['score'] > 0 and win['ci95'][0] > 0
choice = {'rows': rows, 'best_score': best, 'tied': [r['cfg'] for r in tied], 'winner_before_gate': win['cfg'], 'gate_passed': bool(gate)}
if gate:
    choice['recipe'] = {'edit': win['edit'], 'k': win['k'], 'recenter': win['rc']}
else:
    # E1 fallback: recenter chosen by mean diag F1 change against E0
    d0 = {}
    for no in TRANSFORMERS6:
        z = conf[no]
        p0 = 200.0 * z['E0'][0].sum() / (2 * z['E0'][0].sum() + z['E0'][1].sum() + z['E0'][2].sum())
        for flag in ('raw', 'rc'):
            d0.setdefault(flag, []).append(point[no]['E1|' + flag] - p0)
    m0 = {f: float(np.mean(v)) for f, v in d0.items()}
    use_rc = m0['rc'] > m0['raw'] + 0.05
    choice['E1_fallback_mean_diag_change_vs_E0'] = m0
    choice['recipe'] = {'edit': 'E1', 'k': None, 'recenter': bool(use_rc)}
json.dump(choice, open(OUT + '/choice_diag.json', 'w'), indent=1)
for r in sorted(rows, key=lambda r: -r['score'])[:8]:
    print('%-12s score %+.3f ci [%+.3f, %+.3f]' % (r['cfg'], r['score'], *r['ci95']))
print('best', best, 'tied', choice['tied'], 'winner', win['cfg'], 'gate', gate, 'recipe', choice['recipe'])
