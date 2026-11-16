"""After the recipe was fixed on diag (out/choice_diag.json): metrics on diag AND guard for all models,
recipe.json, decisions.json, metrics.json, summary.json. Holdout is never touched."""
from analysis import *

assert os.path.exists(OUT + '/choice_diag.json'), 'choose the recipe on diag first'
choice = json.load(open(OUT + '/choice_diag.json'))
rec = choice['recipe']
rcfg = '%s|%s' % (variant_name(rec['edit'], rec['k']), 'rc' if rec['recenter'] else 'raw')
models = [m for m in ALL_MODELS if os.path.exists('%s/margins_M%d.npz' % (OUT, m))]
GROUPS = {'transformers6': TRANSFORMERS6, 'word_level8': [0, 2, 3, 4, 5, 8, 12, 13],
          'all_evaluated': [m for m in models]}
metrics, boots = {}, {}
for pn in ('diag', 'guard'):
    metrics[pn] = {}
    conf, joint = {}, {}
    for no in models:
        res, shifts = part_metrics(no, pn)
        conf[no], n, _ = margins_and_conf(no, pn)
        if no == models[0]:
            w = boot_weights(n, 2000, SEEDS[pn])
        Fb = {cfg: boot_f1(w, c) for cfg, c in conf[no].items()}
        joint[no] = Fb
        e0 = res['E0']
        for cfg, r in res.items():
            flag = cfg.split('|')[1] if '|' in cfg else 'raw'
            b1 = 'E1|' + flag
            r['d_f1_E0'], r['d_auc_E0'], r['d_ap_E0'] = r['f1'] - e0['f1'], r['auc'] - e0['auc'], r['ap'] - e0['ap']
            r['d_f1_E1'], r['d_auc_E1'], r['d_ap_E1'] = r['f1'] - res[b1]['f1'], r['auc'] - res[b1]['auc'], r['ap'] - res[b1]['ap']
            dd = Fb[cfg] - Fb[b1]
            r['d_f1_E1_ci95'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            d0 = Fb[cfg] - Fb['E0']
            r['d_f1_E0_ci95'] = [float(np.percentile(d0, 2.5)), float(np.percentile(d0, 97.5))]
        res['_shifts'] = shifts
        metrics[pn][str(no)] = res
    # group summaries per config
    summ = {}
    cfgs = [c for c in metrics[pn][str(models[0])] if c != '_shifts' and not c.startswith('E1_cnnonly')]
    for g, ms in GROUPS.items():
        summ[g] = {}
        for cfg in cfgs:
            flag = cfg.split('|')[1] if '|' in cfg else 'raw'
            d1 = np.array([metrics[pn][str(m)][cfg]['d_f1_E1'] for m in ms])
            d0 = np.array([metrics[pn][str(m)][cfg]['d_f1_E0'] for m in ms])
            bd1 = np.mean([joint[m][cfg] - joint[m]['E1|' + flag] for m in ms], axis=0)
            bd0 = np.mean([joint[m][cfg] - joint[m]['E0'] for m in ms], axis=0)
            summ[g][cfg] = {'mean_d_f1_E0': float(d0.mean()), 'n_improve_E0': int((d0 > 0).sum()),
                            'mean_d_f1_E1': float(d1.mean()), 'n_improve_E1': int((d1 > 0).sum()),
                            'ci95_mean_d_f1_E1': [float(np.percentile(bd1, 2.5)), float(np.percentile(bd1, 97.5))],
                            'ci95_mean_d_f1_E0': [float(np.percentile(bd0, 2.5)), float(np.percentile(bd0, 97.5))],
                            'n_models': len(ms)}
    boots[pn] = summ
json.dump(metrics, open(OUT + '/metrics.json', 'w'), indent=1)
json.dump(boots, open(OUT + '/summary.json', 'w'), indent=1)

# ---- M15: not applicable, original kept; F1 from the cached head output of the unedited model
z = np.load(W + '/bundle_codebert/base_feats_M15.npz')
zb = np.load(W + '/bundle_codebert/bundle.npz')
p15, y15 = zb['part'], zb['y'].astype(np.int64)
m15 = np.full(len(y15), np.nan)
dev15 = np.where(p15 < HOLDOUT)[0]
m15[dev15] = margin_of(z['head'][dev15])
f15 = {pn: f1_pct(p15, part_idx(p15, pn), y15, m15) for pn in ('diag', 'guard')}

# ---- recipe.json
S = boots
def sel(pn, g='transformers6'):
    return S[pn][g][rcfg]
recipe = {'edit': rec['edit'], 'k': rec['k'], 'recenter': rec['recenter'], 'config': rcfg,
          'rule': open(HERE + '/RULE.md').read(),
          'diag': {'per_model_d_f1_vs_E1': {m: metrics['diag'][m][rcfg]['d_f1_E1'] for m in map(str, TRANSFORMERS6)},
                   'transformers6': sel('diag'), 'word_level8': sel('diag', 'word_level8'),
                   'best_score': choice['best_score'], 'gate_passed': choice['gate_passed']},
          'guard': {'per_model_d_f1_vs_E1': {m: metrics['guard'][m][rcfg]['d_f1_E1'] for m in map(str, TRANSFORMERS6)},
                    'transformers6': sel('guard'), 'word_level8': sel('guard', 'word_level8'),
                    'all_evaluated': sel('guard', 'all_evaluated')}}
g = recipe['guard']['transformers6']
if g['mean_d_f1_E1'] > 0 and g['ci95_mean_d_f1_E1'][0] > 0:
    conf_txt = 'confirmed'
elif g['mean_d_f1_E1'] > 0:
    conf_txt = 'direction confirmed, interval includes 0 (not significant)'
else:
    conf_txt = 'NOT confirmed'
recipe['guard_confirms_diag'] = conf_txt
json.dump(recipe, open(OUT + '/recipe.json', 'w'), indent=1)

# ---- decisions.json (12 entries)
dec = {}
for no in ALL_MODELS:
    if no == 15:
        dec['15'] = {'applied': 'original', 'applicable': False,
                     'reason': 'not applicable: CodeBERT embeddings come from pre-training (rows never seen in fine-tuning are meaningful vectors, not random N(0,1)); padding is masked by the attention mask so the pad row is not read; Round 1 also made no commit. Not evaluated; original kept.',
                     'diag_f1_before': f15['diag'], 'diag_f1_after': f15['diag'], 'guard_f1_before': f15['guard'], 'guard_f1_after': f15['guard']}
        continue
    if no not in models:
        dec[str(no)] = {'applied': 'missing', 'reason': 'margins not computed'}
        continue
    md, mg = metrics['diag'][str(no)], metrics['guard'][str(no)]
    ok = mg[rcfg]['f1'] >= mg['E0']['f1']
    dec[str(no)] = {'applied': rcfg if ok else 'original', 'applicable': True,
                    'reason': ('guard F1 %.3f >= original %.3f: recipe accepted' if ok else 'guard F1 %.3f < original %.3f: recipe refused, original weights kept') % (mg[rcfg]['f1'], mg['E0']['f1']),
                    'diag_f1_before': md['E0']['f1'], 'diag_f1_after': md[rcfg]['f1'] if ok else md['E0']['f1'],
                    'guard_f1_before': mg['E0']['f1'], 'guard_f1_after': mg[rcfg]['f1'] if ok else mg['E0']['f1'],
                    'recipe_if_applied_diag_f1': md[rcfg]['f1'], 'recipe_if_applied_guard_f1': mg[rcfg]['f1'],
                    'E1_diag_f1': md['E1|' + ('rc' if rec['recenter'] else 'raw')]['f1'], 'E1_guard_f1': mg['E1|' + ('rc' if rec['recenter'] else 'raw')]['f1']}
json.dump(dec, open(OUT + '/decisions.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in recipe.items() if k != 'rule'}, indent=1))
print(json.dumps({k: (v['applied'], round(v['guard_f1_before'], 2), round(v['guard_f1_after'], 2)) for k, v in dec.items() if 'guard_f1_before' in v}))
