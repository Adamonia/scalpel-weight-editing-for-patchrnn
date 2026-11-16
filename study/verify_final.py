"""Independent check of the final results.

For each model: load the edited checkpoint file with the plain model class (no Session, no caches),
run the holdout rows, compare with final_predictions.npz, and recompute the metrics with scikit-learn.

Usage: verify_final.py MODEL [MODEL ...]     (needs scikit-learn)
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCH = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
OUT = os.environ.get('SCALPEL_RESULTS', os.path.join(_HERE, '..', 'results'))

import json, sys, time
import numpy as np, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
sys.path.insert(0, os.path.join(_HERE, '..'))
from scalpel.models import REGISTRY, load_state_dict
z = np.load(os.path.join(BUNDLE, 'bundle.npz'))
y, part = z['y'], z['part']
rows = np.where(part == 2)[0]
out = {}
for no in [int(a) for a in sys.argv[1:]]:
    fname, cls, kw, label = REGISTRY[no]
    run = os.path.join(RUNS, 'M%d' % no)
    sd = load_state_dict(f'{run}/edited_{fname}')
    model = cls(sd, **kw).eval()
    bs = 64 if no in (0, 2) else 16
    logits = []
    t = time.time()
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            r = rows[s:s + bs]
            b = {k: torch.from_numpy(z[k][r]) for k in ('tok_b', 'typ_b', 'tok_a', 'typ_a', 'msg')}
            acts = {}
            for st in model.STAGES: acts[st] = model.run_stage(st, b, acts)
            logits.append(acts['head'].numpy())
    logits = np.concatenate(logits); pred = logits.argmax(1)
    fp = np.load(f'{run}/final_predictions.npz')
    fin = json.load(open(f'{run}/FINAL.json'))['parts']['holdout']['after']
    yh = y[rows]
    mine = {'acc': 100 * accuracy_score(yh, pred), 'precision': 100 * precision_score(yh, pred), 'recall': 100 * recall_score(yh, pred),
            'f1': 100 * f1_score(yh, pred), 'mcc': matthews_corrcoef(yh, pred)}
    ok_pred = bool((pred == fp['pred_edited'][rows]).all())
    gap = float(np.abs(logits - fp['logits_edited'][rows]).max())
    ok_metrics = all(abs(mine[k] - fin[k]) < 6e-4 for k in mine)
    # the original checkpoint must differ only in the tensors listed in weight_diff
    sd0 = load_state_dict(f'{ARCH}/Model{no}/content/temp/{fname}')
    changed = sorted(k for k in sd if not torch.equal(sd[k], sd0[k]))
    listed = sorted(w['tensor'] for w in json.load(open(f'{run}/FINAL.json'))['weight_diff'])
    out[no] = {'pred_equal': ok_pred, 'max_logit_gap': gap, 'metrics_equal': ok_metrics, 'changed_tensors_match': changed == listed,
               'auc': roc_auc_score(yh, logits[:, 1] - logits[:, 0]), 'acc': mine['acc'], 'seconds': round(time.time() - t)}
    print(f'M{no} {label}: predictions equal {ok_pred}; max logit gap {gap:.2e}; sklearn metrics equal {ok_metrics}; changed tensors {changed} match {changed == listed}; acc {mine["acc"]:.2f} auc {out[no]["auc"]:.4f}; {out[no]["seconds"]}s', flush=True)
json.dump(out, open(os.path.join(OUT, 'verify_final.json'), 'w'), indent=1)
