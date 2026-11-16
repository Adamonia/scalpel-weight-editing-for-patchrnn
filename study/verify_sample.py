"""Independent check of final sessions: load the edited checkpoint files into a new model, run the full
forward pass on a random sample of holdout rows, and compare with the predictions that finalize stored.

Usage: verify_sample.py N RUN [RUN ...]"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import sys, os, json, glob, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
import numpy as np, torch
from scalpel.models import build_model, load_state_dict
n_s = int(sys.argv[1]); out = {}
for run in sys.argv[2:]:
    d = os.path.join(RUNS, run)
    cfg = json.load(open(d + '/config.json')); fin = json.load(open(d + '/FINAL.json'))
    m = build_model(cfg['archive_dir'], cfg['model_no'])
    files = fin.get('edited_files', [fin['edited_checkpoint']])
    for f in files:
        if f.endswith('.safetensors'):
            from safetensors.torch import load_file
            m.load_state_dict(load_file(d + '/' + f), strict=False)
        elif f.startswith('edited_sub_Model'):
            no = int(f.split('_')[2][5:]); prefix = {2: 'cnn', 5: 'trans', 0: 'lstm'}[no]
            getattr(m, prefix).load_state_dict(load_state_dict(d + '/' + f))
        elif cfg['model_no'] == 14:
            m.load_state_dict(load_state_dict(d + '/' + f), strict=False)
        else:
            m.load_state_dict(load_state_dict(d + '/' + f))
    z = np.load(cfg['bundle_dir'] + '/bundle.npz'); fp = np.load(d + '/final_predictions.npz')
    rows = np.sort(np.random.RandomState(7).choice(np.where(z['part'] == 2)[0], n_s, replace=False))
    logits = []
    with torch.no_grad():
        for a in range(0, len(rows), 8):
            r = rows[a:a + 8]
            b = {k: torch.from_numpy(z[k][r]) for k in ('tok_b', 'typ_b', 'tok_a', 'typ_a', 'msg')}
            acts = {st: m.run_stage(st, b, None) for st in ('twin_b', 'twin_a', 'msg')}
            logits.append(m.run_stage('head', None, acts).numpy())
    lg = np.concatenate(logits)
    res = {'rows': n_s, 'files': files, 'pred_equal': int((lg.argmax(1) == fp['pred_edited'][rows]).sum()),
           'max_abs_logit_gap': float(np.abs(lg - fp['logits_edited'][rows]).max()),
           'changed_vs_original_in_sample': int((fp['pred_edited'][rows] != fp['pred_base'][rows]).sum())}
    out[run] = res; print(run, res, flush=True)
json.dump(out, open(os.path.join(_HERE, '..', 'results', 'verify_sample_%s.json' % '_'.join(sys.argv[2:4])), 'w'), indent=1)
