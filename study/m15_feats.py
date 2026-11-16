"""Base features of Model15 (CodeBERT). The pass is slow on a CPU, thus it writes chunks and can continue."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import os, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
import numpy as np, torch
from scalpel.models import build_model
B = os.path.join(_HERE, '..', 'bundle_codebert') + os.sep; D = B + 'chunks/'; os.makedirs(D, exist_ok=True)
ids = np.load(B + 'bundle.npz')['msg']
m = build_model(ARCHIVE, 15)
STEP = 200
for a in range(0, len(ids), STEP):
    f = D + 'f_%05d.npy' % a
    if os.path.exists(f): continue
    out = []
    with torch.no_grad():
        for b in range(a, min(a + STEP, len(ids)), 8):
            out.append(m.run_stage('msg', {'msg': torch.from_numpy(ids[b:b + 8])}, None).numpy().copy())
    np.save(f + '.tmp.npy', np.concatenate(out)); os.replace(f + '.tmp.npy', f)
    print('chunk', a, flush=True)
feat = np.concatenate([np.load(D + 'f_%05d.npy' % a) for a in range(0, len(ids), STEP)])
n = len(ids); one = np.zeros((n, 1), dtype=np.float32)
with torch.no_grad():
    head = m.run_stage('head', None, {'msg': torch.from_numpy(feat)}).numpy()
np.savez(B + 'base_feats_M15.npz', twin_b=one, twin_a=one, msg=feat, head=head)
print('FEATSDONE')
