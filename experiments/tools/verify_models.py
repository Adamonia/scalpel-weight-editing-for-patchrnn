"""Check each rebuilt model against predicted.txt of the archive."""
import sys, time
import numpy as np, torch
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(ROOT, '..', 'model-archive'))
PUB_TEMP = os.environ.get('SCALPEL_PUB_TEMP', os.path.join(WORK, 'pub_temp'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scalpel.data import build_dataset
from scalpel.models import build_model, REGISTRY
ARCH = ARCHIVE
d = build_dataset(PUB_TEMP, os.path.join(WORK, 'cache', 'dataset.npz'))
n = int(sys.argv[1]); nums = [int(x) for x in sys.argv[2:]] or list(REGISTRY)
ti = d['test_idx'][:n]
for no in nums:
    m = build_model(ARCH, no)
    pf = np.array([int(x) for x in open(f'{ARCH}/Model{no}/content/predicted.txt').read().split()])[:n]
    t = time.time(); preds = []
    with torch.no_grad():
        for s in range(0, n, 64):
            idx = ti[s:s+64]
            b = {k: torch.from_numpy(d[k][idx]) for k in ['tok_b','typ_b','tok_a','typ_a','msg']}
            acts = {}
            for st in m.STAGES: acts[st] = m.run_stage(st, b, acts)
            preds.extend(acts['head'].argmax(1).tolist())
    preds = np.array(preds)
    print(f'Model{no:<2} {REGISTRY[no][3]:<26} agreement {int((preds==pf).sum())}/{n}  time {time.time()-t:.1f}s', flush=True)
