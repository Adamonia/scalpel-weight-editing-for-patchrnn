"""Check each rebuilt model against predicted.txt of the archive."""
import sys, time
import numpy as np, torch
sys.path.insert(0, '/home/claude/work/scalpel_project')
from scalpel.data import build_dataset
from scalpel.models import build_model, REGISTRY
ARCH = '/mnt/user-data/uploads/patchrnn-archive/model-archive'
d = build_dataset('/home/claude/ext/pub_temp', '/home/claude/work/cache/dataset.npz')
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
