"""Compute the base stage outputs of each archive model and compare with predicted.txt."""
import sys, time, os, json
import numpy as np
sys.path.insert(0, '/home/claude/work/scalpel_project')
from scalpel.engine import Session
ARCH = '/mnt/user-data/uploads/patchrnn-archive/model-archive'; B = '/home/claude/work/bundle'
res = {}
for no in [int(x) for x in sys.argv[1:]]:
    t = time.time()
    run = f'/home/claude/work/runs/_pre{no}'
    if not os.path.exists(run + '/config.json'):
        s = Session.create(run, ARCH, B, no)
    else:
        s = Session(run)
    pf = np.array([int(x) for x in open(f'{ARCH}/Model{no}/content/predicted.txt').read().split()])
    agree = int((s.base_pred == pf).sum())
    res[no] = agree
    print(f'Model{no} {s.label}: agreement with predicted.txt {agree}/{len(pf)}; {time.time()-t:.0f}s', flush=True)
json.dump(res, open('/home/claude/work/agreement_%s.json' % '_'.join(sys.argv[1:]), 'w'))
