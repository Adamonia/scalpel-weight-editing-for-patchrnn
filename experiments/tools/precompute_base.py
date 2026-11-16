"""Compute the base stage outputs of each archive model and compare with predicted.txt."""
import sys, time, os, json
import numpy as np
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(ROOT, '..', 'model-archive'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(ROOT, 'bundle'))
PUB_TEMP = os.environ.get('SCALPEL_PUB_TEMP', os.path.join(WORK, 'pub_temp'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scalpel.engine import Session
ARCH = ARCHIVE; B = BUNDLE
res = {}
for no in [int(x) for x in sys.argv[1:]]:
    t = time.time()
    run = os.path.join(os.environ.get('SCALPEL_RUNS', os.path.join(WORK, 'runs')), f'_pre{no}')
    if not os.path.exists(run + '/config.json'):
        s = Session.create(run, ARCH, B, no)
    else:
        s = Session(run)
    pf = np.array([int(x) for x in open(f'{ARCH}/Model{no}/content/predicted.txt').read().split()])
    agree = int((s.base_pred == pf).sum())
    res[no] = agree
    print(f'Model{no} {s.label}: agreement with predicted.txt {agree}/{len(pf)}; {time.time()-t:.0f}s', flush=True)
json.dump(res, open(os.path.join(WORK, 'agreement_%s.json' % '_'.join(sys.argv[1:])), 'w'))
