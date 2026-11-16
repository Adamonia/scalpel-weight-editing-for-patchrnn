"""Build caches: raw text of the test samples and token statistics of the training set."""
import json, sys
import numpy as np
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
from scalpel.safe_npy import load_object_npy
from scalpel.data import build_dataset
d = build_dataset(PUB_TEMP, os.path.join(WORK, 'cache', 'dataset.npz'))
raw = load_object_npy(os.path.join(PUB_TEMP, 'data.npy'))
out = []
for i in d['test_idx']:
    msg_lines, hunks, label = raw[i]
    assert label == d['labels'][i]
    msg = ''.join(msg_lines)
    diff = '\n@@\n'.join(''.join(h) for h in hunks)
    out.append({'msg': msg[:4000], 'diff': diff[:12000], 'diff_chars': len(diff), 'n_hunks': len(hunks)})
with open(os.path.join(WORK, 'cache', 'raw_test.json'), 'w', encoding='utf-8') as fp:
    json.dump(out, fp, ensure_ascii=False)
print('raw test samples', len(out))
