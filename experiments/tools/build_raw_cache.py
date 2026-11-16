"""Build caches: raw text of the test samples and token statistics of the training set."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/claude/work/scalpel_project')
from scalpel.safe_npy import load_object_npy
from scalpel.data import build_dataset
d = build_dataset('/home/claude/ext/pub_temp', '/home/claude/work/cache/dataset.npz')
raw = load_object_npy('/home/claude/ext/pub_temp/data.npy')
out = []
for i in d['test_idx']:
    msg_lines, hunks, label = raw[i]
    assert label == d['labels'][i]
    msg = ''.join(msg_lines)
    diff = '\n@@\n'.join(''.join(h) for h in hunks)
    out.append({'msg': msg[:4000], 'diff': diff[:12000], 'diff_chars': len(diff), 'n_hunks': len(hunks)})
with open('/home/claude/work/cache/raw_test.json', 'w', encoding='utf-8') as fp:
    json.dump(out, fp, ensure_ascii=False)
print('raw test samples', len(out))
