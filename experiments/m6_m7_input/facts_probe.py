# Probe: what does the training-time text look like, and how does the tokenizer split it?
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import sys, json, numpy as np
from scalpel import retokenize as R
z = np.load(os.path.join(WORK, 'cache', 'dataset.npz')); v = json.load(open(os.path.join(WORK, 'cache', 'dataset.npz.vocab.json')))
print(v.keys())
code_v = v['diff_vocab']; msg_v = v['msg_vocab']
i = z['test_idx'][3]
ids = [t for t in z['tok_a'][i] if t]
print(ids[:40]); print([code_v[t] for t in ids[:40]])
mids=[t for t in z['msg'][i] if t]; print([msg_v[t] for t in mids[:30]])
for kind, files in (('bert',[ROOT + '/tokenizers/bert/vocab.txt']),('roberta',[ROOT + '/tokenizers/roberta/vocab.json',ROOT + '/tokenizers/roberta/merges.txt'])):
    tok = R.load_tokenizer(kind, files)
    s_join = ''.join(code_v[t] for t in ids[:25]); s_sp = ' '.join(code_v[t] for t in ids[:25])
    print(kind); print(' JOIN', s_join); print(tok.convert_ids_to_tokens(tok.encode(s_join, add_special_tokens=False)))
    print(' SPACE', s_sp); print(tok.convert_ids_to_tokens(tok.encode(s_sp, add_special_tokens=False)))
