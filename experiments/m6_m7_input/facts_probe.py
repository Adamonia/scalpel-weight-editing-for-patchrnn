# Probe: what does the training-time text look like, and how does the tokenizer split it?
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/work/scalpel_project')
from scalpel import retokenize as R
z = np.load('/home/claude/work/cache/dataset.npz'); v = json.load(open('/home/claude/work/cache/dataset.npz.vocab.json'))
print(v.keys())
code_v = v['diff_vocab']; msg_v = v['msg_vocab']
i = z['test_idx'][3]
ids = [t for t in z['tok_a'][i] if t]
print(ids[:40]); print([code_v[t] for t in ids[:40]])
mids=[t for t in z['msg'][i] if t]; print([msg_v[t] for t in mids[:30]])
for kind, files in (('bert',['/home/claude/work/tokenizers/bert/vocab.txt']),('roberta',['/home/claude/work/tokenizers/roberta/vocab.json','/home/claude/work/tokenizers/roberta/merges.txt'])):
    tok = R.load_tokenizer(kind, files)
    s_join = ''.join(code_v[t] for t in ids[:25]); s_sp = ' '.join(code_v[t] for t in ids[:25])
    print(kind); print(' JOIN', s_join); print(tok.convert_ids_to_tokens(tok.encode(s_join, add_special_tokens=False)))
    print(' SPACE', s_sp); print(tok.convert_ids_to_tokens(tok.encode(s_sp, add_special_tokens=False)))
