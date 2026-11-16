# Check: my encoding of the training format equals the project bundle, and the model reproduces predicted.txt.
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for common.py
from common import *
for m in (6, 7):
    b = load_bundle(m); ids = encode_variant(m, 'orig')
    same = {k: bool((ids[k] == b[k]).all()) for k in ('tok_b', 'tok_a', 'msg')}
    print('M%d ids equal bundle' % m, same, flush=True)
    t = time.time(); model = build_model(ARCH, m)
    lg = run_model(model, {k: b[k] for k in ('tok_b','tok_a','msg')}, None, log=True)
    pf = np.array([int(x) for x in open('%s/Model%d/content/predicted.txt' % (ARCH, m)).read().split()])
    print('M%d preds vs predicted.txt: %d / %d match; %.0fs' % (m, (lg.argmax(1) == pf[:len(lg)]).sum(), len(lg), time.time() - t), len(pf), flush=True)
    np.save('%s/logits_M%d_orig.npy' % (CACHE, m), lg)
