import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
import numpy as np, torch
from scalpel.engine import Session, confusion_metrics
from scalpel.tools import dispatch
s = Session(os.path.join(RUNS, 'R2_M0')); A = 'claude-fable-5-2'
m = s.model
r = dispatch(s, 'unit_report', {'stage': 'msg', 'top': 64}, actor=A)
h = {u['unit']: u['mean_help'] for u in r['most_harmful'] + r['most_helpful']}
print('msg units help: forward(0-31) %.3f  reverse(32-63) %.3f; negative units %d' % (sum(h[u] for u in range(32)), sum(h[u] for u in range(32, 64)), r['n_units_with_negative_help']))
for st in ('twin_b', 'twin_a'):
    r = dispatch(s, 'unit_report', {'stage': st, 'top': 64}, actor=A)
    h = {u['unit']: u['mean_help'] for u in r['most_harmful'] + r['most_helpful']}
    print('%s units help: forward %.3f reverse %.3f' % (st, sum(h[u] for u in range(32)), sum(h[u] for u in range(32, 64))))
# Padding row and its effect on the forward state.
w = s.tensor('embedMsg.weight')
print('pad row norm msg %.3f, mean row norm %.3f; code pad norm %.3f' % (w[0].norm(), w[1:2000].norm(dim=1).mean(), s.tensor('embedTwin.weight')[0].norm()))
rows = s.rows_of('diag'); y = s.y[rows]
msg = s.b['msg'][rows]; L = (msg != 0).sum(1)
# Forward final state with and without the padding steps.
with torch.no_grad():
    x = m.embedMsg(torch.from_numpy(msg).long())
    out, (hn, _) = m.lstmMsg(x.permute(1, 0, 2))           # out: [T, N, 64]
    out = out.permute(1, 0, 2).numpy()
    idx = np.clip(L - 1, 0, None)
    h_last_real = out[np.arange(len(rows)), idx, :32]       # forward state at the last real token
    h_final = hn[0].numpy()
d = np.linalg.norm(h_final - h_last_real, axis=1); n0 = np.linalg.norm(h_last_real, axis=1)
for lo, hi in ((1, 5), (6, 20), (21, 60), (61, 199)):
    k = (L >= lo) & (L <= hi)
    print('len %3d-%3d n=%4d: |h_final - h_lastreal| %.3f, |h_lastreal| %.3f, |h_final| %.3f, std over samples of h_final %.3f' % (lo, hi, k.sum(), d[k].mean(), n0[k].mean(), np.linalg.norm(h_final[k], axis=1).mean(), h_final[k].std(0).mean()))
print('std of h_final over all diag %.4f ; std of h_last_real %.4f' % (h_final.std(0).mean(), h_last_real.std(0).mean()))
# Diagnostic only: head on the forward state without padding drift.
f = s.feats['msg'][rows].copy(); f2 = f.copy(); f2[:, :32] = h_last_real
lg0 = s.head_from(s.feats['twin_b'][rows], s.feats['twin_a'][rows], f)
lg2 = s.head_from(s.feats['twin_b'][rows], s.feats['twin_a'][rows], f2)
f3 = f.copy(); f3[:, :32] = f[:, :32].mean(0)
lg3 = s.head_from(s.feats['twin_b'][rows], s.feats['twin_a'][rows], f3)
f4 = f.copy(); f4[:, 32:] = f[:, 32:].mean(0)
lg4 = s.head_from(s.feats['twin_b'][rows], s.feats['twin_a'][rows], f4)
for nm, lg in (('base', lg0), ('forward state without padding (diagnostic)', lg2), ('forward half = mean', lg3), ('reverse half = mean', lg4)):
    c = confusion_metrics(y, lg.argmax(1)); print('  %-45s acc %.2f mcc %.4f' % (nm, c['acc'], c['mcc']))
