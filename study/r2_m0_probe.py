import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
import numpy as np, torch
from scalpel.engine import Session, confusion_metrics
s = Session(os.path.join(RUNS, 'R2_M0')); m = s.model
dev = np.where(s.part < 2)[0]; y = s.y[dev]; part = s.part[dev]
msg = s.b['msg'][dev]; L = (msg != 0).sum(1)
with torch.no_grad():
    x = m.embedMsg(torch.from_numpy(msg).long())
    out, (hn, cn) = m.lstmMsg(x.permute(1, 0, 2))
    out = out.permute(1, 0, 2).numpy()
h = out[np.arange(len(dev)), np.clip(L - 1, 0, None), :32]
h[L == 0] = 0
z = s.base_feats['head'][dev]; z = z[:, 1] - z[:, 0]
d, g = part == 0, part == 1
def fit(X, t, lam):
    Xc = np.c_[X, np.ones(len(X))]
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ t)
t = np.where(y == 1, 1.0, -1.0)
print('base: diag %s guard %s' % (confusion_metrics(y[d], z[d] > 0)['mcc'], confusion_metrics(y[g], z[g] > 0)['mcc']))
for lam in (1.0, 10.0, 100.0):
    X = np.c_[z, h]
    b = fit(X[d], t[d], lam); p = np.c_[X, np.ones(len(X))] @ b
    cd, cg = confusion_metrics(y[d], p[d] > 0), confusion_metrics(y[g], p[g] > 0)
    # Control: the same fit with the margin only (threshold and scale).
    b0 = fit(z[d, None], t[d], lam); p0 = np.c_[z, np.ones(len(z))] @ b0
    c0 = confusion_metrics(y[g], p0[g] > 0)
    print('ridge %5.1f: diag acc %.2f mcc %.4f | guard acc %.2f mcc %.4f | control (margin only) guard acc %.2f mcc %.4f' % (lam, cd['acc'], cd['mcc'], cg['acc'], cg['mcc'], c0['acc'], c0['mcc']))
