# M3 message embedding rows: original and after the edit of each round. Rows are sorted by training count c.
import os
import sys
# Paths. Set the SCALPEL_* variables to use other folders.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # repo root
WORK = os.environ.get('SCALPEL_WORK', os.path.join(ROOT, 'work'))
import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'round3'))
import r3lib
s = r3lib.get_session(3)
name, window, cnt = r3lib.table_specs(s, 3)[0]
W = dict(s.model.named_parameters())[name].detach().cpu().double().numpy() if hasattr(s, "model") else None
pad = s.pad_id
vocab = s.vocab["msg"]
rng = np.random.RandomState(4)
def pick(lo, hi, n):
    ids = [i for i in range(len(cnt)) if lo <= cnt[i] <= hi and i != pad]
    return list(rng.choice(ids, n, replace=False))
groups = [(0, 0, 3), (1, 10, 3), (11, 100, 3), (101, 10**9, 3)]
ids = []
for lo, hi, n in groups: ids += sorted(pick(lo, hi, n), key=lambda i: cnt[i])
D = 24
orig = W[ids, :D]
c = cnt[ids].astype(float)
R1 = orig.copy(); R1[c == 0] = 0
R2 = orig.copy(); R2[c <= 10] = 0
R3 = orig * (c / (c + 3))[:, None]
v = np.abs(orig).max()
panels = [("Original", orig), ("Round 1\nuntrained rows = 0", R1), ("Round 2\nrows with c ≤ 10 = 0", R2), ("Round 3\nrow × c / (c + 3)", R3)]
fig, axs = plt.subplots(1, 4, figsize=(15, 5.6), sharey=True)
for ax, (t, M) in zip(axs, panels):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_title(t, fontsize=13); ax.set_xticks([0, 11, 23]); ax.set_xticklabels([1, 12, 24]); ax.set_xlabel("Embedding dimension", fontsize=11)
    for b in (2.5, 5.5, 8.5): ax.axhline(b, color="black", lw=1.5)
lbl = [f"{(vocab[i] if len(vocab[i]) <= 10 else vocab[i][:8] + '..')}  c={int(cnt[i])}" for i in ids]
axs[0].set_yticks(range(len(ids))); axs[0].set_yticklabels(lbl, fontsize=11)
cax = fig.add_axes([0.925, 0.2, 0.012, 0.6]); fig.colorbar(im, cax=cax, label="Weight value")
fig.text(0.01, 0.01, "Not visible here: the bias. Round 1 also subtracted 0.716 from one bias value (prior correction). Round 2 also moved the decision threshold. Round 3 changed no bias.", fontsize=10.5, color="#444")
fig.subplots_adjust(left=0.13, right=0.91, top=0.86, bottom=0.14)
fig.savefig(os.path.join(_HERE, 'fig10_rounds_embedding.png'), dpi=170, facecolor="white")
print(list(zip([vocab[i] for i in ids], c)))
