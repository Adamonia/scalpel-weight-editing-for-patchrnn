"""Figures for the Scalpel weight-editing method. Read-only on inputs; writes only to this folder.
Run: python3 make_figs.py
"""
import sys, os, json, shutil, tempfile
sys.path.insert(0, '/home/claude/work/ext')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import precision_recall_curve

FIG = '/home/claude/work/fig'
OUT = '/home/claude/work/ext/out'
BLUE, ORANGE, GREY = '#2a6fbb', '#e08a1e', '#8a8a8a'
MODELS = [0, 2, 3, 4, 5, 8, 12, 13]
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 15,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False})
notes = []      # lines for NUMBERS.md


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig('%s/%s.%s' % (FIG, name, ext), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load(no):
    d = np.load('%s/logits_M%d.npz' % (OUT, no))
    return {k: d[k] for k in d.files}


bundle = np.load('/home/claude/work/bundle/bundle.npz')
part = bundle['part']
ho = part == 2


def f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return 100 * 2 * tp / (2 * tp + fp + fn)


def prec_rec(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return tp / (tp + fp), tp / (tp + fn)


# ---------------- Fig 1: PR curves on holdout ----------------
fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
notes.append('## Fig 1: precision-recall on holdout (n=%d), threshold 0' % ho.sum())
for ax, no in zip(axs, (3, 4)):
    L = load(no); y = L['y'][ho]
    for key, col, lab in (('orig', BLUE, 'Original'), ('untrained_zero', ORANGE, 'Edited')):
        m = L[key][ho]
        p, r, _ = precision_recall_curve(y, m)
        ax.plot(r, p, color=col, lw=2.5, label=lab)
        pp, rr = prec_rec(y, (m > 0).astype(int))
        ax.plot(rr, pp, 'o', color=col, ms=11, mec='white', mew=1.5, zorder=5)
        notes.append('- M%d %s: precision %.4f, recall %.4f (F1 %.2f)' % (no, key, pp, rr, f1(y, (m > 0).astype(int))))
    ax.set_title('Model M%d' % no); ax.set_xlabel('Recall')
    ax.set_xlim(0, 1); ax.set_ylim(0.3, 1.02)
axs[0].set_ylabel('Precision')
axs[0].legend(loc='lower left', frameon=False)
fig.text(0.5, -0.02, 'Dots mark the operating threshold (margin = 0)', ha='center', fontsize=13)
fig.tight_layout(); save(fig, 'fig1_pr_curve')

# ---------------- Fig 2: placebo strip ----------------
rows = []
notes.append('\n## Fig 2: F1 change in pp, full set (n=7608). Edited = untrained_zero; placebo = 20 freqmatch seeds')
for no in MODELS:
    L = load(no); y = L['y']; base = f1(y, (L['orig'] > 0).astype(int))
    ed = f1(y, (L['untrained_zero'] > 0).astype(int)) - base
    pl = [f1(y, (L['placebo_freqmatch_%d' % k] > 0).astype(int)) - base for k in range(20)]
    rows.append((no, ed, pl))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axhline(0, color='black', lw=1)
rng = np.random.RandomState(0)
for i, (no, ed, pl) in enumerate(rows):
    ax.plot(i + rng.uniform(-.15, .15, 20), pl, 'o', color=GREY, ms=6, alpha=.8, mec='none',
            label='Placebo (20 seeds)' if i == 0 else None)
    ax.plot(i, ed, 'o', color=ORANGE, ms=13, mec='white', mew=1.5, zorder=5, label='Edited' if i == 0 else None)
    notes.append('- M%d: edited %+.2f pp; placebo mean %+.2f, min %+.2f, max %+.2f' % (no, ed, np.mean(pl), min(pl), max(pl)))
ax.set_xticks(range(len(rows))); ax.set_xticklabels(['M%d' % r[0] for r in rows])
ax.set_ylabel('F1 change (percentage points)'); ax.set_title('Edit effect vs matched placebo edits')
ax.legend(frameon=False, loc='best')
fig.tight_layout(); save(fig, 'fig2_placebo')

# ---------------- Fig 3: flow on M3 holdout ----------------
L = load(3); y = L['y'][ho]; a = L['orig'][ho] > 0; b = L['untrained_zero'][ho] > 0
ra, rb = a == y, b == y
cnt = [('Wrong to right', int((~ra & rb).sum()), ORANGE), ('Right to wrong', int((ra & ~rb).sum()), '#b03a2e'),
       ('Unchanged right', int((ra & rb).sum()), BLUE), ('Unchanged wrong', int((~ra & ~rb).sum()), GREY)]
assert sum(c[1] for c in cnt) == ho.sum()
fig, ax = plt.subplots(figsize=(11, 3.2))
order = [2, 3, 0, 1]   # big bars first, flipped ones at the right
left = 0
total = ho.sum()
for i in order:
    n, c = cnt[i][1], cnt[i][2]
    ax.barh(0, n, left=left, color=c, height=.5)
    left += n
ax.set_xlim(0, total); ax.set_ylim(-1.2, 0.5); ax.axis('off')
# labels in a table under the bar (small bars are too thin for text)
xs = [0.02, 0.27, 0.52, 0.77]
for x, i in zip(xs, order):
    ax.text(x * total, -0.45, '%d' % cnt[i][1], color=cnt[i][2], fontsize=24, fontweight='bold', va='top')
    ax.text(x * total, -0.85, cnt[i][0], fontsize=15, va='top')
ax.set_title('M3 holdout (%d samples): who changes correctness after the edit' % total, loc='left')
notes.append('\n## Fig 3: M3 holdout, n=%d' % total)
for n_, c_, _ in cnt: notes.append('- %s: %d' % (n_, c_))
fig.tight_layout(); save(fig, 'fig3_flow')

# ---------------- Fig 4: example ----------------
tmp = tempfile.mkdtemp()
shutil.copytree('/home/claude/work/ext/tmp/S_M3', tmp + '/S_M3')     # copy so the original is never touched
from common import Session, untrained_ids, msg_window
s = Session(tmp + '/S_M3')
U = set(int(i) for i in untrained_ids(s))
w = msg_window(s)
ho_idx = np.where(ho)[0]
d = L['untrained_zero'][ho] - L['orig'][ho]
yy = L['y'][ho]
w_h = w[ho]
mlen = (w_h != s.pad_id).sum(1)
ok = np.where(~ra & rb)[0]
# score: margin moved in the right direction, large, with a short message
sign = np.where(yy == 1, 1, -1)
cand = [(sign[i] * d[i], mlen[i], i) for i in ok if 4 <= mlen[i] <= 14]
cand.sort(key=lambda t: -t[0])
best = cand[0][2]
gi = int(ho_idx[best])
toks = [int(t) for t in w_h[best] if t != s.pad_id]
words = [s.vocab['msg'][t] for t in toks]
untr = [t in U for t in toks]
m0, m1 = float(L['orig'][gi]), float(L['untrained_zero'][gi])
notes.append('\n## Fig 4: example, M3 holdout sample')
notes.append('- global sample index %d (position %d inside holdout), true label %d' % (gi, best, yy[best]))
notes.append('- token ids: %s' % toks)
notes.append('- tokens: %s' % words)
notes.append('- untrained flags: %s' % untr)
notes.append('- margin before %.3f, after %.3f (change %+.3f); prediction %d -> %d' % (m0, m1, m1 - m0, m0 > 0, m1 > 0))
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0.2, 4.2)
x, yrow = 0.2, 3.0
for wd, u in zip(words, untr):
    wd = wd if len(wd) <= 22 else wd[:20] + '..'
    wbox = 0.3 + 0.17 * len(wd)
    if x + wbox > 11.8:
        x, yrow = 0.2, yrow - 0.8
    ax.add_patch(FancyBboxPatch((x, yrow), wbox, 0.55, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=ORANGE if u else '#eeeeee', ec='black' if u else GREY, lw=1.2))
    ax.text(x + wbox / 2, yrow + .275, wd, ha='center', va='center', fontsize=14, color='black')
    x += wbox + 0.15
ax.text(0.2, 3.85, 'Commit message tokens (orange = untrained, set to zero)', fontsize=15, fontweight='bold')
lab = 'security patch' if yy[best] == 1 else 'not a security patch'
ax.text(0.2, 1.9, 'True label: %s' % lab, fontsize=15)
ax.text(0.2, 1.1, 'Margin before: %+.2f  (wrong)' % m0, fontsize=17, color=BLUE, fontweight='bold')
ax.text(6.2, 1.1, 'Margin after: %+.2f  (right)' % m1, fontsize=17, color=ORANGE, fontweight='bold')
fig.tight_layout(); save(fig, 'fig4_example')

# ---------------- Fig 5: embedding heatmap ----------------
W = s.model.embedMsg.weight.detach().numpy()
tr = np.where(s.trained['msg'])[0]; tr = tr[tr != s.pad_id]
rng = np.random.RandomState(1)
# pick 6 trained and 6 untrained tokens that occur in the example message pool (random, fixed seed)
tr_ids = rng.choice(tr, 6, replace=False)
un_ids = rng.choice(np.array(sorted(U)), 6, replace=False)
ids = np.concatenate([tr_ids, un_ids]); D = 24
before = W[ids, :D]
after = before.copy(); after[6:] = 0
v = max(abs(before).max(), 1e-6)
vocab = s.vocab['msg']
fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, M, t in zip(axs, (before, after), ('Before edit', 'After edit')):
    im = ax.imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    ax.set_title(t); ax.set_xlabel('Embedding dimension (first 24)')
    ax.set_xticks([0, 7, 15, 23]); ax.set_xticklabels([1, 8, 16, 24])
    ax.axhline(5.5, color='black', lw=2)
axs[0].set_yticks(range(12))
lbl = [(vocab[i] if len(vocab[i]) <= 12 else vocab[i][:10] + '..') for i in ids]
axs[0].set_yticklabels(lbl, fontsize=12)
for ax in axs:
    ax.text(24.2, 2.5, 'trained', rotation=-90, va='center', fontsize=13, clip_on=False)
    ax.text(24.2, 8.5, 'untrained', rotation=-90, va='center', fontsize=13, clip_on=False, color=ORANGE)
fig.colorbar(im, ax=axs, shrink=.8, label='Weight value')
save(fig, 'fig5_embedding')
notes.append('\n## Fig 5: M3 message embedding rows (first 24 of %d dims); colour range +-%.4f' % (W.shape[1], v))
notes.append('- trained rows (unchanged): ' + ', '.join('%s (id %d)' % (vocab[i], i) for i in tr_ids))
notes.append('- untrained rows (set to zero): ' + ', '.join('%s (id %d)' % (vocab[i], i) for i in un_ids))

# ---------------- Fig 6: agent loop ----------------
fig, ax = plt.subplots(figsize=(14, 4.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.2); ax.axis('off')
steps = [('Diagnose', 'on diag split', BLUE), ('Propose edit', 'which rows to change', BLUE),
         ('Apply to copy', 'no gradient training', BLUE), ('Guard check', 'accept or refuse', ORANGE),
         ('Final test', 'sealed holdout, once', '#555555')]
bw, bh, gap = 2.3, 1.4, 0.52
x0 = 0.2
for i, (t, sub, c) in enumerate(steps):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, 1.5), bw, bh, boxstyle='round,pad=0.03,rounding_size=0.15', fc='white', ec=c, lw=3))
    ax.text(x + bw / 2, 2.45, t, ha='center', va='center', fontsize=17, fontweight='bold')
    ax.text(x + bw / 2, 1.95, sub, ha='center', va='center', fontsize=13)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + bw + .03, 2.2), (x + bw + gap - .03, 2.2), arrowstyle='-|>', mutation_scale=22, lw=2.5, color='black'))
# refuse arrow: guard check back to propose
gx = x0 + 3 * (bw + gap) + bw / 2; px = x0 + 1 * (bw + gap) + bw / 2
# refuse path runs below the boxes: down, left, up
ax.plot([gx, gx, px], [1.5, 0.9, 0.9], color=ORANGE, lw=2.5)
ax.add_patch(FancyArrowPatch((px, 0.9), (px, 1.47), arrowstyle='-|>', mutation_scale=22, lw=2.5, color=ORANGE))
ax.text((gx + px) / 2, 0.4, 'Refused: propose a different edit', ha='center', fontsize=14, color=ORANGE)
fig.tight_layout(); save(fig, 'fig6_loop')

# ---------------- NUMBERS.md ----------------
notes.append('\n## Not done / caveats')
notes.append('- Fig 6 is a schematic, no numbers. Accept/refuse logic is drawn as described in the task, not read from code.')
notes.append('- Fig 5 rows are a random draw (seed 1) of trained and untrained tokens; dims are the first 24 only.')
notes.append('- Fig 4 token text decoded with the M3 message vocabulary; window is first 199 tokens, padding removed.')
open(FIG + '/NUMBERS.md', 'w').write('# Numbers shown in the figures\n\n' + '\n'.join(notes) + '\n')
shutil.rmtree(tmp, ignore_errors=True)
