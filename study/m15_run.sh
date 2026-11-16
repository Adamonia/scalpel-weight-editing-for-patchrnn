#!/bin/bash
# Model15 sessions (round 1: prior correction as a control; round 2: F1 threshold) and a census by text length.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
RUNS=${SCALPEL_RUNS:-$ROOT/runs}; A=${SCALPEL_ARCHIVE:-$ROOT/../model-archive}; P=${PYTHON:-python}; export OMP_NUM_THREADS=2
rm -rf $RUNS/M15 $RUNS/R2_M15
$P -m scalpel.cli init --run $RUNS/M15 --archive $A --bundle $ROOT/bundle_codebert --model 15 2>&1 | grep -v -i warn | cut -c1-700
$P -m scalpel.cli init --run $RUNS/R2_M15 --archive $A --bundle $ROOT/bundle_codebert --model 15 --guard-metric f1 2>&1 | grep -c '"run"'
cd $ROOT/study
$P protocol.py 15 bias 2>&1 | grep -v -i warn | cut -c1-400
$P r2_protocol.py 15 thr 2>&1 | grep -v -i warn
$P - <<'PY' 2>&1 | grep -v -i warn
import numpy as np
z = np.load('../bundle_codebert/bundle.npz'); f = np.load('../bundle_codebert/base_feats_M15.npz')
y = z['y'].astype(int); pred = f['head'].argmax(1); part = z['part']; L = z['len_m']; M = z['msg_tokens']
tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
print('full test: TN %d FN %d FP %d TP %d acc %.3f (notebook: TN 4749 FN 361 FP 374 TP 2124, 90.339)' % (tn, fn, fp, tp, 100.0 * (tp + tn) / len(y)))
d = part == 0
def rate(mask, name):
    k = d & mask; e = pred[k] != y[k]
    print('  %-28s n=%4d pos=%4d err=%.1f%% FN=%d FP=%d' % (name, k.sum(), y[k].sum(), 100 * e.mean() if k.sum() else 0, int((e & (y[k] == 1)).sum()), int((e & (y[k] == 0)).sum())))
print('diag part, by length of the full text (tokens):')
for lo, hi in ((0, 255), (256, 510), (511, 1000), (1001, 3000), (3001, 10**9)): rate((L >= lo) & (L <= hi), '%d-%d' % (lo, hi))
print('diag part, by tokens of the message part:')
for lo, hi in ((0, 20), (21, 60), (61, 150), (151, 400), (401, 10**9)): rate((M >= lo) & (M <= hi), '%d-%d' % (lo, hi))
print('share of test texts cut at 512 tokens: %.1f%%; message alone fills the window: %.1f%%' % (100 * (L > 510).mean(), 100 * (M >= 510).mean()))
PY
echo M15DONE
