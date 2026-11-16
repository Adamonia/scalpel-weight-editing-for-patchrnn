"""Margins of E0 and all edit variants on diag+guard samples only. Usage: compute_margins.py MODEL_NO
Holdout samples are never forwarded; their margin cells stay NaN in the saved file."""
import sys
import time
from r3lib import *

no = int(sys.argv[1])
t0 = time.time()
s = get_session(no)
part, y, n = s.part, s.y, s.n
dev = np.where(part < HOLDOUT)[0]
check_idx(part, dev)
msg_dev = s.b['msg'][dev]                                   # only non-holdout rows are kept in memory
fb, fa, fm0 = (s.base_feats[k][dev] for k in ('twin_b', 'twin_a', 'msg'))
bs = 16 if no in (12, 14) else 32
model = s.model.eval()
params = dict(model.named_parameters())
specs = table_specs(s, no)
pad = s.pad_id
meta = {'model': no, 'n_dev': int(len(dev)), 'pad_id': pad, 'tables': [sp[0] for sp in specs]}

# --- pad id check: fill value of the msg array
last = s.b['msg'][dev][:, -1]
vals, cnts = np.unique(last, return_counts=True)
meta['msg_last_column_mode'] = int(vals[cnts.argmax()])
meta['msg_pad_share_of_positions_dev'] = float((msg_dev == pad).mean())
meta['pad_equals_fill_value'] = bool(vals[cnts.argmax()] == pad)
meta['pad_row_training_count'] = [int(c[pad]) for _, _, c in specs]
meta['n_train_samples'] = int(s.b['n_train_pos'] + s.b['n_train_neg'])


def head_margin(fm):
    return margin_of(s.head_from(fb, fa, fm))


def full_margin(m):
    out = np.full(n, np.nan)
    out[dev] = m
    return out


res = {'E0': head_margin(fm0)}
meta['E0_vs_cached_head_max_abs_margin_gap'] = float(np.abs(res['E0'] - margin_of(s.base_feats['head'][dev])).max())
# recompute the whole message stage once with the untouched model and compare with the cache
fm_re = msg_features(model, msg_dev, np.arange(len(dev)), bs)
meta['E0_recompute_msg_stage_margin_gap'] = float(np.abs(head_margin(fm_re) - res['E0']).max())

variants = [(variant_name(e, k), e, k, None) for e, k in all_variants()]
if no == 14:
    variants.append(('E1_cnnonly', 'E1', None, [0]))        # Round 1/2 edit of M14 (CNN table only), for reproduction
saved = {i: params[sp[0]].data.clone() for i, sp in enumerate(specs)}
nhit = {}
for name, edit, k, tabs in variants:
    tv = time.time()
    changed_masks, newW = {}, {}
    for i, (pname, win, cnt) in enumerate(specs):
        if tabs is not None and i not in tabs:
            continue
        newW[i] = new_table(saved[i], cnt, pad, edit, k)
        changed_masks[i] = (newW[i] != saved[i]).any(1).numpy()
        assert not changed_masks[i][pad], 'pad row changed'
    hit = np.zeros(len(dev), bool)
    for i, m in changed_masks.items():
        hit |= m[msg_dev[:, :specs[i][1]]].any(1)
    hit = np.where(hit)[0]
    fm = fm0.copy()
    try:
        for i, W1 in newW.items():
            params[specs[i][0]].data.copy_(W1)
        if len(hit):
            fm[hit] = msg_features(model, msg_dev, hit, bs)
    finally:
        for i, sp in enumerate(specs):
            params[sp[0]].data.copy_(saved[i])
    res[name] = head_margin(fm)
    nhit[name] = int(len(hit))
    print('M%d %-10s hit %5d/%d  %.0fs' % (no, name, len(hit), len(dev), time.time() - tv), flush=True)
meta['n_samples_hit'] = nhit
meta['n_rows_changed_E1'] = [int(((saved[i] != new_table(saved[i], specs[i][2], pad, 'E1')).any(1)).sum()) for i in range(len(specs))]

# --- reproduce the earlier ext margins (8 word-level models), diag+guard samples only
ext = '%s/ext/out/logits_M%d.npz' % (W, no)
if os.path.exists(ext):
    z = np.load(ext)
    meta['E0_vs_ext_orig_max_abs_gap'] = float(np.abs(res['E0'] - z['orig'][dev].astype(np.float64)).max())
    meta['E1_vs_ext_untrained_zero_max_abs_gap'] = float(np.abs(res['E1'] - z['untrained_zero'][dev].astype(np.float64)).max())
meta['seconds'] = round(time.time() - t0, 1)
np.savez('%s/margins_M%d.npz' % (OUT, no), y=y, part=part, **{k: full_margin(v) for k, v in res.items()})
json.dump(meta, open('%s/meta_M%d.json' % (OUT, no), 'w'), indent=1)
print(json.dumps(meta, indent=1))
