"""Forward helper for the final evaluation. Same edit path as r3/compute_margins.py (E3, k=3, no recenter).
Only the rows given are forwarded. Code-branch and message features of the original model come from the cache."""
from r3lib import *

K_FROZEN = 3


def edited_margins(no, rows, want=('R',)):
    """Margins of the original model (E0) and of the edits in `want` on the given rows.
    R  = frozen recipe E3 k=3 on all message tables of the model (same as r3).
    E1 = untrained rows := 0 on all tables.   E1c = same, first table only (M14 round 1/2 edit)."""
    s = get_session(no)
    bs = 16 if no in (12, 14) else 32
    model = s.model.eval()
    params = dict(model.named_parameters())
    specs = table_specs(s, no)
    pad = s.pad_id
    msg = s.b['msg'][rows]
    fb, fa, fm0 = (s.base_feats[k][rows] for k in ('twin_b', 'twin_a', 'msg'))
    head = lambda fm: margin_of(s.head_from(fb, fa, fm))
    out = {'E0': head(fm0), 'cached_head_gap': float(np.abs(head(fm0) - margin_of(s.base_feats['head'][rows])).max()),
           'y': s.y[rows], 'part': s.part[rows], 'hit': {}}
    saved = {i: params[sp[0]].data.clone() for i, sp in enumerate(specs)}
    plans = {'R': ('E3', K_FROZEN, None), 'E1': ('E1', None, None), 'E1c': ('E1', None, [0])}
    for name in want:
        edit, k, tabs = plans[name]
        newW, masks = {}, {}
        for i, (pname, win, cnt) in enumerate(specs):
            if tabs is not None and i not in tabs:
                continue
            newW[i] = new_table(saved[i], cnt, pad, edit, k)
            masks[i] = (newW[i] != saved[i]).any(1).numpy()
            assert not masks[i][pad], 'pad row changed'
        hit = np.zeros(len(rows), bool)
        for i, m in masks.items():
            hit |= m[msg[:, :specs[i][1]]].any(1)
        hit = np.where(hit)[0]
        fm = fm0.copy()
        try:
            for i, W1 in newW.items():
                params[specs[i][0]].data.copy_(W1)
            if len(hit):
                fm[hit] = msg_features(model, msg, hit, bs)
        finally:
            for i, sp in enumerate(specs):
                params[sp[0]].data.copy_(saved[i])
        out[name] = head(fm)
        out['hit'][name] = int(len(hit))
    return out
