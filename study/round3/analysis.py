"""Metrics of every model / variant / recenter flag on one part (diag or guard)."""
from r3lib import *

SEEDS = {'diag': 2026, 'guard': 2027}


def load_margins(no):
    z = np.load('%s/margins_M%d.npz' % (OUT, no))
    return {k: z[k] for k in z.files}


def cfg_list():
    out = ['E0']
    for e, k in all_variants():
        for rc in ('raw', 'rc'):
            out.append('%s|%s' % (variant_name(e, k), rc))
    return out


def cfg_margin(z, cfg, shift_cache):
    name, flag = (cfg.split('|') + [None])[:2]
    if name == 'E0':
        return z['E0']
    m = z[name]
    if flag == 'rc':
        if name not in shift_cache:
            shift_cache[name] = recenter_shift(z['part'], z['E0'], m)
        return m + shift_cache[name]
    return m


def all_cfgs(z):
    c = cfg_list()
    if 'E1_cnnonly' in z:
        c += ['E1_cnnonly|raw', 'E1_cnnonly|rc']
    return c


def part_metrics(no, part_name):
    z = load_margins(no)
    part, y = z['part'], z['y']
    idx = part_idx(part, part_name)
    shifts, res = {}, {}
    for cfg in all_cfgs(z):
        m = cfg_margin(z, cfg, shifts)
        res[cfg] = {'f1': f1_pct(part, idx, y, m), 'auc': auc_pct(part, idx, y, m), 'ap': ap_pct(part, idx, y, m)}
    return res, shifts


def margins_and_conf(no, part_name):
    """Confusion columns of every config on the part (for the paired bootstrap)."""
    z = load_margins(no)
    part, y = z['part'], z['y']
    idx = part_idx(part, part_name)
    shifts = {}
    return {cfg: conf_cols(part, idx, y, cfg_margin(z, cfg, shifts)) for cfg in all_cfgs(z)}, len(idx), y[idx]
