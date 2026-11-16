"""Step 2: 50 random stratified fit/eval splits of the full test set, plus a paired bootstrap.

Usage: splits.py [model_no ...]    (default: all word-level models that have logits)
Writes out/splits_M{n}.json for each model and out/splits.json (all models together).
Metrics are in percent points. All differences are method minus orig at threshold 0.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = '/home/claude/work/ext/out'
MODELS = [0, 2, 3, 4, 5, 8, 12, 13]
N_SPLITS = 50
N_BOOT = 2000
LF = ['untrained_zero', 'rare_zero', 'untrained_impute_mean', 'untrained_impute_knn']   # label-free methods
RIDGES = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
METRICS = ['f1', 'acc', 'precision', 'recall']


def metrics(y, m):
    """F1, accuracy, precision, recall (percent) of the prediction margin > 0."""
    p = m > 0
    tp = np.sum(p & y)
    fp = np.sum(p & ~y)
    fn = np.sum(~p & y)
    tn = np.sum(~p & ~y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'f1': 100 * f1, 'acc': 100 * (tp + tn) / len(y), 'precision': 100 * prec, 'recall': 100 * rec}


def best_threshold(y, m, n_grid=801):
    """Margin threshold with the best F1 on (y, m). Grid between the 1 % and 99 % quantiles plus 0.

    Returns a shift d such that the prediction is m + d > 0. Ties go to the smallest |d|.
    """
    lo, hi = np.quantile(m, [0.01, 0.99])
    thr = np.unique(np.concatenate([np.linspace(lo, hi, n_grid), [0.0]]))
    order = np.argsort(-m)
    ms, ys = m[order], y[order]
    ctp = np.concatenate([[0], np.cumsum(ys)])
    k = np.searchsorted(-ms, -thr, side='left')          # number of samples with margin > thr
    tp = ctp[k]
    f1 = 2 * tp / (k + ys.sum())
    best = f1.max()
    cand = np.where(f1 >= best - 1e-12)[0]
    return -thr[cand[np.argmin(np.abs(thr[cand]))]]


def ridge_fit(X, t, lam):
    """Closed-form ridge on standardized X with an unpenalized intercept."""
    xm = X.mean(0)
    tm = t.mean()
    A = (X - xm).T @ (X - xm) + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, (X - xm).T @ (t - tm))
    return w, tm - xm @ w


def lstsq_head(Xf, yf, rng):
    """Ridge least squares of the +-1 target on the fit features. Lambda chosen on an inner 80/20 split."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    Z = (Xf - mu) / sd
    t = np.where(yf, 1.0, -1.0)
    idx = rng.permutation(len(yf))
    cut = int(0.8 * len(yf))
    tr, va = idx[:cut], idx[cut:]
    best, lam_best = None, None
    for lam in RIDGES:
        w, b = ridge_fit(Z[tr], t[tr], lam)
        mse = np.mean((Z[va] @ w + b - t[va]) ** 2)
        if best is None or mse < best:
            best, lam_best = mse, lam
    w, b = ridge_fit(Z, t, lam_best)
    return (lambda X: ((X - mu) / sd) @ w + b), lam_best


def stratified_split(y, seed):
    rng = np.random.RandomState(seed)
    fit = np.zeros(len(y), dtype=bool)
    for c in (False, True):
        idx = rng.permutation(np.where(y == c)[0])
        fit[idx[:len(idx) // 2]] = True
    return fit


def summarize(d):
    d = np.asarray(d)
    return {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)), 'p97.5': float(np.percentile(d, 97.5)),
            'frac_pos': float((d > 0).mean())}


def run_model(no):
    z = np.load('%s/logits_M%d.npz' % (OUT, no))
    f = np.load('%s/feat_M%d.npz' % (OUT, no))
    y = z['y'].astype(bool)
    X = f['feat'].astype(np.float64)
    L = {k: z[k].astype(np.float64) for k in z.files if k != 'y'}
    seeds = sorted(int(k.split('_')[-1]) for k in L if k.startswith('placebo_untrained_'))
    n_seeds = len(seeds)
    PU = np.stack([L['placebo_untrained_%d' % k] for k in seeds], 1)       # n x seeds
    PF = np.stack([L['placebo_freqmatch_%d' % k] for k in seeds], 1)
    # records[method][metric] -> list over splits of the paired difference vs orig; absolute values kept too
    diff, absv = {}, {}
    auc, ap = {}, {}            # threshold-free, absolute values on the eval half
    ps_diff = {'placebo_untrained': {m: [] for m in METRICS}, 'placebo_freqmatch': {m: [] for m in METRICS}}  # per split x seed
    ps_diff_rc = {'placebo_untrained': {m: [] for m in METRICS}, 'placebo_freqmatch': {m: [] for m in METRICS}}
    ps_auc = {'placebo_untrained': [], 'placebo_freqmatch': []}
    ps_ap = {'placebo_untrained': [], 'placebo_freqmatch': []}
    lam_chosen = []

    def add(name, ev_metrics, base):
        for m in METRICS:
            diff.setdefault(name, {}).setdefault(m, []).append(ev_metrics[m] - base[m])
            absv.setdefault(name, {}).setdefault(m, []).append(ev_metrics[m])

    def add_rank(name, ye, s):
        auc.setdefault(name, []).append(roc_auc_score(ye, s))
        ap.setdefault(name, []).append(average_precision_score(ye, s))

    for sp in range(N_SPLITS):
        fit = stratified_split(y, sp)
        ev = ~fit
        yf, ye = y[fit], y[ev]
        base = metrics(ye, L['orig'][ev])
        add('orig', base, base)
        add('prior', metrics(ye, L['prior'][ev]), base)
        add_rank('orig', ye, L['orig'][ev])
        mu_orig_fit = L['orig'][fit].mean()
        # label-free methods: no recenter, and recentered with the fit-half mean of orig
        for name in LF:
            m = L[name]
            add(name + '__norc', metrics(ye, m[ev]), base)
            shift = mu_orig_fit - m[fit].mean()
            add(name, metrics(ye, m[ev] + shift), base)
            add_rank(name, ye, m[ev])
        # placebos: every seed, then the seed average
        for tag, P in (('placebo_untrained', PU), ('placebo_freqmatch', PF)):
            mets_n = [metrics(ye, P[ev, k]) for k in range(n_seeds)]
            mets_r = [metrics(ye, P[ev, k] + (mu_orig_fit - P[fit, k].mean())) for k in range(n_seeds)]
            for m in METRICS:
                ps_diff[tag][m].append([mm[m] - base[m] for mm in mets_n])
                ps_diff_rc[tag][m].append([mm[m] - base[m] for mm in mets_r])
            ps_auc[tag].append([roc_auc_score(ye, P[ev, k]) for k in range(n_seeds)])
            ps_ap[tag].append([average_precision_score(ye, P[ev, k]) for k in range(n_seeds)])
            for suffix, src in (('__norc', ps_diff), ('', ps_diff_rc)):
                for m in METRICS:
                    d = np.mean(src[tag][m][-1])
                    diff.setdefault(tag + suffix, {}).setdefault(m, []).append(d)
                    absv.setdefault(tag + suffix, {}).setdefault(m, []).append(d + base[m])
            auc.setdefault(tag, []).append(float(np.mean(ps_auc[tag][-1])))
            ap.setdefault(tag, []).append(float(np.mean(ps_ap[tag][-1])))
        # label-using baselines fitted on the fit half only
        d = best_threshold(yf, L['orig'][fit])
        add('thr_only', metrics(ye, L['orig'][ev] + d), base)
        d = best_threshold(yf, L['untrained_zero'][fit])
        add('untrained_zero+thr', metrics(ye, L['untrained_zero'][ev] + d), base)
        rng = np.random.RandomState(1000 + sp)
        score, lam = lstsq_head(X[fit], yf, rng)
        lam_chosen.append(lam)
        s_fit, s_ev = score(X[fit]), score(X[ev])
        d = best_threshold(yf, s_fit)
        add('lstsq_head+thr', metrics(ye, s_ev + d), base)
        add('lstsq_head_argmax', metrics(ye, s_ev), base)
        add_rank('lstsq_head', ye, s_ev)

    res = {'methods': {}, 'per_split_f1_diff': {}}
    for name in diff:
        res['methods'][name] = {m: summarize(diff[name][m]) for m in METRICS}
        res['methods'][name]['abs_mean'] = {m: float(np.mean(absv[name][m])) for m in METRICS}
        res['per_split_f1_diff'][name] = [float(v) for v in diff[name]['f1']]
    for name in auc:
        base_auc = np.array(auc['orig']); base_ap = np.array(ap['orig'])
        res['methods'].setdefault(name, {})['auc'] = dict(summarize(np.array(auc[name]) - base_auc), abs_mean=float(np.mean(auc[name])))
        res['methods'][name]['ap'] = dict(summarize(np.array(ap[name]) - base_ap), abs_mean=float(np.mean(ap[name])))
    # per-seed placebo effects: mean over the 50 splits for each seed, then spread over seeds
    per_seed = {}
    base_auc = np.array(auc['orig']); base_ap = np.array(ap['orig'])
    for tag in ('placebo_untrained', 'placebo_freqmatch'):
        e = {}
        for suffix, src in (('__norc', ps_diff), ('', ps_diff_rc)):
            for m in METRICS:
                e[m + suffix] = np.mean(np.array(src[tag][m]), 0)            # per seed
        e['auc'] = np.mean(np.array(ps_auc[tag]) - base_auc[:, None], 0)
        e['ap'] = np.mean(np.array(ps_ap[tag]) - base_ap[:, None], 0)
        per_seed[tag] = {k: [float(v) for v in vals] for k, vals in e.items()}
    res['placebo_per_seed_mean_over_splits'] = per_seed
    # How often the untrained_zero effect is larger than the placebo effect: per seed and per split
    cmp_ = {}
    for tag in ('placebo_untrained', 'placebo_freqmatch'):
        c = {}
        for key, mine in (('f1', np.array(diff['untrained_zero']['f1'])), ('f1__norc', np.array(diff['untrained_zero__norc']['f1']))):
            src = ps_diff_rc if key == 'f1' else ps_diff
            plac = np.array(src[tag]['f1'])                                    # split x seed
            c[key] = {'frac_seeds_placebo_mean_effect_ge_untrained': float((np.mean(plac, 0) >= mine.mean()).mean()),
                      'untrained_minus_placebo_seedmean': summarize(mine - plac.mean(1))}
        cmp_[tag] = c
    res['untrained_vs_placebo'] = cmp_
    np.savez_compressed('%s/splits_perseed_M%d.npz' % (OUT, no),
                        **{'%s_%s_%s' % (tag, kind, m): np.array(src[tag][m]) for tag in ps_diff
                           for kind, src in (('norc', ps_diff), ('rc', ps_diff_rc)) for m in METRICS},
                        **{'%s_auc' % t: np.array(ps_auc[t]) for t in ps_auc}, **{'%s_ap' % t: np.array(ps_ap[t]) for t in ps_ap})
    res['lstsq_ridge_chosen_counts'] = {str(l): int(lam_chosen.count(l)) for l in RIDGES}
    res['n_splits'] = N_SPLITS
    res['n_seeds'] = n_seeds
    res['bootstrap_full_test'] = bootstrap(y, L, PU, PF)
    return res


def f1_of(tp, pp, ap_):
    return 100 * 2 * tp / np.maximum(pp + ap_, 1)


def bootstrap(y, L, PU, PF):
    """Paired bootstrap (samples resampled) of the F1 difference vs orig, full 7,608 samples, label-free methods.

    Recentered = shift so that the mean margin on all samples equals the mean orig margin.
    """
    n = len(y)
    rng = np.random.RandomState(777)
    mu = L['orig'].mean()
    methods = {}
    for name in LF:
        methods[name] = L[name] + (mu - L[name].mean())
        methods[name + '__norc'] = L[name]
    for tag, P in (('placebo_untrained', PU), ('placebo_freqmatch', PF)):
        for k in range(P.shape[1]):
            methods['%s_%d' % (tag, k)] = P[:, k] + (mu - P[:, k].mean())
            methods['%s_%d__norc' % (tag, k)] = P[:, k]
    methods['orig'] = L['orig']
    pred = {k: v > 0 for k, v in methods.items()}
    acc = {k: [] for k in pred}
    for _ in range(N_BOOT // 250):
        idx = rng.randint(0, n, size=(250, n))
        apos = y[idx].sum(1)
        for k, p in pred.items():
            tp = (p & y)[idx].sum(1)
            pp = p[idx].sum(1)
            acc[k].append(f1_of(tp, pp, apos))
    f1b = {k: np.concatenate(v) for k, v in acc.items()}
    out = {}

    def ci(v):
        return {'mean': float(v.mean()), 'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    for k in list(LF) + [x + '__norc' for x in LF]:
        out[k] = ci(f1b[k] - f1b['orig'])
    for tag, P in (('placebo_untrained', PU), ('placebo_freqmatch', PF)):
        for suffix in ('', '__norc'):
            mean_eff = np.mean([f1b['%s_%d%s' % (tag, k, suffix)] for k in range(P.shape[1])], 0) - f1b['orig']
            out[tag + suffix] = ci(mean_eff)
            out['untrained_zero%s_minus_%s%s' % (suffix, tag, suffix)] = ci((f1b['untrained_zero' + suffix] - f1b['orig']) - mean_eff)
    out['point_f1_orig'] = float(f1_of((pred['orig'] & y).sum(), pred['orig'].sum(), y.sum()))
    out['n_boot'] = N_BOOT
    return out


if __name__ == '__main__':
    models = [int(a) for a in sys.argv[1:]] or MODELS
    for no in models:
        res = run_model(no)
        json.dump(res, open('%s/splits_M%d.json' % (OUT, no), 'w'))
        print('done M%d' % no, flush=True)
    allr = {}
    for no in MODELS:
        p = '%s/splits_M%d.json' % (OUT, no)
        if os.path.exists(p):
            allr[str(no)] = json.load(open(p))
    json.dump({'note': 'Splits overlap, so percentiles over splits describe split-to-split variation, not independent-sample confidence intervals. Bootstrap intervals are for the full 7,608 samples. Metrics in percent points; differences are method minus orig at threshold 0.',
               'models': allr}, open(OUT + '/splits.json', 'w'))
