"""Final evaluation. finalize() opens the sealed holdout part one time and locks the session."""
import json
import math
import os
import time

import numpy as np
import torch

from .engine import PARTS, ScalpelError, confusion_metrics
from .models import checkpoint_path, load_state_dict

N_BOOT = 10000
BOOT_SEED = 12345


def mcnemar_exact(n_fixed, n_broken):
    """Two-sided exact McNemar test on the discordant pairs."""
    n = n_fixed + n_broken
    if n == 0:
        return 1.0
    k = min(n_fixed, n_broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)        # Integer division: no overflow for large n.
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (mid - half), 2), round(100 * (mid + half), 2)]


def roc_auc(y, score):
    """Area under the ROC curve (rank statistic, ties get the mean rank)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=np.float64)
    s_sorted = np.asarray(score)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _vector_metrics(y, pred):
    """Metrics for a matrix of bootstrap samples (rows = resamples)."""
    tp = (pred & y).sum(1).astype(np.float64)
    fp = (pred & ~y).sum(1).astype(np.float64)
    fn = (~pred & y).sum(1).astype(np.float64)
    tn = (~pred & ~y).sum(1).astype(np.float64)
    n = tp + fp + fn + tn
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    return {'acc': 100 * (tp + tn) / n, 'precision': 100 * prec, 'recall': 100 * rec, 'f1': 100 * f1, 'mcc': mcc}


def paired_bootstrap(y, pred_old, pred_new, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile 95 % intervals of the metric differences (new - old). Samples are paired."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(bool)
    po = np.asarray(pred_old).astype(bool)
    pn = np.asarray(pred_new).astype(bool)
    n = len(y)
    diffs = {k: [] for k in ('acc', 'precision', 'recall', 'f1', 'mcc')}
    for _ in range(n_boot // 500):
        idx = rng.randint(0, n, size=(500, n))
        mo = _vector_metrics(y[idx], po[idx])
        mn = _vector_metrics(y[idx], pn[idx])
        for k in diffs:
            diffs[k].append(mn[k] - mo[k])
    out = {}
    for k, v in diffs.items():
        v = np.concatenate(v)
        out[k] = {'ci95': [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                  'p_improve': round(float((v > 0).mean()), 4)}
    return out


def compare(y, pred_old, pred_new, bootstrap=True):
    y = np.asarray(y)
    old_ok = pred_old == y
    new_ok = pred_new == y
    fixed = int((~old_ok & new_ok).sum())
    broken = int((old_ok & ~new_ok).sum())
    before = confusion_metrics(y, pred_old)
    after = confusion_metrics(y, pred_new)
    out = {'before': before, 'after': after,
           'delta': {k: round(after[k] - before[k], 5) for k in ('acc', 'precision', 'recall', 'f1', 'mcc')},
           'n_fixed': fixed, 'n_broken': broken, 'mcnemar_p': round(mcnemar_exact(fixed, broken), 6),
           'acc_wilson95': {'before': wilson(before['tp'] + before['tn'], before['n']),
                            'after': wilson(after['tp'] + after['tn'], after['n'])},
           'error_flow': {'FN_fixed': int((~old_ok & new_ok & (y == 1)).sum()),
                          'FP_fixed': int((~old_ok & new_ok & (y == 0)).sum()),
                          'new_FN': int((old_ok & ~new_ok & (y == 1)).sum()),
                          'new_FP': int((old_ok & ~new_ok & (y == 0)).sum())}}
    if bootstrap:
        out['bootstrap'] = paired_bootstrap(y, pred_old, pred_new)
    return out


def weight_diff(s):
    out = []
    for name, p in s.params.items():
        d = p.data - s.base[name]
        n_changed = int((d != 0).sum().item())
        if n_changed:
            rows = int((d.reshape(d.shape[0], -1) != 0).any(1).sum().item()) if d.dim() > 1 else None
            out.append({'tensor': name, 'shape': list(p.shape), 'elements_changed': n_changed,
                        'rows_changed': rows, 'share_changed_pct': round(100.0 * n_changed / d.numel(), 4),
                        'l2_change': round(d.norm().item(), 4),
                        'relative_l2': round((d.norm() / s.base[name].norm().clamp_min(1e-12)).item(), 6)})
    return out


def finalize(s, note=''):
    """Evaluate the committed weights on the sealed holdout part and lock the session."""
    if s.final is not None:
        return s.final
    if s.pending:
        raise ScalpelError('There are pending edits. Call commit or rollback first.')
    t0 = time.time()
    s._holdout_open = True
    rows = np.arange(s.n)
    # A stage that no committed tensor feeds has the features of the original weights.
    touched = set(st for name in s.committed_delta for st in s.model.stage_of(name))
    feats = {st: (s._compute_stage(st, rows) if st in touched else s.base_feats[st]) for st in s.model.STAGES[:-1]}
    head = s.head_from(feats['twin_b'], feats['twin_a'], feats['msg'])
    dev = np.where(s.part < 2)[0]
    s.refresh(dev)
    cache_gap = float(np.abs(head[dev] - s.feats['head'][dev]).max())
    pred_new = head.argmax(1)
    pred_old = s.base_pred
    result = {'model_no': s.model_no, 'label': s.label, 'note': note,
              'n_commits': len(s.commits), 'n_evaluations': s.n_evaluations,
              'cache_consistency_max_abs_logit_gap': round(cache_gap, 6),
              'guard_metric': s.limits.get('guard_metric', 'mcc'), 'stages_recomputed': sorted(touched - {'head'}),
              'parts': {}, 'weight_diff': weight_diff(s)}
    for name, code in PARTS.items():
        m = s.part == code
        result['parts'][name] = compare(s.y[m], pred_old[m], pred_new[m], bootstrap=(name == 'holdout'))
    result['parts']['test_all'] = compare(s.y, pred_old, pred_new, bootstrap=False)
    m_old = s.base_feats['head'][:, 1] - s.base_feats['head'][:, 0]
    m_new = head[:, 1] - head[:, 0]
    for name, code in PARTS.items():
        m = s.part == code
        result['parts'][name]['auc'] = {'before': round(roc_auc(s.y[m], m_old[m]), 5),
                                        'after': round(roc_auc(s.y[m], m_new[m]), 5)}
    result['error_types'] = [{'commit': c['commit'], 'error_type': c['error_type'], 'rationale': c['rationale'],
                              'n_edits': len(c['edits']),
                              'diag_delta_errors': c['diag']['n_broken'] - c['diag']['n_fixed'],
                              'guard_delta_errors': c['guard']['n_broken'] - c['guard']['n_fixed']}
                             for c in s.commits]

    if hasattr(s.model, 'export'):
        files = s.model.export(s.cfg['archive_dir'], s.model_no, s.run_dir)
        out_path = s._p(files[0])
        result['edited_files'] = files
    else:
        sd = load_state_dict(checkpoint_path(s.cfg['archive_dir'], s.model_no))
        for name in sd:
            sd[name] = s.params[name].data.clone() if name in s.params else sd[name]
        out_path = s._p('edited_' + os.path.basename(checkpoint_path(s.cfg['archive_dir'], s.model_no)))
        torch.save(sd, out_path)
    np.savez(s._p('final_predictions.npz'), y=s.y, part=s.part, pred_base=pred_old, pred_edited=pred_new,
             logits_base=s.base_feats['head'], logits_edited=head)
    result['edited_checkpoint'] = os.path.basename(out_path)
    result['seconds'] = round(time.time() - t0, 1)
    result['finalized_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(s._p('FINAL.json'), 'w', encoding='utf-8') as fp:
        json.dump(result, fp, indent=1)
    s.final = result
    s.log('finalize', {'holdout': result['parts']['holdout'], 'weight_diff': result['weight_diff']})
    return result
