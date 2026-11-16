"""Standard diagnostic battery for one session. Each call goes through the tool dispatcher (journal)."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))

import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
from scalpel.engine import Session
from scalpel.tools import dispatch
no = int(sys.argv[1]); steps = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ov', 'census', 'branch', 'untr', 'freq', 'units', 'dead', 'ablate', 'bias']
s = Session(os.path.join(RUNS, 'M%d' % no))
A = 'claude-fable-5-2'
def call(tool, args=None):
    r = dispatch(s, tool, args or {}, actor=A)
    if 'error' in r: print('  ERROR %s: %s' % (tool, r['error']))
    return r
def ev(tag):
    r = call('evaluate')
    for p in ('diag', 'guard'):
        b, a = r[p]['before'], r[p]['after']
        print('  %-28s %-5s acc %.2f->%.2f f1 %.2f->%.2f mcc %.4f->%.4f fixed %d broken %d (FN %d->%d FP %d->%d)' % (tag, p, b['acc'], a['acc'], b['f1'], a['f1'], b['mcc'], a['mcc'], r[p]['n_fixed'], r[p]['n_broken'], b['fn'], a['fn'], b['fp'], a['fp']))
    return r
if 'ov' in steps:
    r = call('overview')
    print('==', r['model']); 
    for p in ('diag', 'guard'): m = r['metrics'][p]; print('  %s: acc %.2f prec %.2f rec %.2f f1 %.2f mcc %.4f (TP %d FP %d FN %d TN %d)' % (p, m['acc'], m['precision'], m['recall'], m['f1'], m['mcc'], m['tp'], m['fp'], m['fn'], m['tn']))
    print('  unused tensors:', [t.split(' ')[0] for t in r['tensors'] if 'UNUSED' in t], '| empty msgs in diag:', r['empty_message_samples_diag'])
if 'census' in steps:
    r = call('error_census')
    for k in ('by_message_length', 'by_code_length', 'by_untrained_msg_tokens', 'by_untrained_code_tokens'):
        print('  %s: ' % k + ' | '.join('%s n=%d pos=%d FN=%d FP=%d err=%.1f%%' % (b['bucket'], b['n'], b['n_pos'], b['FN'], b['FP'], b['err_pct']) for b in r[k]))
if 'branch' in steps:
    r = call('branch_attribution')
    print('  branch attribution (margin at mean feats %.2f):' % r['margin_at_mean_features'])
    for g, v in r['groups'].items(): print('    %s n=%d margin %.2f | twin_b %+.2f twin_a %+.2f msg %+.2f' % (g, v['n'], v['mean_margin'], v['twin_b'], v['twin_a'], v['msg']))
if 'untr' in steps:
    for t in ('msg', 'code'):
        r = call('untrained_summary', {'table': t})
        print('  untrained %s: rows %d of %d; in diag inputs %d rows; occurrences %d of %d (%.1f%%); samples with >=1: %d of %d; err with %.1f%% vs without %.1f%%' % (t, r['rows_untrained'], r['rows_total'], r['untrained_rows_in_diag_inputs'], r['token_occurrences_in_diag']['untrained'], r['token_occurrences_in_diag']['all'], 100.0 * r['token_occurrences_in_diag']['untrained'] / max(1, r['token_occurrences_in_diag']['all']), r['diag_samples_with_untrained'], r['diag_samples'], r['err_pct_with'] or 0, r['err_pct_without']))
if 'freq' in steps:
    for t in ('msg', 'code'):
        r = call('effect_by_frequency', {'table': t, 'per_bucket': 300})
        print('  effect by train count (%s):' % t)
        for b in r['buckets']: print('    %-9s occ %6d (%5.2f%%) |eff| %7.3f help %+7.3f ±%.3f helpful %5.1f%%' % (b['train_count'], b['occurrences'], b['share_pct'], b['mean_abs_effect'], b['mean_help'], b['help_stderr'], b['share_helpful_pct']))
if 'units' in steps:
    for st in ('twin_b', 'twin_a', 'msg'):
        r = call('unit_report', {'stage': st, 'top': 3})
        print('  units %s: n=%d sum_help %.3f, negative-help units %d, std<0.05: %d, std quantiles %s; top helpful %s' % (st, r['n_units'], r['sum_help_all_units'], r['n_units_with_negative_help'], r['units_with_std_below_0.05'], r['act_std_quantiles_0_10_50_90_100'], [(u['unit'], u['mean_help']) for u in r['most_helpful']]))
if 'dead' in steps:
    unused = [n for n in s.params if not s.model.stage_of(n)]
    if unused:
        for n in unused: call('edit_tensor', {'name': n, 'op': 'scale', 'value': 0.0})
        ev('zero unused %s' % unused); call('rollback')
    if s.is_tf:
        call('edit_tensor', {'name': 'embedTwin.weight', 'op': 'scale', 'value': 0.0, 'index': [None, {'range': [123, 128]}]})
        ev('zero last 5 embedTwin columns'); call('rollback')
if 'ablate1' in steps:
    for st in ('twin_b', 'twin_a'): call('edit_branch_gain', {'stage': st, 'factor': 0.0})
    ev('gain twin_b+twin_a x0'); call('rollback')
if 'ablate' in steps:
    for stages, f in ((('twin_b', 'twin_a'), 0.0), (('twin_b', 'twin_a'), 2.0), (('twin_b',), 0.0), (('twin_a',), 0.0), (('msg',), 0.0)):
        for st in stages: call('edit_branch_gain', {'stage': st, 'factor': f})
        ev('gain %s x%g' % ('+'.join(stages), f)); call('rollback')
if 'bias' in steps:
    r = call('bias_scan')
    print('  bias scan (%s), margin std %.2f, quantiles %s' % (r['output_bias'], r['margin_std'], r['margin_quantiles_5_25_50_75_95']))
    print('    ' + ' | '.join('d=%+.2f acc %.2f f1 %.2f mcc %.4f' % (t['delta'], t['acc'], t['f1'], t['mcc']) for t in r['table']))
s.save()
