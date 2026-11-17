"""Step 3: write out/REPORT.md. Every number comes from out/round3_holdout.json."""
import json
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(os.path.dirname(_HERE), 'out') + '/'
d = json.load(open(O + 'round3_holdout.json'))
m = d['models']
L = ['# Round 3 final evaluation on the sealed holdout (3,804 samples)', '',
     'Frozen recipe and decisions: see FROZEN.md. All numbers below are read from round3_holdout.json. F1 in percent, change in F1 points, 95 % paired bootstrap interval (10,000 resamples, seed 12345). p = two-sided p (2*min(p_up, 1-p_up); no changed prediction: 1). Holm36 = Holm over the 24 published + 12 new F1 comparisons, Holm12 = over the 12 new alone, flag * = significant at 0.05.', '',
     '| Model | Decision | F1 orig | F1 round 2 (pub.) | F1 round 3 | Change vs orig [95 % interval] | fixed/broken | p (F1) | McNemar p | Holm36 p | Holm12 p | Round 3 - round 2 |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|']
for k, v in m.items():
    dec = 'recipe' if v['decision'] != 'original' else ('original (refused)' if v['applicable'] else 'original (n/a)')
    ci = v['ci95']
    r2 = v['vs_round2']
    pi = r2['paired_interval']
    r2s = '%+.3f' % r2['d_f1_r3_minus_r2'] + (' [%+.2f, %+.2f]' % tuple(pi['ci95']) if pi else '')
    star = lambda s: '*' if s else ''
    L.append('| M%s %s | %s | %.3f | %.3f | %.3f | %+.3f [%+.2f, %+.2f] | %d/%d | %.4f | %.4g | %.4f%s | %.4f%s | %s |' % (
        k, v['label'], dec, v['f1_original'], v['f1_round2_published'], v['f1_round3'], v['d_f1_vs_original'], ci[0], ci[1], v['fixed'], v['broken'],
        v['p_f1_two_sided'], v['mcnemar_p'], v['holm36_p_f1'], star(v['holm36_sig_f1']), v['holm12_p_f1'], star(v['holm12_sig_f1']), r2s))
g = d['original_vs_published_round1']
rb = [k for k, v in m.items() if not isinstance(v['vs_round2']['rebuild'], str) and v['vs_round2']['rebuild']['exact']]
L += ['', '## Checks', '',
      '- Original holdout F1 equals the published round 1 original F1 for all 12 models: max abs gap %s (other metrics: %s).' % (g['max_abs_gap_f1'], g['max_abs_gap_other_metrics']),
      '- Round 3 minus round 2 is a difference of published F1 values. A paired interval is given only where round 2 predictions were rebuilt exactly (F1, fixed, broken, FP, FN all equal): models %s. For the others the rebuild from "untrained rows to zero" did not reproduce the published round 2 numbers (see round3_holdout.json, vs_round2.rebuild), or round 2 used an F1 threshold, so no interval is available.' % (', '.join('M' + k for k in rb) or 'none'),
      '- McNemar p = 0 after rounding to 6 decimals is entered as 1e-6 in Holm, as in ext/holm.py.',
      '- Published rows whose Holm F1 flag changes when the family grows from 24 to 36: %s.' % (json.dumps(d['published_rows_whose_holm_f1_flag_changes_in_36_family']) if d['published_rows_whose_holm_f1_flag_changes_in_36_family'] else 'none'),
      '- Significant F1 changes after Holm over 36: %s. Over the 12 new alone: %s.' % (
          ', '.join('M' + k for k, v in m.items() if v['holm36_sig_f1']) or 'none', ', '.join('M' + k for k, v in m.items() if v['holm12_sig_f1']) or 'none'),
      '- Holdout margins were computed once per model (step2a_forward.py, saved in holdout_margins_M*.npz). No crash, no method change.']
open(O + 'REPORT.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
