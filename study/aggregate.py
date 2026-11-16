"""Collect the final results of all sessions into tables (JSON and Markdown).

Usage: aggregate.py RUNS_DIR BUNDLE_DIR OUT_DIR
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scalpel.engine import confusion_metrics
from scalpel.report import compare, mcnemar_exact, roc_auc

RUNS, BUNDLE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODELS = [0, 2, 3, 4, 5, 8, 12, 13]
LEXICON = (r'^(overflow|underflow|overrun|overread|leak|memleak|crash|null|deref|derefer|race|deadlock|corrupt|'
           r'uniniti|use-after-fre|uaf|oob|out-of-bound|bound|secur|vulner|exploit|attack|malici|sanit|dos|privileg|'
           r'inject|infinit|panic|oop|segfault|segv|fault|dangl|double-fre|unsaf)$')
RECIPE = [('class-weight prior', 'prior correction'), ('rare-token noise', 'rare rows (count <= 10) to zero'),
          ('untrained message embedding rows', 'untrained rows to zero'), ('inert code branches', 'code branches removed'),
          ('operating-point shift', 'recentering')]


def recipe_of(error_type):
    for key, name in RECIPE:
        if error_type.startswith(key):
            return name
    return error_type


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(BUNDLE, 'bundle.npz'))
    y, part, msg = z['y'], z['part'], z['msg']
    with open(os.path.join(BUNDLE, 'vocab.json'), encoding='utf-8') as fp:
        vocab = json.load(fp)['msg']
    rx = re.compile(LEXICON)
    lex_ids = np.array([i for i, t in enumerate(vocab) if rx.search(t)])
    summary = {'models': {}, 'lexicon_regex': LEXICON}
    preds = {}
    for no in MODELS:
        run = os.path.join(RUNS, 'M%d' % no)
        with open(os.path.join(run, 'FINAL.json'), encoding='utf-8') as fp:
            fin = json.load(fp)
        fp_ = np.load(os.path.join(run, 'final_predictions.npz'))
        preds[no] = (fp_['pred_base'], fp_['pred_edited'], fp_['logits_base'], fp_['logits_edited'])
        win = 199 if no not in (0, 2) else 200
        has_lex = np.isin(msg[:, :win], lex_ids).any(1)
        nonempty = (msg[:, :win] > 0).any(1)
        slices = {}
        for pname, pmask in (('diag', part == 0), ('holdout', part == 2)):
            for tag, m in (('security_with_lexicon', (y == 1) & has_lex), ('security_without_lexicon', (y == 1) & ~has_lex & nonempty),
                           ('nonsecurity_with_lexicon', (y == 0) & has_lex), ('nonsecurity_without_lexicon', (y == 0) & ~has_lex & nonempty),
                           ('empty_message', ~nonempty)):
                mm = m & pmask
                slices['%s/%s' % (pname, tag)] = {
                    'n': int(mm.sum()), 'n_security': int((y[mm] == 1).sum()),
                    'err_base_pct': round(100.0 * (fp_['pred_base'][mm] != y[mm]).mean(), 2),
                    'err_edited_pct': round(100.0 * (fp_['pred_edited'][mm] != y[mm]).mean(), 2)}
        # Journal: attempts (commit calls) with the result.
        attempts = []
        last_eval = None
        with open(os.path.join(run, 'journal.jsonl'), encoding='utf-8') as fp:
            for line in fp:
                rec = json.loads(line)
                if rec.get('kind') != 'tool':
                    continue
                if rec['tool'] == 'evaluate' and 'diag' in rec.get('result', {}):
                    last_eval = rec['result']
                if rec['tool'] == 'commit' and last_eval is not None:
                    ok = 'error' not in rec['result']
                    if ok:
                        # An accepted commit carries its own evaluation. Use it, not an earlier evaluate call.
                        last_eval = rec['result']
                    attempts.append({'recipe': recipe_of(rec['args'].get('error_type', '')), 'accepted': ok,
                                     'diag_net': last_eval['diag']['n_fixed'] - last_eval['diag']['n_broken'],
                                     'guard_net': last_eval['guard']['n_fixed'] - last_eval['guard']['n_broken'],
                                     'diag_mcc': [last_eval['diag']['before']['mcc'], last_eval['diag']['after']['mcc']],
                                     'guard_mcc': [last_eval['guard']['before']['mcc'], last_eval['guard']['after']['mcc']]})
        n_tool_calls = sum(1 for line in open(os.path.join(run, 'journal.jsonl'), encoding='utf-8')
                           if json.loads(line).get('kind') == 'tool')
        # Holdout effect of each commit. A bias edit shifts all margins by a constant, so the
        # states between the commits follow from the base logits and the final logits.
        with open(os.path.join(run, 'state.json'), encoding='utf-8') as fp:
            commits = json.load(fp)['commits']
        shifts, nonbias = [], []
        for c in commits:
            sh, nb = 0.0, False
            for e in c['edits']:
                if e['op'] == 'recenter':
                    sh += e['bias_shift']
                elif e['op'] == 'tensor' and e['tensor'].endswith('.bias') and e.get('index') == [1] and e['mode'] == 'add' \
                        and e['tensor'] in ('fc4.bias', 'fc2.bias', 'classifier.4.bias'):
                    sh += float(e['value'])
                else:
                    nb = True
            shifts.append(sh)
            nonbias.append(nb)
        assert sum(nonbias) <= 1, 'more than one commit with non-bias edits'
        m_base = fp_['logits_base'][:, 1] - fp_['logits_base'][:, 0]
        m_final = fp_['logits_edited'][:, 1] - fp_['logits_edited'][:, 0]
        total = sum(shifts)
        j = nonbias.index(True) if any(nonbias) else len(commits)
        hold = part == 2
        per_commit = []
        prev = m_base
        for i, c in enumerate(commits):
            cum = sum(shifts[:i + 1])
            cur = (m_base if i < j else (m_final - total)) + cum
            cmp_ = compare(y[hold], (prev[hold] > 0).astype(int), (cur[hold] > 0).astype(int), bootstrap=False)
            per_commit.append({'commit': i + 1, 'recipe': recipe_of(c['error_type']), 'bias_shift': round(shifts[i], 4),
                               'n_fixed': cmp_['n_fixed'], 'n_broken': cmp_['n_broken'], 'delta': cmp_['delta'],
                               'mcnemar_p': cmp_['mcnemar_p'],
                               'auc_before': round(roc_auc(y[hold], prev[hold]), 5), 'auc_after': round(roc_auc(y[hold], cur[hold]), 5)})
            prev = cur
        assert np.abs(prev - m_final).max() < 1e-3, 'commit decomposition does not reach the final margins'
        summary['models'][str(no)] = {'label': fin['label'], 'parts': fin['parts'], 'weight_diff': fin['weight_diff'],
                                      'commits': fin['error_types'], 'per_commit_holdout': per_commit, 'attempts': attempts, 'slices': slices,
                                      'n_tool_calls': n_tool_calls, 'n_evaluations': fin['n_evaluations']}

    # Error overlap on the holdout part, base and edited.
    hold = part == 2
    for tag, idx in (('base', 0), ('edited', 1)):
        wrong = np.stack([preds[no][idx][hold] != y[hold] for no in MODELS])
        k = wrong.sum(0)
        summary['holdout_overlap_' + tag] = {'wrong_in_k_models': {int(i): int((k == i).sum()) for i in range(len(MODELS) + 1)},
                                             'majority_vote_acc': round(100.0 * (k < len(MODELS) / 2).mean(), 3),
                                             'mean_acc': round(100.0 * (1 - wrong.mean()), 3)}
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=1)

    # Markdown tables.
    L = []
    L.append('## Holdout part (3,804 samples): original and edited weights\n')
    L.append('| Model | Acc orig | Acc edit | d Acc [95% CI] | F1 orig | F1 edit | d F1 [95% CI] | MCC orig | MCC edit | d MCC [95% CI] | AUC orig | AUC edit | fixed | broken | McNemar p |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for no in MODELS:
        m = summary['models'][str(no)]
        h = m['parts']['holdout']
        b, a, bs = h['before'], h['after'], h['bootstrap']
        L.append('| M%d %s | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %.3f | %.3f | %+.3f [%+.3f, %+.3f] | %.4f | %.4f | %d | %d | %.2g |' % (
            no, m['label'], b['acc'], a['acc'], h['delta']['acc'], bs['acc']['ci95'][0], bs['acc']['ci95'][1],
            b['f1'], a['f1'], h['delta']['f1'], bs['f1']['ci95'][0], bs['f1']['ci95'][1],
            b['mcc'], a['mcc'], h['delta']['mcc'], bs['mcc']['ci95'][0], bs['mcc']['ci95'][1],
            h['auc']['before'], h['auc']['after'], h['n_fixed'], h['n_broken'], mcnemar_exact(h['n_fixed'], h['n_broken'])))
    L.append('\n## Holdout part: precision, recall and error flow\n')
    L.append('| Model | Prec orig | Prec edit | Rec orig | Rec edit | FP orig | FP edit | FN orig | FN edit |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for no in MODELS:
        m = summary['models'][str(no)]
        b, a = m['parts']['holdout']['before'], m['parts']['holdout']['after']
        L.append('| M%d %s | %.2f | %.2f | %.2f | %.2f | %d | %d | %d | %d |' % (no, m['label'], b['precision'], a['precision'], b['recall'], a['recall'], b['fp'], a['fp'], b['fn'], a['fn']))
    L.append('\n## Accuracy by part (original -> edited)\n')
    L.append('| Model | diag | guard | holdout | full test |')
    L.append('|---|---|---|---|---|')
    for no in MODELS:
        m = summary['models'][str(no)]
        L.append('| M%d %s | %s |' % (no, m['label'], ' | '.join('%.2f -> %.2f' % (m['parts'][p]['before']['acc'], m['parts'][p]['after']['acc']) for p in ('diag', 'guard', 'holdout', 'test_all'))))
    L.append('\n## Edit attempts (net corrected samples on diag / guard; A = accepted, R = refused by the guard rule)\n')
    names = [n for _, n in RECIPE]
    L.append('| Model | ' + ' | '.join(names) + ' |')
    L.append('|---|' + '---|' * len(names))
    for no in MODELS:
        m = summary['models'][str(no)]
        cells = []
        for n in names:
            at = [x for x in m['attempts'] if x['recipe'] == n]
            cells.append('; '.join('%s %+d / %+d' % ('A' if x['accepted'] else 'R', x['diag_net'], x['guard_net']) for x in at) or '-')
        L.append('| M%d %s | %s |' % (no, m['label'], ' | '.join(cells)))
    L.append('\n## Holdout effect of each commit (in commit order)\n')
    L.append('| Model | Commit | Recipe | Bias shift | fixed | broken | d Acc | d F1 | d MCC | AUC before | AUC after | McNemar p |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for no in MODELS:
        m = summary['models'][str(no)]
        for c in m['per_commit_holdout']:
            L.append('| M%d | %d | %s | %+.3f | %d | %d | %+.2f | %+.2f | %+.4f | %.4f | %.4f | %.4g |' % (
                no, c['commit'], c['recipe'], c['bias_shift'], c['n_fixed'], c['n_broken'], c['delta']['acc'], c['delta']['f1'], c['delta']['mcc'], c['auc_before'], c['auc_after'], mcnemar_exact(c['n_fixed'], c['n_broken'])))
    L.append('\n## Weight changes of the edited checkpoints\n')
    L.append('| Model | Tensor | Elements changed | Rows changed | L2 of change |')
    L.append('|---|---|---|---|---|')
    for no in MODELS:
        m = summary['models'][str(no)]
        for w in m['weight_diff']:
            L.append('| M%d | %s %s | %d (%.2f%%) | %s | %.3f |' % (no, w['tensor'], w['shape'], w['elements_changed'], w['share_changed_pct'], w['rows_changed'] if w['rows_changed'] is not None else '-', w['l2_change']))
    L.append('\n## Keyword slices on the holdout part: error rate in percent (original -> edited)\n')
    tags = ['security_with_lexicon', 'security_without_lexicon', 'nonsecurity_with_lexicon', 'nonsecurity_without_lexicon', 'empty_message']
    L.append('| Model | ' + ' | '.join(tags) + ' |')
    L.append('|---|' + '---|' * len(tags))
    for no in MODELS:
        m = summary['models'][str(no)]
        L.append('| M%d | %s |' % (no, ' | '.join('%.1f -> %.1f (n=%d)' % (m['slices']['holdout/' + t]['err_base_pct'], m['slices']['holdout/' + t]['err_edited_pct'], m['slices']['holdout/' + t]['n']) for t in tags)))
    L.append('\n## Error overlap on the holdout part\n')
    for tag in ('base', 'edited'):
        o = summary['holdout_overlap_' + tag]
        L.append('- %s: wrong in k of 8 models: %s; majority vote accuracy %.2f; mean accuracy %.2f' % (tag, o['wrong_in_k_models'], o['majority_vote_acc'], o['mean_acc']))
    with open(os.path.join(OUT, 'summary.md'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
