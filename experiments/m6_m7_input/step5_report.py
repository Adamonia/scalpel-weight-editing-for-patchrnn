# Build REPORT.md. Tables come from the JSON files; nothing is typed in by hand there.
import json
R_ = json.load(open('out/results.json')); B_ = json.load(open('out/branch_ablation.json')); F_ = json.load(open('out/facts.json'))
NAME = {('orig', 'none'): 'A0 training format, no edit (original)', ('orig', 'twin'): 'A1 training format + twin copy',
        ('space', 'none'): 'N0 normal tokenization (single-space join)', ('space', 'twin'): 'N1 normal + twin copy',
        ('space', 'twin+mean'): 'N2 normal + twin copy + mean fill', ('pertok', 'none'): 'P0 each abstract token encoded alone',
        ('pertok', 'twin'): 'P1 each token alone + twin copy'}
def table(m, part):
    s = '| variant | F1 | acc | prec | rec | AUC | dF1 vs original [95% CI] | flipped preds |\n|---|---|---|---|---|---|---|---|\n'
    for r in R_[str(m)]['rows']:
        x = r[part]
        s += '| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %d |\n' % (NAME[(r['format'], r['edit'])], x['f1'], x['acc'], x['prec'], x['rec'], x['auc'], x['dF1'], x['lo'], x['hi'], x['flips'])
    return s
def btable(m):
    s = '| branches in normal format | all 7,608: F1 (dF1 [CI]) | holdout 3,804: F1 (dF1 [CI]) |\n|---|---|---|\n'
    for k, r in B_[str(m)].items():
        s += '| %s | %.2f (%+.2f [%+.2f, %+.2f]) | %.2f (%+.2f [%+.2f, %+.2f]) |\n' % (k, r['all']['f1'], r['all']['dF1'], r['all']['lo'], r['all']['hi'], r['hold']['f1'], r['hold']['dF1'], r['hold']['lo'], r['hold']['hi'])
    return s
def facts_table():
    s = '| quantity | M6 (BERT) | M7 (RoBERTa) |\n|---|---|---|\n'
    g = lambda m, *k: F_[str(m)][k[0]][k[1]] if len(k) == 2 else F_[str(m)][k[0]]
    rows = [('vocabulary rows', lambda m: '%d' % F_[str(m)]['vocab']),
            ('code table (embedTwin): rows never seen in training', lambda m: '%d (%.1f%%)' % (g(m, 'rows_code', 'untrained'), 100 * g(m, 'rows_code', 'frac_untrained'))),
            ('message table (embedMsg): rows never seen in training', lambda m: '%d (%.1f%%)' % (g(m, 'rows_msg', 'untrained'), 100 * g(m, 'rows_msg', 'frac_untrained'))),
            ('code, normal format: positions on an untrained row', lambda m: '%.2f%% / %.2f%%' % (50 * (g(m, 'pos_tok_b', 'frac_pos_untrained_row_space') + g(m, 'pos_tok_a', 'frac_pos_untrained_row_space')), 0) if False else '%.2f%% (training format), %.2f%% (normal format)' % (50 * (g(m, 'pos_tok_b', 'frac_pos_untrained_row_orig') + g(m, 'pos_tok_a', 'frac_pos_untrained_row_orig')), 50 * (g(m, 'pos_tok_b', 'frac_pos_untrained_row_space') + g(m, 'pos_tok_a', 'frac_pos_untrained_row_space')))),
            ('message, normal format: positions on an untrained row (training / normal format)', lambda m: '%.2f%% / %.2f%%' % (100 * g(m, 'pos_msg', 'frac_pos_untrained_row_orig'), 100 * g(m, 'pos_msg', 'frac_pos_untrained_row_space'))),
            ('code (tok_b / tok_a): share of normal-format tokens with no partner in the training-format sequence', lambda m: '%.1f%% / %.1f%%' % (100 * g(m, 'pos_tok_b', 'frac_multiset_mismatch_space'), 100 * g(m, 'pos_tok_a', 'frac_multiset_mismatch_space'))),
            ('message: same share', lambda m: '%.1f%%' % (100 * g(m, 'pos_msg', 'frac_multiset_mismatch_space'))),
            ('code (tok_b/tok_a): samples whose id sequence is identical in both formats', lambda m: '%.1f%% / %.1f%%' % (100 * g(m, 'pos_tok_b', 'frac_samples_identical'), 100 * g(m, 'pos_tok_a', 'frac_samples_identical'))),
            ('code (tok_a): border-marker tokens (## or Ġ), training / normal format', lambda m: '%.1f%% / %.1f%%' % (100 * g(m, 'pos_tok_a', 'frac_border_marker_orig'), 100 * g(m, 'pos_tok_a', 'frac_border_marker_space'))),
            ('message: border-marker tokens, training / normal format', lambda m: '%.1f%% / %.1f%%' % (100 * g(m, 'pos_msg', 'frac_border_marker_orig'), 100 * g(m, 'pos_msg', 'frac_border_marker_space'))),
            ('untrained code rows with a trained twin (of those with any twin)', lambda m: '%d of %d' % (g(m, 'twins_code', 'twin_trained'), g(m, 'twins_code', 'has_twin'))),
            ('untrained message rows with a trained twin (of those with any twin)', lambda m: '%d of %d' % (g(m, 'twins_msg', 'twin_trained'), g(m, 'twins_msg', 'has_twin')))]
    for n, f in rows: s += '| %s | %s | %s |\n' % (n, f(6), f(7))
    return s
txt = open('report_template.md').read()
for k, v in {'FACTS': facts_table(), 'M6ALL': table(6, 'all'), 'M6HOLD': table(6, 'hold'), 'M7ALL': table(7, 'all'), 'M7HOLD': table(7, 'hold'), 'B6': btable(6), 'B7': btable(7)}.items():
    txt = txt.replace('{{%s}}' % k, v)
open('out/REPORT.md', 'w').write(txt)
