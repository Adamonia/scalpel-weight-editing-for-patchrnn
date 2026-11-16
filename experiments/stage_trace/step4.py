"""Step 4: build out/REPORT.md from the saved json files only (trace_M*.json, readout_M*.json, weights.json).
The text of the verdict is read from verdict.md (written by hand after reading the tables)."""
import json, os
import numpy as np

O = '/home/claude/work/exth2/out'
MODELS = [3, 4, 5, 8, 12, 13]
GA, GB = [3, 4, 13], [5, 8, 12]          # the two groups of the question
T = {n: json.load(open('%s/trace_M%d.json' % (O, n))) for n in MODELS}
R = {n: json.load(open('%s/readout_M%d.json' % (O, n))) for n in MODELS}
Wt = json.load(open('%s/weights.json' % O))
L = []
w = L.append
f = lambda x, k=3: ('%.' + str(k) + 'f') % x


def table(head, rows):
    w('| ' + ' | '.join(head) + ' |'); w('|' + '---|' * len(head))
    for r in rows:
        w('| ' + ' | '.join(str(c) for c in r) + ' |')
    w('')


def sep_mark(vals):
    """vals: dict model -> value. 'A>B' if all of {M3,M4,M13} are above all of {M5,M8,M12}, 'A<B' if all below."""
    a = [vals[n] for n in GA if vals.get(n) is not None]; b = [vals[n] for n in GB if vals.get(n) is not None]
    if len(a) < 3 or len(b) < 3:
        return 'n/a'
    if min(a) > max(b):
        return 'A>B'
    if max(a) < min(b):
        return 'A<B'
    return 'overlap'


hdr = ['Model'] + ['M%d' % n for n in MODELS]
w('# Why is noise-to-signal higher in M3, M4, M13 than in M5, M8, M12?\n')
w('All numbers are read from `out/trace_M*.json` (step1.py), `out/readout_M*.json` (step2.py) and `out/weights.json` (step3.py). No gradient training. '
  'K = 10 fresh N(0,1) draws (seeds 0..9) in the untrained message-embedding rows against the zeroed ("clean") model, on all 7608 test samples. '
  'Group A = {M3, M4, M13}, group B = {M5, M8, M12}. One trained model per architecture, n = 6: nothing here is a general claim about architectures.\n')

# ---------------------------------------------------------------- 0. checks
w('## 0. Checks\n')
table(['Model', 'heads (registry)', 'layers', 'untrained rows', 'hit samples', 'max abs gap, zeroed manual forward vs cached `untrained_zero` (all 7608)'],
      [['M%d' % n, T[n]['n_heads_registry'], T[n]['n_layers'], T[n]['n_untrained_rows'], T[n]['n_hit'], '%.1e' % T[n]['gap_zero_vs_cache_all_samples']] for n in MODELS])
w('Head-count check by function. The checkpoint shapes cannot give the head count (in_proj_weight is (3d, d) for any number of heads). '
  'Instead the clean message branch was recomputed with H = 1, 2, 4, 8, 16 on 256 hit samples; max abs margin gap to the cached margin:\n')
table(['Model', 'registry H'] + ['H=%s' % h for h in ('1', '2', '4', '8', '16')],
      [['M%d' % n, T[n]['n_heads_registry']] + ['%.1e' % T[n]['head_count_check_max_abs_margin_gap'][h] for h in ('1', '2', '4', '8', '16')] for n in MODELS])
w('The registry head count is the only value that reproduces the cached margins. (This tests the head count the weights were trained with, not whether the head split is "meaningful".)\n')

# ---------------------------------------------------------------- S1
w('## S1. Stage trace of the message branch\n')
w('Stages: `emb` = embedding + position; per layer l: `Ll.a` attention output (after out_proj), `Ll.n1` after norm1(x+a), `Ll.ff` feed-forward output, `Ll.n2` after norm2 (layer output); '
  '`pool` = final LayerNorm of the CLS vector; classifier `c0` = LayerNorm(384) of [twin_b, twin_a, msg], `c2` = Linear(384,384), `c3` = LayerNorm(384); `margin` = logit1 - logit0. '
  'Stage vectors are taken at the CLS position (the only position that is read out); for `emb` the CLS vector is unchanged by construction, so its sequence-level numbers are given in a separate table. '
  'Statistics are over the 4328 hit samples (the other samples have change 0) and the 10 draws. "Change" = noise minus clean. '
  '`ratio_raw` = mean |change| / mean |clean|. `ratio_centered` = mean |change| / mean |clean - mean over samples of clean|, i.e. relative to the part of the clean activation that varies between samples (the signal). '
  'Amplification = ratio at this stage / ratio at the previous stage that carries a change. Mark: A>B means all of M3, M4, M13 are above all of M5, M8, M12 at this stage.\n')
stages = ['emb', 'L0.a', 'L0.n1', 'L0.ff', 'L0.n2', 'L1.a', 'L1.n1', 'L1.ff', 'L1.n2', 'pool', 'c0', 'c2', 'c3', 'margin']


def getst(n, st):
    for r in T[n]['S1_trace_hit_samples']:
        if r['stage'] == st:
            return r
    return None


def stage_table(key, title, k=3, skip_emb=True):
    w('**%s**\n' % title)
    rows = []
    for st in stages:
        if skip_emb and st == 'emb':
            continue
        vals = {n: (getst(n, st).get(key) if getst(n, st) else None) for n in MODELS}
        if all(v is None for v in vals.values()):
            continue
        rows.append([st] + [f(vals[n], k) if vals[n] is not None else '-' for n in MODELS] + [sep_mark(vals)])
    table(['stage'] + ['M%d' % n for n in MODELS] + ['A vs B'], rows)


stage_table('nsr_sd_rms_centered', 'PRIMARY. Noise-to-signal at each stage, same definition as the margin NSR: rms over samples of the sd over draws (norm of the per-coordinate sd vector) / rms over samples of |clean centered|', 3)
stage_table('amp_nsr_sd_rms_vs_prev', 'Stage-to-stage amplification of the primary noise-to-signal', 2)
stage_table('nsr_sd_mean_centered', 'Same with means of norms instead of rms: mean sd-norm / mean |clean centered|', 3)
stage_table('sd_norm_mean', 'Mean norm of the sd vector over draws (noise around its own mean)', 3)
w('The tables below use the change relative to the clean (zeroed) model, as asked. This includes the systematic shift between the zeroed model and the mean of the draws, so it is larger than the draw-to-draw sd.\n')
w('**Where the primary noise-to-signal is built (read from the primary table).** Factors are ratios of the primary NSR between stages.\n')
def nsr_of(n, st):
    r = getst(n, st)
    return r['nsr_sd_rms_centered']
rows = []
for n in MODELS:
    two = T[n]['n_layers'] > 1
    l0 = nsr_of(n, 'L0.a'); l0e = nsr_of(n, 'L0.n2'); le = nsr_of(n, 'L1.n2') if two else l0e; pl = nsr_of(n, 'pool'); mg = nsr_of(n, 'margin')
    rows.append(['M%d' % n, f(l0), f(l0e), f(nsr_of(n, 'L1.a')) if two else '-', f(le), f(pl), f(mg), f(l0e / l0, 2), f(le / l0e, 2) if two else '-', f(pl / l0, 2), f(mg / pl, 2)])
table(['Model', 'NSR L0.a', 'NSR L0.n2', 'NSR L1.a', 'NSR last layer out', 'NSR pool', 'NSR margin (hit samples)', 'L0.n2 / L0.a', 'layer 1 factor', 'pool / L0.a (all after first attention)', 'margin / pool (classifier)'], rows)
BS = json.load(open(O + '/bootstrap.json'))
w('**Bootstrap over test samples (step5.py), NSR of the margin, all 7608 samples.** 95% intervals over 1000 resamples; they cover test-sample uncertainty only, not the choice of the 10 draws and not the training run.\n')
table(['Model', 'NSR', '95% interval'], [['M%d' % n, f(T[n]['S5']['nsr_all']), '%s - %s' % (f(BS['nsr_ci95']['M%d' % n][0]), f(BS['nsr_ci95']['M%d' % n][1]))] for n in MODELS])
w('Group ratio (mean NSR of A / mean NSR of B): median %s, 95%% interval %s - %s. Share of resamples with min(A) > max(B): %s.\n' % (f(BS['group_ratio_A_over_B_mean_nsr']['median'], 2), f(BS['group_ratio_A_over_B_mean_nsr']['ci95'][0], 2), f(BS['group_ratio_A_over_B_mean_nsr']['ci95'][1], 2), f(BS['min_A_minus_max_B']['share_positive'], 2)))
stage_table('clean_norm_mean', 'Mean norm of the clean activation', 2)
stage_table('clean_centered_norm_mean', 'Mean norm of the clean activation, centered over samples', 2)
stage_table('delta_norm_mean', 'Mean norm of the change', 3)
stage_table('ratio_raw', 'Ratio |change| / |clean| (raw)', 3)
stage_table('ratio_centered', 'Ratio |change| / |clean centered| (signal-relative)', 3)
stage_table('amp_centered_vs_prev', 'Stage-to-stage amplification of ratio_centered', 2)
stage_table('amp_raw_vs_prev', 'Stage-to-stage amplification of ratio_raw', 2)
w('**Sequence level (all 200 positions, Frobenius norm per sample), embedding and layer 0.** Shows how much change enters the layer and how much of it is at the CLS position.\n')
rows = []
for st in ['emb', 'L0.a', 'L0.n1', 'L0.ff', 'L0.n2']:
    vals = {n: getst(n, st).get('seq_ratio') for n in MODELS}
    rows.append([st] + [f(vals[n]) for n in MODELS] + [sep_mark(vals)])
table(['stage (seq ratio)'] + ['M%d' % n for n in MODELS] + ['A vs B'], rows)

w('**Layer-0 CLS attention output: where the change comes from.** Exact split of change of a = out_proj(sum_h w_h V_h): value path = out_proj(w_new (V_new - V_old)), weight path = out_proj((w_new - w_old) V_old). '
  'The CLS query itself does not change at layer 0 (the CLS input is not touched). Norms are means over hit samples and draws.\n')
table(['Model', '|change of a|', '|value path|', '|weight path|', 'cos(value, weight)', 'value / total', 'weight / total'],
      [['M%d' % n, f(T[n]['S1_layer0_cls_attention_paths']['mean_norm_delta_a']), f(T[n]['S1_layer0_cls_attention_paths']['mean_norm_value_path']),
        f(T[n]['S1_layer0_cls_attention_paths']['mean_norm_weight_path']), f(T[n]['S1_layer0_cls_attention_paths']['mean_cos_value_weight_path']),
        f(T[n]['S1_layer0_cls_attention_paths']['value_path_over_delta'], 2), f(T[n]['S1_layer0_cls_attention_paths']['weight_path_over_delta'], 2)] for n in MODELS])

w('**Branch contribution to the clean margin variance (all 7608 samples).** The classifier has LayerNorms, so the shares need not add to 1. "msg varies only" = margin with twin_b and twin_a features fixed at their means.\n')
table(['Model', 'var(margin)', 'var, msg varies only', 'share msg', 'var, code varies only', 'share code', 'R2 full vs msg-only', 'R2 full vs code-only'],
      [['M%d' % n] + [f(T[n]['S1_branch_variance'][k], 3 if 'var' in k else 3) for k in ('var_margin_clean', 'var_msg_varies_only', 'share_msg', 'var_code_varies_only', 'share_code', 'corr2_full_vs_msg_only', 'corr2_full_vs_code_only')] for n in MODELS])

# ---------------------------------------------------------------- S2
w('## S2. Readout gain at the pooled vector\n')
w('g_i = gradient of the margin with respect to the pooled message vector (128-d) at the clean point, per sample (input gradient, no weights trained). '
  'Alongness = |change . g/|g|| / |change| per sample. For a random direction in 128-d the mean absolute cosine is %s. '
  'Margin sd = per-sample sd over draws, rms over all 7608 samples. Decomposition: NSR = (rms |dp| / rms |p centered|) x (gain on noise / gain on signal), '
  'where gain on noise = margin sd / rms |dp| and gain on signal = clean margin sd / rms |p centered| (all 7608 samples).\n' % f(R[3]['random_direction_mean_abs_cos']))
table(['Model', 'mean |g|', 'mean |dp|', 'mean |along g|', 'mean |orthogonal|', '|along|/|dp| (ratio of means)', 'mean per-sample |cos|', 'margin sd, actual', 'margin sd, along-g part only', 'corr(linear, actual change)'],
      [['M%d' % n, f(R[n]['grad_norm_mean'], 2), f(R[n]['mean_dp_norm'], 2), f(R[n]['mean_abs_along']), f(R[n]['mean_ortho']), f(R[n]['mean_abs_along_over_mean_dp_norm'], 2),
        f(R[n]['mean_per_sample_abs_cos_dp_grad'], 2), f(R[n]['margin_sd_actual_rms_all']), f(R[n]['margin_sd_from_along_gradient_rms_all']), f(R[n]['corr_linear_vs_actual_change'], 3)] for n in MODELS])
table(['Model', 'rms |dp| (all)', 'rms |p centered|', 'relative change at pooled', 'gain on noise', 'gain on signal', 'gain ratio noise/signal', 'NSR (= product)'],
      [['M%d' % n, f(R[n]['rms_dp_all']), f(R[n]['rms_clean_pooled_centered']), f(R[n]['relative_change_at_pooled']), f(R[n]['gain_noise_margin_per_unit_dp']),
        f(R[n]['gain_signal_margin_per_unit_p']), f(R[n]['gain_ratio_noise_over_signal']), f(R[n]['nsr_all'])] for n in MODELS])
table(['Model', 'participation ratio of dp', 'top-1 PC share of dp', 'top-5 share of dp', 'participation ratio, clean pooled', 'top-5 share, clean pooled', 'cos(g, clean centered pooled)', '|mean dp over draws|', '|dp - mean over draws|'],
      [['M%d' % n, f(R[n]['dp_spectrum']['participation_ratio'], 1), f(R[n]['dp_spectrum']['top1_share'], 2), f(R[n]['dp_spectrum']['top5_share'], 2),
        f(R[n]['clean_pooled_spectrum_all_samples']['participation_ratio'], 1), f(R[n]['clean_pooled_spectrum_all_samples']['top5_share'], 2),
        f(R[n]['mean_cos_grad_vs_centered_clean_pooled'], 2), f(R[n]['norm_mean_dp_over_draws'], 2), f(R[n]['norm_dp_minus_mean_over_draws'], 2)] for n in MODELS])

# ---------------------------------------------------------------- S3
w('## S3. Attention shape (CLS query row, the row that feeds pooling)\n')
w('Rows: 200 keys (CLS + 199 message positions, padding not masked); maximal entropy = %s nats. Entropy and max weight are means over samples (hit samples) and heads. '
  'Mass on untrained = sum of the attention weights on untrained key positions; distribution over samples and heads (hit samples; "fresh" pools the 10 draws, "clean" is the zeroed model). '
  'Chance level (untrained keys / keys, mean over hit samples) = %s. "Max single" = largest weight on any one untrained token. Per-head values are in the json.\n' % (f(T[3]['S3']['max_entropy_nats'], 2), f(T[3]['S3']['chance_mass_on_U_mean'], 3)))
rows = []
for n in MODELS:
    for li in range(T[n]['n_layers']):
        for cond in ('fresh', 'clean'):
            c = T[n]['S3']['layer%d' % li][cond]
            m = c['massU_dist_over_samples_and_heads']; ms = c['max_single_U_weight_dist_over_samples_and_heads']
            rows.append(['M%d' % n, li, cond, f(c['entropy_mean'], 2), '%s-%s' % (f(min(c['entropy_per_head']), 2), f(max(c['entropy_per_head']), 2)), f(c['maxw_mean'], 2),
                         f(m['mean']), f(m['sd']), f(m['p90']), f(m['p99']), f(ms['mean']), f(ms['p99'])])
table(['Model', 'layer', 'rows', 'entropy mean', 'entropy per-head range', 'max weight mean', 'mass U mean', 'sd', 'p90', 'p99', 'max single U weight mean', 'p99'], rows)

# ---------------------------------------------------------------- S4
w('## S4. Weights (message branch and classifier)\n')
w('Frobenius / spectral norm. in_proj is split into Q, K, V blocks (128 x 128 each). LN gain = mean of the LayerNorm weight. No scale factor is applied to the embeddings in the forward code. '
  'Classifier input order is [twin_b, twin_a, msg]; "msg columns" are columns 256..383 of Linear(384,384).\n')
rows = []
for li in (0, 1):
    for name, key in (('Q', 'Q'), ('K', 'K'), ('V', 'V'), ('out_proj', 'out_proj'), ('linear1 (512x128)', 'linear1'), ('linear2 (128x512)', 'linear2')):
        r = ['layer%d %s' % (li, name)]
        for n in MODELS:
            ls = Wt['M%d' % n]['layers']
            r.append('%s / %s' % (f(ls[li][key]['fro'], 2), f(ls[li][key]['spec'], 2)) if li < len(ls) else '-')
        rows.append(r)
    for name, key in (('norm1 gain mean', 'norm1_gain_mean'), ('norm2 gain mean', 'norm2_gain_mean')):
        rows.append(['layer%d %s' % (li, name)] + [f(Wt['M%d' % n]['layers'][li][key], 3) if li < len(Wt['M%d' % n]['layers']) else '-' for n in MODELS])
rows.append(['final norm gain mean'] + [f(Wt['M%d' % n]['final_norm_gain_mean'], 3) for n in MODELS])
for name, key in (('classifier LN0 gain mean', 'ln0_gain_mean'), ('classifier LN3 gain mean', 'ln3_gain_mean'), ('LN0 gain mean, msg part', 'ln0_gain_mean_msg_part'), ('LN0 gain mean, twin part', 'ln0_gain_mean_twin_part'),
                  ('|w1 - w0| of last linear', 'margin_vector_norm (|w1-w0| of linear_4)')):
    rows.append([name] + [f(Wt['M%d' % n]['classifier'][key], 3) for n in MODELS])
for name, key in (('classifier Linear(384,384)', 'linear_2 (384x384)'), ('  msg columns', 'linear_2_msg_columns'), ('  twin columns', 'linear_2_twin_columns'), ('classifier Linear(384,2)', 'linear_4 (2x384)')):
    rows.append([name] + ['%s / %s' % (f(Wt['M%d' % n]['classifier'][key]['fro'], 2), f(Wt['M%d' % n]['classifier'][key]['spec'], 2)) for n in MODELS])
for name, key in (('trained msg rows, mean norm', 'trained_rows_mean_norm'), ('untrained rows (original N(0,1)), mean norm', 'untrained_rows_mean_norm_original'),
                  ('position embedding, mean row norm', 'position_embedding_mean_row_norm'), ('CLS token norm', 'cls_token_norm')):
    rows.append([name] + [f(Wt['M%d' % n]['embedding'][key], 2) for n in MODELS])
rows.append(['trained rows (count)'] + [Wt['M%d' % n]['embedding']['n_trained_rows'] for n in MODELS])
table(['tensor'] + ['M%d' % n for n in MODELS], rows)

# ---------------------------------------------------------------- S5
w('## S5. Clean signal and noise term (absolute units)\n')
w('Noise sd = per-sample sd of the margin over the 10 draws, rms over all 7608 samples (numerator). Clean sd = sd of the clean margin over samples (denominator). F1 and AUC in percent, clean = zeroed model, threshold margin > 0.\n')
table(['Model', 'F1 clean', 'AUC clean', 'F1 fresh mean (sd)', 'F1 gap', 'clean margin sd', 'noise sd rms (numerator)', 'NSR', 'sd of msg-only clean margin', 'noise / msg-only sd', 'noise sd rms, hit only', 'clean sd, hit only'],
      [['M%d' % n, f(T[n]['S5']['f1_clean'], 2), f(T[n]['S5']['auc_clean'], 2), '%s (%s)' % (f(T[n]['S5']['f1_fresh_mean'], 2), f(T[n]['S5']['f1_fresh_sd'], 2)), f(T[n]['S5']['f1_gap_clean_minus_fresh'], 2),
        f(T[n]['S5']['clean_margin_sd'], 3), f(T[n]['S5']['noise_sd_rms_all'], 3), f(T[n]['S5']['nsr_all'], 3), f(T[n]['S1_branch_variance']['sd_msg_only_clean'], 3), f(T[n]['S5']['nsr_over_msg_only_sd'], 3),
        f(T[n]['S5']['noise_sd_rms_hit'], 3), f(T[n]['S5']['sd_clean_hit_samples'], 3)] for n in MODELS])
for key, name in (('noise_sd_rms_all', 'noise sd rms'), ('clean_margin_sd', 'clean margin sd'), ('nsr_all', 'NSR')):
    a = [T[n]['S5'][key] for n in GA]; b = [T[n]['S5'][key] for n in GB]
    w('- %s: group A mean %s (range %s-%s), group B mean %s (range %s-%s).' % (name, f(np.mean(a)), f(min(a)), f(max(a)), f(np.mean(b)), f(min(b)), f(max(b))))
w('')

# ---------------------------------------------------------------- S6
w('## S6. Sample composition (number of untrained tokens in the window)\n')
w('Noise sd = rms over samples of the per-sample sd over 10 draws. NSR(all) uses the clean sd of all 7608 samples; NSR(group) uses the clean sd inside the group. Ratio to 1 token vs sqrt(k): independent equal-size token contributions would give sqrt(k) for exactly k tokens; the 3+ group has a mean of about 7 tokens.\n')
rows = []
for n in MODELS:
    for g in ('1', '2', '3+'):
        c = T[n]['S6'][g]
        rows.append(['M%d' % n, g, c['n'], f(c['noise_sd_rms'], 3), f(c['noise_sd_median'], 3), f(c['nsr_over_sd_clean_all'], 3), f(c['nsr_over_sd_clean_in_group'], 3), f(c['delta_pool_norm_mean'], 2),
                     f(c['noise_sd_rms'] / T[n]['S6']['1']['noise_sd_rms'], 2)])
table(['Model', 'untrained tokens', 'samples', 'margin sd rms', 'median', 'NSR (all)', 'NSR (group)', 'mean |dp| at pooled', 'sd / sd(1 token)'], rows)
a1 = [T[n]['S6']['1']['nsr_over_sd_clean_in_group'] for n in GA]; b1 = [T[n]['S6']['1']['nsr_over_sd_clean_in_group'] for n in GB]
w('Exactly one untrained token: NSR(group) A = %s, B = %s; min A %s vs max B %s.\n' % (', '.join(f(x, 3) for x in a1), ', '.join(f(x, 3) for x in b1), f(min(a1), 3), f(max(b1), 3)))

if os.path.exists('/home/claude/work/exth2/verdict.md'):
    w(open('/home/claude/work/exth2/verdict.md').read())
open('%s/REPORT.md' % O, 'w').write('\n'.join(L))
print('written', len(L))
