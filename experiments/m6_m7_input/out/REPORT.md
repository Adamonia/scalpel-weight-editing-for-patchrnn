# M6 / M7 input-format feasibility study

Models: M6 (word Transformer on BERT WordPiece ids) and M7 (same model on RoBERTa BPE ids). Test set 7,608 samples; holdout = 3,804 samples (part == 2, DEV_SEED 20261111). All numbers below were computed by the scripts in `/home/claude/work/ext67/` (step0 to step5). Metrics are in percent, positive class = security patch, AUC from the logit margin. dF1 is the paired bootstrap difference to the original model (2,000 resamples of test samples, seed 1, 95% percentile interval).

## Verdict

**No. The input defect cannot be repaired without training, for either model.**

- The earlier claim is **partly right**. It is right for M7 and for the message branch of M6. It is wrong for the code branches of M6.
- The training-time format is not a bug that sits between a good model and its input. The models were trained only on this format, and they are consistent with it (the archive predictions are reproduced 7608/7608 for both models). Normal tokenization is a different input distribution, and it hurts.
- Normal tokenization (single-space join), holdout F1: M6 65.13 -> 58.32; M7 68.39 -> 37.37.
- A label-free twin copy recovers part of the loss for M7 only: holdout F1 37.37 -> 54.92, which is still 13.5 points below the original (CI of the difference -16.2 to -10.8). About 57% of the lost F1 is recovered ((54.92 - 37.37) / (68.39 - 37.37)), but the result stays far below the original. For M6 the twin copy recovers nothing (58.32 -> 58.32).
- No inference-time variant is significantly better than the original model. The only positive number is M7 with each abstract token encoded alone (holdout +0.44 F1, CI -1.35 to +2.32; all samples +0.58, CI -0.73 to +1.81). That is not distinguishable from zero.

## 1. Facts

### How the training-time text is made
Source: `scalpel/retokenize.py` (its docstring, and the code, which I re-ran). I could not read the training notebooks: the model archive holds only `predicted.txt` and `temp/model_TwinTransformer.pth` for Model6 and Model7. The format is inferred from retokenize.py and confirmed by reproduction (below).

1. Each input (code before, code after, message) is a list of abstracted tokens (`VAR0`, `->`, `FUNC1`, `LITERAL`, `COMMENT`, message words that are already stemmed like `valu`, `profil`).
2. The tokens are joined with the empty string: `''.join(tokens)`. No space, no separator.
3. The pre-trained tokenizer encodes this one long string (no special tokens, 800 ids, BERT lowercases). Only its vocabulary and sub-word split are used. The embedding tables are 128-dimensional and trained from scratch, so no BERT/RoBERTa weights are involved.

Normal use would separate the tokens with spaces. Examples (real test sample):

| text | format | tokens |
|---|---|---|
| code `... ; COMMENT VAR2 = FUNC0 ( VAR0 ) ;` | M6 training | `comment ##var ##2 = fun ##c ##0 ( var ##0 ) ;` |
| | M6 normal | `comment var ##2 = fun ##c ##0 ( var ##0 ) ;` |
| code `VAR0 = FUNC0 ( VAR1 , LITERAL ,` | M7 training | `V AR 0 = FUN C 0 ( V AR 1 , L IT ERAL ,-` |
| | M7 normal | `V AR 0 Ġ= ĠFUN C 0 Ġ( ĠV AR 1 Ġ, ĠL IT ERAL Ġ,` |
| message `media cx231xx return error can't` | M6 training | `media ##c ##x ##23 ##1 ##xx ##ret ##urne ##rro ##rca ##n ' t` |
| | M6 normal | `media c ##x ##23 ##1 ##xx return error can ' t` |
| | M7 training | `medi ac x 231 xx return error can 't` |
| | M7 normal | `media Ġcx 231 xx Ġreturn Ġerror Ġcan 't` |

### Quantities on the 7,608 test samples

"Normal format" = single-space join (my definition of normal; see caveats). Positions are the non-padding positions in the model window (799 for code, 199 for message). "Untrained row" = embedding row whose id never occurs in the training windows (document frequency 0 in both classes, from the bundle); rows used in the unused last 5 code columns are ignored.

| quantity | M6 (BERT) | M7 (RoBERTa) |
|---|---|---|
| vocabulary rows | 30522 | 50265 |
| code table (embedTwin): rows never seen in training | 26921 (88.2%) | 45456 (90.4%) |
| message table (embedMsg): rows never seen in training | 22632 (74.1%) | 42535 (84.6%) |
| code, normal format: positions on an untrained row | 0.02% (training format), 0.02% (normal format) | 0.02% (training format), 15.55% (normal format) |
| message, normal format: positions on an untrained row (training / normal format) | 0.15% / 1.61% | 0.08% / 53.39% |
| code (tok_b / tok_a): share of normal-format tokens with no partner in the training-format sequence | 7.7% / 8.3% | 57.0% / 56.9% |
| message: same share | 68.5% | 61.8% |
| code (tok_b/tok_a): samples whose id sequence is identical in both formats | 12.8% / 10.5% | 0.1% / 0.0% |
| code (tok_a): border-marker tokens (## or Ġ), training / normal format | 36.9% / 29.3% | 0.1% / 56.9% |
| message: border-marker tokens, training / normal format | 79.6% / 36.3% | 0.0% / 53.3% |
| untrained code rows with a trained twin (of those with any twin) | 464 of 4300 | 2062 of 14012 |
| untrained message rows with a trained twin (of those with any twin) | 505 of 2727 | 3779 of 13293 |

Untrained rows are plain initial values: standard deviation 1.00 and mean norm about 11.07 for M6 and M7 code tables, the same as trained rows (11.08). So the checkpoint gives no sign of training for them; they are random N(0,1) vectors.

### Is the claim right?
- **M7 (RoBERTa): right.** The training format contains no word-initial `Ġ` token (0.1% of code positions against 57% in normal use). In normal use 15.5% of code positions and 53% of message positions land on rows that were never trained. Because spaces are removed, BPE also merges across token borders (`);`, `,-`, `){`) and these merges differ from normal ones. It is a systematic mismatch, not a few rare words.
- **M6, code branches: wrong.** WordPiece splits on punctuation and spaces already. Joining without a separator changes only places where two alphanumeric abstract tokens touch (`COMMENTVAR2`). The `##` marker is used in the normal way (36-37% of positions in training, 29% in normal use). Only 0.01-0.02% of code positions hit untrained rows in normal use; 7.7-8.3% of tokens differ.
- **M6, message branch: right.** Message words are glued (`mediacx231xx...`), giving word pieces that never occur in normal text. 68% of message tokens differ in normal use. (Only 1.6% land on untrained rows, because glued words get replaced by other, trained pieces, so the damage is a different segmentation, not a missing row.)
- Whether this is a "defect": it lowers nothing by itself, since the models were trained and tested in the same format. It is only a defect in the sense that the sub-word vocabulary is used in a way that the tokenizer was not made for, and most embedding rows (74-90%) are never used. The data do not show that a cleaner format would give a better model, because that needs retraining and I did not do it.

## 2. Inference-time changes

Variants:
- A0: original (training format).
- A1: A0 plus twin copy.
- N0 (task a): normal tokenization at inference, single-space join.
- N1 (task b): N0 plus twin copy. Twin copy: for every untrained row whose boundary twin (with/without `Ġ`, with/without `##`) is trained, copy the twin's row. Done for both embedding tables (code: 464 rows M6, 2,062 rows M7; message: 505 rows M6, 3,779 rows M7). No labels are used, nothing is tuned.
- N2 (own idea): N1, and all other untrained rows set to the mean of the trained rows of that table (M6: 26,457 code / 22,127 message rows; M7: 43,394 / 38,756). Idea: an unknown token gets an average vector instead of random noise.
- P0 (own idea): each abstract token encoded alone, then the ids are concatenated (no spaces, no gluing). P1: P0 plus twin copy.

Token types are zero in all variants, as in the original bundle. The original (A0) reproduces `predicted.txt` for 7608/7608 samples for both models, and the holdout F1 is 65.13 (M6) and 68.39 (M7).

### M6, all 7,608 samples
| variant | F1 | acc | prec | rec | AUC | dF1 vs original [95% CI] | flipped preds |
|---|---|---|---|---|---|---|---|
| A0 training format, no edit (original) | 65.16 | 75.85 | 61.62 | 69.13 | 82.09 | +0.00 [+0.00, +0.00] | 0 |
| A1 training format + twin copy | 65.17 | 75.87 | 61.64 | 69.13 | 82.09 | +0.01 [+0.00, +0.04] | 1 |
| N0 normal tokenization (single-space join) | 58.44 | 74.28 | 61.87 | 55.37 | 77.24 | -6.72 [-8.41, -5.02] | 1772 |
| N1 normal + twin copy | 58.35 | 74.24 | 61.82 | 55.25 | 77.25 | -6.81 [-8.50, -5.13] | 1773 |
| N2 normal + twin copy + mean fill | 58.54 | 74.29 | 61.85 | 55.57 | 77.32 | -6.62 [-8.31, -4.93] | 1777 |
| P0 each abstract token encoded alone | 58.44 | 74.28 | 61.87 | 55.37 | 77.24 | -6.72 [-8.41, -5.02] | 1772 |
| P1 each token alone + twin copy | 58.35 | 74.24 | 61.82 | 55.25 | 77.25 | -6.81 [-8.50, -5.13] | 1773 |

### M6, holdout 3,804 samples
| variant | F1 | acc | prec | rec | AUC | dF1 vs original [95% CI] | flipped preds |
|---|---|---|---|---|---|---|---|
| A0 training format, no edit (original) | 65.13 | 75.74 | 61.00 | 69.85 | 82.50 | +0.00 [+0.00, +0.00] | 0 |
| A1 training format + twin copy | 65.13 | 75.74 | 61.00 | 69.85 | 82.50 | +0.00 [+0.00, +0.00] | 0 |
| N0 normal tokenization (single-space join) | 58.32 | 74.26 | 61.43 | 55.51 | 77.18 | -6.81 [-9.09, -4.66] | 888 |
| N1 normal + twin copy | 58.32 | 74.26 | 61.43 | 55.51 | 77.20 | -6.81 [-9.11, -4.68] | 888 |
| N2 normal + twin copy + mean fill | 58.50 | 74.26 | 61.33 | 55.92 | 77.29 | -6.63 [-8.92, -4.46] | 888 |
| P0 each abstract token encoded alone | 58.32 | 74.26 | 61.43 | 55.51 | 77.18 | -6.81 [-9.09, -4.66] | 888 |
| P1 each token alone + twin copy | 58.32 | 74.26 | 61.43 | 55.51 | 77.20 | -6.81 [-9.11, -4.68] | 888 |

### M7, all 7,608 samples
| variant | F1 | acc | prec | rec | AUC | dF1 vs original [95% CI] | flipped preds |
|---|---|---|---|---|---|---|---|
| A0 training format, no edit (original) | 68.26 | 79.04 | 67.52 | 69.01 | 84.82 | +0.00 [+0.00, +0.00] | 0 |
| A1 training format + twin copy | 68.26 | 79.04 | 67.52 | 69.01 | 84.82 | +0.00 [+0.00, +0.00] | 0 |
| N0 normal tokenization (single-space join) | 38.45 | 69.70 | 57.14 | 28.97 | 66.95 | -29.81 [-31.93, -27.70] | 2138 |
| N1 normal + twin copy | 55.85 | 75.71 | 68.72 | 47.04 | 78.52 | -12.41 [-14.26, -10.67] | 1597 |
| N2 normal + twin copy + mean fill | 56.03 | 76.03 | 69.87 | 46.76 | 79.15 | -12.23 [-14.04, -10.44] | 1579 |
| P0 each abstract token encoded alone | 68.84 | 79.90 | 69.74 | 67.97 | 84.79 | +0.58 [-0.73, +1.81] | 1094 |
| P1 each token alone + twin copy | 68.84 | 79.90 | 69.74 | 67.97 | 84.79 | +0.58 [-0.73, +1.81] | 1094 |

### M7, holdout 3,804 samples
| variant | F1 | acc | prec | rec | AUC | dF1 vs original [95% CI] | flipped preds |
|---|---|---|---|---|---|---|---|
| A0 training format, no edit (original) | 68.39 | 79.07 | 67.06 | 69.77 | 85.20 | +0.00 [+0.00, +0.00] | 0 |
| A1 training format + twin copy | 68.39 | 79.07 | 67.06 | 69.77 | 85.20 | +0.00 [+0.00, +0.00] | 0 |
| N0 normal tokenization (single-space join) | 37.37 | 69.43 | 55.70 | 28.12 | 66.29 | -31.02 [-33.92, -28.08] | 1065 |
| N1 normal + twin copy | 54.92 | 75.53 | 68.23 | 45.95 | 78.36 | -13.47 [-16.17, -10.81] | 829 |
| N2 normal + twin copy + mean fill | 54.85 | 75.76 | 69.31 | 45.38 | 79.17 | -13.54 [-16.24, -10.89] | 830 |
| P0 each abstract token encoded alone | 68.83 | 80.02 | 69.68 | 67.99 | 84.67 | +0.44 [-1.35, +2.32] | 566 |
| P1 each token alone + twin copy | 68.83 | 80.02 | 69.68 | 67.99 | 84.67 | +0.44 [-1.35, +2.32] | 566 |

Notes on the tables:
- M6 P0 and P1 are identical to N0 and N1 for M6 by construction: WordPiece splits on spaces first, so encoding each word alone gives the same ids as a space-joined text. For M7, P1 equals P0 because P0 contains no `Ġ` tokens, so there is nothing for the twin copy to change.
- P0 for M7 is not normal tokenization: it keeps the training format's lack of `Ġ` and only stops BPE from merging across abstract-token borders. It is close to the training format, which is why it is close to the original (1,094 predictions flip, F1 and AUC unchanged within noise).
- A1 equals A0 except for one flipped prediction (M6, all samples): the original format uses almost no untrained rows in the test set.

### Which branch causes the loss (normal format in one branch only)
Features of each branch were computed in both formats and mixed before the head (step4). The first and last rows reproduce A0 and N0 exactly.

M6:
| branches in normal format | all 7,608: F1 (dF1 [CI]) | holdout 3,804: F1 (dF1 [CI]) |
|---|---|---|
| all orig (check) | 65.16 (+0.00 [+0.00, +0.00]) | 65.13 (+0.00 [+0.00, +0.00]) |
| code orig + msg normal | 58.43 (-6.74 [-8.39, -5.00]) | 57.90 (-7.23 [-9.52, -5.02]) |
| code normal + msg orig | 64.74 (-0.42 [-1.09, +0.25]) | 65.07 (-0.06 [-1.06, +0.87]) |
| all normal (check) | 58.44 (-6.72 [-8.41, -5.02]) | 58.32 (-6.81 [-9.09, -4.66]) |

M7:
| branches in normal format | all 7,608: F1 (dF1 [CI]) | holdout 3,804: F1 (dF1 [CI]) |
|---|---|---|
| all orig (check) | 68.26 (+0.00 [+0.00, +0.00]) | 68.39 (+0.00 [+0.00, +0.00]) |
| code orig + msg normal | 46.75 (-21.51 [-23.44, -19.63]) | 45.88 (-22.51 [-25.09, -19.84]) |
| code normal + msg orig | 65.62 (-2.64 [-3.81, -1.48]) | 65.02 (-3.37 [-5.02, -1.72]) |
| all normal (check) | 38.45 (-29.81 [-31.93, -27.70]) | 37.37 (-31.02 [-33.92, -28.08]) |

M6: the whole loss comes from the message branch; normal code alone changes F1 by -0.42 (all, CI -1.09 to +0.25) and -0.06 (holdout), not significant. M7: both branches contribute, the message branch most (-21.5 F1 alone), code alone -2.6 (all) / -3.4 (holdout); effects are not additive.

## 3. Interpretation

- The models rely on the exact glued format. In M7 the normal text moves more than half of the code positions onto Ġ rows that are random vectors. The twin copy repairs this only partly. Two likely reasons, not tested separately: (1) only some untrained rows have a trained twin (M7 code: 2,062 of 14,012 untrained rows that have any twin); the most frequent unseen tokens in normal M7 code (`Ġ)`, `Ġ,`, `ĠFUN`, `Ġ{`, `Ġ}`) are exactly the ones that training only saw merged (`);`, `,-`, `){`), so a copied twin row may be a poor stand-in; (2) the segmentation itself differs, which no embedding edit can change.
- M6 has almost no untrained-row problem in code, so the twin copy has nothing to do there; its loss is from message segmentation, which is not a row-level problem.
- The weight-edit result (no gain for M6 and M7) is consistent with this: the weakness is not in a few weights but in the use of the vocabulary. Whether a model trained on normal tokenization would be better was not tested.

## 4. What I could not verify, and what failed
- The training notebooks are not in the archive for Model6 and Model7. The training format is taken from `retokenize.py` and the 7608/7608 reproduction of `predicted.txt`; this proves that the format reproduces the saved predictions, not how the notebook code was written.
- "Normal tokenization" has no single definition. I used one space between abstract tokens. The original source text is not available after abstraction (identifiers become `VARn`), so a truly natural text cannot be built. Message words are already stemmed, so even the normal format is not natural English.
- The conclusion holds for the variants I tried (space join, per-token, twin copy, mean fill). I did not try other label-free edits (for example learned or similarity-based mapping of rows) or any training; a variant with training is outside this study.
- Untrained rows were found from the bundle document frequencies (training windows of 799/199 positions). I checked their weights only for the statistics shown above, not against the original initialization (not available).
- Bootstrap intervals describe sampling noise over test samples only, for one fixed set of trained weights. I looked at all 7,608 and the holdout side by side; no variant or setting was chosen from them. Holdout was not kept sealed for this study, because I report it as requested.
- One run failed: the first attempt of the branch analysis (step4) was killed for lack of memory (stored output views kept large tensors alive). I fixed it with `.clone()` and re-ran it; the reported numbers come from the re-run, and its A0 and N0 rows match the main runs.

## 5. Files
- `common.py` shared code (tokenization variants, model run, metrics, bootstrap)
- `step0_check.py` reproduction check; `step1_facts.py` facts (`out/facts.json`); `step2_eval.py` variant logits; `step3_analyze.py` metrics and bootstrap (`out/results.json`, `out/results.txt`); `step4_branch.py` branch analysis (`out/branch_ablation.json`); `step5_report.py` + `report_template.md` build this report.
- Cached logits and ids are in `/home/claude/work/ext67/cache/`.
