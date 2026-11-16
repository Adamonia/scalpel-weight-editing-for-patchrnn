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

{{FACTS}}
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
{{M6ALL}}
### M6, holdout 3,804 samples
{{M6HOLD}}
### M7, all 7,608 samples
{{M7ALL}}
### M7, holdout 3,804 samples
{{M7HOLD}}
Notes on the tables:
- M6 P0 and P1 are identical to N0 and N1 for M6 by construction: WordPiece splits on spaces first, so encoding each word alone gives the same ids as a space-joined text. For M7, P1 equals P0 because P0 contains no `Ġ` tokens, so there is nothing for the twin copy to change.
- P0 for M7 is not normal tokenization: it keeps the training format's lack of `Ġ` and only stops BPE from merging across abstract-token borders. It is close to the training format, which is why it is close to the original (1,094 predictions flip, F1 and AUC unchanged within noise).
- A1 equals A0 except for one flipped prediction (M6, all samples): the original format uses almost no untrained rows in the test set.

### Which branch causes the loss (normal format in one branch only)
Features of each branch were computed in both formats and mixed before the head (step4). The first and last rows reproduce A0 and N0 exactly.

M6:
{{B6}}
M7:
{{B7}}
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
