# Scalpel: direct weight editing of security-patch classifiers by an LLM agent

> Run records (`runs/`), data bundles (`bundle*/`), the model archive and model checkpoints are not in this repository because of their size. Paths in this file that point to them are for a local copy.

## Summary

Scalpel lets an LLM agent find the parts of a trained network that do the task incorrectly and correct them
with direct edits of the weights. There is no gradient training. The agent reads diagnostics, writes
values into weight tensors, and each edit must pass a guard before it is kept.

The tool works on 12 trained security-patch classifiers from the PatchRNN archive (PatchDB data):
Model0 (LSTM, PatchRNN), Model2 (Conv2D-E), Model3/4/5/8/12/13 (Transformer encoders), and the models M6, M7, M14, M15.
The data are split in three parts: diag (2,282 patches), guard (1,522) and a sealed holdout (3,804).

Main finding. The embedding rows of message tokens that were never seen in training are random noise.
Zeroing these rows raises F1, mostly for models with few attention heads (round 3: M3 +2.33 F1, M4 +1.50).
The noise goes through the attention weights. Many heads average it away.
The effect was confirmed on 50 newly trained models (10 architectures x 5 seeds). The gain is positive in all 50 models,
the mean gain is +0.93 F1, and each doubling of the heads lowers the gain by 0.185 F1 (exact permutation p = 0.0004).

## Three rounds

- Round 1: guard rule MCC. The agent chose the edits. See the table below.
- Round 2: guard rule F1. The agent chose the edits. See the table below.
- Round 3: one fixed recipe for all models. Every message-embedding row is multiplied by c/(c+3), where c is the training count of the token. See `experiments/round3/`.

## Layout

```
.
|-- scalpel/             The tool: engine, tool functions, edits, agent loop, model classes, report code.
|-- tests/               Engine tests and a scripted test of the agent loop.
|-- study/               Scripts of rounds 1 and 2 (diagnostics, recipes, tables, checks).
|-- tokenizers/          Tokenizer files for Model6, Model7 and Model15.
|-- results/             Final tables of rounds 1 and 2 (summary_all.md, summary_all.json, verify_final.json).
|-- experiments/         All later studies. See experiments/README.md.
|   |-- controls/        Placebo rows, threshold-only, least-squares head, 50 splits, Holm correction.
|   |-- m6_m7_input/     Can M6 and M7 be repaired by a change of the input format?
|   |-- fresh_noise/     Fresh random rows in place of the untrained rows.
|   |-- stage_trace/     Where the noise grows inside the network.
|   |-- round3/          The fixed recipe, chosen on diag, confirmed on guard, tested once on the holdout.
|   |-- retrain_kit/     Training and analysis of the 50 new models (10 architectures x 5 seeds).
|   |-- figures/         Scripts and images of the figures.
|   `-- tools/           Helper scripts that rebuild and check the models and start earlier steps.
|-- requirements.txt     Python packages.
`-- .gitignore
```

Local only (not in the repository): `runs/` (session records), `bundle/`, `bundle_bert/`, `bundle_roberta/`,
`bundle_codebert/` (data bundles), `model-archive/` (original models).

## How to start

1. To read the results, open `results/summary_all.md` (rounds 1 and 2) and `experiments/README.md` (all later studies).
2. To run the tool, install the packages (`pip install -r requirements.txt`), get the model archive and make the data bundle (next sections), then run a session.
3. To check the tool, run the tests (section Tests).
4. To train and test the 50 models, read `experiments/retrain_kit/README.md`.

## Data protocol

The original test set (7,608 samples, split seed 10) is divided one time with a fixed seed:

| Part | Samples | Use |
|---|---|---|
| diag | 2,282 | The agent reads samples, labels and model outputs. |
| guard | 1,522 | Aggregate metrics only. A commit is refused if the MCC of diag or guard decreases. |
| holdout | 3,804 | Sealed. `finalize` reads it one time and locks the session. |

## Install

```
pip install -r requirements.txt                 # scikit-learn is needed only for study/verify_final.py
# experiments/figures also needs matplotlib
```

## Prepare the data bundle (one time)

```
git clone https://github.com/shuwang127/PatchRNN
unzip PatchRNN/temp/data.zip -d temp; unzip PatchRNN/temp/props.zip -d temp; unzip PatchRNN/temp/msgs.zip -d temp
python -m scalpel.prepare temp bundle
```

The bundle holds the test arrays, the vocabularies, the token statistics of the training set and the
raw text of the test samples. The pipeline reproduces the notebooks: 35,576 code tokens, 80,831 message
tokens, and the labels of `reals.txt` (7,608 of 7,608).

## Run a session

```
python -m scalpel.cli init  --run runs/M3 --archive model-archive --bundle bundle --model 3
python -m scalpel.cli tools                      # list of all tools
python -m scalpel.cli call  --run runs/M3 overview
python -m scalpel.cli call  --run runs/M3 error_census
python -m scalpel.cli call  --run runs/M3 edit_tensor '{"name":"classifier.4.bias","op":"add","value":-0.7159,"index":[1]}'
python -m scalpel.cli call  --run runs/M3 evaluate
python -m scalpel.cli call  --run runs/M3 commit '{"rationale":"...","error_type":"..."}'
python -m scalpel.cli call  --run runs/M3 finalize
```

`call` is the interface for an external LLM agent or for a person.

Autonomous loop (uses the `anthropic` Python package and needs the variable `ANTHROPIC_API_KEY`):

```
python -m scalpel.cli agent --run runs/M3 --llm claude-opus-5-5 --max-turns 60
```

The loop and `call` use the same dispatcher (`scalpel/tools.py`). The loop is tested with a scripted
client (`tests/test_agent_loop.py`). It was not run against the live API in the study.

## Study of the model archive (rounds 1 and 2)

An LLM agent (configured model id `claude-fable-5-2`) ran one session for each model through `scalpel.cli call`.
The study has 12 models: M0, M2, M3, M4, M5, M6, M7, M8, M12, M13, M14, M15. Model1 cannot be reconstructed.
There are two rounds here. Round 3 is in `experiments/round3/`.

- Round 1 (guard rule MCC): recipes prior correction, rare or untrained rows, code branches. Records are in `runs/M*`.
- Round 2 (guard rule F1): recipes untrained rows, rare rows, F1 threshold. The sessions start with `--guard-metric f1` at init. Records are in `runs/R2_M*`.

All sessions were finalized after the last edit. Holdout part, 3,804 samples, original -> edited weights.
d F1 and d Acc are changes with 95 % intervals.

## Round 1 (MCC rule; recipes: prior correction, rare or untrained rows, code branches)

| Model | Accepted edits | F1 orig | F1 edit | d F1 [95% CI] | Acc orig | Acc edit | d Acc [95% CI] | Prec | Rec | AUC | fixed/broken | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 LSTM baseline (PatchRNN) | code branches removed | 73.75 | 72.88 | -0.86 [-1.69, -0.06] | 83.04 | 82.65 | -0.39 [-0.92, +0.13] | 74.1 -> 73.9 | 73.4 -> 71.9 | 0.8766 -> 0.8730 | 45/60 | 0.17 |
| M2 Conv2D-E | rare rows to zero | 73.32 | 73.79 | +0.47 [-0.72, +1.68] | 83.62 | 83.44 | -0.18 [-0.92, +0.55] | 77.7 -> 75.8 | 69.4 -> 71.9 | 0.8830 -> 0.8921 | 99/106 | 0.68 |
| M3 Trans-1H | prior correction; untrained rows to zero | 72.67 | 72.38 | -0.29 [-1.69, +1.04] | 81.57 | 83.47 | +1.89 [+1.00, +2.76] | 70.0 -> 79.0 | 75.5 -> 66.8 | 0.8871 -> 0.8968 | 185/113 | 3.6e-05 |
| M4 Trans-2H | prior correction; rare rows to zero; recentering | 71.62 | 72.05 | +0.43 [-0.90, +1.72] | 81.02 | 83.33 | +2.31 [+1.45, +3.18] | 69.5 -> 79.0 | 73.8 -> 66.2 | 0.8671 -> 0.8812 | 183/95 | 0 |
| M5 Trans-8H | prior correction; untrained rows to zero | 74.04 | 74.44 | +0.40 [-0.82, +1.64] | 81.49 | 83.57 | +2.08 [+1.24, +2.97] | 67.9 -> 75.1 | 81.4 -> 73.7 | 0.8897 -> 0.8932 | 185/106 | 4e-06 |
| M6 Trans-8H+B | prior correction | 65.13 | 60.84 | -4.29 [-6.23, -2.42] | 75.74 | 78.34 | +2.60 [+1.42, +3.79] | 61.0 -> 73.6 | 69.9 -> 51.9 | 0.8250 -> 0.8250 | 321/222 | 2.5e-05 |
| M7 Trans-8H+R | prior correction; untrained rows to zero | 68.39 | 66.39 | -2.00 [-3.55, -0.50] | 79.08 | 80.68 | +1.60 [+0.66, +2.55] | 67.1 -> 76.2 | 69.8 -> 58.8 | 0.8520 -> 0.8521 | 196/135 | 0.00095 |
| M8 Trans-16H | prior correction; untrained rows to zero | 71.75 | 72.18 | +0.43 [-0.70, +1.54] | 81.41 | 83.12 | +1.71 [+0.97, +2.44] | 70.8 -> 77.6 | 72.8 -> 67.5 | 0.8694 -> 0.8748 | 136/71 | 7e-06 |
| M12 Trans-8H-2E | prior correction; untrained rows to zero | 74.54 | 74.64 | +0.10 [-1.00, +1.18] | 82.10 | 83.44 | +1.34 [+0.58, +2.13] | 69.2 -> 74.2 | 80.8 -> 75.1 | 0.8886 -> 0.8899 | 140/89 | 0.00091 |
| M13 Trans-1H-2E | rare rows to zero | 72.83 | 72.98 | +0.15 [-0.95, +1.26] | 81.23 | 81.41 | +0.18 [-0.63, +1.00] | 68.7 -> 69.1 | 77.6 -> 77.4 | 0.8737 -> 0.8802 | 124/117 | 0.7 |
| M14 ParCombCtl (hybrid) | untrained rows to zero (CNN sub-model) | 76.58 | 76.69 | +0.11 [-0.45, +0.68] | 84.96 | 84.96 | +0.00 [-0.37, +0.37] | 77.4 -> 77.1 | 75.8 -> 76.3 | 0.9094 -> 0.9100 | 26/26 | 1 |
| M15 CodeBERT-SC | none | 85.23 | 85.23 | +0.00 [+0.00, +0.00] | 90.30 | 90.30 | +0.00 [+0.00, +0.00] | 84.2 -> 84.2 | 86.3 -> 86.3 | 0.9617 -> 0.9617 | 0/0 | 1 |

## Round 2 (F1 rule; recipes: untrained rows, rare rows, F1 threshold)

| Model | Accepted edits | F1 orig | F1 edit | d F1 [95% CI] | Acc orig | Acc edit | d Acc [95% CI] | Prec | Rec | AUC | fixed/broken | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 LSTM baseline (PatchRNN) | F1 threshold | 73.75 | 74.19 | +0.45 [-0.16, +1.07] | 83.04 | 82.94 | -0.10 [-0.50, +0.29] | 74.1 -> 72.8 | 73.4 -> 75.6 | 0.8766 -> 0.8766 | 27/31 | 0.69 |
| M2 Conv2D-E | untrained rows to zero; F1 threshold | 73.32 | 73.88 | +0.56 [-0.37, +1.49] | 83.62 | 83.57 | -0.05 [-0.60, +0.50] | 77.7 -> 76.3 | 69.4 -> 71.6 | 0.8830 -> 0.8860 | 57/59 | 0.93 |
| M3 Trans-1H | untrained rows to zero; rare rows to zero; F1 threshold | 72.67 | 74.49 | +1.82 [+0.72, +2.92] | 81.57 | 82.68 | +1.10 [+0.32, +1.89] | 70.0 -> 71.3 | 75.5 -> 78.0 | 0.8871 -> 0.8982 | 136/94 | 0.0067 |
| M4 Trans-2H | untrained rows to zero | 71.62 | 73.11 | +1.50 [+0.59, +2.42] | 81.02 | 82.39 | +1.37 [+0.71, +2.02] | 69.5 -> 72.4 | 73.8 -> 73.8 | 0.8671 -> 0.8781 | 110/58 | 7.4e-05 |
| M5 Trans-8H | untrained rows to zero; F1 threshold | 74.04 | 74.40 | +0.36 [-0.84, +1.60] | 81.49 | 83.52 | +2.02 [+1.18, +2.92] | 67.9 -> 75.0 | 81.4 -> 73.8 | 0.8897 -> 0.8932 | 182/105 | 6e-06 |
| M6 Trans-8H+B | untrained rows to zero | 65.13 | 65.13 | -0.00 [-0.15, +0.12] | 75.74 | 75.76 | +0.03 [-0.08, +0.16] | 61.0 -> 61.1 | 69.9 -> 69.8 | 0.8250 -> 0.8250 | 3/2 | 1 |
| M7 Trans-8H+R | untrained rows to zero | 68.39 | 68.44 | +0.05 [+0.00, +0.14] | 79.08 | 79.13 | +0.05 [+0.00, +0.13] | 67.1 -> 67.2 | 69.8 -> 69.8 | 0.8520 -> 0.8521 | 2/0 | 0.5 |
| M8 Trans-16H | untrained rows to zero | 71.75 | 72.54 | +0.78 [+0.04, +1.54] | 81.41 | 82.05 | +0.63 [+0.11, +1.16] | 70.8 -> 72.0 | 72.8 -> 73.1 | 0.8694 -> 0.8748 | 64/40 | 0.024 |
| M12 Trans-8H-2E | untrained rows to zero | 74.54 | 74.27 | -0.27 [-0.99, +0.45] | 82.10 | 81.97 | -0.13 [-0.71, +0.42] | 69.2 -> 69.1 | 80.8 -> 80.2 | 0.8886 -> 0.8899 | 57/62 | 0.71 |
| M13 Trans-1H-2E | untrained rows to zero | 72.83 | 73.51 | +0.68 [-0.07, +1.41] | 81.23 | 81.81 | +0.58 [+0.03, +1.13] | 68.7 -> 69.7 | 77.6 -> 77.8 | 0.8737 -> 0.8778 | 68/46 | 0.049 |
| M14 ParCombCtl (hybrid) | untrained rows to zero (CNN sub-model) | 76.58 | 76.69 | +0.11 [-0.45, +0.68] | 84.96 | 84.96 | +0.00 [-0.37, +0.37] | 77.4 -> 77.1 | 75.8 -> 76.3 | 0.9094 -> 0.9100 | 26/26 | 1 |
| M15 CodeBERT-SC | F1 threshold | 85.23 | 85.16 | -0.08 [-0.81, +0.64] | 90.30 | 90.59 | +0.29 [-0.18, +0.76] | 84.2 -> 87.2 | 86.3 -> 83.2 | 0.9617 -> 0.9617 | 49/38 | 0.28 |

F1 is the main metric. In round 1 accuracy rises but F1 does not. In round 2 the F1 interval is above zero for M3, M4 and M8.

See `results/summary_all.md` for the same tables. Round 3 and all later studies are in `experiments/` (see the section Later studies).


## Later studies

All studies after round 2 are in `experiments/`. `experiments/README.md` has a table with the question and the result of each folder.
The scripts there use absolute paths of the study machine. Change them before a run.

## Folders of the first two rounds

| Path | Content |
|---|---|
| `scalpel/` | The tool. |
| `tests/` | Engine tests and the scripted test of the API loop. |
| `study/` | `battery.py` (diagnostics), `protocol.py` (round 1 recipes), `r2_protocol.py` (round 2 recipes), `overlap.py`, `show_samples.py`, `aggregate.py` (round 1 tables), `aggregate2.py` (`results/summary_all.*` from both rounds), `verify_final.py` (independent check with scikit-learn), `verify_sample.py` (check on a random sample with a new forward pass), `m14_sub.py` and `r2_m14_extra.py` (sub-model edits in Model14), `m15_feats.py` and `m15_run.sh` (Model15), `r2_m0_probe.py` and `r2_m0_diag.py` (probes of Model0). |
| `tokenizers/` | `bert/vocab.txt`, `roberta/vocab.json`, `roberta/merges.txt`. |
| `results/` | `summary_all.md`, `summary_all.json`, `verify_final.json`. |
| `bundle/` (local) | Data bundle made by `scalpel.prepare`, with the cached outputs of the original models. |
| `bundle_bert/`, `bundle_roberta/`, `bundle_codebert/` (local) | Bundles of Model6, Model7 and Model15. |
| `runs/M*/` (local) | Records of the round 1 sessions: `journal.jsonl`, `FINAL.json`, `final_predictions.npz`, edited checkpoint (M0 to M5, M8, M12, M13). M6, M7, M14 and M15 hold records only. Use a new folder for a new session. |
| `runs/R2_M*/` (local) | Records of the round 2 sessions: `journal.jsonl`, `FINAL.json`, `final_predictions.npz`, `config.json`. No edited checkpoint. `study/r2_protocol.py` and `study/protocol.py` make the edits again. |

`raw_test.json` is not in the three new bundle folders. It is equal to `bundle/raw_test.json`. Copy it from
`bundle/` into `bundle_bert/`, `bundle_roberta/` and `bundle_codebert/` before `show_sample` is used.

The npz files are compressed (`numpy.savez_compressed`). `numpy.load` reads them as usual. The base features
of Model14 are split in three files because of the size limit: `bundle/base_feats_M14_a.npz` (head, twin_b),
`_b.npz` (twin_a), `_c.npz` (msg). Merge them into one `base_feats_M14.npz` before a session of Model14:
`np.savez(path, **{k: v for f in 'abc' for k, v in np.load('base_feats_M14_%s.npz' % f).items()})`.

The study scripts read `SCALPEL_RUNS`, `SCALPEL_BUNDLE` and `SCALPEL_ARCHIVE`. The defaults are `runs/`,
`bundle/` and `../model-archive/`.

To load an edited checkpoint in a notebook of the first article, put `runs/M3/edited_model_TwinTransformer.pth`
in the place of `model_TwinTransformer.pth`.

## Tools

Instruments (read only): `overview`, `metrics`, `error_census`, `slice_metrics`, `list_errors`,
`show_sample`, `branch_attribution`, `token_attribution`, `vocab_search`, `untrained_summary`,
`token_effect`, `effect_by_frequency`, `solo_probe`, `row_neighbors`, `unit_report`, `bias_scan`,
`weight_read`.

Edits: `edit_embedding_rows` (blend, mean_of, mean_trained, zero, pad, scale, compose, nearest_numeric),
`edit_tensor` (set, add, scale on a slice), `edit_feature_gain`, `edit_branch_gain`, `edit_recenter`,
`edit_rank1` (closed-form rank-one edit of a head layer).

Control: `evaluate`, `commit`, `rollback`, `finalize`.

## Files of a run folder

| File | Content |
|---|---|
| `journal.jsonl` | Each tool call, edit, commit and rollback with arguments and results. |
| `FINAL.json` | Holdout result: metrics, McNemar test, bootstrap intervals, AUC, weight changes. |
| `edited_model_*.pth` | Edited checkpoint. Same keys as the original file. |
| `final_predictions.npz` | Labels, parts, predictions and logits of the original and edited weights. |

## Tests

```
python tests/test_engine.py ARCHIVE BUNDLE 0
python tests/test_agent_loop.py ARCHIVE BUNDLE
```

## Add a model

Add a class to `scalpel/models.py` with `STAGES`, `stage_of`, `run_stage`, `HEAD_LINEARS`, `OUT_BIAS`,
`DESCRIPTION`, and add it to `REGISTRY`. The modules `scalpel.retokenize` (Model6, Model7) and
`scalpel.textbundle` (Model15) build their bundles from the files in `tokenizers/`. Model15 needs the checkpoint
at `<archive>/Model15/content/results/checkpoint-4760/model.safetensors`. One pass over the test set takes about
one hour on two CPU cores.
