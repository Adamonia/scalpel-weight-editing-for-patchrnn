# Scalpel: direct weight editing of security-patch classifiers by an LLM agent

Scalpel lets Claude find the parts of a trained network that do the task incorrectly and correct them
with direct edits of the weights. There is no gradient training. The agent reads diagnostics, writes
values into weight tensors, and each edit must pass a guard before it is kept.

The tool works on the models of the repository `Security-Patch-Identification-Using-Deep-Learning`
(`model-archive`): Model0 (LSTM, PatchRNN), Model2 (Conv2D-E), Model3/4/5/8/12/13 (Transformer encoders).

## Data protocol

The original test set (7,608 samples, split seed 10) is divided one time with a fixed seed:

| Part | Samples | Use |
|---|---|---|
| diag | 2,282 | The agent reads samples, labels and model outputs. |
| guard | 1,522 | Aggregate metrics only. A commit is refused if the MCC of diag or guard decreases. |
| holdout | 3,804 | Sealed. `finalize` reads it one time and locks the session. |

## Install

```
pip install torch numpy nltk anthropic
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

`call` is the interface for an external agent (for example Claude Code) or for a person.

Autonomous loop on the Anthropic API (needs `ANTHROPIC_API_KEY`):

```
python -m scalpel.cli agent --run runs/M3 --llm claude-opus-5-5 --max-turns 60
```

The loop and `call` use the same dispatcher (`scalpel/tools.py`). The loop is tested with a scripted
client (`tests/test_agent_loop.py`). It was not run against the live API in the study.

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
`DESCRIPTION`, and add it to `REGISTRY`. Model6, Model7 and Model15 need the BERT, RoBERTa and CodeBERT
tokenizer files and are not in the registry.
