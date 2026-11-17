# laser (low-rank copy of one weight matrix)

What it tests: can a direct weight edit other than the zero edit improve the models, also the BERT-based ones
(M6, M7, M15)? The edit is LASER (Sharma et al., 2023, "The Truth is in There"): one weight matrix W is replaced
by its best rank-r copy from the SVD, r = round(rho x max rank). Nothing is trained.

Sweep: all 12 archive models, every weight matrix (embedding tables, attention, feed-forward, LSTM, CNN filters,
head; not the 2-row output layers, not LayerNorm or position vectors), rho in {0.01, 0.05, 0.1, 0.25, 0.5}.
CodeBERT (M15): only the top 4 encoder layers (feed-forward and attention output), the word embeddings and the head.
214 matrices, 1,070 edited models. Only the stages that use the edited matrix are computed again.

Choice (in `laser_report.py`): on diag only. Per model: the largest diag F1 gain with diag MCC not falling, used
only if guard F1 and guard MCC do not fall. One recipe: (type, branch, layer, rho) with the largest mean diag F1 gain
over at least 6 models. Holdout is reported, never used for a choice. This study is exploratory.

## Run (Windows cmd, in the repository root)
    git pull && experiments\laser\run_laser.bat
The script installs `transformers` and `safetensors` if they are missing, unpacks `laser_bundles.zip` (the four data
bundles) into the repository root, looks for the model archive (`SCALPEL_ARCHIVE`, else `..\model-archive` and a few
usual folders), runs the sweep and the report, then commits and pushes the small result files.
Run time on one RTX 4090: about 30 minutes (most of it CodeBERT). On a CPU: many hours.
If it stops, start the same line again: finished models are skipped. Out of GPU memory: it tries again with smaller batches.
Linux: `python3 laser_sweep.py --archive ../../../model-archive --bundles ../.. && python3 laser_report.py --bundles ../..`.
Smoke test (minutes on a CPU): `python laser_sweep.py --models 3 --limit-rows 300 --max-params 2 --out out_test`.

## Output (`out/`)
- `laser_rows.csv`: one row per (model, matrix, rho) and one row `original` per model; F1 (%) and MCC on diag, guard, holdout.
- `REPORT.md`, `summary.json`: per-model choice and the top recipes, with the holdout F1 change and its 95 % interval.
- `margins_M<n>.npz`: margins of all 7,608 samples for every row (float16, about 1 to 4 MB each; not needed in the repository).
- `log.txt`: time per model.
Check: the `original` rows must give the holdout F1 of the round tables (for example M3 72.67, MCC 0.589).
