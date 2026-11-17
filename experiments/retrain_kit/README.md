# retrain_kit (Scalpel retraining kit)

Trains word-level TwinTransformer classifiers (10 architectures x 5 seeds = 50 runs), then measures per run how much a
zero edit of the untrained message-embedding rows helps, why (mechanism), and what the round-3 recipe does.

## Architectures (heads x layers, the grid {1,2,4,8,16} x {1,2})
- one layer:  M3 (1 head), M4 (2), H4L1 (4), M5 (8), M8 (16)
- two layers: M13 (1 head), H2L2 (2), H4L2 (4), M12 (8), H16L2 (16)
A second layer is built the same way for all (nn.TransformerEncoder with 2 layers, one final LayerNorm).
Runs of M3, M4, M5, M8, M12, M13 seeds 0-4 that finished earlier are skipped; only the missing ones are trained.

## One command (Windows cmd, in the kit folder)
    run_all.bat
(PowerShell: .\run_all.ps1.) Order: version print, train_seeds.py, analyse_seeds.py, mechanism_seeds.py, apply_round3.py.
Weights go to %USERPROFILE%\scalpel_runs_v2. A step that fails stops the script. Start it again to resume.
Only the 4 new architectures x 5 seeds need training (20 runs, about 17 s per epoch, up to 20 epochs).
Linux: python3 train_seeds.py --data dataset.npz --out runs, then the same steps with --runs runs --out-dir .

## Files in this folder
- `train_seeds.py`: trains the runs. `analyse_seeds.py`: zero edit, noise-to-signal, trend tests. `mechanism_seeds.py`: attention interventions and analytic prediction. `apply_round3.py`: the round 3 recipe on every run.
- `model.py`, `common.py`: model class and shared helpers. `recipe.json`: the round 3 recipe (E3, k = 3, no recenter).
- `run_all.bat`, `run_all.ps1`: run all steps in order on Windows.
- `out/`: results of the 50 runs (see below). `out/six_architectures/`: results of the first 30 runs (6 architectures x 5 seeds), kept for comparison.
- The dataset file `dataset.npz` and the model weights are not in the repository.

## Files to copy back after a run (small files, in the kit folder)
results.csv, summary.json, mechanism.csv, mechanism_summary.json, round3_results.csv, round3_summary.json
Also the log.json files of all runs (small). Do not copy model.pth.

## What the outputs hold (copies are in `out/`)
- results.csv: one row per run (heads, layers, F1/AUC, zero edit, gain, noise-to-signal nsr, spectral norm of V).
- summary.json: per architecture mean and sd, permutation F test between architectures, and TREND for gain_f1, nsr, f1
  (run-level fit on log2(heads) and layers; architecture-level permutation test of the heads slope inside each layer
  stratum; exact sign test of the layer effect). The old contrast {M3,M4,M13} vs {M5,M8,M12} is kept only if all six old runs exist.
- mechanism.csv: one row per run and setting. The row temp_1 is the unchanged model and holds the analytic prediction and mediators.
  Settings: attention temperature 0.5/1/2/4, head ablation (with and without rescale by H/k), uniform attention. They act on
  layer 0 and are OUTSIDE what the models were trained on. For two-layer models the analytic part and mediators use layer 0 only.
- mechanism_summary.json: prediction vs measurement, mediator correlations, interventions per architecture.
- round3_*.csv/json: recipe of recipe.json applied to every run.

## Options
train_seeds.py: --archs M3,H4L1 --seeds 0-4 --epochs --lr --batch --dropout --weight-decay --device auto|cuda|cpu --shuffle
--select-on val|test --micro-batch 16 (memory only; 8 if the GPU runs out of memory) --amp (mixed precision on cuda).
mechanism_seeds.py: --batch 32 (samples per pass; 4 on a small CPU) --limit-runs N --device.
analyse_seeds.py --from-csv results.csv recomputes summary.json from a results.csv (no runs needed).
Recipe: Adam, lr 5e-4, batch 256, no weight decay, no shuffling, dropout 0.2 / 0.5, best validation accuracy, early stop rule of
the original code, max 20 epochs for all. --select-on val (default) uses 10 % of train; the test set is used only for analysis.

## UNK experiment (does training with an UNK token give the same as the zero edit?)
What it tests: the 50 runs are trained again with one change. Every message token with training count c <= 1 is
mapped to one new UNK id (train, validation and test). There is no edit. Then F1(UNK) is compared with F1(base + zero edit),
F1(base) and F1(base + round 3), per (architecture, seed) pair. The rule is in `unk_rule.md` (written before any result).
- c = number of fit rows (training rows without the validation rows) that contain the token in the model window (`token_counts` of apply_round3.py).
- UNK id = vocabulary size (80831 for this data): one new row, N(0,1) init, added after the model is built.
  So all other initial weights and the dropout numbers are the same as in the base run of the same seed. Pad id 0 and the code input do not change.
- Same split, epochs, validation selection and hyperparameters. fp16 is off. The training order of the rows is the same.
How to run (kit folder, needs dataset.npz, the first 50 runs in scalpel_runs_v2, and out\round3_results.csv or the file in the kit folder):
    run_unk.bat          (PowerShell: .\run_unk.ps1)
Weights go to %USERPROFILE%\scalpel_runs_unk. scalpel_runs_v2 is never written. Finished runs are skipped, so start it again after a stop.
Run time: 50 runs, the same time per run as the first 50 runs. The analysis takes a few minutes.
Output: out\unk_results.csv (one row per pair) and out\unk_summary.json (primary test, secondary tests, slope over log2(heads)).
Send back these two files and the log.json files of scalpel_runs_unk.
Single steps: `python train_unk.py --data dataset.npz` (same as `train_seeds.py --unk 1 --out ...`), then `python analyse_unk.py --data dataset.npz --runs RUNS --unk-runs UNKRUNS --out-dir out`.

Result (50 pairs, out/unk_summary.json): F1 UNK minus zero edit +0.40 [0.16, 0.63]; UNK minus round 3 +0.07 [-0.20, 0.35]; UNK minus base +1.23 [0.96, 1.50].
