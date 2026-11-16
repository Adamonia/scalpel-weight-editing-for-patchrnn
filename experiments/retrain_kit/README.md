# Scalpel retraining kit

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

## Send back (in the kit folder, small files)
results.csv, summary.json, mechanism.csv, mechanism_summary.json, round3_results.csv, round3_summary.json
Also the log.json files of all runs (small). Do not send model.pth.

## What the outputs hold
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
--select-on val|test --micro-batch 16 (memory only; 8 if the GPU runs out of memory).
mechanism_seeds.py: --batch 32 (samples per pass; 4 on a small CPU) --limit-runs N --device.
analyse_seeds.py --from-csv results.csv recomputes summary.json from a results.csv (no runs needed).
Recipe: Adam, lr 5e-4, batch 256, no weight decay, no shuffling, dropout 0.2 / 0.5, best validation accuracy, early stop rule of
the original code, max 20 epochs for all. --select-on val (default) uses 10 % of train; the test set is used only for analysis.
