# Scalpel retraining kit

Trains 6 Transformer architectures x 5 seeds (30 runs), then measures per run how much a zero edit of
the untrained message-embedding rows helps: are architecture differences larger than seed differences?

Recipe read from the original notebooks (ModelNotebooks/Model3..13 of the project repository): Adam, lr 5e-4,
batch 256, no weight decay, no shuffling (add --shuffle), dropout 0.2 in the encoders and 0.5 in the classifier,
model kept at best validation ACCURACY, early stop when the accuracy is below the minimum of the last 10 epochs.
Maximum 20 epochs for all six architectures (the original M3 notebook used 50 epochs and a window of 20; the
kit uses one setting for all, so that architectures are compared fairly). Position and class vectors start at N(0,1).
--select-on val (default): 10 % of the training set is validation. --select-on test replicates the original
(test set used for selection, biased). Use the same setting for the analysis (it is read from log.json).

Loss: cross-entropy with class weight 1 - class share (ratio pos/neg = n_neg/n_pos, about 2.05, as in the original). Why: the
scalpel docs treat the output bias as carrying ln(20442/9991) = 0.7159 (they subtract it from the
security logit). This is the prior shift that a loss weight of n_neg/n_pos puts into a model. Weight
decay never touches embeddings, so untrained rows stay at their N(0,1) start (checked per run in log.json).
## Requirements
Python 3.9+, torch, numpy. Works on CPU, Windows, Linux. A CUDA GPU is used automatically.

## Commands (PowerShell, in the kit folder)
    python train_seeds.py --data dataset.npz --out runs
    python analyse_seeds.py --data dataset.npz --runs runs --out-dir .
## Linux
    python3 train_seeds.py --data dataset.npz --out runs
    python3 analyse_seeds.py --data dataset.npz --runs runs --out-dir .

Training can be stopped and started again: finished runs (runs/<arch>_seed<n>/log.json) are skipped.
Options: --archs M3,M4  --seeds 0-4  --epochs  --lr  --batch  --dropout  --weight-decay  --device auto|cuda|cpu
--shuffle  --select-on val|test  --micro-batch 16 (memory only; the batch of 256 is built by accumulation.
Use 8 if the GPU runs out of memory, 32 if it has plenty).
## Outputs
- runs/<arch>_seed<n>/model.pth (best validation-accuracy epoch, archive key format) and log.json
- results.csv: one row per run (F1/AUC, after zero edit, gain, noise-to-signal, spectral norm of V)
- summary.json: mean and sd per architecture, permutation test between architectures (10,000 permutations),
  share of variance between architectures, contrast {M3,M4,M13} vs {M5,M8,M12}
The test set is used only in analyse_seeds.py. Model choice uses the validation part (see above).

## Send back
results.csv, summary.json, and all runs/*/log.json (small). Do not send the model.pth files.
