# Run all trainings, then the analysis.
# The weights go to a local folder, not to OneDrive (about 2 GB).
$out = Join-Path $env:USERPROFILE "scalpel_runs_v2"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python train_seeds.py --data dataset.npz --out $out
python analyse_seeds.py --data dataset.npz --runs $out --out-dir .
