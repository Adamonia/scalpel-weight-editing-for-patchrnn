# Run everything, in order. Weights go to a local folder (not OneDrive, about 3 GB).
# Finished runs are skipped, so this file can be started again after a stop.
# Every step stops the script when it fails.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$runs = Join-Path $env:USERPROFILE "scalpel_runs_v2"

function Step {
    param([string[]]$Cmd)
    & python @Cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: python $($Cmd -join ' ')" }
}

Step @("-c", "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())")
Step @("train_seeds.py", "--data", "dataset.npz", "--out", $runs)
Step @("analyse_seeds.py", "--data", "dataset.npz", "--runs", $runs, "--out-dir", ".")
Step @("mechanism_seeds.py", "--data", "dataset.npz", "--runs", $runs, "--results", "results.csv", "--out-dir", ".")
Step @("apply_round3.py", "--data", "dataset.npz", "--runs", $runs, "--out-dir", ".")
Write-Host "All steps finished. Send back the six files listed in README.md."
