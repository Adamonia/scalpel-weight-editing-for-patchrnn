# UNK experiment: train with an UNK token (K = 1), then compare with the first 50 runs.
# Finished runs are skipped, so this file can be started again after a stop.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$runs = Join-Path $env:USERPROFILE "scalpel_runs_v2"
$unkRuns = Join-Path $env:USERPROFILE "scalpel_runs_unk"

function Step {
    param([string[]]$Cmd)
    & python @Cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: python $($Cmd -join ' ')" }
}

Step @("-c", "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())")
Step @("train_unk.py", "--data", "dataset.npz", "--out", $unkRuns)
Step @("analyse_unk.py", "--data", "dataset.npz", "--runs", $runs, "--unk-runs", $unkRuns, "--out-dir", "out")
Write-Host "UNK steps finished. Send back out/unk_results.csv, out/unk_summary.json and the log.json files of scalpel_runs_unk."
