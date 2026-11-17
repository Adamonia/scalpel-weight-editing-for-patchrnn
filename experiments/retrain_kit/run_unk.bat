@echo off
rem UNK experiment: train with an UNK token (K = 1), then compare with the first 50 runs.
rem Needs the first 50 runs in scalpel_runs_v2 and out\results.csv, out\round3_results.csv (or the kit folder copies).
rem Finished runs are skipped, so this file can be started again after a stop.
cd /d "%~dp0"
set "RUNS=%USERPROFILE%\scalpel_runs_v2"
set "UNKRUNS=%USERPROFILE%\scalpel_runs_unk"

python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 goto :fail

python train_unk.py --data dataset.npz --out "%UNKRUNS%"
if errorlevel 1 goto :fail

python analyse_unk.py --data dataset.npz --runs "%RUNS%" --unk-runs "%UNKRUNS%" --out-dir out
if errorlevel 1 goto :fail

echo.
echo UNK steps finished. Send back out\unk_results.csv, out\unk_summary.json and the log.json files of scalpel_runs_unk.
exit /b 0

:fail
echo.
echo STOPPED: a step failed (see the message above).
exit /b 1
