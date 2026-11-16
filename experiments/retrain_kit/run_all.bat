@echo off
rem Run everything, in order. Weights go to a local folder (not OneDrive, about 3 GB).
rem Finished runs are skipped, so this file can be started again after a stop.
rem Every step stops the script when it fails.
cd /d "%~dp0"
set "RUNS=%USERPROFILE%\scalpel_runs_v2"

python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 goto :fail

python train_seeds.py --data dataset.npz --out "%RUNS%"
if errorlevel 1 goto :fail

python analyse_seeds.py --data dataset.npz --runs "%RUNS%" --out-dir .
if errorlevel 1 goto :fail

python mechanism_seeds.py --data dataset.npz --runs "%RUNS%" --results results.csv --out-dir .
if errorlevel 1 goto :fail

python apply_round3.py --data dataset.npz --runs "%RUNS%" --out-dir .
if errorlevel 1 goto :fail

echo.
echo All steps finished. Send back the six files listed in README.md.
exit /b 0

:fail
echo.
echo STOPPED: a step failed (see the message above).
exit /b 1
