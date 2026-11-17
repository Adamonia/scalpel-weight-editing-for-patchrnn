@echo off
rem One command for the whole LASER study. The folder is the one that holds scalpel\ and experiments\
rem (for example ...\Downloads\patchrnn-archive\llm-weight-editing). The model archive is looked up next to it
rem (...\patchrnn-archive\model-archive). Start it from that folder:
rem     experiments\laser\run_laser.bat
rem Steps: packages, bundles (from laser_bundles.zip), model archive, sweep, report, commit and push of the results.
rem Finished models are skipped, so this file can be started again after a stop.
setlocal
cd /d "%~dp0"
set "REPO=%~dp0..\.."

echo [1/6] Python packages
python -c "import torch, transformers, safetensors, numpy" 2>nul || python -m pip install -q transformers safetensors numpy
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 goto :fail

echo [2/6] Data bundles
for %%B in (bundle bundle_bert bundle_roberta bundle_codebert) do (
    if not exist "%REPO%\%%B\bundle.npz" tar -xf laser_bundles.zip -C "%REPO%" || goto :fail
)

echo [3/6] Model archive
if defined SCALPEL_ARCHIVE goto :have_archive
for %%A in ("%REPO%\..\model-archive" "%REPO%\model-archive" "%REPO%\..\patchrnn-archive\model-archive" "%USERPROFILE%\model-archive" "%USERPROFILE%\patchrnn-archive\model-archive" "%USERPROFILE%\Downloads\patchrnn-archive\model-archive" "%USERPROFILE%\Desktop\patchrnn-archive\model-archive") do (
    if exist "%%~A\Model15" if not defined SCALPEL_ARCHIVE set "SCALPEL_ARCHIVE=%%~A"
)
:have_archive
if not exist "%SCALPEL_ARCHIVE%\Model15" (
    echo Model archive not found. Run again with:  set SCALPEL_ARCHIVE=C:\path\to\model-archive ^&^& experiments\laser\run_laser.bat
    goto :fail
)
echo archive: %SCALPEL_ARCHIVE%

echo [4/6] Sweep (about 30 min on one RTX 4090)
python laser_sweep.py --archive "%SCALPEL_ARCHIVE%" --bundles "%REPO%"
if errorlevel 1 (
    echo Second try with smaller batches ^(out of GPU memory?^)
    python laser_sweep.py --archive "%SCALPEL_ARCHIVE%" --bundles "%REPO%" --batch 16 --batch-bert 8
    if errorlevel 1 goto :fail
)

echo [5/6] Report
python laser_report.py --bundles "%REPO%"
if errorlevel 1 goto :fail

echo [6/6] Commit and push of the results
if not exist "%REPO%\.git" (
    echo This folder is not a git clone. Send back out\REPORT.md, out\summary.json, out\laser_rows.csv and out\log.txt.
    goto :done
)
git add out\REPORT.md out\summary.json out\laser_rows.csv out\log.txt
git commit -q -m "Add LASER sweep results"
git push -q || echo Push failed: send out\REPORT.md, out\summary.json, out\laser_rows.csv and out\log.txt by hand.
:done

echo.
echo Finished. Results: experiments\laser\out\REPORT.md
exit /b 0

:fail
echo.
echo STOPPED: a step failed (see the message above).
exit /b 1
