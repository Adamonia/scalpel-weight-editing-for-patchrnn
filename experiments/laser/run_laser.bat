@echo off
rem LASER sweep on all 12 archive models, then the report.
rem Needs: the model archive and the four bundle folders (see README.md).
rem Finished models are skipped, so this file can be started again after a stop.
cd /d "%~dp0"
if not defined SCALPEL_ARCHIVE set "SCALPEL_ARCHIVE=%~dp0..\..\..\model-archive"
set "BUNDLES=%~dp0..\.."

python -c "import sys, torch, transformers, safetensors; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 goto :fail

python laser_sweep.py --archive "%SCALPEL_ARCHIVE%" --bundles "%BUNDLES%"
if errorlevel 1 goto :fail

python laser_report.py --bundles "%BUNDLES%"
if errorlevel 1 goto :fail

echo.
echo Finished. Send back out\REPORT.md, out\summary.json, out\laser_rows.csv and out\log.txt.
exit /b 0

:fail
echo.
echo STOPPED: a step failed (see the message above).
exit /b 1
