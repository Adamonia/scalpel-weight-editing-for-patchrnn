@echo off
rem Deeper edits of CodeBERT (M15): deep LASER, pruning, interpolation with microsoft/codebert-base (WiSE-FT).
rem Start it in the folder that holds scalpel\ and bundle_codebert\ (for example laser_kit):
rem     experiments\laser\run_codebert_deep.bat
rem The base model (about 500 MB) is downloaded from Hugging Face on the first run. Needs internet.
setlocal
cd /d "%~dp0"
set "REPO=%~dp0..\.."
if not defined SCALPEL_ARCHIVE set "SCALPEL_ARCHIVE=%REPO%\..\model-archive"
if not exist "%SCALPEL_ARCHIVE%\Model15" (
    echo Model archive not found. Run again with:  set SCALPEL_ARCHIVE=C:\path\to\model-archive ^&^& experiments\laser\run_codebert_deep.bat
    exit /b 1
)
python codebert_deep.py --archive "%SCALPEL_ARCHIVE%" --bundle "%REPO%\bundle_codebert"
if errorlevel 1 (
    echo STOPPED: see the message above. Out of GPU memory? Try: python codebert_deep.py --batch 16
    exit /b 1
)
echo.
echo Finished. Results: experiments\laser\out_codebert\REPORT.md
exit /b 0
