@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=D:\anaconda\envs\imu_wfst\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ============================================================
echo RePressDB J1-R Low-Memory Paired Experiment v1.7.0
echo Python: %PYTHON_EXE%
echo Project: %CD%
echo ============================================================
echo.
echo Do not open another RePressDB experiment while this runs.
echo Completed cells are resumable.
echo.

"%PYTHON_EXE%" "%~dp0RUN_THIS_ONLY_REPRESSDB_J1R_LOW_MEMORY_PAIRED_V1_7_0.py"

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo ============================================================
echo Process exit code: %EXIT_CODE%
echo ============================================================
pause
exit /b %EXIT_CODE%
