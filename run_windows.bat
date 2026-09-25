@echo off
REM ============================================================================
REM  Hybrid optimizer study - Windows launcher
REM
REM  Double-click this file, or from a terminal:
REM      run_windows.bat --jobs 20
REM
REM  It only sets the environment and calls START.py, which does the real work:
REM  dependency check, tests, experiment, analysis.
REM
REM  Safe to interrupt. Run it again and it resumes where it stopped.
REM ============================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM -- 1. is Python reachable? -------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [X] "python" was not found on PATH.
    echo.
    echo     Install Python 3.11 or newer from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH" in the installer.
    echo.
    goto :fail
)

REM -- 2. are we in a checkout that actually contains the study? ---------------
REM The default branch holds only the original single-file script. Everything
REM this launcher needs lives on the development branch, so a plain "git pull"
REM leaves a checkout with no START.py in it.
if not exist "START.py" (
    echo.
    echo [X] START.py is not in this folder:
    echo     %CD%
    echo.
    echo     The study lives on the development branch, not on the default one.
    echo     Switch to it:
    echo.
    echo         git fetch origin
    echo         git checkout claude/adoring-archimedes-942q36
    echo         git pull
    echo.
    echo     Then run this file again.
    echo.
    goto :fail
)

REM -- 3. one BLAS thread per worker ------------------------------------------
REM Without this, NumPy's internal threads fight the process pool and the run
REM gets SLOWER as the worker count increases.
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set PYTHONIOENCODING=utf-8

python START.py %*
if errorlevel 1 goto :fail

echo.
echo Done. Results are in results_cec2017\ and reports\.
pause
exit /b 0

:fail
echo.
echo FAILED. Nothing was lost - fix the problem above and run this file again.
pause
exit /b 1
