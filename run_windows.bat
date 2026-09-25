@echo off
REM ============================================================================
REM  TEMOA benchmark study - Windows launcher
REM  Tested target: Windows x64, i7-13700F (16 cores / 24 threads), 32 GB RAM.
REM  Expected wall-clock for the full protocol on that machine: about 2 hours.
REM ============================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM One BLAS thread per worker. Without this, NumPy's internal threads fight the
REM process pool and the run gets SLOWER as --jobs increases.
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set PYTHONIOENCODING=utf-8

echo [1/5] Installing dependencies...
python -m pip install -q -r requirements.txt || goto :fail

echo [2/5] Running the test suite...
python tests\test_all.py || goto :fail
echo       ...and checking the port matches the original (a few minutes)...
python tests\test_port_fidelity.py || goto :fail

echo [3/5] Smoke test (about 3 minutes)...
python run_all.py --smoke --jobs 20 || goto :fail

echo [4/5] Main study: 30D, 50D, 100D, 30 runs (about 2 hours)...
echo      Interrupted? Re-run this file - it resumes automatically.
python run_all.py --dims 30 50 100 --runs 30 --jobs 20 --out results --resume || goto :fail

echo [5/5] Analysis...
python analyze.py --out results --control TEMOA_V11 || goto :fail
python analyze.py --out results --control TEMOA_V10 || goto :fail

echo.
echo DONE. Tables: results\tables   Figures: results\figures
echo Optional, about 30 minutes more:
echo    python experiments\ablation.py    --dim 30 --runs 15 --jobs 20
echo    python experiments\sensitivity.py --dim 30 --runs 15 --jobs 20
goto :eof

:fail
echo.
echo FAILED with error %errorlevel%. Nothing was lost - re-run to resume.
exit /b %errorlevel%
