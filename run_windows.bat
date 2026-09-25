@echo off
REM ============================================================================
REM  TEMOA benchmark study - Windows launcher
REM
REM  This file only sets the environment and calls START.py, which does the
REM  actual work: dependency check, tests, experiment, analysis.
REM
REM  Safe to interrupt. Run it again and it resumes where it stopped.
REM ============================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM One BLAS thread per worker. Without this, NumPy's internal threads fight the
REM process pool and the run gets SLOWER as the worker count increases.
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set PYTHONIOENCODING=utf-8

python START.py %*
if errorlevel 1 (
    echo.
    echo FAILED. Nothing was lost - run this file again to resume.
    exit /b 1
)
