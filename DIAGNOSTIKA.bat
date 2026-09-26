@echo off
REM ============================================================================
REM  DIAGNOSTIKA.bat - F4/F5 sababini aniqlash
REM
REM  Ikki marta bosing. Skript o'zi loyihani yangilaydi va diagnostikani
REM  ishga tushiradi.
REM
REM  O'lchangan narx: 14 konfiguratsiya x 10 funksiya x 31 yurish = 4340 yurish,
REM  har biri 100 000 baholash. 20 ishchida ~15-25 daqiqa.
REM
REM  Ctrl+C bilan to'xtatsa bo'ladi. Natija results_diagnose\ papkasida.
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set BRANCH=claude/adoring-archimedes-942q36

echo ==============================================================================
echo   F4/F5 diagnostikasi
echo ==============================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [X] "python" PATH da topilmadi.
    echo     https://www.python.org/downloads/ dan o'rnating, "Add python.exe to PATH" ni belgilang.
    goto :fail
)

if not exist "experiments\diagnose.py" (
    echo [X] experiments\diagnose.py topilmadi. Papka yoki branch noto'g'ri:
    echo     %CD%
    echo.
    echo     git fetch origin
    echo     git checkout %BRANCH%
    echo     git pull
    goto :fail
)

REM -- loyihani yangilash (internet bo'lmasa ham davom etadi) -------------------
where git >nul 2>nul
if not errorlevel 1 (
    echo [*] loyiha yangilanmoqda ...
    git fetch origin >nul 2>nul
    git checkout %BRANCH% >nul 2>nul
    git pull origin %BRANCH%
    echo.
)

REM -- bitta BLAS tred: busiz ishchi soni ko'paygani sari run SEKINLASHADI ------
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set PYTHONIOENCODING=utf-8

REM -- ishchi soni: mantiqiy yadrolarning ~85 foizi ----------------------------
set /a JOBS=%NUMBER_OF_PROCESSORS% * 85 / 100
if %JOBS% LSS 1 set JOBS=1
if not "%~1"=="" set JOBS=%~1

echo [*] %NUMBER_OF_PROCESSORS% mantiqiy yadro topildi, %JOBS% ta ishchi ishlatiladi.
echo [*] 4340 yurish x 100 000 baholash. Sabr qiling.
echo.

python experiments\diagnose.py --runs 31 --jobs %JOBS%
if errorlevel 1 goto :fail

echo.
echo ==============================================================================
echo   TUGADI. Natija: %CD%\results_diagnose\tables\
echo.
echo   Shu papkani menga yuboring:
echo     - success_rate_by_config.csv   ^<- eng muhimi
echo     - median_by_config.csv
echo     - effect_vs_baseline.csv
echo     - internals.csv
echo ==============================================================================
pause
exit /b 0

:fail
echo.
echo XATOLIK. Yuqoridagi xabarni o'qing va menga yuboring.
pause
exit /b 1
