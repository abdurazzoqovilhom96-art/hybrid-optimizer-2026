@echo off
REM ============================================================================
REM  BOSHLA.bat - bitta fayl, bitta bosish.
REM
REM  Bu faylni istalgan papkaga saqlang va ikki marta bosing. U o'zi:
REM    1. git va python borligini tekshiradi
REM    2. loyihani to'g'ri branchdan yuklaydi (yoki mavjudini yangilaydi)
REM    3. kutubxonalarni o'rnatadi, 72 ta testni o'tkazadi
REM    4. eksperimentni boshlaydi
REM
REM  Ctrl+C bilan to'xtatsangiz hech narsa yo'qolmaydi - shu faylni qayta bosing.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set REPO=https://github.com/abdurazzoqovilhom96-art/hybrid-optimizer-2026.git
set BRANCH=claude/adoring-archimedes-942q36
set FOLDER=hybrid-optimizer-2026

echo ==============================================================================
echo   Gibrid optimizator - eksperimentni boshlash
echo ==============================================================================
echo.

REM -- git ---------------------------------------------------------------------
where git >nul 2>nul
if errorlevel 1 (
    echo [X] "git" PATH da topilmadi.
    echo     https://git-scm.com/download/win dan o'rnating va shu faylni qayta bosing.
    goto :fail
)
echo [ok] git topildi

REM -- python ------------------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [X] "python" PATH da topilmadi.
    echo     https://www.python.org/downloads/ dan o'rnating.
    echo     O'rnatishda "Add python.exe to PATH" katagini BELGILANG.
    goto :fail
)
echo [ok] python topildi

REM -- loyiha ------------------------------------------------------------------
if exist "%FOLDER%\.git" (
    echo [*] mavjud nusxa yangilanmoqda ...
    cd /d "%FOLDER%"
    git fetch origin
    if errorlevel 1 goto :gitfail
    git checkout %BRANCH%
    if errorlevel 1 goto :gitfail
    git pull origin %BRANCH%
    if errorlevel 1 goto :gitfail
) else (
    echo [*] loyiha yuklanmoqda ^(bir marta, bir necha soniya^) ...
    git clone -b %BRANCH% "%REPO%" "%FOLDER%"
    if errorlevel 1 goto :gitfail
    cd /d "%FOLDER%"
)
echo [ok] loyiha tayyor: %CD%
echo.

if not exist "START.py" (
    echo [X] START.py topilmadi. Branch noto'g'ri yuklangan.
    goto :fail
)

REM -- bitta BLAS tred ---------------------------------------------------------
REM Busiz NumPy ning ichki tredlari process pool bilan urishadi va ishchi soni
REM ko'paygani sari run SEKINLASHADI.
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set PYTHONIOENCODING=utf-8

REM -- ishchi soni: mantiqiy yadrolarning ~80%% -------------------------------
set JOBS=%1
if "%JOBS%"=="" set JOBS=20

echo [*] %JOBS% ta ishchi bilan boshlanmoqda.
echo     Bosqichlar: environment -^> 72 ta test -^> eksperiment -^> tahlil
echo.

python START.py --jobs %JOBS% --yes
if errorlevel 1 goto :fail

echo.
echo ==============================================================================
echo   TUGADI. Natija: %CD%\results_cec2017\  va  %CD%\reports\
echo ==============================================================================
pause
exit /b 0

:gitfail
echo.
echo [X] git amali bajarilmadi. Yuqoridagi xabarni o'qing.
echo     Ko'p uchraydigan sabab: GitHub login talab qilinmoqda yoki internet yo'q.
goto :fail

:fail
echo.
echo XATOLIK. Hech narsa yo'qolmadi - sababni tuzatib shu faylni qayta bosing.
pause
exit /b 1
