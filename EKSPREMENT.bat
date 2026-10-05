@echo off
REM ============================================================================
REM  EKSPREMENT.bat - to'liq tadqiqotni boshdan oxirigacha o'tkazadi.
REM
REM  OLDIN TEKSHIR.bat ni bosing.
REM
REM  To'qqiz bosqich, har biri keyingisining darvozasi:
REM    1 preflight  muhit (Python 3.14, opfunu 1.0.4) - mos kelmasa to'xtaydi
REM    2 test       85 ta test - algoritm o'zgargan bo'lsa to'xtaydi
REM    3 tune       L-SHADE-GB skalyarlari, F21-F30 da (keyingi bosqichlar
REM                 o'lchamaydigan funksiyalar - held-out)
REM    4 screen     armlar D=50 da
REM    5 select     arm qidiruvining o'z javobi
REM    6 gate       butun saf, 29 funksiya, D=10/30/50, 51 run
REM    7 analyze    ikki tahlil: reference saf va nomzod qo'shilgani
REM    8 aos        operator tanlash korrelyatsiyasi
REM    9 report     majburiy mezon -> RESULT.md, qaysi tomonga tushsa ham
REM
REM  UZILSA HECH NARSA YO'QOLMAYDI. Ctrl+C, tok o'chishi, Windows qayta ishga
REM  tushishi - shu faylni qayta bosing, to'xtagan joyidan davom etadi.
REM  Hisoblangan birorta qator qayta hisoblanmaydi.
REM
REM  BOSHLASHDAN OLDIN (bir marta, administrator cmd'da):
REM      powercfg /change standby-timeout-ac 0
REM      powercfg /change hibernate-timeout-ac 0
REM  va Windows Update uchun "Active hours" ni kengaytiring.
REM  Bu oynani YOPMANG.
REM
REM  Boshqa yadro soni kerak bo'lsa:  EKSPREMENT.bat 16
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

if not exist "study_2026.py" goto :nofile

set PY=
py -3.14 -c "import sys" >nul 2>nul
if not errorlevel 1 set PY=py -3.14
if defined PY goto :havepy
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 set PY=py -3
if defined PY goto :havepy
where python >nul 2>nul
if not errorlevel 1 set PY=python
if defined PY goto :havepy
echo [X] Python topilmadi. TEKSHIR.bat ni bosing.
goto :end

:havepy
set /a JOBS=%NUMBER_OF_PROCESSORS%-2
if %JOBS% LSS 1 set JOBS=1
if not "%~1"=="" set JOBS=%~1
if not exist "logs" mkdir "logs"
set LOG=logs\eksprement.log

echo ==============================================================================
echo   EKSPERIMENT - L-SHADE-GB, CEC'2017, D=10/30/50
echo   %NUMBER_OF_PROCESSORS% yadro, --jobs %JOBS%
echo   Boshlandi: %DATE% %TIME%
echo   Jurnal:    %LOG%
echo ==============================================================================
echo.
echo   Uzilsa shu faylni qayta bosing - to'xtagan joyidan davom etadi.
echo.

REM -u: chiqish quvurga yozilganda Python buferlab qo'ymasin, bosqich
REM nomlari darhol ko'rinsin. Quvurning ikkinchi tomoni ekranga ham,
REM jurnalga ham yozadi, chunki cmd'da tee yo'q.
%PY% -u study_2026.py experiment --jobs %JOBS% 2>&1 | %PY% -c "import sys;f=open(sys.argv[1],'a',encoding='utf-8',errors='replace');w=sys.stdout.write;[(w(l),sys.stdout.flush(),f.write(l),f.flush()) for l in iter(sys.stdin.readline,'')];f.close()" "%LOG%"

echo.
echo ==============================================================================
echo   Tugadi: %DATE% %TIME%
echo   Natija:   results_pipeline\RESULT.md
echo   Jadvallar: results_cec2017\tables_k8\   (reference: tables_k7\)
echo   Jurnal:   %LOG%
echo ==============================================================================
goto :end

:nofile
echo [X] study_2026.py shu papkada yo'q: %CD%

:end
echo.
pause
endlocal
