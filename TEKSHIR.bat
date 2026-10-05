@echo off
REM ============================================================================
REM  TEKSHIR.bat - eksperimentdan OLDIN bir marta bosiladi.
REM
REM  Uzoq hisob-kitob qilmaydi. To'rtta savolga javob beradi:
REM    1. Qaysi Python? Yozilgan qatorlar 3.14 da o'lchangan, preflight bosqichi
REM       boshqa versiyani ataylab rad etadi - bu xato emas, natijaning qismi.
REM    2. Kutubxonalar bormi? opfunu 1.0.4 shart: 1.0.1 CEC'2017 ning 29
REM       funksiyasidan 13 tasini boshqacha aniqlaydi.
REM    3. Bu papkada qanday o'lchangan natijalar bor? Gate shularning ustiga
REM       davom etadi, shuning uchun bu raqam eksperiment narxini belgilaydi.
REM    4. 85 ta test o'tadimi? (~2 daqiqa)
REM
REM  Oxirida ekrandagi hammasini nusxalab Claude'ga tashlang.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================================================
echo   TEKSHIRUV - eksperiment muhiti
echo ==============================================================================
echo.

if not exist "study_2026.py" goto :nofile

REM -- 1. Python --------------------------------------------------------------
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
goto :nopy

:havepy
echo [*] Python buyrug'i: %PY%
%PY% -c "import sys;print('    versiya:', sys.version.split()[0])"
echo.

REM -- 2. Kutubxonalar -------------------------------------------------------
echo [*] Kutubxonalar:
%PY% -c "import importlib;[print('   ',m.ljust(12),getattr(importlib.import_module(m),'__version__','?')) for m in ('numpy','scipy','pandas','joblib','matplotlib','opfunu','cma')]"
if errorlevel 1 echo     [!] Biri yo'q. O'rnatish:  %PY% -m pip install -r requirements.txt
echo.

REM -- 3. Yadrolar -----------------------------------------------------------
set /a JOBS=%NUMBER_OF_PROCESSORS%-2
if %JOBS% LSS 1 set JOBS=1
echo [*] Protsessor: %NUMBER_OF_PROCESSORS% ta mantiqiy yadro
echo     eksperiment --jobs %JOBS% bilan ishlaydi, 2 tasi tizimga qoladi.
echo     --jobs hech qanday chop etilgan raqamni o'zgartirmaydi, faqat vaqtni:
echo     urug'lar algoritm nomidan olinadi, ishchilar sonidan emas.
echo.

REM -- 4. Mavjud o'lchangan natijalar ---------------------------------------
echo [*] Mavjud o'lchangan qatorlar:
%PY% -c "import glob;f=sorted(glob.glob('results*/raw/results.csv'));print('    (topilmadi - butun saf noldan olchanadi)') if not f else [print('   ',p,'->',sum(1 for _ in open(p,encoding='utf-8'))-1,'qator') for p in f]"
echo.
echo [*] Ular qanday muhitda o'lchangan:
%PY% -c "import glob,json;f=sorted(glob.glob('results*/manifest.json'));print('    (manifest yoq)') if not f else [print('   ',p,'-> python',json.load(open(p,encoding='utf-8')).get('python'),' opfunu',json.load(open(p,encoding='utf-8')).get('opfunu'),' runs',json.load(open(p,encoding='utf-8')).get('runs')) for p in f]"
echo.

echo [*] O'lchangan narx - sizning o'z Seconds ustunidan hisoblanadi,
echo     meros qolgan taxmin emas:
%PY% -c "import glob,pandas as pd;f=sorted(glob.glob('results*/raw/results.csv'));d=pd.concat([pd.read_csv(p,usecols=['Dimension','Seconds']) for p in f]) if f else None;g=None if d is None else d.groupby('Dimension')['Seconds'].agg(['count','mean']);print('    (natija yoq - narx hali olchanmagan)') if g is None else [print('    D=%-4d n=%-6d ortacha %7.1f s/run  ->  29x51 run = %6.1f CPU-soat' % (i,g.loc[i,'count'],g.loc[i,'mean'],29*51*g.loc[i,'mean']/3600)) for i in g.index] and print('    JAMI bitta algoritm uchun shu olchamlarda: %.1f CPU-soat' % (sum(29*51*g.loc[i,'mean'] for i in g.index)/3600))"
echo.

REM -- 5. Kod versiyasi ------------------------------------------------------
where git >nul 2>nul
if errorlevel 1 goto :notests
echo [*] Kod versiyasi:
git log --oneline -1
git status --short
echo.

:notests
echo ==============================================================================
echo   TESTLAR - 85 ta, taxminan 2 daqiqa.
echo   Bittasi ham yiqilsa eksperiment boshlanmaydi: yozilgan algoritm
echo   o'zgarmaganligi shu yerda pinlanadi.
echo ==============================================================================
echo.
%PY% -u study_2026.py test --full
if errorlevel 1 goto :testfail

echo.
echo ==============================================================================
echo   TEKSHIRUV TUGADI.
echo   Yuqoridagi hammasini nusxalab Claude'ga tashlang, keyin EKSPREMENT.bat.
echo ==============================================================================
goto :end

:nofile
echo [X] study_2026.py shu papkada yo'q:
echo     %CD%
echo     Bu fayl study_2026.py bilan bitta papkada turishi kerak.
goto :end

:nopy
echo [X] Python topilmadi.
echo     https://www.python.org/downloads/ dan 3.14 ni o'rnating,
echo     o'rnatishda "Add python.exe to PATH" ni belgilang.
goto :end

:testfail
echo.
echo [X] Testlar yiqildi. Ekrandagi xatoni Claude'ga tashlang.
echo     Eksperimentni boshlamang: yiqilgan test = ishonib bo'lmaydigan jadval.

:end
echo.
pause
endlocal
