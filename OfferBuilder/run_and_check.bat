@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   OfferBuilder - sborka i proverka
echo ============================================
echo.

rem --- 1. Python ---------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python ne nayden v PATH. Ustanovite Python 3.10+ i povtorite.
    goto :fail
)
for /f "delims=" %%V in ('python --version 2^>^&1') do echo [OK] %%V naiden
echo.

rem --- 2. Zavisimosti (Pillow, PyYAML) -------------------------------------
python -c "import PIL, yaml" >nul 2>nul
if errorlevel 1 (
    echo [WARN] Ne naideny Pillow/PyYAML. Ustanavlivayu...
    python -m pip install --quiet pillow pyyaml
    if errorlevel 1 (
        echo [FAIL] Ne udalos ustanovit zavisimosti ^(pillow, pyyaml^).
        goto :fail
    )
    echo [OK] Zavisimosti ustanovleny.
) else (
    echo [OK] Zavisimosti na meste ^(Pillow, PyYAML^).
)
echo.

rem --- 3. Papki s resursami ------------------------------------------------
set "MISSING="
if not exist "C:\Playrix\Cutted"          set "MISSING=!MISSING! Cutted"
if not exist "C:\Playrix\RuleForBuilding" set "MISSING=!MISSING! RuleForBuilding"
if not exist "C:\Playrix\Icons"           set "MISSING=!MISSING! Icons"
if defined MISSING (
    echo [FAIL] Ne naideny papki:!MISSING!
    goto :fail
)
echo [OK] Papki s resursami na meste ^(Cutted, RuleForBuilding, Icons^).
echo.

rem --- 4. Fayl skripta ------------------------------------------------------
if not exist "%~dp0build_offer.py" (
    echo [FAIL] Ne nayden build_offer.py ryadom s etim bat-faylom.
    goto :fail
)

rem --- 5. Config i vyhodnoy fayl -------------------------------------------
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=example_config.yaml"
if not exist "%CONFIG%" (
    echo [FAIL] Config ne nayden: %CONFIG%
    goto :fail
)
set "OUT=result.png"
if not "%~2"=="" set "OUT=%~2"

echo Sborka offera iz "%CONFIG%" -^> "%OUT%" ...
echo.
python build_offer.py --config "%CONFIG%" --out "%OUT%" -v
if errorlevel 1 (
    echo.
    echo [FAIL] build_offer.py zavershilsya s oshibkoy ^(sm. log vyshe^).
    goto :fail
)
echo.

rem --- 6. Proverka rezultata -------------------------------------------------
if not exist "%OUT%" (
    echo [FAIL] Fayl rezultata "%OUT%" ne sozdan.
    goto :fail
)
for %%A in ("%OUT%") do set "SIZE=%%~zA"
if !SIZE! LSS 10000 (
    echo [FAIL] Fayl rezultata podozritelno malenkiy: !SIZE! bayt.
    goto :fail
)
echo [OK] Rezultat sozdan: %OUT% ^(!SIZE! bayt^)
echo.
echo ============================================
echo   VSYO GOTOVO. Offer sobran i proveren.
echo ============================================
echo.
start "" "%OUT%"
goto :end

:fail
echo.
echo ============================================
echo   PROVERKA NE PROYDENA. Sm. soobscheniya vyshe.
echo ============================================
pause
exit /b 1

:end
pause
exit /b 0
