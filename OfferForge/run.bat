@echo off
setlocal
cd /d "%~dp0"

rem UTF-8 в консоли — иначе русские echo превращаются в кракозябры
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set PORT=8848

echo ============================================
echo   OfferForge
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [!] Python не найден в PATH. Поставь Python 3.11+ и повтори.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [*] Создаю виртуальное окружение...
    python -m venv .venv
    if errorlevel 1 goto :fail
)

call .venv\Scripts\activate.bat

rem Ставим зависимости всегда. pip идемпотентен.
echo [*] Проверяю зависимости...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 goto :fail

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo.
        echo [!] Создан .env — впиши туда OPENROUTER_API_KEY.
        echo     Без ключа доступен только пресет "Оффлайн (mock)".
        echo.
    )
)

rem Порт занят старым uvicorn — снимаем, иначе WinError 10048.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo [*] Порт %PORT% занят PID %%p — останавливаю...
    taskkill /F /PID %%p >nul 2>nul
)

echo [*] Сервер: http://127.0.0.1:%PORT%
echo     Остановить — Ctrl+C
echo.

start "" http://127.0.0.1:%PORT%
python -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%

rem Ctrl+C даёт 0 или 2 — штатная остановка.
if errorlevel 3 (
    echo.
    echo [!] Сервер остановился с ошибкой. Текст ошибки — выше.
    pause
)

goto :eof

:fail
echo.
echo [!] Не удалось подготовить окружение.
pause
exit /b 1
