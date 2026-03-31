@echo off
echo ============================================
echo  Trade Buddy v3.0 — 7 Regimenes HMM
echo ============================================
echo.

set PYTHON_CMD=

if exist python_path.txt (
    set /p PYTHON_CMD=<python_path.txt
)

if "%PYTHON_CMD%"=="" (
    python --version >nul 2>&1
    if not errorlevel 1 set PYTHON_CMD=python
)

if "%PYTHON_CMD%"=="" (
    py --version >nul 2>&1
    if not errorlevel 1 set PYTHON_CMD=py
)

if "%PYTHON_CMD%"=="" (
    for /d %%i in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
        if exist "%%i\python.exe" set PYTHON_CMD="%%i\python.exe"
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python no encontrado. Ejecuta instalar.bat primero.
    pause
    exit /b 1
)

echo [OK] Usando Python: %PYTHON_CMD%
echo.

%PYTHON_CMD% -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Streamlit no encontrado, instalando...
    %PYTHON_CMD% -m pip install "streamlit>=1.32"
)

echo Iniciando Trade Buddy en http://localhost:8501
echo Para acceder desde celular: http://[IP-de-tu-PC]:8501
echo Para detener: Ctrl + C
echo.
%PYTHON_CMD% -m streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
pause
