@echo off
echo ============================================
echo  Trade Buddy v3.0 — 7 Regimenes HMM
echo  Instalador
echo ============================================
echo.

set PYTHON_CMD=

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :found
)

py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :found
)

for /d %%i in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%i\python.exe" (
        set PYTHON_CMD="%%i\python.exe"
        goto :found
    )
)

echo [ERROR] Python no encontrado. Descarga Python desde https://www.python.org/downloads/
echo         Marca "Add Python to PATH" durante la instalacion.
pause & exit /b 1

:found
echo [OK] Python encontrado: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

echo Paso 1: Actualizando pip, setuptools y wheel...
%PYTHON_CMD% -m pip install --upgrade pip setuptools wheel
echo.

echo Paso 2: Instalando numpy...
%PYTHON_CMD% -m pip install "numpy>=1.24,<2.0"
echo.

echo Paso 3: Instalando scipy...
%PYTHON_CMD% -m pip install "scipy>=1.10"
echo.

echo Paso 4: Instalando pandas...
%PYTHON_CMD% -m pip install "pandas>=2.0"
echo.

echo Paso 5: Instalando scikit-learn y joblib...
%PYTHON_CMD% -m pip install "scikit-learn>=1.3" "joblib>=1.3"
echo.

echo Paso 6: Instalando hmmlearn...
%PYTHON_CMD% -m pip install "hmmlearn>=0.3"
echo.

echo Paso 7: Instalando plotly y streamlit...
%PYTHON_CMD% -m pip install "plotly>=5.18" "streamlit>=1.32"
echo.

echo Paso 8: Instalando fuentes de datos...
%PYTHON_CMD% -m pip install "ccxt>=4.0" "yfinance>=0.2.36" "requests>=2.31"
echo.

echo Guardando configuracion...
echo %PYTHON_CMD% > python_path.txt

echo.
echo ============================================
echo  INSTALACION COMPLETA
echo  Ejecuta: iniciar.bat
echo ============================================
pause
