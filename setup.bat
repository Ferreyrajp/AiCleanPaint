@echo off
rem Instalador de AiCleanPaint para Windows.
rem Crea el entorno, instala PyTorch con la version de CUDA de tu driver y las dependencias.
rem Uso: setup.bat [opciones]    ej: setup.bat --cuda cu126   ^|   setup.bat --download-model
rem      setup.bat --help   para ver todas las opciones.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo No se encontro Python 3.10 o superior.
    echo Instalalo desde https://www.python.org/downloads/ ^(marca "Add python.exe to PATH"^)
    echo o con:  winget install Python.Python.3.12
    if not defined NOPAUSE pause
    exit /b 1
)

%PY% install.py %*
set "RC=%errorlevel%"

echo.
if not "%RC%"=="0" echo La instalacion termino con errores ^(codigo %RC%^). Es seguro volver a ejecutarla.
if not defined NOPAUSE pause
exit /b %RC%