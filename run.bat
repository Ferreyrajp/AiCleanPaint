@echo off
rem Lanza Qwen Eraser y abre el navegador automaticamente.
rem Uso: run.bat [opciones de app.py]   ej: run.bat --quant int8
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo No se encontro el entorno virtual ^(.venv^). Ejecuta primero setup.bat para instalarlo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python app.py %*

if errorlevel 1 (
    echo.
    echo La aplicacion termino con error.
    pause
)