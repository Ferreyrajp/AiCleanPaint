#!/usr/bin/env bash
# Lanza Qwen Eraser y abre el navegador automáticamente.
# Uso: ./run.sh [opciones de app.py]   ej: ./run.sh --quant int8
cd "$(dirname "$0")" || exit 1

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
else
    echo "No se encontró el entorno virtual (.venv). Ejecutá primero ./setup.sh para instalarlo."
    exit 1
fi

python app.py "$@"
status=$?
if [ $status -ne 0 ]; then
    echo
    echo "La aplicación terminó con error (código $status)."
    read -r -p "Presioná Enter para salir..." _
fi
exit $status
