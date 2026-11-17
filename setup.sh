#!/usr/bin/env bash
# Instalador de AiCleanPaint para Linux (y macOS).
# Crea el entorno, instala PyTorch con la versión de CUDA de tu driver y las dependencias.
# Uso: ./setup.sh [opciones]     ej: ./setup.sh --cuda cu126   |   ./setup.sh --download-model
#      ./setup.sh --help   para ver todas las opciones.
cd "$(dirname "$0")" || exit 1

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "No se encontró Python 3.10 o superior."
    echo "Instalalo con tu gestor de paquetes, por ejemplo:"
    echo "  Debian/Ubuntu:  sudo apt install python3 python3-venv python3-pip"
    echo "  Fedora:         sudo dnf install python3"
    exit 1
fi

export PYTHONUTF8=1
"$PY" install.py "$@"
status=$?
if [ $status -ne 0 ]; then
    echo
    echo "La instalación terminó con errores (código $status). Es seguro volver a ejecutarla."
fi
exit $status
