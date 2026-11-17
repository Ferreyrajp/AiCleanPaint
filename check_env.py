#!/usr/bin/env python3
"""Diagnóstico del entorno de AiCleanPaint. Uso: python check_env.py

Imprime qué está bien y qué falta. Si pedís ayuda, pegá la salida completa.
"""
from __future__ import annotations

import importlib
import os
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:  # evita errores de codificación en consolas de Windows
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

OK, WARN, FAIL, INFO = "OK", "AVISO", "FALLA", "INFO"
results: list[tuple[str, str, str]] = []


def add(status: str, name: str, msg: str = "") -> None:
    results.append((status, name, msg))
    tag = {OK: "[OK]   ", WARN: "[AVISO]", FAIL: "[FALLA]", INFO: "[INFO] "}[status]
    print(f"{tag} {name}" + (f": {msg}" if msg else ""), flush=True)


def version_of(mod_name: str) -> str:
    try:
        return getattr(importlib.import_module(mod_name), "__version__", "?")
    except Exception:
        return "no instalado"


def total_ram_gb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEM()
            m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024**3
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    return None


def check_python() -> None:
    v = sys.version_info
    if v < (3, 10):
        add(FAIL, "Python", f"{v.major}.{v.minor} es demasiado viejo (se necesita 3.10 o superior)")
    else:
        add(OK, "Python", f"{v.major}.{v.minor}.{v.micro}")


def check_torch() -> bool:
    try:
        import torch
    except Exception as e:
        add(FAIL, "PyTorch", f"no se pudo importar ({type(e).__name__}: {e})")
        return False
    build = torch.__version__
    add(OK, "PyTorch", build)
    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / 1024**3
        add(OK, "GPU (CUDA)", f"{props.name}, {gb:.1f} GB de VRAM, CUDA {torch.version.cuda}")
        if gb < 10:
            add(WARN, "VRAM", "menos de 10 GB: probá `run.bat --offload sequential` y bajá la resolución del recorte")
    else:
        why = "esta instalación de PyTorch es solo para CPU" if "+cpu" in build else \
              "PyTorch no ve ninguna GPU (¿driver de NVIDIA desactualizado?)"
        add(FAIL, "GPU (CUDA)", why + ". Sin GPU, Qwen-Image no es utilizable en la práctica.")
    return cuda_ok


def check_import(name: str, import_name: str | None = None, critical: bool = True, hint: str = "") -> bool:
    mod = import_name or name
    try:
        m = importlib.import_module(mod)
        add(OK, name, getattr(m, "__version__", ""))
        return True
    except Exception as e:
        add(FAIL if critical else WARN, name, f"no disponible ({type(e).__name__}: {e})" + (f". {hint}" if hint else ""))
        return False


def check_pipeline() -> None:
    try:
        from diffusers import QwenImage21Pipeline  # noqa: F401
        add(OK, "QwenImage21Pipeline", "disponible en diffusers " + version_of("diffusers"))
    except Exception as e:
        add(FAIL, "QwenImage21Pipeline", f"no disponible ({type(e).__name__}). "
            "Actualizá con: pip install -U diffusers transformers")


def check_optional_selection() -> None:
    try:
        from transformers import Sam2Model, Sam2Processor  # noqa: F401
        add(OK, "SAM 2 (selección de objetos)", "disponible en transformers " + version_of("transformers"))
    except Exception:
        add(WARN, "SAM 2 (selección de objetos)", "no disponible; es opcional. pip install -U transformers")
    try:
        from transformers import AutoModelForZeroShotObjectDetection  # noqa: F401
        add(OK, "Detección por texto (Grounding DINO)", "disponible")
    except Exception:
        add(WARN, "Detección por texto (Grounding DINO)", "no disponible; es opcional")


def check_disk_and_models() -> None:
    hub = Path(os.environ.get("HF_HOME", ROOT / "models")) / "hub" / "models--Qwen--Qwen-Image-2.1"
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if hub.exists():
        add(OK, "Modelo Qwen-Image-2.1", f"ya descargado en {hub.parent.parent}")
    else:
        msg = f"todavía no descargado (~33 GB, se baja la primera vez que ejecutás); espacio libre: {free_gb:.0f} GB"
        add(WARN if free_gb < 40 else INFO, "Modelo Qwen-Image-2.1", msg)


def check_ram() -> None:
    gb = total_ram_gb()
    if gb is None:
        add(INFO, "RAM", "no se pudo medir")
    elif gb < 16:
        add(WARN, "RAM", f"{gb:.0f} GB; se recomiendan 32 GB (parte del modelo vive en memoria del sistema)")
    else:
        add(OK, "RAM", f"{gb:.0f} GB")


def check_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        busy = s.connect_ex(("127.0.0.1", 7860)) == 0
    if busy:
        add(INFO, "Puerto 7860", "ocupado (otra instancia abierta); la app usará el siguiente libre")
    else:
        add(OK, "Puerto 7860", "libre")


def main() -> int:
    print("== Diagnóstico de AiCleanPaint ==\n")
    check_python()
    cuda_ok = check_torch()
    check_import("torchvision", hint="Instalalo desde el mismo índice de CUDA que torch")
    if cuda_ok:
        check_import("bitsandbytes", hint="Necesario para la cuantización de 4 bits")
    else:
        add(INFO, "bitsandbytes", "se omite: requiere GPU NVIDIA")
    check_pipeline()
    check_import("transformers")
    check_import("accelerate")
    check_import("gradio")
    check_import("PIL", "PIL")
    check_import("opencv (cv2)", "cv2", critical=False, hint="Opcional: relleno rápido de manchas pequeñas")
    check_import("sentencepiece", critical=False, hint="Necesario para la traducción de instrucciones")
    check_optional_selection()
    check_disk_and_models()
    check_ram()
    check_port()

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print("\n== Versiones (útil para pedir ayuda) ==")
    for m in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "bitsandbytes", "gradio", "PIL"):
        print(f"  {m:<14} {version_of(m)}")
    print(f"\nResumen: {len(results) - len(fails) - len(warns)} bien, {len(warns)} avisos, {len(fails)} fallas.")
    if fails:
        print("Hay fallas que impiden usar la app; revisá las líneas [FALLA] de arriba.")
        return 1
    print("Todo listo." if not warns else "Funciona, pero revisá los avisos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
