#!/usr/bin/env python3
"""Instalador de AiCleanPaint (Windows, Linux y macOS).

Crea el entorno virtual `.venv`, instala PyTorch con la versión de CUDA que corresponde a tu driver,
instala el resto de las dependencias y verifica que todo funcione.

Uso:  python install.py [opciones]      (o setup.bat / ./setup.sh, que lo invocan)
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MIN_PY = (3, 10)
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
VALID_TAGS = ("cpu", "cu126", "cu128", "cu130")
MODEL_ID = "Qwen/Qwen-Image-2.1"

try:  # consolas de Windows con codificaciones viejas
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    say(f"\n=== {msg} ===")


# ------------------------------------------------------------------ detección (funciones puras, testeables)
def parse_cuda_version(text: str) -> tuple[int, int] | None:
    """Extrae la versión máxima de CUDA que soporta el driver, de la salida de `nvidia-smi`."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_gpu_line(text: str) -> tuple[str, float] | None:
    """'NVIDIA GeForce RTX 3060, 12288' -> ('NVIDIA GeForce RTX 3060', 12.0)."""
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[-1].replace(".", "", 1).isdigit():
            return ", ".join(parts[:-1]), float(parts[-1]) / 1024.0
    return None


def pick_backend(cuda: tuple[int, int] | None, system: str, override: str | None = None):
    """Elige el índice de PyTorch. Devuelve (tag, motivo). tag=None si el driver es demasiado viejo.

    Regla conservadora: cu128 para drivers con CUDA >= 12.8 (combinación validada), cu126 para >= 12.6.
    cu130 solo se usa si se pide con --cuda cu130.
    """
    if override:
        o = override.lower()
        if o not in VALID_TAGS:
            raise ValueError(f"--cuda debe ser uno de: {', '.join(VALID_TAGS)}")
        return o, f"elegido por vos (--cuda {o})"
    if system == "Darwin":
        return "default", "macOS: PyTorch estándar (sin CUDA)"
    if cuda is None:
        return "cpu", "no se detectó una GPU NVIDIA (nvidia-smi no responde)"
    if cuda >= (12, 8):
        return "cu128", f"tu driver admite CUDA {cuda[0]}.{cuda[1]}"
    if cuda >= (12, 6):
        return "cu126", f"tu driver admite CUDA {cuda[0]}.{cuda[1]}"
    return None, f"tu driver solo admite CUDA {cuda[0]}.{cuda[1]} (se necesita 12.6 o superior)"


def torch_tag_from_version(version: str | None) -> str | None:
    """'2.11.0+cu128' -> 'cu128'; '2.11.0+cpu' -> 'cpu'; sin sufijo -> None."""
    if not version or "+" not in version:
        return None
    return version.split("+", 1)[1].strip()


# ------------------------------------------------------------------ utilidades de ejecución
def run_capture(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return 127, ""


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str], dry: bool, env: dict | None = None) -> int:
    say("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry:
        return 0
    return subprocess.run(cmd, env=env).returncode


def _rm_readonly(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# ------------------------------------------------------------------ pasos
def ensure_python() -> None:
    if sys.version_info < MIN_PY:
        say(f"[X]  Python {sys.version_info.major}.{sys.version_info.minor} es demasiado viejo: "
            f"se necesita {MIN_PY[0]}.{MIN_PY[1]} o superior.")
        sys.exit(1)
    say(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({sys.executable})")
    if sys.version_info >= (3, 15):
        say("[!!] Python muy nuevo: puede que PyTorch todavía no publique paquetes para esta versión. "
            "Si falla, usá Python 3.12 o 3.13.")


def ensure_venv(recreate: bool, dry: bool) -> None:
    if recreate and VENV.exists():
        say(f"Borrando el entorno anterior ({VENV}) ...")
        if not dry:
            shutil.rmtree(VENV, onerror=_rm_readonly)
    if venv_python().exists():
        rc, out = run_capture([str(venv_python()), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
        say(f"[OK] Entorno virtual existente: {VENV}")
        m = re.match(r"(\d+)\.(\d+)", out.strip()) if rc == 0 else None
        if m and (int(m.group(1)), int(m.group(2))) < MIN_PY:
            say("[!!] El entorno existente usa un Python viejo. Volvé a correr con --recreate.")
        return
    say(f"Creando el entorno virtual en {VENV} ...")
    if dry:
        say(f"$ {sys.executable} -m venv {VENV}")
        return
    rc = subprocess.run([sys.executable, "-m", "venv", str(VENV)]).returncode
    if rc != 0 or not venv_python().exists():
        say("[X]  No se pudo crear el entorno virtual.")
        if platform.system() == "Linux":
            say("     En Debian/Ubuntu instalá el módulo venv:  sudo apt install python3-venv")
        sys.exit(2)
    say("[OK] Entorno virtual creado.")


def detect_gpu() -> dict:
    info = {"cuda": None, "name": None, "vram_gb": None}
    rc, out = run_capture(["nvidia-smi"])
    if rc == 0:
        info["cuda"] = parse_cuda_version(out)
        rc2, out2 = run_capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        g = parse_gpu_line(out2) if rc2 == 0 else None
        if g:
            info["name"], info["vram_gb"] = g
    return info


def installed_torch_tag() -> tuple[bool, str | None]:
    """(¿hay torch en el entorno?, etiqueta de su build)."""
    if not venv_python().exists():
        return False, None
    rc, out = run_capture([str(venv_python()), "-c", "import torch; print(torch.__version__)"], timeout=120)
    if rc != 0:
        return False, None
    return True, torch_tag_from_version(out.strip().splitlines()[-1] if out.strip() else "")


def install_torch(args, gpu: dict) -> None:
    step("PyTorch")
    system = platform.system()
    if gpu["name"]:
        say(f"[OK] GPU detectada: {gpu['name']} ({gpu['vram_gb']:.0f} GB de VRAM)")
        if gpu["vram_gb"] and gpu["vram_gb"] < 10:
            say("[!!] Menos de 10 GB de VRAM: puede que necesites `--offload sequential` y recortes más chicos.")
    try:
        tag, reason = pick_backend(gpu["cuda"], system, args.cuda)
    except ValueError as e:
        say(f"[X]  {e}")
        sys.exit(1)
    if tag is None:
        say(f"[X]  {reason}.")
        say("     Actualizá el driver de NVIDIA (https://www.nvidia.com/drivers) y volvé a ejecutar este instalador,")
        say("     o forzá una build de CPU (muy lenta, no recomendada) con:  --cuda cpu")
        sys.exit(3)
    say(f"Build de PyTorch elegida: {tag}  ({reason})")
    if tag == "cpu":
        say("[!!] Se instalará PyTorch solo para CPU. Qwen-Image no es utilizable en la práctica sin una GPU NVIDIA.")
    if system == "Darwin":
        say("[!!] macOS no es un objetivo soportado: la cuantización de 4 bits (bitsandbytes) requiere CUDA.")

    present, current = installed_torch_tag()
    wanted = None if tag == "default" else tag
    cmd = [str(venv_python()), "-m", "pip", "install", "torch", "torchvision"]
    if tag != "default":
        cmd += ["--index-url", TORCH_INDEX.format(tag=tag)]
    if present and (wanted is None or current == wanted):
        say(f"[OK] PyTorch ya está instalado con la build correcta ({current or 'estándar'}); "
            "solo se verifica que torchvision coincida.")
    elif present:
        say(f"Se reemplaza la build instalada ({current or 'sin etiqueta'}) por {wanted}.")
        cmd.append("--force-reinstall")
    else:
        say("Descargando PyTorch (varios GB; puede tardar). Se instalan torch y torchvision juntos desde el mismo "
            "índice para que no queden con versiones incompatibles.")
    if run(cmd, args.dry_run) != 0:
        say("\n[X]  Falló la instalación de PyTorch. Causas comunes: sin internet, poco espacio en disco, o un "
            "Python demasiado nuevo.")
        say("     Probá otra build con:  --cuda cu126   (o cu128 / cu130)")
        sys.exit(4)


def install_deps(args) -> None:
    step("Dependencias del proyecto")
    py = str(venv_python())
    run([py, "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)
    if run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], args.dry_run) != 0:
        say("\n[X]  Falló la instalación de dependencias. Revisá el error de arriba y volvé a ejecutar el instalador "
            "(es seguro repetirlo).")
        sys.exit(5)


def verify(args) -> int:
    step("Verificación")
    if args.dry_run:
        say(f"$ {venv_python()} {ROOT / 'check_env.py'}")
        return 0
    return subprocess.run([str(venv_python()), str(ROOT / "check_env.py")]).returncode


def download_model(args) -> None:
    step(f"Descarga anticipada del modelo ({MODEL_ID})")
    free = shutil.disk_usage(ROOT).free / 1024**3
    say(f"Espacio libre en disco: {free:.0f} GB (el modelo ocupa ~33 GB).")
    if free < 40:
        say("[!!] Espacio justo. Si se corta la descarga, liberá espacio y volvé a ejecutar.")
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(ROOT / "models"))
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    code = f"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_ID}')"
    say(f"Se guarda en: {env['HF_HOME']}")
    if run([str(venv_python()), "-c", code], args.dry_run, env=env) != 0:
        say("[!!] La descarga no terminó. No es grave: la app la retoma sola la primera vez que la uses.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Instalador de AiCleanPaint")
    p.add_argument("--cuda", choices=VALID_TAGS, help="forzar la build de PyTorch (por defecto: automática)")
    p.add_argument("--recreate", action="store_true", help="borrar y recrear el entorno virtual .venv")
    p.add_argument("--skip-torch", action="store_true", help="no instalar PyTorch")
    p.add_argument("--skip-deps", action="store_true", help="no instalar requirements.txt")
    p.add_argument("--download-model", action="store_true", help="descargar ahora el modelo (~33 GB)")
    p.add_argument("--check", action="store_true", help="solo diagnosticar el entorno existente")
    p.add_argument("--dry-run", action="store_true", help="mostrar los comandos sin ejecutarlos")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    say("== Instalador de AiCleanPaint ==")
    say(f"Sistema: {platform.system()} {platform.release()} · Carpeta: {ROOT}")
    if args.dry_run:
        say("[Modo simulacro: no se ejecuta nada]")
    ensure_python()
    if args.check:
        if not venv_python().exists():
            say("[X]  Todavía no hay entorno (.venv). Ejecutá el instalador sin --check.")
            return 1
        return verify(args)

    step("Entorno virtual")
    ensure_venv(args.recreate, args.dry_run)
    gpu = detect_gpu()
    if not args.skip_torch:
        install_torch(args, gpu)
    if not args.skip_deps:
        install_deps(args)
    if args.download_model:
        download_model(args)
    rc = verify(args)

    step("Listo")
    if rc == 0:
        say("Instalación completa. Para abrir la app:")
    else:
        say("La instalación terminó, pero el diagnóstico encontró fallas (arriba, líneas [FALLA]).")
        say("Corregilas y volvé a ejecutar el instalador (es seguro repetirlo). Cuando esté todo bien:")
    say("   Windows:  run.bat" if os.name == "nt" else "   Linux:    ./run.sh")
    say("La primera vez que apretés «Ejecutar» se descarga el modelo (~33 GB) si no usaste --download-model.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
