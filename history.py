"""Guardado automático de resultados e historial deshacer/rehacer (por sesión)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PIL import Image, ImageChops, PngImagePlugin

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MAX_ITEMS = 30  # versiones que se recuerdan por sesión (la más vieja se descarta)


def new_state() -> dict:
    return {"items": [], "idx": -1}


def _tmp_dir() -> Path:
    base = Path(os.environ.get("GRADIO_TEMP_DIR", ROOT / ".gradio_tmp")) / "history"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _unique_path(folder: Path, stem: str, suffix: str = ".png") -> Path:
    p = folder / f"{stem}{suffix}"
    n = 2
    while p.exists():
        p = folder / f"{stem}_{n}{suffix}"
        n += 1
    return p


def save_temp(img: Image.Image, label: str) -> Path:
    path = _unique_path(_tmp_dir(), f"{time.strftime('%Y%m%d-%H%M%S')}_{label}")
    img.convert("RGB").save(path, format="PNG", compress_level=1)  # rápido: es temporal
    return path


def save_result(img: Image.Image, meta: dict) -> Path:
    """Guarda en outputs/ un PNG sin pérdida con los parámetros embebidos."""
    OUTPUTS.mkdir(exist_ok=True)
    kind = "rellenar" if meta.get("action") == "fill" else "borrar"
    if meta.get("engine") == "lama" and kind == "borrar":
        kind += "-lama"
    path = _unique_path(OUTPUTS, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{kind}")
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", json.dumps(meta, ensure_ascii=False))
    img.convert("RGB").save(path, format="PNG", pnginfo=info)
    return path


def same_image(path: str, img: Image.Image, tol: int = 2) -> bool:
    """True si la imagen guardada en `path` es (casi) idéntica a `img`. Tolera ±tol por canal."""
    try:
        with Image.open(path) as a:
            a = a.convert("RGB")
            b = img.convert("RGB")
            if a.size != b.size:
                return False
            diff = ImageChops.difference(a, b).point(lambda v: 255 if v > tol else 0)
            return diff.getbbox() is None
    except Exception:
        return False


def push(state: dict, input_img: Image.Image, result_img: Image.Image, meta: dict,
         autosave: bool = True) -> tuple[dict, Path]:
    """Registra una edición. Continúa la cadena si la entrada es la versión actual; si no, reinicia."""
    items = list(state["items"])
    idx = state["idx"]
    head = items[idx] if 0 <= idx < len(items) else None
    if head is not None and same_image(head, input_img):
        items = items[: idx + 1]  # si venía de un "deshacer", se descarta la rama rehacer
    else:
        items = [str(save_temp(input_img, "original"))]
    path = save_result(result_img, meta) if autosave else save_temp(result_img, "resultado")
    items.append(str(path))
    if len(items) > MAX_ITEMS:
        items = items[-MAX_ITEMS:]
    return {"items": items, "idx": len(items) - 1}, path


def step(state: dict, delta: int) -> tuple[dict, Image.Image | None]:
    """Mueve el puntero (delta=-1 deshacer, +1 rehacer). Devuelve (estado, imagen) o (estado, None)."""
    items, idx = state["items"], state["idx"]
    new_idx = idx + delta
    if not items or not (0 <= new_idx < len(items)):
        return state, None
    try:
        with Image.open(items[new_idx]) as im:
            img = im.convert("RGB")
    except Exception:  # el archivo fue movido o borrado
        return state, None
    return {"items": items, "idx": new_idx}, img


def pair_paths(state: dict) -> tuple[str, str] | None:
    """(antes, después) de la versión actual, o None si es el original."""
    items, idx = state["items"], state["idx"]
    if idx >= 1 and idx < len(items):
        return items[idx - 1], items[idx]
    return None


def status(state: dict, saved: Path | None = None) -> str:
    items, idx = state["items"], state["idx"]
    if not items:
        return "Sin historial todavía."
    total = len(items) - 1
    base = "Original (sin ediciones)" if idx == 0 else f"Edición {idx} de {total}"
    if saved is not None:
        try:
            shown = saved.relative_to(ROOT)
        except ValueError:
            shown = saved
        base += f" · guardado en `{shown}`"
    return base
