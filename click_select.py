"""Lógica de la selección por clics (sin interfaz): estado, clics de incluir/excluir y máscara final."""
from __future__ import annotations

import numpy as np
from PIL import Image

import mask_tools as mt

CAND_CHOICES = ["Auto", "1", "2", "3"]


def new_state() -> dict:
    return {"base": None, "points": [], "labels": [], "cands": [], "scores": []}


def start(segmenter, image: Image.Image) -> dict:
    """Prepara una imagen (calcula embeddings) y reinicia los clics."""
    segmenter.set_image(image)
    st = new_state()
    st["base"] = image.convert("RGB")
    return st


def _choose(cands, scores, choice):
    if not cands:
        return None
    if choice in ("1", "2", "3"):
        i = int(choice) - 1
        if i < len(cands):
            return cands[i]
    return cands[int(np.argmax(scores))]


def current_mask(state: dict, choice: str = "Auto") -> np.ndarray | None:
    return _choose(state["cands"], state["scores"], choice)


def render(state: dict, choice: str = "Auto") -> Image.Image:
    return mt.overlay(state["base"], current_mask(state, choice), state["points"], state["labels"])


def _with(state: dict, **changes) -> dict:
    new = dict(state)
    new.update(changes)
    return new


def add_point(state: dict, segmenter, x: float, y: float, positive: bool, choice: str = "Auto"):
    if state["base"] is None:
        raise ValueError("Primero cargá la imagen del editor.")
    if not positive and not state["points"]:
        raise ValueError("El primer clic tiene que ser de tipo «Incluir (+)», sobre el objeto.")
    w, h = state["base"].size
    x = min(max(int(round(x)), 0), w - 1)
    y = min(max(int(round(y)), 0), h - 1)
    pts = state["points"] + [(x, y)]
    labs = state["labels"] + [1 if positive else 0]
    cands, scores = segmenter.predict(pts, labs)
    new = _with(state, points=pts, labels=labs, cands=cands, scores=scores)
    return new, render(new, choice)


def remove_last_point(state: dict, segmenter, choice: str = "Auto"):
    if state["base"] is None or not state["points"]:
        return state, (render(state, choice) if state["base"] is not None else None)
    pts, labs = state["points"][:-1], state["labels"][:-1]
    cands, scores = segmenter.predict(pts, labs) if pts else ([], [])
    new = _with(state, points=pts, labels=labs, cands=cands, scores=scores)
    return new, render(new, choice)


def clear(state: dict, choice: str = "Auto"):
    if state["base"] is None:
        return state, None
    new = _with(state, points=[], labels=[], cands=[], scores=[])
    return new, render(new, choice)


def to_editor_value(state: dict, choice: str = "Auto", clean: bool = True) -> dict:
    """Valor para gr.ImageEditor: imagen de fondo + la máscara como capa pintada."""
    m = current_mask(state, choice)
    if m is None or not m.any():
        raise ValueError("Todavía no hay máscara: hacé clic sobre el objeto.")
    if clean:
        m = mt.clean_mask(m)
    return {"background": state["base"], "layers": [mt.mask_to_layer(m)], "composite": None}


def describe(state: dict) -> str:
    if state["base"] is None:
        return "Cargá la imagen del editor para empezar."
    n = len(state["points"])
    if n == 0:
        return "Imagen lista. **Hacé clic sobre el objeto** que querés seleccionar."
    m = current_mask(state)
    pct = 100.0 * float(m.sum()) / m.size if m is not None else 0.0
    extra = ""
    if len(state["cands"]) > 1:
        extra = f" · {len(state['cands'])} máscaras sugeridas (elegí otra si no es la que querías)"
    return f"{n} clic(s) · la máscara cubre el {pct:.1f}% de la imagen{extra}."
