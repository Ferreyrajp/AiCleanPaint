"""Selección por texto: etiquetas en español -> detección (Grounding DINO) -> máscaras (SAM 2).

Sin interfaz ni dependencias pesadas: el detector y SAM se reciben como parámetros.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

import mask_tools as mt

# Sustantivos frecuentes: se resuelven sin traductor (instantáneo y más confiable que traducir una palabra suelta).
_ES_EN = {
    "persona": "person", "gente": "person", "humano": "person", "hombre": "man", "mujer": "woman",
    "chica": "girl", "niña": "girl", "chico": "boy", "niño": "boy", "bebé": "baby", "bebe": "baby",
    "perro": "dog", "gato": "cat", "caballo": "horse", "pájaro": "bird", "pajaro": "bird", "ave": "bird",
    "vaca": "cow", "oveja": "sheep",
    "auto": "car", "coche": "car", "carro": "car", "automóvil": "car", "automovil": "car",
    "moto": "motorcycle", "motocicleta": "motorcycle", "bicicleta": "bicycle", "bici": "bicycle",
    "camión": "truck", "camion": "truck", "colectivo": "bus", "autobús": "bus", "autobus": "bus",
    "barco": "boat", "bote": "boat", "avión": "airplane", "avion": "airplane",
    "árbol": "tree", "arbol": "tree", "flor": "flower", "planta": "plant", "pasto": "grass", "césped": "grass",
    "casa": "house", "edificio": "building", "ventana": "window", "puerta": "door",
    "mesa": "table", "silla": "chair", "sillón": "couch", "sillon": "couch", "sofá": "couch", "sofa": "couch",
    "cama": "bed", "botella": "bottle", "vaso": "glass", "taza": "cup", "plato": "plate",
    "mochila": "backpack", "bolso": "handbag", "cartera": "handbag", "paraguas": "umbrella",
    "sombrilla": "umbrella", "valija": "suitcase", "maleta": "suitcase",
    "teléfono": "phone", "telefono": "phone", "celular": "phone", "computadora": "laptop", "laptop": "laptop",
    "cartel": "sign", "letrero": "sign", "texto": "text", "logo": "logo", "cable": "cable",
    "poste": "pole", "farol": "street lamp", "basura": "trash", "pelota": "ball",
    "cara": "face", "rostro": "face", "mano": "hand", "sombrero": "hat", "gorra": "cap",
    "gafas": "glasses", "anteojos": "glasses", "lentes": "glasses", "ropa": "clothes",
    "cielo": "sky", "nube": "cloud", "montaña": "mountain", "montana": "mountain",
}
_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del"}
_SEP_RE = re.compile(r"\s*(?:,|;|\by\b|\be\b|\bo\b)\s*", re.IGNORECASE)


def _lookup(word: str) -> str | None:
    """Busca la palabra tolerando plurales simples (perros, árboles, flores)."""
    w = word.lower().strip()
    for cand in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None):
        if cand and cand in _ES_EN:
            return _ES_EN[cand]
    return None


def to_labels(text: str | None, translator=None) -> list[str]:
    """'personas, el perro y la botella roja' -> ['person', 'dog', 'red bottle'] (en inglés, sin duplicados).

    `translator(texto_es) -> texto_en` se usa solo para frases que no están en el diccionario.
    """
    out: list[str] = []
    for part in _SEP_RE.split((text or "").strip()):
        words = [w for w in part.strip().split() if w.lower() not in _ARTICLES]
        if not words:
            continue
        phrase = " ".join(words)
        label = _lookup(phrase) if len(words) == 1 else None
        if label is None and translator is not None:
            try:
                label = (translator(phrase) or phrase).strip()
            except Exception:
                label = phrase
            label = re.sub(r"^(the|a|an)\s+", "", label.lower().strip()).rstrip(".").strip()
        label = (label or phrase).lower()
        if label and label not in out:
            out.append(label)
    return out


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets: list[dict], iou_thr: float = 0.6) -> list[dict]:
    """Quita cajas casi superpuestas (conserva la de mayor puntaje)."""
    keep: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(_iou(d["box"], k["box"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def new_state() -> dict:
    return {"base": None, "dets": [], "selected": [], "labels": []}


def search(image: Image.Image, labels: list[str], detector, segmenter, threshold: float = 0.30,
           text_threshold: float = 0.25, max_dets: int = 12, nms_iou: float = 0.6) -> dict:
    """Detecta, enmascara cada detección con SAM y numera de izquierda a derecha."""
    st = new_state()
    st["base"] = image.convert("RGB")
    st["labels"] = list(labels)
    dets = nms(detector.detect(st["base"], labels, threshold, text_threshold), nms_iou)
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)[:max_dets]
    dets = sorted(dets, key=lambda d: (d["box"][0] + d["box"][2]) / 2)  # numeración izquierda -> derecha
    if not dets:
        return st
    segmenter.set_image(st["base"])
    masks, mscores = segmenter.predict_boxes([d["box"] for d in dets])
    for d, m, s in zip(dets, masks, mscores):
        d["mask"], d["mask_score"] = m, s
    st["dets"] = dets
    st["selected"] = list(range(len(dets)))
    return st


def choices(state: dict) -> list[tuple[str, int]]:
    out = []
    for i, d in enumerate(state["dets"]):
        pct = 100.0 * float(d["mask"].sum()) / d["mask"].size
        out.append((f"{i + 1} · {d['label']} ({d['score']:.0%}) · {pct:.0f}% de la imagen", i))
    return out


def set_selected(state: dict, selected) -> dict:
    n = len(state["dets"])
    sel = sorted({int(i) for i in (selected or []) if 0 <= int(i) < n})
    new = dict(state)
    new["selected"] = sel
    return new


def render(state: dict) -> Image.Image | None:
    if state["base"] is None:
        return None
    return mt.overlay_instances(state["base"], [d["mask"] for d in state["dets"]],
                                [d["box"] for d in state["dets"]], state["selected"])


def final_mask(state: dict) -> np.ndarray | None:
    masks = [state["dets"][i]["mask"] for i in state["selected"]]
    if not masks:
        return None
    return np.logical_or.reduce(masks)


def to_editor_value(state: dict, clean: bool = True) -> dict:
    m = final_mask(state)
    if m is None or not m.any():
        raise ValueError("No hay nada seleccionado: buscá y marcá al menos un resultado.")
    if clean:
        m = mt.clean_mask(m)
    return {"background": state["base"], "layers": [mt.mask_to_layer(m)], "composite": None}


def describe(state: dict, shown_prompt: str = "") -> str:
    if state["base"] is None:
        return "Escribí qué querés seleccionar y apretá **Buscar**."
    q = f" Busqué: `{shown_prompt}`." if shown_prompt else ""
    if not state["dets"]:
        return ("No encontré nada." + q + " Probá bajar la **sensibilidad**, usar otro término o "
                "seleccionar con clics.")
    n, k = len(state["dets"]), len(state["selected"])
    return (f"Encontré **{n}** resultado(s), numerados de izquierda a derecha;{q} "
            f"seleccionados: **{k}**. Desmarcá los que no quieras y apretá «Usar la selección».")
