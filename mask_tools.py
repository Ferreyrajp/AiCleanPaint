"""Utilidades de máscaras (numpy/PIL, con OpenCV opcional). No dependen de torch ni de SAM."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

HILITE = (255, 0, 255)  # mismo magenta que usa el resto de la app


def clean_mask(mask: np.ndarray, min_island_frac: float = 0.0005, max_hole_frac: float = 0.0005) -> np.ndarray:
    """Quita islas diminutas y rellena agujeros pequeños. Sin OpenCV devuelve la máscara tal cual."""
    mask = np.asarray(mask, dtype=bool)
    if cv2 is None or not mask.any():
        return mask
    h, w = mask.shape
    area = h * w
    u8 = mask.astype(np.uint8)

    # 1) islas: se conserva la componente más grande y las que superen el umbral
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n > 2:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = np.zeros(n, dtype=bool)
        keep[biggest] = True
        keep |= stats[:, cv2.CC_STAT_AREA] >= max(64, min_island_frac * area)
        keep[0] = False
        u8 = keep[labels].astype(np.uint8)

    # 2) agujeros: fondo encerrado (que no toca el borde) y chico
    inv = (1 - u8).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches_border and a <= max(64, max_hole_frac * area):
            u8[labels == i] = 1
    return u8.astype(bool)


def overlay(image: Image.Image, mask: np.ndarray | None, points: list, labels: list,
            alpha: float = 0.55) -> Image.Image:
    """Vista previa: máscara teñida de magenta + los clics (verde = incluir, rojo = excluir)."""
    base = image.convert("RGB")
    if mask is not None and mask.any():
        arr = np.asarray(base).astype(np.float32)
        m = np.asarray(mask, dtype=bool)
        arr[m] = arr[m] * (1 - alpha) + np.array(HILITE, np.float32) * alpha
        base = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(base)
    r = max(6, int(min(base.size) * 0.012))
    for (x, y), lab in zip(points, labels):
        col = (40, 200, 70) if lab == 1 else (230, 40, 40)
        draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), fill=(255, 255, 255))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=col)
    return base


def mask_to_layer(mask: np.ndarray) -> Image.Image:
    """Capa RGBA (magenta opaco donde hay máscara) para cargar en el editor con pincel."""
    m = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[m] = (*HILITE, 255)
    return Image.fromarray(rgba, "RGBA")
