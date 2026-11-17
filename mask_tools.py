"""Utilidades de máscaras (numpy/PIL, con OpenCV opcional). No dependen de torch ni de SAM."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


_PALETTE = [(255, 0, 255), (0, 190, 255), (255, 170, 0), (60, 210, 90), (255, 70, 70), (140, 90, 255),
            (0, 220, 200), (240, 220, 40), (255, 120, 190), (120, 180, 40), (90, 140, 255), (200, 120, 60)]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow viejo: sin parámetro de tamaño
        return ImageFont.load_default()


def overlay_instances(image: Image.Image, masks: list, boxes: list, selected: list[int],
                      alpha: float = 0.5) -> Image.Image:
    """Vista previa de varias detecciones numeradas (1, 2, 3...). Las elegidas se tiñen; las otras, en gris."""
    base = image.convert("RGB")
    arr = np.asarray(base).astype(np.float32)
    sel = set(selected)
    for i in selected:
        if i < len(masks):
            col = np.array(_PALETTE[i % len(_PALETTE)], np.float32)
            m = np.asarray(masks[i], dtype=bool)
            arr[m] = arr[m] * (1 - alpha) + col * alpha
    out = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    fs = max(16, int(min(out.size) * 0.035))
    font = _font(fs)
    lw = max(2, int(min(out.size) * 0.004))
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        col = _PALETTE[i % len(_PALETTE)] if i in sel else (150, 150, 150)
        draw.rectangle((x0, y0, x1, y1), outline=col, width=lw if i in sel else 1)
        label = str(i + 1)
        tw = draw.textlength(label, font=font)
        pad = max(3, fs // 5)
        bx0, by0 = max(0, x0), max(0, y0)
        draw.rectangle((bx0, by0, bx0 + tw + 2 * pad, by0 + fs + 2 * pad), fill=col)
        draw.text((bx0 + pad, by0 + pad), label, fill=(255, 255, 255), font=font)
    return out
