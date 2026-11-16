"""Envoltorio de SAM 2.1 (Hugging Face transformers) para seleccionar objetos con clics.

Opcional: no se importa ni se descarga nada hasta que se usa por primera vez.
"""
from __future__ import annotations

import gc
import hashlib

import numpy as np
from PIL import Image

MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}
# Tamaño aproximado de la descarga (MB), solo informativo.
APPROX_MB = {"tiny": 155, "small": 200, "base-plus": 336, "large": 900}
DEFAULT_SIZE = "small"


class Segmenter:
    def __init__(self, size: str = DEFAULT_SIZE, device: str = "cpu"):
        self.size = size if size in MODELS else DEFAULT_SIZE
        self.device = device
        self.model = None
        self.processor = None
        self._emb = None
        self._sizes = None
        self._key = None

    # ------------------------------------------------------------------ carga
    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def ready(self) -> bool:
        """True si hay modelo cargado y una imagen preparada (embeddings calculados)."""
        return self.model is not None and self._emb is not None

    def set_size(self, size: str) -> None:
        if size in MODELS and size != self.size:
            self.unload()
            self.size = size

    def unload(self) -> None:
        self.model = self.processor = None
        self._emb = self._sizes = self._key = None
        gc.collect()

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as e:
            raise RuntimeError(
                "Tu versión de transformers no incluye SAM 2. Actualizá con: pip install -U transformers"
            ) from e
        model_id = MODELS[self.size]
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()

    # -------------------------------------------------------------- inferencia
    def set_image(self, image: Image.Image) -> None:
        """Calcula (una sola vez por imagen) los embeddings que reutilizan todos los clics."""
        import torch

        self.load()
        image = image.convert("RGB")
        key = (image.size, hashlib.md5(image.tobytes()).hexdigest(), self.size)
        if key == self._key:
            return
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt")
            self._emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        self._sizes = inputs["original_sizes"].cpu()
        self._key = key

    def predict(self, points: list[tuple[float, float]], labels: list[int],
                multimask: bool | None = None) -> tuple[list[np.ndarray], list[float]]:
        """Devuelve (máscaras booleanas HxW, puntajes). Con un solo clic ofrece 3 candidatas."""
        import torch

        if self._emb is None:
            raise RuntimeError("Primero hay que cargar una imagen (set_image).")
        if not points:
            return [], []
        if multimask is None:
            multimask = len(points) == 1
        pts = [[float(x), float(y)] for x, y in points]
        # Formatos que exige Sam2Processor: puntos [imagen][objeto][punto][x,y]; etiquetas [imagen][objeto][punto]
        labs = [int(v) for v in labels]
        enc = self.processor(
            original_sizes=self._sizes,
            input_points=[[pts]],   # profundidad 4: [imagen][objeto][punto][x, y]
            input_labels=[[labs]],  # profundidad 3: [imagen][objeto][punto]
            return_tensors="pt",
        )
        with torch.inference_mode():
            out = self.model(
                image_embeddings=self._emb,
                input_points=enc["input_points"].to(self.device),
                input_labels=enc["input_labels"].to(self.device),
                multimask_output=bool(multimask),
            )
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), self._sizes)[0]  # (1, N, H, W)
        arr = masks[0]
        arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        scores = out.iou_scores[0, 0].cpu().numpy()
        n = arr.shape[0]
        return [arr[i].astype(bool) for i in range(n)], [float(s) for s in scores[:n]]
