"""Detector de objetos por texto (Grounding DINO). Opcional: no se descarga ni carga hasta usarse."""
from __future__ import annotations

import gc

from PIL import Image

MODELS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}
# Tamaño aproximado de la descarga (MB), estimado de los parámetros en fp32 (172 M y 232 M). Solo informativo.
APPROX_MB = {"tiny": 690, "base": 930}
DEFAULT_SIZE = "tiny"


def build_prompt(labels: list[str]) -> str:
    """Formato que exige Grounding DINO: minúsculas y cada frase terminada en punto ('person. dog.')."""
    parts = [l.strip().lower().rstrip(".").strip() for l in labels if l and l.strip()]
    return " ".join(p + "." for p in parts if p)


class Detector:
    def __init__(self, size: str = DEFAULT_SIZE, device: str = "cpu"):
        self.size = size if size in MODELS else DEFAULT_SIZE
        self.device = device
        self.model = None
        self.processor = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def set_size(self, size: str) -> None:
        if size in MODELS and size != self.size:
            self.unload()
            self.size = size

    def unload(self) -> None:
        self.model = self.processor = None
        gc.collect()

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "Tu versión de transformers no incluye la detección por texto. "
                "Actualizá con: pip install -U transformers"
            ) from e
        model_id = MODELS[self.size]
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image: Image.Image, labels: list[str], threshold: float = 0.30,
               text_threshold: float = 0.25) -> list[dict]:
        """Devuelve [{'box': [x0,y0,x1,y1], 'score': float, 'label': str}] en píxeles de la imagen."""
        import torch

        prompt = build_prompt(labels)
        if not prompt:
            return []
        self.load()
        image = image.convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        sizes = [image.size[::-1]]  # (alto, ancho)
        try:  # transformers >= 5: `threshold`
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
                target_sizes=sizes)[0]
        except TypeError:  # transformers 4.x: `box_threshold`
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=threshold, text_threshold=text_threshold,
                target_sizes=sizes)[0]
        boxes = res["boxes"].cpu().tolist()
        scores = res["scores"].cpu().tolist()
        names = list(res["text_labels"])
        fallback = labels[0] if labels else ""
        return [
            {"box": [float(v) for v in b], "score": float(s), "label": (n or fallback).strip()}
            for b, s, n in zip(boxes, scores, names)
        ]
