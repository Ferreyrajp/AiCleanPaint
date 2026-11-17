"""Modo rápido: inpainting clásico con LaMa (big-lama, TorchScript). Opcional y apagado por defecto.

- Los pesos (~200 MB, Apache-2.0) se descargan a `models/` recién la primera vez que se usa.
- Como un `.pt` TorchScript usa pickle, se VERIFICA su hash SHA-256 contra un valor fijo antes de cargarlo.
- Contrato del modelo (según simple-lama-inpainting, MIT): imagen RGB float32 [0,1] de forma (1,3,H,W) y máscara
  entera 0/1 (1,1,H,W), con H y W múltiplos de 8; devuelve (1,3,H,W) en [0,1].
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

# Repositorio fijado a un commit concreto (inmutable) y con hash conocido.
REPO_ID = "JosephCatrambone/big-lama-torchscript"
FILENAME = "lama.pt"
REVISION = "7dfbdc6dc27602d370909e8688f8f84404a1f995"
SHA256 = "79ed9fa18680ca19a269eb88fc4f708fb6580197b2220a51d3e000eb0ffdf33b"
SIZE_BYTES = 206134115
APPROX_MB = 200


# ------------------------------------------------------------------ preparación de datos (sin torch)
def ceil_modulo(x: int, mod: int) -> int:
    return x if x % mod == 0 else (x // mod + 1) * mod


def pad_to_modulo(arr: np.ndarray, mod: int = 8) -> np.ndarray:
    """Rellena (C, H, W) hasta múltiplos de `mod` repitiendo el borde en espejo, como el LaMa original."""
    _, h, w = arr.shape
    return np.pad(arr, ((0, 0), (0, ceil_modulo(h, mod) - h), (0, ceil_modulo(w, mod) - w)), mode="symmetric")


def prepare(image: Image.Image, mask: Image.Image) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """-> (imagen (1,3,H',W') float32 [0,1], máscara (1,1,H',W') int64 0/1, (alto, ancho) originales)."""
    img = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    msk = (np.asarray(mask.convert("L")) > 127).astype(np.float32)[None]
    if msk.shape[1:] != img.shape[1:]:
        raise ValueError("La imagen y la máscara deben tener el mismo tamaño.")
    h, w = img.shape[1:]
    img = pad_to_modulo(img)[None].astype(np.float32)
    msk = (pad_to_modulo(msk)[None] > 0).astype(np.int64)
    return np.ascontiguousarray(img), np.ascontiguousarray(msk), (h, w)


def to_image(out: np.ndarray, hw: tuple[int, int]) -> Image.Image:
    """(1,3,H',W') o (3,H',W') en [0,1] -> PIL RGB recortado al tamaño original (se quita el relleno)."""
    a = np.asarray(out)
    if a.ndim == 4:
        a = a[0]
    h, w = hw
    a = np.clip(a.transpose(1, 2, 0)[:h, :w] * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGB")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ modelo
class LamaInpainter:
    def __init__(self, device: str = "auto"):
        self.device_pref = device  # "auto" | "cpu" | "cuda"
        self.device: str | None = None
        self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _resolve_device(self) -> str:
        import torch

        if self.device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_pref == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device_pref

    def ensure_weights(self) -> Path:
        """Descarga (si hace falta) los pesos a la caché de Hugging Face y verifica su integridad."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError("Falta huggingface_hub (viene con transformers): pip install -U transformers") from e
        path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION))
        if path.stat().st_size != SIZE_BYTES or sha256_file(path) != SHA256:
            raise RuntimeError(
                "El archivo de LaMa descargado no coincide con el hash esperado, así que NO se carga "
                f"(podría estar dañado o alterado). Borralo y reintentá: {path}"
            )
        return path

    def load(self) -> None:
        if self.loaded:
            return
        import torch

        path = self.ensure_weights()
        self.device = self._resolve_device()
        model = torch.jit.load(str(path), map_location=self.device)
        model.eval()
        model.to(self.device)
        self.model = model

    def unload(self) -> None:
        self.model = None
        gc.collect()

    def _run(self, img: np.ndarray, msk: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            out = self.model(torch.from_numpy(img).to(self.device), torch.from_numpy(msk).to(self.device))
            return out[0].detach().cpu().numpy()

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Rellena la zona blanca de `mask`. Devuelve una imagen del mismo tamaño que `image`."""
        self.load()
        img, msk, hw = prepare(image, mask)
        try:
            out = self._run(img, msk)
        except RuntimeError as e:
            # Sin memoria en la GPU (p. ej. mientras Qwen está cargado): se reintenta en la CPU.
            if "out of memory" not in str(e).lower() or self.device != "cuda":
                raise
            import torch

            torch.cuda.empty_cache()
            self.model.to("cpu")
            self.device = "cpu"
            out = self._run(img, msk)
        return to_image(out, hw)
