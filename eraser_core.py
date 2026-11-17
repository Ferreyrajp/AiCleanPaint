"""Núcleo: borrado de objetos con Qwen-Image-2.1 (pensado para GPUs de 12 GB)."""
from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

try:  # opcional: relleno clásico para manchas pequeñas
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

MODEL_ID = "Qwen/Qwen-Image-2.1"

# Color del resaltado: magenta, porque casi no aparece en fotos reales (el rojo se confunde
# con ropa, flores, etc.).
HILITE_RGB = (255, 0, 255)
PROMPT_OVERLAY = (
    "Object removal. Remove everything covered by the solid magenta (pink-purple) overlay, "
    "including any person, body parts, clothing or fabric under it. Fill that area by naturally "
    "continuing the surrounding background (same surfaces, textures, lighting and perspective). "
    "The filled area must not contain any person, clothing or new object. "
    "Do not change anything else. No magenta color should remain."
)
PROMPT_FILL_OVERLAY = (
    "Image editing. Replace the area covered by the solid magenta (pink-purple) overlay with the "
    "following content: {what}. Match the skin tone, lighting, shadows, perspective, scale, "
    "sharpness and photographic style of the rest of the image, and keep the anatomy natural and "
    "consistent with the rest of the body. Do not change anything outside the highlighted area. "
    "No magenta color should remain."
)
PROMPT_FILL_REFERENCE = (
    "Image 2 is a mask. In image 1, replace the area where image 2 is white with the following "
    "content: {what}. Match the skin tone, lighting, shadows, perspective, scale, sharpness and "
    "photographic style of the rest of the image, and keep the anatomy natural. "
    "Do not change anything outside that area."
)
PROMPT_REFERENCE = (
    "Image 2 is a mask. Remove the object in image 1 located where image 2 is white, "
    "and fill that area by naturally continuing the surrounding background, matching "
    "lighting, texture and perspective. Do not change anything else."
)


@dataclass
class Settings:
    steps: int = 30
    max_side: int = 1024       # lado mayor del recorte que se envía al modelo
    context: float = 0.75      # contexto alrededor de la máscara (fracción del tamaño del objeto)
    grow_px: int = 6           # cuánto se agranda la máscara (cubre bordes y sombras)
    feather_px: int = 8        # suavizado del borde al pegar el resultado
    seed: int = 42
    mode: str = "overlay"      # "overlay" (pintado en rojo) | "reference" (máscara como 2ª imagen)
    extra_prompt: str = ""
    action: str = "erase"      # "erase" (borrar objeto) | "fill" (rellenar/agregar según instrucción)
    small_px: int = 64         # zonas más chicas que esto (lado mayor, en px) usan relleno clásico; 0 = nunca
    engine: str = "qwen"       # "qwen" (calidad, por defecto) | "lama" (modo rápido, solo para borrar)


def _round32(v: int) -> int:
    return max(256, int(round(v / 32)) * 32)


def _grow(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(2 * px + 1))


def _magenta_fraction(img: Image.Image, mask: Image.Image) -> float:
    """Fracción de píxeles dentro de la máscara que siguen siendo magenta (resaltado sobrante)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    m = np.asarray(mask.convert("L")) > 127
    if not m.any():
        return 0.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mag = ((r - g) > 80) & ((b - g) > 80) & (np.abs(r - b) < 110)
    return float((mag & m).sum() / m.sum())


def _classical_inpaint(img: Image.Image, mask: Image.Image, radius: int = 5) -> Image.Image | None:
    """Relleno clásico (Telea) con OpenCV. Útil para manchas pequeñas. None si no hay OpenCV."""
    if cv2 is None:
        return None
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    m = ((np.asarray(mask.convert("L")) > 127).astype(np.uint8)) * 255
    return Image.fromarray(cv2.inpaint(arr, m, radius, cv2.INPAINT_TELEA))


def _paste_patch(image: Image.Image, patch: Image.Image, mcrop: Image.Image,
                 box: tuple[int, int, int, int], feather_px: int) -> Image.Image:
    """Pega `patch` solo dentro de la máscara (borde suavizado) sobre una copia de `image`."""
    soft = mcrop
    if feather_px > 0:
        soft = _grow(mcrop, feather_px // 2).filter(ImageFilter.GaussianBlur(feather_px))
    result = image.copy()
    result.paste(patch, box[:2], mask=soft)
    return result


class Eraser:
    def __init__(self, model_id: str = MODEL_ID, quant: str = "nf4", offload: str = "model"):
        self.model_id = model_id
        self.quant = quant          # "nf4" | "int8" | "none"
        self.offload = offload      # "model" | "sequential" | "none"
        self.pipe = None
        self.lama = None            # LamaInpainter opcional (modo rápido); lo asigna la app

    # ------------------------------------------------------------------ carga
    def load(self):
        if self.pipe is not None:
            return
        from diffusers import QwenImage21Pipeline

        kwargs = {"torch_dtype": torch.bfloat16}
        if self.quant != "none":
            from diffusers.quantizers import PipelineQuantizationConfig

            if self.quant == "nf4":
                backend = "bitsandbytes_4bit"
                qkw = {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                }
            else:
                backend = "bitsandbytes_8bit"
                qkw = {"load_in_8bit": True}

            last_err = None
            for comps in (["transformer", "text_encoder"], ["transformer"]):
                try:
                    kwargs["quantization_config"] = PipelineQuantizationConfig(
                        quant_backend=backend, quant_kwargs=qkw, components_to_quantize=comps
                    )
                    self.pipe = QwenImage21Pipeline.from_pretrained(self.model_id, **kwargs)
                    break
                except Exception as e:  # nombres de componentes distintos, etc.
                    last_err = e
                    self.pipe = None
            if self.pipe is None:
                raise RuntimeError(f"No se pudo cargar con cuantización: {last_err}")
        else:
            self.pipe = QwenImage21Pipeline.from_pretrained(self.model_id, **kwargs)

        if self.offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif self.offload == "model":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass

    # ------------------------------------------------------------- utilidades
    @staticmethod
    def mask_from_editor(value) -> tuple[Image.Image, Image.Image]:
        """Devuelve (imagen RGB, máscara L 0/255) desde el valor de gr.ImageEditor."""
        if value is None or value.get("background") is None:
            raise ValueError("Subí una imagen primero.")
        bg = value["background"].convert("RGB")
        mask = Image.new("L", bg.size, 0)
        for layer in value.get("layers") or []:
            if layer is None:
                continue
            alpha = layer.convert("RGBA").split()[-1]
            if alpha.size != bg.size:
                alpha = alpha.resize(bg.size)
            mask = ImageChops.lighter(mask, alpha)
        mask = mask.point(lambda v: 255 if v > 10 else 0)
        if mask.getbbox() is None:
            raise ValueError("Pintá con el pincel el objeto que querés borrar.")
        return bg, mask

    @staticmethod
    def _crop_box(mask: Image.Image, size: tuple[int, int], context: float) -> tuple[int, int, int, int]:
        W, H = size
        x0, y0, x1, y1 = mask.getbbox()
        bw, bh = x1 - x0, y1 - y0
        margin = max(128, int(context * max(bw, bh)))
        x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
        # recorte mínimo de 640 px por lado (si la imagen lo permite)
        for _ in range(2):
            if x1 - x0 < 640:
                d = (640 - (x1 - x0)) // 2 + 1
                x0, x1 = x0 - d, x1 + d
            if y1 - y0 < 640:
                d = (640 - (y1 - y0)) // 2 + 1
                y0, y1 = y0 - d, y1 + d
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    # --------------------------------------------------------------- inferencia
    def _lama_patch(self, crop: Image.Image, mcrop: Image.Image, max_side: int) -> Image.Image:
        """Parche de LaMa para el recorte. Si el recorte es grande se reduce (nunca se agranda) y se vuelve a escalar."""
        cw, ch = crop.size
        k = min(1.0, max_side / max(cw, ch))
        if k < 1.0:
            tw, th = max(8, round(cw * k)), max(8, round(ch * k))
            patch = self.lama.inpaint(crop.resize((tw, th), Image.LANCZOS), mcrop.resize((tw, th), Image.NEAREST))
        else:
            patch = self.lama.inpaint(crop, mcrop)
        return patch if patch.size == (cw, ch) else patch.resize((cw, ch), Image.LANCZOS)

    @torch.inference_mode()
    def erase(self, image: Image.Image, mask: Image.Image, s: Settings, progress=None) -> Image.Image:
        image = image.convert("RGB")
        fill = s.action == "fill"
        what = s.extra_prompt.strip()
        if fill and not what:
            raise ValueError("En modo 'Rellenar' escribí en 'Instrucción' qué debe aparecer (en inglés).")

        x0, y0, x1, y1 = mask.getbbox()
        small = max(x1 - x0, y1 - y0) <= max(1, s.small_px) if s.small_px > 0 else False
        # En zonas chicas conviene tapar un poco más alrededor.
        grow = max(s.grow_px, 10) if max(x1 - x0, y1 - y0) < 150 else s.grow_px
        mask_big = _grow(mask, grow)

        box = self._crop_box(mask_big, image.size, s.context)
        crop = image.crop(box)
        mcrop = mask_big.crop(box)
        cw, ch = crop.size

        # Modo rápido (LaMa): no usa Qwen, ni instrucciones, ni semilla. "Rellenar con instrucción" siempre usa Qwen.
        if s.engine == "lama" and not fill:
            if self.lama is None:
                raise RuntimeError("El modo rápido (LaMa) no está disponible en esta ejecución.")
            return _paste_patch(image, self._lama_patch(crop, mcrop, s.max_side), mcrop, box, s.feather_px)

        # Manchas pequeñas: el modelo suele fallar (deja el resaltado o dibuja garabatos).
        # Se usa un relleno clásico, instantáneo y sin cargar el modelo.
        if small and not fill and cv2 is not None:
            patch = _classical_inpaint(crop, mcrop)
            if patch is not None:
                return _paste_patch(image, patch, mcrop, box, s.feather_px)

        self.load()

        scale = min(2.0, s.max_side / max(cw, ch))
        tw, th = _round32(int(cw * scale)), _round32(int(ch * scale))
        crop_in = crop.resize((tw, th), Image.LANCZOS)
        mask_in = mcrop.resize((tw, th), Image.NEAREST)

        if s.mode == "reference":
            images = [crop_in, mask_in.convert("RGB")]
            prompt = PROMPT_FILL_REFERENCE.format(what=what) if fill else PROMPT_REFERENCE
        else:
            marked = crop_in.copy()
            marked.paste(HILITE_RGB, mask=mask_in)
            images = marked
            prompt = PROMPT_FILL_OVERLAY.format(what=what) if fill else PROMPT_OVERLAY
        if what and not fill:
            prompt += " " + what

        attempts = 3 if s.mode == "overlay" else 1   # reintenta si queda magenta sin quitar
        best = None
        for attempt in range(attempts):
            def _cb(pipe, step, timestep, cb_kwargs, _a=attempt):
                if progress is not None:
                    tag = f"Intento {_a + 1}/{attempts} · " if attempt else ""
                    progress((step + 1) / s.steps, desc=f"{tag}Paso {step + 1}/{s.steps}")
                return cb_kwargs

            call_kwargs = dict(
                prompt=prompt,
                image=images,
                width=tw,
                height=th,
                num_inference_steps=s.steps,
                generator=torch.Generator("cpu").manual_seed(s.seed + attempt),
            )
            try:
                out = self.pipe(**call_kwargs, callback_on_step_end=_cb).images[0]
            except TypeError:  # versiones sin callback_on_step_end
                out = self.pipe(**call_kwargs).images[0]
            out = out.convert("RGB").resize((cw, ch), Image.LANCZOS)

            frac = _magenta_fraction(out, mcrop) if s.mode == "overlay" else 0.0
            if best is None or frac < best[0]:
                best = (frac, out)
            if frac < 0.02:
                break

        frac, out = best
        if frac >= 0.02 and not fill:
            # El modelo dejó el resaltado: se usa el relleno clásico como último recurso.
            patch = _classical_inpaint(crop, mcrop)
            if patch is not None:
                out = patch

        result = _paste_patch(image, out, mcrop, box, s.feather_px)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
