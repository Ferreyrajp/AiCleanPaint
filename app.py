"""Interfaz Gradio con pincel para borrar objetos con Qwen-Image-2.1."""
from __future__ import annotations

import os
from pathlib import Path

# Los modelos se descargan en ./models (junto a este archivo) en vez de la caché del usuario.
# Debe definirse ANTES de importar diffusers/transformers. Si ya definiste HF_HOME, se respeta.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import functools

import gradio as gr

from eraser_core import MODEL_ID, Eraser, Settings

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=MODEL_ID)
parser.add_argument("--quant", choices=["nf4", "int8", "none"], default="nf4",
                    help="nf4 recomendado para 12 GB de VRAM")
parser.add_argument("--offload", choices=["model", "sequential", "none"], default="model")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--share", action="store_true")
parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador automáticamente")
args = parser.parse_args()

eraser = Eraser(args.model, quant=args.quant, offload=args.offload)


# En algunas versiones de Gradio el navegador envía un `composite` corrupto y el preprocesado
# falla con "OSError: unrecognized data stream contents". Esta app solo necesita `background`
# y `layers`, así que se descarta `composite`. Se parchea el método (no se crea una subclase:
# Gradio trataría la subclase como componente personalizado y el editor no cargaría).
_orig_preprocess = gr.ImageEditor.preprocess


@functools.wraps(_orig_preprocess)
def _safe_preprocess(self, payload):
    if payload is not None:
        try:
            payload.composite = None
        except Exception:
            pass
    return _orig_preprocess(self, payload)


gr.ImageEditor.preprocess = _safe_preprocess


def run(editor, action, extra, steps, max_side, context, grow_px, feather_px, seed, mode, progress=gr.Progress()):
    image, mask = Eraser.mask_from_editor(editor)
    if eraser.pipe is None:
        progress(0, desc="Cargando modelo (la primera vez descarga varios GB)...")
        eraser.load()
    s = Settings(
        steps=int(steps), max_side=int(max_side), context=float(context),
        grow_px=int(grow_px), feather_px=int(feather_px), seed=int(seed),
        mode=mode, extra_prompt=extra or "",
        action="fill" if action.startswith("Rellenar") else "erase",
    )
    return eraser.erase(image, mask, s, progress=progress)


def reuse(result):
    if result is None:
        return gr.update()
    return {"background": result, "layers": [], "composite": None}


with gr.Blocks(title="AiCleanPaint") as demo:
    gr.Markdown("## AiCleanPaint\nSubí una imagen, **pintá encima del objeto** que querés borrar y presioná *Borrar*.")
    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                label="Imagen + pincel",
                type="pil",
                sources=["upload", "clipboard"],
                brush=gr.Brush(colors=["#ff00ff"], default_color="#ff00ff",
                               default_size=40, color_mode="fixed"),
                eraser=gr.Eraser(default_size=30),
                transforms=[],
                height=600,
            )
            action = gr.Radio(
                ["Borrar objeto", "Rellenar con instrucción"], value="Borrar objeto",
                label="Acción",
                info="Rellenar sirve para agregar o corregir algo (ej. una pierna que falta).",
            )
            extra = gr.Textbox(
                label="Instrucción (inglés). Obligatoria en 'Rellenar'; opcional al borrar",
                placeholder="e.g. her second leg, standing naturally next to the other leg",
            )
            btn = gr.Button("Ejecutar", variant="primary")
        with gr.Column():
            out = gr.Image(label="Resultado", type="pil", height=600, format="png")
            again = gr.Button("Usar resultado como nueva entrada")

    with gr.Accordion("Ajustes", open=False):
        steps = gr.Slider(10, 50, value=30, step=1, label="Pasos (más = mejor y más lento)")
        max_side = gr.Slider(512, 1536, value=1024, step=64,
                             label="Resolución del recorte enviado al modelo (baja si te quedás sin VRAM)")
        context = gr.Slider(0.25, 2.0, value=0.75, step=0.05, label="Contexto alrededor del objeto")
        grow_px = gr.Slider(0, 40, value=6, step=1, label="Agrandar máscara (px) — ayuda con sombras/bordes")
        feather_px = gr.Slider(0, 40, value=8, step=1, label="Suavizado del borde (px)")
        seed = gr.Number(value=42, precision=0, label="Semilla (cambiala si el resultado no convence)")
        mode = gr.Radio(["overlay", "reference"], value="overlay",
                        label="Cómo se le indica la zona al modelo",
                        info="overlay: pintado en rojo sobre la imagen. reference: máscara como segunda imagen.")

    inputs = [editor, action, extra, steps, max_side, context, grow_px, feather_px, seed, mode]
    btn.click(run, inputs, out)
    again.click(reuse, out, editor)

if __name__ == "__main__":
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        inbrowser=not args.no_browser,
                        favicon_path=str(Path(__file__).resolve().parent / "assets" / "favicon.png"))
