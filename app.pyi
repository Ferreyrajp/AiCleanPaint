"""Interfaz Gradio con pincel para borrar objetos con Qwen-Image-2.1."""
from __future__ import annotations

import argparse

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
args = parser.parse_args()

eraser = Eraser(args.model, quant=args.quant, offload=args.offload)

from gradio.events import Dependency

class SafeImageEditor(gr.ImageEditor):
    """ImageEditor que ignora el campo `composite`.

    En algunas versiones de Gradio el navegador envía un `composite` corrupto y el
    preprocesado falla con "OSError: unrecognized data stream contents". Esta app solo
    necesita `background` y `layers`, así que se descarta antes de que Gradio lo procese.
    """

    def preprocess(self, payload):
        if payload is not None:
            try:
                payload.composite = None
            except Exception:
                pass
        return super().preprocess(payload)
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component


def run(editor, steps, max_side, context, grow_px, feather_px, seed, mode, extra, progress=gr.Progress()):
    image, mask = Eraser.mask_from_editor(editor)
    if eraser.pipe is None:
        progress(0, desc="Cargando modelo (la primera vez descarga varios GB)...")
        eraser.load()
    s = Settings(
        steps=int(steps), max_side=int(max_side), context=float(context),
        grow_px=int(grow_px), feather_px=int(feather_px), seed=int(seed),
        mode=mode, extra_prompt=extra or "",
    )
    return eraser.erase(image, mask, s, progress=progress)


def reuse(result):
    if result is None:
        return gr.update()
    return {"background": result, "layers": [], "composite": None}


with gr.Blocks(title="Qwen Eraser") as demo:
    gr.Markdown("## Qwen Eraser\nSubí una imagen, **pintá encima del objeto** que querés borrar y presioná *Borrar*.")
    with gr.Row():
        with gr.Column():
            editor = SafeImageEditor(
                label="Imagen + pincel",
                type="pil",
                sources=["upload", "clipboard"],
                brush=gr.Brush(colors=["#ff0000"], default_color="#ff0000",
                               default_size=40, color_mode="fixed"),
                eraser=gr.Eraser(default_size=30),
                transforms=[],
                height=600,
            )
            btn = gr.Button("Borrar", variant="primary")
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
        extra = gr.Textbox(label="Instrucción extra (opcional, en inglés)", placeholder="e.g. the background is a grassy field")

    inputs = [editor, steps, max_side, context, grow_px, feather_px, seed, mode, extra]
    btn.click(run, inputs, out)
    again.click(reuse, out, editor)

if __name__ == "__main__":
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)