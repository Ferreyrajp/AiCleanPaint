"""Interfaz Gradio con pincel para borrar objetos con Qwen-Image-2.1."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Los modelos se descargan en ./models (junto a este archivo) en vez de la caché del usuario.
# Debe definirse ANTES de importar diffusers/transformers. Si ya definiste HF_HOME, se respeta.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Archivos temporales de Gradio dentro del proyecto. Se vacía al iniciar para no reutilizar
# archivos dañados de sesiones anteriores (causa de "broken PNG file").
if "GRADIO_TEMP_DIR" not in os.environ:
    _tmp = Path(__file__).resolve().parent / ".gradio_tmp"
    shutil.rmtree(_tmp, ignore_errors=True)
    _tmp.mkdir(exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(_tmp)

import argparse
import dataclasses
import functools
import json
import subprocess
import sys
import time

import gradio as gr

# --- Parche para Windows ---------------------------------------------------------------
# Gradio guarda cada subida en <cache>/<hash del contenido>/<nombre>. El editor vuelve a subir
# el mismo fondo en cada cambio, así que el destino ya existe: en Windows os.rename falla y el
# copiado de respaldo choca con el archivo en uso (WinError 32), pudiendo dejar un PNG dañado.
# Como el hash coincide, el contenido es idéntico: se descarta el temporal y listo.
try:
    from gradio import route_utils as _route_utils

    class _ShutilProxy:
        def __getattr__(self, name):
            return getattr(shutil, name)

        @staticmethod
        def move(src, dst, *args, **kwargs):
            try:
                if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                    try:
                        os.remove(src)
                    except OSError:
                        pass
                    return str(dst)
            except OSError:
                pass
            last = None
            for _ in range(3):
                try:
                    return shutil.move(src, dst, *args, **kwargs)
                except (PermissionError, FileExistsError) as e:
                    last = e
                    try:
                        os.replace(src, dst)  # reemplazo atómico
                        return str(dst)
                    except OSError:
                        time.sleep(0.3)
            raise last

    _route_utils.shutil = _ShutilProxy()
except Exception:  # si cambia la estructura interna de Gradio, la app sigue funcionando
    pass

import history
from compare_view import EMPTY_HTML, build_compare_html
from eraser_core import MODEL_ID, Eraser, Settings
from translate import prepare_instruction

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=MODEL_ID)
parser.add_argument("--quant", choices=["nf4", "int8", "none"], default="nf4",
                    help="nf4 recomendado para 12 GB de VRAM")
parser.add_argument("--offload", choices=["model", "sequential", "none"], default="model")
parser.add_argument("--port", type=int, default=None,
                    help="puerto fijo; si se omite, usa el primero libre desde 7860")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--share", action="store_true")
parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador automáticamente")
args = parser.parse_args()

eraser = Eraser(args.model, quant=args.quant, offload=args.offload)

# --- Ajustes recordados entre sesiones (hoy: la semilla) -------------------------------
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
DEFAULT_SEED = 42


def load_seed() -> int:
    try:
        return int(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("seed", DEFAULT_SEED))
    except Exception:
        return DEFAULT_SEED


def save_seed(value) -> None:
    if value is None:
        return
    try:
        data = {}
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        data["seed"] = int(value)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass  # no es crítico: si no se puede guardar, la app sigue funcionando


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
    last_err = None
    for _ in range(3):  # el archivo temporal puede estar aún escribiéndose: se reintenta
        try:
            return _orig_preprocess(self, payload)
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise gr.Error(
        "La imagen del editor llegó dañada. Quitala con la ✕ del editor y volvé a cargarla "
        f"(detalle: {type(last_err).__name__})."
    )


gr.ImageEditor.preprocess = _safe_preprocess


def run(editor, action, extra, steps, max_side, context, grow_px, feather_px, seed, mode, small_px,
        translate, autosave, hist, progress=gr.Progress()):
    image, mask = Eraser.mask_from_editor(editor)
    if eraser.pipe is None:
        progress(0, desc="Cargando modelo (la primera vez descarga varios GB)...")
        eraser.load()
    instruction, translated, err = prepare_instruction(extra, enabled=translate)
    if err:
        gr.Warning("No se pudo traducir la instrucción; se envía tal cual. Detalle: " + err[:200])
    s = Settings(
        steps=int(steps), max_side=int(max_side), context=float(context),
        grow_px=int(grow_px), feather_px=int(feather_px), seed=int(seed),
        mode=mode, extra_prompt=instruction,
        action="fill" if action.startswith("Rellenar") else "erase",
        small_px=int(small_px),
    )
    result = eraser.erase(image, mask, s, progress=progress)
    try:
        compare = build_compare_html(image, result)
    except Exception:  # el comparador es accesorio: nunca debe romper la edición
        compare = EMPTY_HTML
    # Historial + guardado automático (accesorios: un fallo acá no debe perder el resultado)
    try:
        meta = dataclasses.asdict(s)
        meta.update({"app": "AiCleanPaint", "instruction_original": extra or "",
                     "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        hist, saved = history.push(hist, image, result, meta, autosave=bool(autosave))
        info = history.status(hist, saved if autosave else None)
    except Exception as e:
        info = f"No se pudo guardar el historial ({type(e).__name__}: {e})"
    return result, compare, instruction, hist, info


def reuse(result):
    if result is None:
        return gr.update()
    return {"background": result, "layers": [], "composite": None}


def _navigate(hist, delta):
    hist, img = history.step(hist, delta)
    if img is None:
        gr.Info("No hay más versiones en esa dirección.")
        return gr.update(), gr.update(), gr.update(), hist, history.status(hist)
    pair = history.pair_paths(hist)
    compare = EMPTY_HTML
    if pair is not None:
        try:
            from PIL import Image as _Image
            with _Image.open(pair[0]) as b, _Image.open(pair[1]) as r:
                compare = build_compare_html(b.convert("RGB"), r.convert("RGB"))
        except Exception:
            compare = EMPTY_HTML
    editor_value = {"background": img, "layers": [], "composite": None}
    return editor_value, img, compare, hist, history.status(hist)


def undo(hist):
    return _navigate(hist, -1)


def redo(hist):
    return _navigate(hist, +1)


def open_outputs_folder():
    history.OUTPUTS.mkdir(exist_ok=True)
    try:
        if sys.platform.startswith("win"):
            os.startfile(history.OUTPUTS)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(history.OUTPUTS)])
        else:
            subprocess.Popen(["xdg-open", str(history.OUTPUTS)])
    except Exception as e:
        gr.Warning(f"No se pudo abrir la carpeta: {e}. Está en: {history.OUTPUTS}")


with gr.Blocks(title="AiCleanPaint") as demo:
    hist = gr.State(history.new_state())
    gr.Markdown("## AiCleanPaint\nSubí una imagen, **pintá encima del objeto** que querés borrar, seleccioná **Borrar objeto** y presioná **Ejecutar**.")
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
                label="Instrucción (podés escribir en español). Obligatoria en 'Rellenar'; opcional al borrar",
                placeholder="ej. su otra pierna, parada de forma natural junto a la otra",
            )
            sent = gr.Textbox(
                label="Instrucción enviada al modelo",
                interactive=False,
                info="Si escribís en español se traduce al inglés; revisá que el sentido sea el correcto.",
            )
            btn = gr.Button("Ejecutar", variant="primary")
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Resultado"):
                    out = gr.Image(label="Resultado", type="pil", height=600, format="png")
                with gr.Tab("Antes / después"):
                    compare = gr.HTML(EMPTY_HTML)
            again = gr.Button("Usar resultado como nueva entrada")
            with gr.Row():
                undo_btn = gr.Button("↶ Deshacer")
                redo_btn = gr.Button("↷ Rehacer")
                open_btn = gr.Button("📂 Abrir carpeta de resultados")
            status = gr.Markdown("Sin historial todavía.")

    with gr.Accordion("Ajustes", open=False):
        steps = gr.Slider(10, 50, value=30, step=1, label="Pasos (más = mejor y más lento)")
        max_side = gr.Slider(512, 1536, value=1024, step=64,
                             label="Resolución del recorte enviado al modelo (baja si te quedás sin VRAM)")
        context = gr.Slider(0.25, 2.0, value=0.75, step=0.05, label="Contexto alrededor del objeto")
        grow_px = gr.Slider(0, 40, value=6, step=1, label="Agrandar máscara (px) — ayuda con sombras/bordes")
        feather_px = gr.Slider(0, 40, value=8, step=1, label="Suavizado del borde (px)")
        small_px = gr.Slider(0, 200, value=64, step=4,
                             label="Manchas pequeñas: usar relleno rápido si miden menos de (px, 0 = nunca)",
                             info="Para detalles diminutos el modelo falla; esto los resuelve sin usar la IA (requiere opencv).")
        translate = gr.Checkbox(
            value=True, label="Traducir la instrucción de español a inglés",
            info="Desactivala para enviarla tal cual. La primera vez descarga un modelo chico (~300 MB).")
        autosave = gr.Checkbox(
            value=True, label="Guardar cada resultado en la carpeta outputs/",
            info="PNG sin pérdida con fecha y hora. Si lo desactivás, el historial sigue funcionando pero no se guarda nada permanente.")
        seed = gr.Number(value=load_seed(), precision=0, label="Semilla (cambiala si el resultado no convence)")
        mode = gr.Radio(["overlay", "reference"], value="overlay",
                        label="Cómo se le indica la zona al modelo",
                        info="overlay: zona resaltada en magenta sobre la imagen. reference: máscara como segunda imagen.")

    inputs = [editor, action, extra, steps, max_side, context, grow_px, feather_px, seed, mode, small_px,
              translate, autosave, hist]
    btn.click(run, inputs, [out, compare, sent, hist, status])
    seed.change(save_seed, seed, None, show_progress="hidden")
    demo.load(load_seed, None, seed)  # al abrir/refrescar la página, usa la última semilla guardada
    again.click(reuse, out, editor)
    nav_outputs = [editor, out, compare, hist, status]
    undo_btn.click(undo, hist, nav_outputs)
    redo_btn.click(redo, hist, nav_outputs)
    open_btn.click(open_outputs_folder, None, None)

if __name__ == "__main__":
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        inbrowser=not args.no_browser,
                        favicon_path=str(Path(__file__).resolve().parent / "assets" / "favicon.png"))
