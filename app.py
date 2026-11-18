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
import inspect
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

import click_select
import history
import text_select
from compare_view import EMPTY_HTML, build_compare_html
from eraser_core import MODEL_ID, Eraser, Settings
from lama_inpaint import APPROX_MB as LAMA_MB, LamaInpainter
from detector import APPROX_MB as GD_MB, DEFAULT_SIZE as GD_DEFAULT, MODELS as GD_MODELS, Detector, build_prompt
from segmenter import APPROX_MB, DEFAULT_SIZE, MODELS, Segmenter
from translate import prepare_instruction, translate_es_en

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=MODEL_ID)
parser.add_argument("--quant", choices=["nf4", "int8", "none"], default="nf4",
                    help="nf4 recomendado para 12 GB de VRAM")
parser.add_argument("--offload", choices=["model", "sequential", "none"], default="model")
parser.add_argument("--port", type=int, default=None,
                    help="puerto fijo; si se omite, usa el primero libre desde 7860")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--share", action="store_true")
parser.add_argument("--sam-device", choices=["cpu", "cuda"], default="cpu",
                    help="dispositivo para SAM (selección con clic). cpu evita competir por VRAM con Qwen")
parser.add_argument("--lama-device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="dispositivo para el modo rápido (LaMa). auto = GPU si hay; si falta memoria, cae a CPU")
parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador automáticamente")
args = parser.parse_args()

eraser = Eraser(args.model, quant=args.quant, offload=args.offload)
# Modo rápido (LaMa): opcional y apagado por defecto. No descarga ni carga nada hasta que se lo usa.
lama = LamaInpainter(device=args.lama_device)
eraser.lama = lama

# --- Ajustes recordados entre sesiones (semilla, selección con clic) -------------------
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
DEFAULT_SEED = 42


def _read_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_setting(key, default=None):
    return _read_settings().get(key, default)


def save_setting(key, value) -> None:
    try:
        data = _read_settings()
        data[key] = value
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass  # no es crítico: si no se puede guardar, la app sigue funcionando


def load_seed() -> int:
    try:
        return int(load_setting("seed", DEFAULT_SEED))
    except Exception:
        return DEFAULT_SEED


def save_seed(value) -> None:
    if value is not None:
        save_setting("seed", int(value))


# Selección con un clic (SAM 2.1): opcional y apagada por defecto. No se importa ni se descarga
# nada hasta que se la habilita en Ajustes y se usa por primera vez.
_saved_size = load_setting("sam_size", DEFAULT_SIZE)
sam = Segmenter(size=_saved_size if _saved_size in MODELS else DEFAULT_SIZE, device=args.sam_device)
_saved_gd = load_setting("gd_size", GD_DEFAULT)
detector = Detector(size=_saved_gd if _saved_gd in GD_MODELS else GD_DEFAULT, device=args.sam_device)


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
        translate, autosave, hist, engine="Calidad (Qwen)", progress=gr.Progress()):
    image, mask = Eraser.mask_from_editor(editor)
    fill = action.startswith("Rellenar")
    wants_fast = str(engine).startswith("Rápido")
    if wants_fast and fill:
        gr.Info("«Rellenar con instrucción» siempre usa Qwen: el modo rápido no entiende instrucciones.")
    use_lama = wants_fast and not fill

    if use_lama:
        # Modo rápido: no se carga Qwen, no hay traducción ni semilla.
        instruction = ""
        shown = "Modo rápido (LaMa): no usa instrucción."
        if (extra or "").strip():
            gr.Info("El modo rápido (LaMa) no usa la instrucción; se ignora.")
        if not lama.loaded:
            progress(0, desc="Cargando LaMa (la primera vez descarga ~%d MB)..." % LAMA_MB)
    else:
        if eraser.pipe is None:
            progress(0, desc="Cargando modelo (la primera vez descarga varios GB)...")
            eraser.load()
        instruction, translated, err = prepare_instruction(extra, enabled=translate)
        if err:
            gr.Warning("No se pudo traducir la instrucción; se envía tal cual. Detalle: " + err[:200])
        shown = instruction
    s = Settings(
        steps=int(steps), max_side=int(max_side), context=float(context),
        grow_px=int(grow_px), feather_px=int(feather_px), seed=int(seed),
        mode=mode, extra_prompt=instruction,
        action="fill" if fill else "erase",
        small_px=int(small_px),
        engine="lama" if use_lama else "qwen",
    )
    try:
        result = eraser.erase(image, mask, s, progress=progress)
    except Exception as e:
        if use_lama:  # descarga, hash, carga o inferencia de LaMa: mensaje claro en la interfaz
            raise gr.Error(f"Modo rápido (LaMa): {e}") from e
        raise
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
    return result, compare, shown, hist, info


def reuse(hist):
    """Carga en el editor la versión actual del historial (la misma que se ve en «Resultado»)."""
    _, img = history.step(hist, 0)
    if img is None:
        gr.Info("Todavía no hay un resultado para usar.")
        return gr.update()
    return {"background": img, "layers": [], "composite": None}


def _result_image_kwargs() -> dict:
    """Botones de la esquina del resultado: descargar y pantalla completa (sin «compartir»).
    Gradio 6 usa `buttons`; Gradio 5 usa `show_*_button`."""
    params = inspect.signature(gr.Image.__init__).parameters
    if "buttons" in params:
        return {"buttons": ["download", "fullscreen"]}
    kw = {}
    if "show_download_button" in params:
        kw["show_download_button"] = True
    if "show_share_button" in params:
        kw["show_share_button"] = False
    if "show_fullscreen_button" in params:
        kw["show_fullscreen_button"] = True
    return kw


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


def _editor_background(editor):
    if editor is None or editor.get("background") is None:
        raise gr.Error("Primero subí una imagen en el editor.")
    return editor["background"].convert("RGB")


def sam_load(editor, state, size, progress=gr.Progress()):
    image = _editor_background(editor)
    progress(0, desc="Preparando SAM (la primera vez descarga el modelo)...")
    try:
        sam.set_size(size)
        state = click_select.start(sam, image)
    except Exception as e:
        raise gr.Error(f"No se pudo preparar SAM: {e}")
    return state["base"], state, click_select.describe(state)


def sam_click(state, kind, cand, evt: gr.SelectData):
    if state["base"] is None or not sam.ready:
        gr.Warning("Primero apretá «1 · Cargar imagen del editor».")
        return gr.update(), state, click_select.describe(state)
    try:
        x, y = evt.index[0], evt.index[1]
        state, preview = click_select.add_point(state, sam, x, y, positive=kind.startswith("Incluir"),
                                                choice=cand)
    except ValueError as e:
        gr.Warning(str(e))
        return gr.update(), state, click_select.describe(state)
    except Exception as e:
        raise gr.Error(f"SAM falló: {e}")
    return preview, state, click_select.describe(state)


def sam_refresh(state, cand):
    if state["base"] is None or not state["points"]:
        return gr.update(), click_select.describe(state)
    return click_select.render(state, cand), click_select.describe(state)


def sam_undo(state, cand):
    try:
        state, preview = click_select.remove_last_point(state, sam, cand)
    except Exception as e:
        raise gr.Error(f"SAM falló: {e}")
    return (preview if preview is not None else gr.update()), state, click_select.describe(state)


def sam_clear(state, cand):
    state, preview = click_select.clear(state, cand)
    return (preview if preview is not None else gr.update()), state, click_select.describe(state)


def sam_use(state, cand):
    try:
        value = click_select.to_editor_value(state, cand)
    except ValueError as e:
        gr.Warning(str(e))
        return gr.update(), click_select.describe(state)
    gr.Info("Máscara cargada en el editor. Revisala, retocala con el pincel si hace falta y apretá «Ejecutar».")
    return value, "Máscara enviada al editor."


def sam_toggle(enabled):
    save_setting("sam_enabled", bool(enabled))
    return gr.update(visible=bool(enabled))


def sam_size_changed(size):
    save_setting("sam_size", size)
    sam.set_size(size)


def sam_restore():
    enabled = bool(load_setting("sam_enabled", False))
    size = load_setting("sam_size", DEFAULT_SIZE)
    return enabled, gr.update(visible=enabled), (size if size in MODELS else DEFAULT_SIZE)


def lama_toggle(enabled):
    save_setting("lama_enabled", bool(enabled))
    return gr.update(visible=bool(enabled), value="Calidad (Qwen)")  # al apagarlo siempre se vuelve a Qwen


def lama_restore():
    enabled = bool(load_setting("lama_enabled", False))
    return enabled, gr.update(visible=enabled)


def gd_size_changed(size):
    save_setting("gd_size", size)
    detector.set_size(size)


def gd_restore():
    size = load_setting("gd_size", GD_DEFAULT)
    return size if size in GD_MODELS else GD_DEFAULT


def _txt_labels(query, translate_on):
    return text_select.to_labels(query, translator=translate_es_en if translate_on else None)


def txt_search(editor, query, thr, translate_on, progress=gr.Progress()):
    image = _editor_background(editor)
    labels = _txt_labels(query, translate_on)
    if not labels:
        gr.Warning("Escribí qué querés seleccionar (ej. personas, perro, botella).")
        return (gr.update(), text_select.new_state(), gr.update(choices=[], value=[]),
                "Escribí qué querés seleccionar y apretá **Buscar**.")
    progress(0, desc="Buscando (la primera vez descarga el detector y SAM)...")
    try:
        state = text_select.search(image, labels, detector, sam, threshold=float(thr))
    except Exception as e:
        raise gr.Error(f"No se pudo buscar: {e}")
    return (text_select.render(state), state,
            gr.update(choices=text_select.choices(state), value=state["selected"]),
            text_select.describe(state, build_prompt(labels)))


def txt_people(editor, thr, translate_on, progress=gr.Progress()):
    return ("personas",) + txt_search(editor, "personas", thr, translate_on, progress)


def txt_refresh(state, selected):
    if state["base"] is None or not state["dets"]:
        return gr.update(), state, gr.update()
    state = text_select.set_selected(state, selected)
    return text_select.render(state), state, text_select.describe(state, build_prompt(state["labels"]))


def txt_use(state):
    try:
        value = text_select.to_editor_value(state)
    except ValueError as e:
        gr.Warning(str(e))
        return gr.update(), text_select.describe(state, build_prompt(state["labels"]))
    gr.Info("Selección cargada en el editor. Revisala, retocala con el pincel si hace falta y apretá «Ejecutar».")
    return value, "Selección enviada al editor."


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
            engine = gr.Radio(
                ["Calidad (Qwen)", "Rápido (LaMa)"], value="Calidad (Qwen)", label="Motor",
                info="Rápido (LaMa) sirve para borrar cosas chicas o fondos simples en pocos segundos; "
                     "para escenas complejas o «Rellenar» usá Calidad.",
                visible=bool(load_setting("lama_enabled", False)),
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
                    out = gr.Image(label="Resultado", type="pil", height=600, format="png", interactive=False,
                                   **_result_image_kwargs())
                with gr.Tab("Antes / después"):
                    compare = gr.HTML(EMPTY_HTML)
            again = gr.Button("Usar resultado como nueva entrada")
            with gr.Row():
                undo_btn = gr.Button("↶ Deshacer")
                redo_btn = gr.Button("↷ Rehacer")
                open_btn = gr.Button("📂 Abrir carpeta de resultados")
            status = gr.Markdown("Sin historial todavía.")

    with gr.Accordion("🖱️ Selección de objetos (SAM)", open=True,
                      visible=bool(load_setting("sam_enabled", False))) as sam_box:
        with gr.Tabs():
            with gr.Tab("Por clics"):
                sam_state = gr.State(click_select.new_state())
                gr.Markdown("**1)** Cargá la imagen del editor. **2)** Hacé clic sobre el objeto; más clics refinan "
                            "la selección. **3)** «Usar esta máscara» la pasa al editor, donde podés retocarla con "
                            "el pincel y apretar «Ejecutar».")
                with gr.Row():
                    sam_load_btn = gr.Button("1 · Cargar imagen del editor")
                    sam_kind = gr.Radio(["Incluir (+)", "Excluir (−)"], value="Incluir (+)", label="Tipo de clic",
                                        info="Excluir quita de la selección lo que tocaste.")
                    sam_cand = gr.Radio(click_select.CAND_CHOICES, value="Auto", label="Máscara sugerida",
                                        info="Si no es la que querías, probá 1, 2 o 3.")
                sam_img = gr.Image(label="Hacé clic sobre el objeto", type="pil", height=520)
                with gr.Row():
                    sam_undo_btn = gr.Button("Quitar último clic")
                    sam_clear_btn = gr.Button("Limpiar clics")
                    sam_use_btn = gr.Button("3 · Usar esta máscara", variant="primary")
                sam_msg = gr.Markdown(click_select.describe(click_select.new_state()))

            with gr.Tab("Por texto"):
                txt_state = gr.State(text_select.new_state())
                gr.Markdown("Escribí **qué querés seleccionar** (en español). Se detecta, se numera de izquierda a "
                            "derecha y elegís cuáles usar. *La primera búsqueda descarga el detector "
                            f"(~{GD_MB[GD_DEFAULT]} MB) y SAM.*")
                with gr.Row():
                    txt_query = gr.Textbox(value="personas", label="¿Qué querés seleccionar?", scale=3,
                                           placeholder="ej. personas, perro, botella (podés separar con comas)")
                    txt_thr = gr.Slider(0.10, 0.60, value=0.30, step=0.05, label="Sensibilidad",
                                        info="Menor = detecta más cosas (y se equivoca más).")
                with gr.Row():
                    txt_people_btn = gr.Button("👥 Todas las personas")
                    txt_search_btn = gr.Button("🔎 Buscar", variant="primary")
                txt_img = gr.Image(label="Resultados numerados", type="pil", height=520, interactive=False)
                txt_sel = gr.CheckboxGroup(choices=[], value=[], label="Elegí cuáles seleccionar")
                txt_use_btn = gr.Button("Usar la selección", variant="primary")
                txt_msg = gr.Markdown(text_select.describe(text_select.new_state()))

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
        lama_enable = gr.Checkbox(
            value=bool(load_setting("lama_enabled", False)),
            label="Habilitar modo rápido (LaMa) para borrar",
            info=f"Apagado por defecto. Agrega el selector «Motor». La primera vez que lo uses descarga ~{LAMA_MB} MB.")
        sam_enable = gr.Checkbox(
            value=bool(load_setting("sam_enabled", False)),
            label="Habilitar selección de objetos (SAM 2.1): por clics y por texto",
            info="Apagado por defecto. Cada modelo se descarga recién la primera vez que lo usás (ver tamaños abajo).")
        _size0 = load_setting("sam_size", DEFAULT_SIZE)
        sam_size = gr.Dropdown(
            choices=[(f"{k} · ~{APPROX_MB[k]} MB" + (" (recomendado)" if k == DEFAULT_SIZE else ""), k)
                     for k in MODELS],
            value=_size0 if _size0 in MODELS else DEFAULT_SIZE, label="Tamaño del modelo SAM",
            info="Más grande = bordes más finos pero más lento. Cambiarlo requiere volver a cargar la imagen.")
        _gd0 = load_setting("gd_size", GD_DEFAULT)
        gd_size = gr.Dropdown(
            choices=[(f"{k} · ~{GD_MB[k]} MB" + (" (recomendado)" if k == GD_DEFAULT else ""), k) for k in GD_MODELS],
            value=_gd0 if _gd0 in GD_MODELS else GD_DEFAULT, label="Tamaño del detector por texto (Grounding DINO)",
            info="Se usa en la pestaña «Por texto». Base detecta mejor objetos chicos pero pesa y tarda más.")
        seed = gr.Number(value=load_seed(), precision=0, label="Semilla (cambiala si el resultado no convence)")
        mode = gr.Radio(["overlay", "reference"], value="overlay",
                        label="Cómo se le indica la zona al modelo",
                        info="overlay: zona resaltada en magenta sobre la imagen. reference: máscara como segunda imagen.")

    inputs = [editor, action, extra, steps, max_side, context, grow_px, feather_px, seed, mode, small_px,
              translate, autosave, hist, engine]
    btn.click(run, inputs, [out, compare, sent, hist, status])
    seed.change(save_seed, seed, None, show_progress="hidden")
    demo.load(load_seed, None, seed)  # al abrir/refrescar la página, usa la última semilla guardada
    again.click(reuse, hist, editor)
    nav_outputs = [editor, out, compare, hist, status]
    undo_btn.click(undo, hist, nav_outputs)
    redo_btn.click(redo, hist, nav_outputs)
    open_btn.click(open_outputs_folder, None, None)

    # Selección con un clic (SAM)
    sam_load_btn.click(sam_load, [editor, sam_state, sam_size], [sam_img, sam_state, sam_msg])
    sam_img.select(sam_click, [sam_state, sam_kind, sam_cand], [sam_img, sam_state, sam_msg])
    sam_cand.change(sam_refresh, [sam_state, sam_cand], [sam_img, sam_msg])
    sam_undo_btn.click(sam_undo, [sam_state, sam_cand], [sam_img, sam_state, sam_msg])
    sam_clear_btn.click(sam_clear, [sam_state, sam_cand], [sam_img, sam_state, sam_msg])
    sam_use_btn.click(sam_use, [sam_state, sam_cand], [editor, sam_msg])
    sam_enable.change(sam_toggle, sam_enable, sam_box)
    sam_size.change(sam_size_changed, sam_size, None)
    demo.load(sam_restore, None, [sam_enable, sam_box, sam_size])

    # Selección por texto
    _txt_outputs = [txt_img, txt_state, txt_sel, txt_msg]
    txt_search_btn.click(txt_search, [editor, txt_query, txt_thr, translate], _txt_outputs)
    txt_query.submit(txt_search, [editor, txt_query, txt_thr, translate], _txt_outputs)
    txt_people_btn.click(txt_people, [editor, txt_thr, translate], [txt_query] + _txt_outputs)
    txt_sel.input(txt_refresh, [txt_state, txt_sel], [txt_img, txt_state, txt_msg])
    txt_use_btn.click(txt_use, txt_state, [editor, txt_msg])
    gd_size.change(gd_size_changed, gd_size, None)
    demo.load(gd_restore, None, gd_size)

    # Modo rápido (LaMa)
    lama_enable.change(lama_toggle, lama_enable, engine)
    demo.load(lama_restore, None, [lama_enable, engine])

if __name__ == "__main__":
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        inbrowser=not args.no_browser,
                        favicon_path=str(Path(__file__).resolve().parent / "assets" / "favicon.png"))
