"""Traducción español -> inglés de la instrucción (corre en CPU, no usa VRAM)."""
from __future__ import annotations

import re
import threading

MODEL_ID = "Helsinki-NLP/opus-mt-es-en"

# Palabras frecuentes en español que casi no existen en inglés (para detectar el idioma).
_SPANISH_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "con", "sin", "que",
    "en", "y", "su", "sus", "por", "para", "es", "está", "están", "lo", "se", "más", "pero", "como",
    "otra", "otro", "otras", "otros", "hacia", "detrás", "sobre", "entre", "cerca", "lejos", "donde",
    "quiero", "agrega", "agregar", "agregá", "poné", "poner", "pon", "completa", "completá", "rellena",
    "rellená", "dibuja", "dibujá", "mismo", "misma", "mismos", "mismas", "parada", "parado", "sentada",
    "sentado", "pierna", "piernas", "brazo", "brazos", "mano", "manos", "pie", "pies", "cara", "pelo",
    "cabello", "ojo", "ojos", "cielo", "fondo", "agua", "muelle", "madera", "pasto", "césped", "persona",
    "personas", "objeto", "nube", "nubes", "sombra", "rojo", "roja", "azul", "verde", "amarillo",
    "amarilla", "negro", "negra", "blanco", "blanca", "izquierda", "izquierdo", "derecha", "derecho",
    "natural", "naturalmente", "falda", "camisa", "sombrero", "arena", "playa", "montaña", "casa",
    "ventana", "puerta", "pared", "piso", "suelo", "mesa", "silla", "perro", "gato", "auto", "coche",
}
_ACCENT_RE = re.compile(r"[áéíóúñ¿¡ü]")

_lock = threading.Lock()
_tok = None
_model = None


def looks_spanish(text: str) -> bool:
    t = text.lower()
    if _ACCENT_RE.search(t):
        return True
    words = re.findall(r"[a-záéíóúñü]+", t)
    if not words:
        return False
    hits = sum(w in _SPANISH_WORDS for w in words)
    return hits >= 1 and hits / len(words) >= 0.2


def _load():
    global _tok, _model
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()


def translate_es_en(text: str) -> str:
    import torch

    with _lock:
        _load()
        batch = _tok([text], return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            out = _model.generate(**batch, num_beams=4, max_new_tokens=160)
        return _tok.decode(out[0], skip_special_tokens=True).strip()


def prepare_instruction(text: str | None, enabled: bool = True, translator=translate_es_en):
    """Devuelve (texto_final, fue_traducido, error). Nunca lanza: si falla, devuelve el original."""
    text = (text or "").strip()
    if not text or not enabled or not looks_spanish(text):
        return text, False, None
    try:
        result = translator(text)
        return (result or text), bool(result), None
    except Exception as e:  # sin internet la primera vez, modelo no descargado, etc.
        return text, False, f"{type(e).__name__}: {e}"
