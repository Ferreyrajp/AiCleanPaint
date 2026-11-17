"""Comparador antes/después en HTML puro: la línea sigue al cursor (o al dedo)."""
from __future__ import annotations

import base64
import io
from html import escape

from PIL import Image

EMPTY_HTML = (
    "<div style='padding:48px 16px;text-align:center;opacity:.65'>"
    "Ejecutá una edición para ver la comparación antes / después."
    "</div>"
)

# Se ejecuta en el navegador en cada movimiento del cursor/dedo sobre el contenedor.
_MOVE_JS = (
    "var el=this,r=el.getBoundingClientRect(),"
    "x=(event.touches&&event.touches.length?event.touches[0].clientX:event.clientX),"
    "p=Math.max(0,Math.min(100,(x-r.left)/r.width*100));"
    "el.querySelector('.cmp-top').style.clipPath='inset(0 '+(100-p)+'% 0 0)';"
    "el.querySelector('.cmp-line').style.left=p+'%';"
)


# Alterna pantalla completa sobre `.cmp-wrap`. Si el navegador no la soporta (o la rechaza), usa una
# "pantalla completa" simulada que cubre toda la ventana (clase cmp-pfs).
_FS_JS = (
    "var w=this.closest('.cmp-wrap'),d=document,fe=d.fullscreenElement||d.webkitFullscreenElement;"
    "if(w&&w.classList.contains('cmp-pfs')){w.classList.remove('cmp-pfs');}"
    "else if(fe){var x=d.exitFullscreen||d.webkitExitFullscreen;if(x){x.call(d);}}"
    "else if(w){var r=w.requestFullscreen||w.webkitRequestFullscreen;"
    "if(r){var p=r.call(w);if(p&&p.catch){p.catch(function(){w.classList.add('cmp-pfs');});}}"
    "else{w.classList.add('cmp-pfs');}}"
)

# En pantalla completa la imagen se ajusta a la pantalla manteniendo la proporción (--r = ancho/alto).
_STYLE = """<style>
.cmp-wrap:fullscreen{background:#000;display:flex;align-items:center;justify-content:center;width:100vw;height:100vh;}
.cmp-wrap:fullscreen .cmp{width:min(100vw,calc(100vh*var(--r))) !important;max-width:none !important;margin:0 !important;border-radius:0 !important;}
.cmp-wrap:-webkit-full-screen{background:#000;display:flex;align-items:center;justify-content:center;width:100vw;height:100vh;}
.cmp-wrap:-webkit-full-screen .cmp{width:min(100vw,calc(100vh*var(--r))) !important;max-width:none !important;margin:0 !important;border-radius:0 !important;}
.cmp-wrap.cmp-pfs{position:fixed;inset:0;z-index:99999;background:#000;display:flex;align-items:center;justify-content:center;}
.cmp-wrap.cmp-pfs .cmp{width:min(100vw,calc(100vh*var(--r))) !important;max-width:none !important;margin:0 !important;border-radius:0 !important;}
.cmp-fs{position:absolute;bottom:10px;right:10px;z-index:5;width:36px;height:36px;border:0;border-radius:8px;cursor:pointer;
  background:rgba(0,0,0,.55);color:#fff;font:700 18px sans-serif;line-height:36px;padding:0;}
.cmp-fs:hover{background:rgba(0,0,0,.8);}
</style>"""

def _data_uri(img: Image.Image, max_side: int, quality: int) -> str:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_compare_html(before: Image.Image, after: Image.Image, max_side: int = 2000,
                       max_height: int = 620, quality: int = 90) -> str:
    """Devuelve HTML con las dos imágenes superpuestas. Izquierda de la línea = antes, derecha = después."""
    after = after.convert("RGB")
    before = before.convert("RGB")
    if before.size != after.size:
        before = before.resize(after.size, Image.LANCZOS)
    w, h = after.size
    width_css = f"min(100%, {max_height * w / h:.0f}px)"
    b64_before = _data_uri(before, max_side, quality)
    b64_after = _data_uri(after, max_side, quality)
    pill = ("position:absolute;top:10px;padding:3px 10px;border-radius:999px;font:600 12px sans-serif;"
            "color:#fff;background:rgba(0,0,0,.55);pointer-events:none;")
    return f"""{_STYLE}
<div class="cmp-wrap" style="--r:{w / h:.5f};">
<div class="cmp" style="position:relative;width:{width_css};aspect-ratio:{w}/{h};margin:0 auto;overflow:hidden;
     border-radius:8px;cursor:ew-resize;user-select:none;touch-action:none;"
     onmousemove="{escape(_MOVE_JS, quote=True)}" ontouchmove="{escape(_MOVE_JS, quote=True)}"
     ondblclick="{escape(_FS_JS, quote=True)}">
  <img src="{b64_after}" alt="Después" draggable="false"
       style="position:absolute;inset:0;width:100%;height:100%;display:block;">
  <img class="cmp-top" src="{b64_before}" alt="Antes" draggable="false"
       style="position:absolute;inset:0;width:100%;height:100%;display:block;clip-path:inset(0 50% 0 0);">
  <div class="cmp-line" style="position:absolute;top:0;bottom:0;left:50%;width:3px;margin-left:-1.5px;
       background:#fff;box-shadow:0 0 6px rgba(0,0,0,.7);pointer-events:none;">
    <div style="position:absolute;top:50%;left:50%;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;
         background:#fff;box-shadow:0 0 8px rgba(0,0,0,.6);color:#444;font:700 14px sans-serif;
         display:flex;align-items:center;justify-content:center;">&#8596;</div>
  </div>
  <span style="{pill}left:10px;">Antes</span>
  <span style="{pill}right:10px;">Después</span>
  <button type="button" class="cmp-fs" title="Pantalla completa (Esc para salir)" aria-label="Pantalla completa"
          onclick="{escape(_FS_JS, quote=True)}" ondblclick="event.stopPropagation()">&#9974;</button>
</div>
</div>
"""
