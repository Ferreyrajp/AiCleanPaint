# AiCleanPaint

Borrador de objetos con relleno generativo (tipo IOPaint) usando **Qwen-Image-2.1**, con interfaz Gradio y pincel.
Funciona en Linux y Windows, ajustado para **GPU de 12 GB (RTX 3060)**.

## Instalación

```bash
python -m venv .venv
# Linux:   source .venv/bin/activate
# Windows: .venv\Scripts\activate

# 1) PyTorch con CUDA (elegí tu versión de CUDA en https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) El resto
pip install -r requirements.txt
```

## Uso

```bash
.\run.bat  (Windows)  |  ./run.sh  (Linux)   # abre el navegador solo
python app.py                 # NF4 + offload (recomendado para 12 GB)
python app.py --quant int8    # algo más de calidad, más VRAM
python app.py --offload sequential   # si te da error de memoria (más lento)
python app.py --share         # link público temporal de Gradio
python app.py --no-browser    # sin abrir el navegador
```

Abrí http://127.0.0.1:7860, subí la imagen, pintá el objeto y presioná **Borrar**.
Los modelos se guardan en la carpeta `models/` del proyecto (no en la caché del usuario). Para cambiarlo, definí la variable `HF_HOME` antes de ejecutar.
La primera ejecución descarga el modelo (decenas de GB: unos 14 GB del transformer + el codificador de texto, en bf16).
Se recomiendan 32 GB de RAM, porque con offload parte del modelo vive en memoria del sistema.

## Cómo funciona

1. Se calcula el rectángulo alrededor de la máscara (con contexto) y solo ese recorte va al modelo, a ≤1024 px.
2. Modo `overlay`: la zona se pinta de rojo y se le pide al modelo que la quite. Modo `reference`: la máscara se envía como segunda imagen.
3. El resultado se pega **solo dentro de la máscara** (con borde suavizado) sobre la imagen original: el resto no se toca.

## Ajustes útiles

- Quedan restos o sombras: subí "Agrandar máscara" o pintá también la sombra.
- Inventa cosas: probá la otra opción de "Cómo se le indica la zona", cambiá la semilla o agregá una instrucción extra.
- Sin VRAM: bajá la resolución del recorte o usá `--offload sequential`.

## Advertencias

- La documentación pública no detalla la sintaxis exacta de máscaras del pipeline, por eso hay dos modos. Si el equipo de Qwen/Diffusers publica un parámetro de máscara dedicado, conviene usarlo en `eraser_core.py` (función `erase`).
- Los nombres de componentes para la cuantización (`transformer`, `text_encoder`) son los habituales en Diffusers; el código reintenta solo con `transformer` si falla.
- **Licencia:** el modelo usa la Qwen Research License (investigación, no comercial según lo reportado). Revisala antes de un uso comercial.
- No pude probar el código con GPU en mi entorno; sí verifiqué que la sintaxis es válida.

## Consejos para borrar bien

- Pintá **toda** la zona (sombras y bordes incluidos) y un poco por fuera.
- Después de "Usar resultado como nueva entrada" hay que volver a pintar: el pincel se reinicia.
- Si el modelo reemplaza el objeto por otra cosa, decile qué debe quedar en "Instrucción extra" (en inglés), por ejemplo: `fill with the wooden pier and the lake water`.
- Probá otra semilla o el modo `reference` si no convence.

## Agregar o corregir cosas (modo "Rellenar")

Elegí la acción **Rellenar con instrucción**, pintá la zona donde debe aparecer lo nuevo y describilo en inglés,
por ejemplo: `her left leg, standing naturally next to the other leg, same skin tone and lighting`.
Pintá un poco de la zona vecina para que el modelo tenga continuidad (por ejemplo, parte del muslo ya existente).
