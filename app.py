from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import mercadopago
import pandas as pd
import requests
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
    session,
    make_response,
)
from flask_cors import CORS
from openai import OpenAI
from PIL import Image, ImageOps, ImageDraw

# ==============================
# Configuración
# ==============================

class Config:
    # Seguridad y servidor
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_secret_key")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024
    SESSION_COOKIE_NAME = "inertiax_session"

    # Archivos
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}

    # Pago / IA / Dominio
    DOMAIN_URL = os.getenv("DOMAIN_URL", "https://inertiax-calculadora-1.onrender.com")
    MP_ACCESS_TOKEN = os.getenv("MP_ACCESS_TOKEN")
    
    # OpenAI Config (tu API key)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # UI
    LOGO_URL = os.getenv("LOGO_URL", "https://scontent-scl3-1.cdninstagram.com/v/t51.2885-19/523933037_17845198551536603_8934147041556657694_n.jpg?efg=eyJ2ZW5jb2RlX3RhZyI6InByb2ZpbGVfcGljLmRqYW5nby4xMDgwLmMyIn0&_nc_ht=scontent-scl3-1.cdninstagram.com&_nc_cat=111&_nc_oc=Q6cZ2QHOy6rgcBc7EW5ZszSp4lJTdyOpbiDr73ZBLQu3R0fFLrhnThZGWbbGejuqVpYJ9a4&_nc_ohc=hYgEXbr2xVoQ7kNvwGzhiGQ&_nc_gid=HUhI8RtTJJyKdKSj0v0qOQ&edm=AP4sbd4BAAAA&ccb=7-5&oh=00_Afd9hP8K3ACP2osMWfw_db9f5k_-SUItXzjPX0kUjnd79A&oe=68F4E13F&_nc_sid=7a9f4b")

    # Códigos de invitado
    GUEST_CODES = set(
        (os.getenv("GUEST_CODES") or "INERTIAXVIP2025,ENTRENADORPRO,INVEXORTEST").split(",")
    )


# ==============================
# App y extensiones
# ==============================

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]

# CORS para permitir uso embebido (Shopify)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Logging bonito
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("inertiax")

# Clientes externos
mp = mercadopago.SDK(app.config["MP_ACCESS_TOKEN"]) if app.config["MP_ACCESS_TOKEN"] else None
ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"])


# ==============================
# Modelos de estado (en disco)
# ==============================

@dataclass
class JobState:
    job_id: str
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    payment_ok: bool = False
    meta_path: Optional[str] = None


def _job_dir(job_id: str) -> str:
    d = os.path.join(app.config["UPLOAD_DIR"], job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _job_meta_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "meta.json")


def _save_meta(job_id: str, meta: Dict) -> str:
    p = _job_meta_path(job_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return p


def _load_meta(job_id: str) -> Dict:
    p = _job_meta_path(job_id)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_job() -> str:
    """Obtiene/crea un job_id mínimo en cookie de sesión."""
    if "job_id" not in session:
        session["job_id"] = uuid.uuid4().hex
    return session["job_id"]


def _allowed_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in app.config["ALLOWED_EXT"]


# ==============================
# Utilidades básicas
# ==============================

def parse_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def fetch_circular_logo(url: str, size_px: int = 220) -> BytesIO:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGBA")
    img = ImageOps.fit(img, (size_px, size_px), method=Image.LANCZOS)
    mask = Image.new("L", (size_px, size_px), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size_px, size_px), fill=255)
    img.putalpha(mask)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ==============================
# NÚCLEO IA - TODO gestionado por IA
# ==============================

def run_complete_ai_analysis(df: pd.DataFrame, meta: dict) -> dict:
    """
    PASA TODOS LOS DATOS A LA IA Y QUE ELLA GESTIONE TODO:
    - Análisis estadístico
    - Gráficos
    - Interpretación
    - Reporte completo
    """
    
    # Preparar datos completos para la IA
    n_rows, n_cols = df.shape
    data_preview = df.head(1000).to_csv(index=False)  # Máximo 1000 filas para contexto
    
    # Información estadística básica para contexto
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    stats_info = ""
    if numeric_cols:
        stats_info = "Estadísticas básicas:\n"
        for col in numeric_cols[:5]:  # Máximo 5 columnas para no saturar
            stats_info += f"{col}: media={df[col].mean():.3f}, std={df[col].std():.3f}, min={df[col].min():.3f}, max={df[col].max():.3f}\n"

    # Construir contexto completo
    contexto = f"""
DATOS DEL ANÁLISIS:
- Atleta: {meta.get('nombre', 'No especificado')}
- Tipo de datos: {meta.get('tipo_datos', 'No especificado')}
- Propósito: {meta.get('proposito', 'No especificado')}
- Detalles: {meta.get('detalles', 'No especificados')}
- Dataset: {n_rows} filas × {n_cols} columnas
- Columnas: {', '.join(df.columns.tolist())}

{stats_info}
"""

    # PROMPT MAESTRO - La IA hace TODO
    system_prompt = """
Eres un analista deportivo de élite ESPECIALIZADO en biomecánica, fisiología del ejercicio y ciencia del deporte. 
Tu tarea es realizar un análisis COMPLETO de los datos y generar un reporte profesional que incluya TODO:

**TAREAS OBLIGATORIAS:**

1. 📊 **ANÁLISIS ESTADÍSTICO COMPLETO:**
   - Estadísticas descriptivas por variable
   - Identificación de outliers y patrones
   - Correlaciones entre variables clave
   - Análisis de tendencias temporales

2. 📈 **GENERACIÓN DE GRÁFICOS (en código Python):**
   - Debes GENERAR código Python completo con matplotlib/seaborn
   - Gráficos ESPECÍFICOS para análisis deportivo
   - Formato: Código listo para ejecutar que retorne imágenes PNG
   - Incluir: evoluciones temporales, relaciones fuerza-velocidad, análisis de fatiga, etc.

3. 🏋️ **INTERPRETACIÓN DEPORTIVA PROFUNDA:**
   - Implicaciones biomecánicas
   - Análisis de rendimiento
   - Identificación de fortalezas/debilidades
   - Recomendaciones específicas de entrenamiento

4. 📋 **REPORTE PROFESIONAL ESTRUCTURADO:**
   - Resumen ejecutivo
   - Hallazgos clave con datos específicos
   - Recomendaciones accionables
   - Plan de seguimiento

**INSTRUCCIONES DE FORMATO:**

Responde EXACTAMENTE en este formato JSON:

{
  "analysis": "Texto completo del análisis...",
  "python_code_for_charts": "Código Python completo para generar gráficos...",
  "charts_description": "Descripción de qué muestra cada gráfico...",
  "recommendations": "Lista de recomendaciones específicas..."
}

El código Python debe:
- Usar matplotlib/seaborn
- Recibir un DataFrame 'df' como input
- Retornar una lista de objetos BytesIO con las imágenes PNG
- Ser ejecutable directamente
"""

    user_prompt = f"""
{contexto}

DATOS CRUDOS (primeras filas):
```csv
{data_preview}
INSTRUCCIONES FINALES:

Analiza TODOS los datos en profundidad

Genera código Python para gráficos deportivos RELEVANTES

Proporciona un análisis que un entrenador profesional pueda usar inmediatamente

Sé específico, técnico y aplicado

Incluye valores numéricos concretos y hallazgos accionables

¡La calidad del análisis depende completamente de ti! Entrega un trabajo de nivel profesional.
"""

    try:
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        log.error(f"Error en análisis IA completo: {e}")

        return {
            "analysis": f"Error en el análisis: {str(e)}",
            "python_code_for_charts": "",
            "charts_description": "",
            "recommendations": []
        }


def execute_ai_charts_code(python_code: str, df: pd.DataFrame) -> List[BytesIO]:
    """Ejecuta el código Python generado por la IA para producir gráficos."""
    if not python_code.strip():
        return []

    try:
        # Crear un entorno seguro para ejecutar el código
        local_vars = {'df': df, 'BytesIO': BytesIO, 'plt': None, 'sns': None}
        
        # Importar librerías dentro del entorno controlado
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        
        local_vars.update({
            'plt': plt,
            'sns': sns,
            'np': np,
            'pd': pd
        })
        
        # Ejecutar el código generado por la IA
        exec(python_code, local_vars)
        
        # Obtener los gráficos resultantes
        charts = local_vars.get('charts', [])
        if not isinstance(charts, list):
            charts = []
            
        return charts
    
    except Exception as e:
        log.error(f"Error ejecutando código de gráficos IA: {e}")
        return []
    

def generate_pdf_from_ai(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera PDF usando el análisis y gráficos creados por la IA."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_ia_{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Reporte Profesional InertiaX - Análisis IA")
    c.setFont("Helvetica", 10)
    c.drawString(100, 730, f"Atleta: {meta.get('nombre', '-')}")
    c.drawString(100, 715, f"Análisis generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # Análisis de IA
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 680, "ANÁLISIS COMPLETO:")
    c.setFont("Helvetica", 9)

    # Función para wrap text
    def _wrap_text(text, x, y, max_width=90):
        lines = []
        words = text.split()
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        
        for line in lines:
            if y < 50:  # Nueva página si queda poco espacio
                c.showPage()
                y = 750
            c.drawString(x, y, line)
            y -= 12
        return y

    y_position = 660
    y_position = _wrap_text(ai_result.get('analysis', 'No analysis generated'), 50, y_position)

    # Gráficos
    for i, chart_buf in enumerate(charts):
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 750, f"Gráfico {i+1} - Análisis IA")
        try:
            chart_buf.seek(0)
            c.drawImage(ImageReader(chart_buf), 50, 400, width=500, height=300)
        except Exception as e:
            c.drawString(50, 500, f"Error cargando gráfico: {e}")

    # Recomendaciones
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "RECOMENDACIONES ESPECÍFICAS")
    c.setFont("Helvetica", 10)

    recommendations = ai_result.get('recommendations', [])
    if isinstance(recommendations, list):
        y_pos = 720
        for rec in recommendations:
            if y_pos < 50:
                c.showPage()
                y_pos = 750
            c.drawString(70, y_pos, f"• {rec}")
            y_pos -= 15
    else:
        _wrap_text(str(recommendations), 50, 720)

    c.save()
    return pdf_path

def generate_docx_from_ai(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera DOCX usando el análisis y gráficos de la IA."""
    from docx import Document
    from docx.shared import Inches

    doc = Document()

    # Título
    doc.add_heading('Reporte Profesional InertiaX - Análisis IA', 0)
    doc.add_paragraph(f"Atleta: {meta.get('nombre', '-')}")
    doc.add_paragraph(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # Análisis
    doc.add_heading('Análisis Completo', level=1)
    doc.add_paragraph(ai_result.get('analysis', 'No analysis generated'))

    # Gráficos
    if charts:
        doc.add_heading('Gráficos de Análisis', level=1)
        for i, chart_buf in enumerate(charts):
            doc.add_heading(f'Gráfico {i+1}', level=2)
            try:
                chart_buf.seek(0)
                doc.add_picture(chart_buf, width=Inches(6.0))
            except Exception as e:
                doc.add_paragraph(f"Error cargando gráfico: {e}")

    # Recomendaciones
    doc.add_heading('Recomendaciones', level=1)
    recommendations = ai_result.get('recommendations', [])
    if isinstance(recommendations, list):
        for rec in recommendations:
            doc.add_paragraph(rec, style='List Bullet')
    else:
        doc.add_paragraph(str(recommendations))

    path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_ia_{uuid.uuid4().hex}.docx")
    doc.save(path)
    return path



@app.route("/healthz")
def healthz():
    return jsonify(ok=True), 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Subida de archivo y procesamiento inicial."""
    job_id = _ensure_job()
    session.modified = True

    form = {
        "tipo_datos": request.form.get("tipo_datos", "").strip(),
        "proposito": request.form.get("proposito", "").strip(),
        "detalles": request.form.get("detalles", "").strip(),
        "nombre": request.form.get("nombre", "").strip(),
        "metricas_interes": request.form.get("metricas_interes", "").strip(),
    }

    # Verificar código de invitado
    code = request.form.get("codigo_invitado", "").strip()
    payment_ok = False
    mensaje = None
    if code and code in app.config["GUEST_CODES"]:
        payment_ok = True
        mensaje = "🔓 Código de invitado válido. Puedes generar tu reporte sin pagar."

    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template("index.html", error="No se subió ningún archivo.")

    if not _allowed_file(f.filename):
        return render_template("index.html", error="Formato no permitido. Usa .csv, .xls o .xlsx")

    ext = os.path.splitext(f.filename)[1].lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(_job_dir(job_id), safe_name)
    f.save(save_path)

    # Persistir meta
    meta = {
        "file_name": f.filename,
        "file_path": save_path,
        "payment_ok": payment_ok,
        "form": form,
    }
    _save_meta(job_id, meta)

    # Previsualización simple
    try:
        df = parse_dataframe(save_path)
        table_html = df.head(10).to_html(classes="table table-striped table-hover", index=False)
        return render_template(
            "index.html",
            table_html=table_html,
            filename=f.filename,
            form_data=form,
            mensaje=mensaje,
            show_payment=(not payment_ok),
        )
    except Exception as e:
        log.exception("Error al procesar DF")
        return render_template("index.html", error=f"Error al procesar el archivo: {e}")
@app.route("/create_preference", methods=["POST"])

def create_preference():
    """Crea preferencia Mercado Pago."""
    if not mp:
        return jsonify(error="Mercado Pago no configurado"), 500


    job_id = session.get("job_id")
    if not job_id:
        return jsonify(error="Sesión inválida"), 400

    meta = _load_meta(job_id)
    form = meta.get("form", {})
    purpose = (form.get("proposito") or "").lower()

    price_map = {
        "generar reporte editable para deportista": 2900,
        "generar análisis estadístico": 3900,
        "generar análisis avanzado": 5900,
        "todo": 6900,
    }
    price = price_map.get(purpose, 2900)

    pref_data = {
        "items": [{
            "title": f"InertiaX - {form.get('proposito') or 'Análisis IA'}",
            "quantity": 1,
            "unit_price": price,
            "currency_id": "CLP",
        }],
        "back_urls": {
            "success": f"{app.config['DOMAIN_URL']}/success",
            "failure": f"{app.config['DOMAIN_URL']}/cancel",
            "pending": f"{app.config['DOMAIN_URL']}/cancel",
        },
        "auto_return": "approved",
    }

    try:
        pref = mp.preference().create(pref_data)
        return jsonify(pref.get("response", {}))
    except Exception as e:
        log.exception("Error creando preferencia MP")
        return jsonify(error=str(e)), 500
    


@app.route("/success")
def success():
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    meta["payment_ok"] = True
    _save_meta(job_id, meta)
    return render_template("success.html")

@app.route("/cancel")
def cancel():
    return render_template("cancel.html")

@app.route("/generate_report")
def generate_report():
    """Endpoint principal que ejecuta TODO el análisis por IA."""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    if not meta.get("payment_ok"):
        return redirect(url_for("index"))

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return render_template("index.html", error="No hay archivo para analizar.")

    try:
        # 1. Cargar datos
        df = parse_dataframe(file_path)
        
        # 2. EJECUTAR ANÁLISIS COMPLETO POR IA
        log.info("Iniciando análisis completo por IA...")
        ai_result = run_complete_ai_analysis(df, meta.get("form", {}))
        
        # 3. EJECUTAR CÓDIGO DE GRÁFICOS GENERADO POR IA
        charts = []
        python_code = ai_result.get("python_code_for_charts", "")
        if python_code:
            log.info("Ejecutando código de gráficos generado por IA...")
            charts = execute_ai_charts_code(python_code, df)
        
        # 4. GENERAR REPORTES CON LOS RESULTADOS DE IA
        pdf_path = generate_pdf_from_ai(ai_result, charts, meta.get("form", {}))
        docx_path = generate_docx_from_ai(ai_result, charts, meta.get("form", {}))
        
        # 5. CREAR ZIP CON REPORTES
        zip_path = os.path.join(_job_dir(job_id), f"reporte_ia_completo_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "reporte_inertiax_ia.pdf")
            zf.write(docx_path, "reporte_inertiax_ia.docx")
        
        return send_file(zip_path, as_attachment=True, download_name="reporte_inertiax_ia_completo.zip")
        
    except Exception as e:
        log.exception("Error generando reporte con IA")
        return render_template("index.html", error=f"Error en el análisis IA: {e}")
    

@app.route("/preview_analysis")
def preview_analysis():
    """Vista previa del análisis de IA (sin descarga)."""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))


    meta = _load_meta(job_id)
    if not meta.get("payment_ok"):
        return redirect(url_for("index"))

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return render_template("index.html", error="No hay archivo para analizar.")

    try:
        df = parse_dataframe(file_path)
        ai_result = run_complete_ai_analysis(df, meta.get("form", {}))
        
        # Mostrar vista previa del análisis
        return render_template(
            "preview.html",
            analysis=ai_result.get("analysis", "No analysis generated"),
            recommendations=ai_result.get("recommendations", []),
            charts_description=ai_result.get("charts_description", ""),
            filename=meta.get("file_name")
        )
        
    except Exception as e:
        log.exception("Error en vista previa IA")
        return render_template("index.html", error=f"Error en vista previa: {e}")


@app.errorhandler(413)
def too_large(_e):
    return make_response(("Archivo demasiado grande.", 413))

@app.errorhandler(404)
def not_found(_e):
    return make_response(("Ruta no encontrada.", 404))

@app.errorhandler(Exception)
def global_error(e):
    log.exception("Error no controlado")
    return make_response((f"Error interno: {e}", 500))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
