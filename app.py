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
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_secret_key_2025")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024
    SESSION_COOKIE_NAME = "inertiax_session"

    # Archivos
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}

    # Pago / IA / Dominio
    DOMAIN_URL = os.getenv("DOMAIN_URL", "https://inertiax-calculadora-1.onrender.com")
    MP_ACCESS_TOKEN = os.getenv("MP_ACCESS_TOKEN")
    MP_PUBLIC_KEY = os.getenv("MP_PUBLIC_KEY")
    
    # OpenAI Config
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("inertiax")

# Clientes externos
mp = mercadopago.SDK(app.config["MP_ACCESS_TOKEN"]) if app.config["MP_ACCESS_TOKEN"] else None
ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"]) if app.config["OPENAI_API_KEY"] else None

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

def preprocess_data_by_origin(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """
    Ajusta el DataFrame según la app o dispositivo que generó el CSV.
    """
    origin = origin.lower()
    if origin == "app_android_encoder_vertical":
        rename_map = {
            "Athlete": "atleta",
            "Exercise": "ejercicio",
            "Date": "fecha",
            "Repetition": "repeticion",
            "Load(kg)": "carga_kg",
            "ConcentricVelocity(m/s)": "velocidad_concentrica_m_s",
            "EccentricVelocity(m/s)": "velocidad_eccentrica_m_s",
            "MaxVelocity(m/s)": "velocidad_maxima_m_s",
            "Duration(s)": "duracion_s",
            "Estimated1RM": "estimado_1rm_kg"
        }
        df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

        num_cols = [
            "carga_kg", "velocidad_concentrica_m_s", "velocidad_eccentrica_m_s",
            "velocidad_maxima_m_s", "duracion_s", "estimado_1rm_kg"
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "velocidad_concentrica_m_s" in df.columns and "carga_kg" in df.columns:
            df["potencia_relativa_w_kg"] = df["carga_kg"] * df["velocidad_concentrica_m_s"]

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        df.dropna(how="all", inplace=True)
        return df

    elif origin == "jump_sensor":
        if "altura_cm" in df.columns:
            df["altura_m"] = df["altura_cm"] / 100
        return df

    else:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

# ==============================
# NÚCLEO IA
# ==============================

def run_complete_ai_analysis(df: pd.DataFrame, meta: dict) -> dict:
    """
    Análisis completo por IA
    """
    if not ai_client:
        return {
            "analysis": "⚠️ Servicio de IA no disponible. Configure OPENAI_API_KEY.",
            "python_code_for_charts": "",
            "charts_description": "",
            "recommendations": ["Configure la API key de OpenAI para habilitar el análisis IA"]
        }
    
    n_rows, n_cols = df.shape
    data_preview = df.head(500).to_csv(index=False)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    stats_info = ""
    if numeric_cols:
        stats_info = "Estadísticas básicas:\n"
        for col in numeric_cols[:5]:
            stats_info += f"{col}: media={df[col].mean():.3f}, std={df[col].std():.3f}, min={df[col].min():.3f}, max={df[col].max():.3f}\n"

    contexto = "\n".join([
        "DATOS DEL ANÁLISIS:",
        f"- Origen del archivo: {meta.get('origen_app', 'No especificado')}",
        f"- Atleta: {meta.get('nombre', 'No especificado')}",
        f"- Tipo de datos: {meta.get('tipo_datos', 'No especificado')}",
        f"- Propósito: {meta.get('proposito', 'No especificado')}",
        f"- Detalles: {meta.get('detalles', 'No especificados')}",
        f"- Dataset: {n_rows} filas × {n_cols} columnas",
        f"- Columnas: {', '.join(df.columns.tolist())}",
        "",
        stats_info
    ])

    system_prompt = """
Eres un analista deportivo especializado en biomecánica y entrenamiento de fuerza.
Analiza los datos proporcionados y genera un reporte profesional.
Responde EXACTAMENTE en formato JSON con estas claves:
- analysis: texto completo del análisis
- python_code_for_charts: código Python para generar gráficos (opcional)
- charts_description: descripción de los gráficos
- recommendations: lista de recomendaciones
"""

    user_prompt = f"{contexto}\n\nDatos CSV (primeras filas):\n```csv\n{data_preview}\n```"

    try:
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        log.error(f"Error en IA: {e}")
        return {
            "analysis": f"Error en el análisis IA: {str(e)}",
            "python_code_for_charts": "",
            "charts_description": "",
            "recommendations": ["Error técnico - contacte al administrador"]
        }

def execute_ai_charts_code(python_code: str, df: pd.DataFrame) -> List[BytesIO]:
    """Ejecuta código Python de gráficos generado por IA"""
    if not python_code.strip():
        return []
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Resetear cualquier figura previa
        plt.close('all')
        
        local_vars = {
            'df': df, 
            'BytesIO': BytesIO, 
            'plt': plt,
            'sns': sns,
            'np': np,
            'pd': pd,
            'charts': []
        }
        
        # Ejecutar el código
        exec(python_code, local_vars)
        
        # Obtener gráficos
        charts = local_vars.get('charts', [])
        if not isinstance(charts, list):
            charts = []
            
        return charts
        
    except Exception as e:
        log.error(f"Error ejecutando código de gráficos: {e}")
        return []

def generate_pdf_from_ai(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera PDF con el análisis de IA"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_ia_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Reporte InertiaX - Análisis IA")
        c.setFont("Helvetica", 10)
        c.drawString(100, 730, f"Atleta: {meta.get('nombre', '-')}")
        c.drawString(100, 715, f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Análisis
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 680, "ANÁLISIS:")
        c.setFont("Helvetica", 9)
        
        # Text wrapping simple
        text = ai_result.get('analysis', 'No analysis available')
        y = 660
        lines = []
        words = text.split()
        line = ""
        
        for word in words:
            test_line = f"{line} {word}".strip()
            if len(test_line) < 80:
                line = test_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        
        for line in lines[:30]:  # Máximo 30 líneas
            if y < 100:
                c.showPage()
                y = 750
            c.drawString(50, y, line)
            y -= 12
        
        # Gráficos
        if charts:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 750, "GRÁFICOS DE ANÁLISIS")
            
            y_pos = 700
            for i, chart in enumerate(charts[:3]):  # Máximo 3 gráficos
                try:
                    chart.seek(0)
                    c.drawImage(ImageReader(chart), 50, y_pos - 250, width=500, height=200)
                    y_pos -= 280
                    if y_pos < 100 and i < len(charts) - 1:
                        c.showPage()
                        y_pos = 750
                except Exception as e:
                    log.error(f"Error dibujando gráfico {i}: {e}")
        
        # Recomendaciones
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 750, "RECOMENDACIONES")
        c.setFont("Helvetica", 10)
        
        recommendations = ai_result.get('recommendations', [])
        y_rec = 720
        if isinstance(recommendations, list):
            for rec in recommendations[:10]:  # Máximo 10 recomendaciones
                if y_rec < 50:
                    c.showPage()
                    y_rec = 750
                c.drawString(70, y_rec, f"• {rec}")
                y_rec -= 20
        else:
            c.drawString(50, 720, str(recommendations))
        
        c.save()
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF: {e}")
        # Crear PDF de error
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "Error generando reporte")
        c.drawString(100, 730, str(e))
        c.save()
        return error_path

def generate_docx_from_ai(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera DOCX con el análisis de IA"""
    try:
        from docx import Document
        from docx.shared import Inches
        
        doc = Document()
        
        # Título
        doc.add_heading('Reporte InertiaX - Análisis IA', 0)
        doc.add_paragraph(f"Atleta: {meta.get('nombre', '-')}")
        doc.add_paragraph(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Análisis
        doc.add_heading('Análisis', level=1)
        doc.add_paragraph(ai_result.get('analysis', 'No analysis available'))
        
        # Gráficos
        if charts:
            doc.add_heading('Gráficos', level=1)
            for i, chart in enumerate(charts[:3]):
                try:
                    chart.seek(0)
                    doc.add_picture(chart, width=Inches(6.0))
                    doc.add_paragraph(f"Gráfico {i+1}")
                except Exception as e:
                    doc.add_paragraph(f"Error cargando gráfico {i+1}: {e}")
        
        # Recomendaciones
        doc.add_heading('Recomendaciones', level=1)
        recommendations = ai_result.get('recommendations', [])
        if isinstance(recommendations, list):
            for rec in recommendations:
                doc.add_paragraph(f"• {rec}", style='List Bullet')
        else:
            doc.add_paragraph(str(recommendations))
        
        path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_ia_{uuid.uuid4().hex}.docx")
        doc.save(path)
        return path
        
    except Exception as e:
        log.error(f"Error generando DOCX: {e}")
        # Crear DOCX simple de error
        doc = Document()
        doc.add_heading('Error', 0)
        doc.add_paragraph(f"Error generando reporte: {e}")
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.docx")
        doc.save(error_path)
        return error_path

# ==============================
# RUTAS PRINCIPALES
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "InertiaX API is running"})

@app.route("/upload", methods=["POST"])
def upload():
    """Subida de archivo y procesamiento inicial"""
    job_id = _ensure_job()
    session.modified = True

    form = {
        "tipo_datos": request.form.get("tipo_datos", "").strip(),
        "proposito": request.form.get("proposito", "").strip(),
        "detalles": request.form.get("detalles", "").strip(),
        "nombre": request.form.get("nombre", "").strip(),
        "metricas_interes": request.form.get("metricas_interes", "").strip(),
        "origen_app": request.form.get("origen_app", "").strip(),
    }

    # Verificar código de invitado
    code = request.form.get("codigo_invitado", "").strip()
    payment_ok = False
    mensaje = None
    if code and code in app.config["GUEST_CODES"]:
        payment_ok = True
        mensaje = "🔓 Código de invitado válido. Puedes generar tu reporte."

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
        df = preprocess_data_by_origin(df, form.get("origen_app", ""))
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
        log.exception("Error al procesar archivo")
        return render_template("index.html", error=f"Error al procesar el archivo: {e}")

@app.route("/create_preference", methods=["POST"])
def create_preference():
    """Crea preferencia Mercado Pago"""
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
    """Pago exitoso"""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    meta["payment_ok"] = True
    _save_meta(job_id, meta)
    return render_template("success.html")

@app.route("/cancel") 
def cancel():
    """Pago cancelado"""
    return render_template("cancel.html")

@app.route("/generate_report")
def generate_report():
    """Genera y descarga el reporte completo"""
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
        df = preprocess_data_by_origin(df, meta.get("form", {}).get("origen_app", ""))
        
        # 2. Ejecutar análisis IA
        log.info("Ejecutando análisis IA...")
        ai_result = run_complete_ai_analysis(df, meta.get("form", {}))
        
        # 3. Generar gráficos si hay código
        charts = []
        python_code = ai_result.get("python_code_for_charts", "")
        if python_code:
            log.info("Generando gráficos...")
            charts = execute_ai_charts_code(python_code, df)
        
        # 4. Generar reportes
        pdf_path = generate_pdf_from_ai(ai_result, charts, meta.get("form", {}))
        docx_path = generate_docx_from_ai(ai_result, charts, meta.get("form", {}))
        
        # 5. Crear ZIP
        zip_path = os.path.join(_job_dir(job_id), f"reporte_completo_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "reporte_inertiax.pdf")
            zf.write(docx_path, "reporte_inertiax.docx")
            # Agregar CSV original también
            zf.write(file_path, os.path.basename(meta.get("file_name", "datos_original.csv")))
        
        # Limpiar archivos temporales
        try:
            os.remove(pdf_path)
            os.remove(docx_path)
        except:
            pass
            
        return send_file(
            zip_path, 
            as_attachment=True, 
            download_name=f"reporte_inertiax_{uuid.uuid4().hex[:8]}.zip"
        )
        
    except Exception as e:
        log.exception("Error generando reporte")
        return render_template("index.html", error=f"Error generando reporte: {e}")

@app.route("/preview_analysis")
def preview_analysis():
    """Vista previa del análisis"""
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
        df = preprocess_data_by_origin(df, meta.get("form", {}).get("origen_app", ""))
        ai_result = run_complete_ai_analysis(df, meta.get("form", {}))
        
        return render_template(
            "preview.html",
            analysis=ai_result.get("analysis", "No analysis generated"),
            recommendations=ai_result.get("recommendations", []),
            charts_description=ai_result.get("charts_description", ""),
            filename=meta.get("file_name")
        )
        
    except Exception as e:
        log.exception("Error en vista previa")
        return render_template("index.html", error=f"Error en vista previa: {e}")

# ==============================
# Manejo de errores
# ==============================

@app.errorhandler(413)
def too_large(_e):
    return render_template("index.html", error="Archivo demasiado grande.")

@app.errorhandler(404)
def not_found(_e):
    return render_template("index.html", error="Página no encontrada.")

@app.errorhandler(500)
def internal_error(_e):
    return render_template("index.html", error="Error interno del servidor.")

@app.errorhandler(Exception)
def global_error(e):
    log.exception("Error no controlado")
    return render_template("index.html", error=f"Error interno: {e}")

# ==============================
# Run
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
