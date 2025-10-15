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
import seaborn as sns
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
from matplotlib import pyplot as plt
from openai import OpenAI
from PIL import Image, ImageOps, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from docx import Document
from docx.shared import Inches


# ==============================
# Configuración
# ==============================

class Config:
    # Seguridad y servidor
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_secret_key")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024  # 20 MB por default
    SESSION_COOKIE_NAME = "inertiax_session"

    # Archivos
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}

    # Pago / IA / Dominio
    DOMAIN_URL = os.getenv("DOMAIN_URL", "https://inertiax-calculadora-1.onrender.com")
    MP_ACCESS_TOKEN = os.getenv("MP_ACCESS_TOKEN")
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
    OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o")

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


# Mercado Pago y OpenAI/OpenRouter
mp = mercadopago.SDK(app.config["MP_ACCESS_TOKEN"]) if app.config["MP_ACCESS_TOKEN"] else None
ai_client = OpenAI(
    base_url=app.config["OPENROUTER_BASE"],
    api_key=app.config["OPENROUTER_KEY"],
)


# ==============================
# Modelos de estado (en disco)
# ==============================

@dataclass
class JobState:
    job_id: str
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    payment_ok: bool = False
    meta_path: Optional[str] = None  # guarda form_data (JSON)


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
    """Obtiene/crea un job_id mínimo en cookie de sesión (sin datos pesados)."""
    if "job_id" not in session:
        session["job_id"] = uuid.uuid4().hex
    return session["job_id"]


def _allowed_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in app.config["ALLOWED_EXT"]


# ==============================
# Utilidades de análisis
# ==============================

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


def parse_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if any(k in c.lower() for k in ["fecha", "date", "dia", "día", "time"]):
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                continue
    return None


def robust_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (describe, corr, missing) con tipos numéricos solo para cálculos."""
    numeric_df = df.select_dtypes(include=["number"])
    desc = numeric_df.describe().transpose()
    corr = numeric_df.corr(numeric_only=True)
    missing = df.isna().mean().to_frame("missing_ratio").sort_values("missing_ratio", ascending=False)
    return desc, corr, missing


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve tabla (col, count_outliers, ratio) basada en IQR."""
    numeric_df = df.select_dtypes(include=["number"])
    rows = []
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
        count = int(mask.sum())
        ratio = float(count / max(1, len(numeric_df[col])))
        rows.append({"variable": col, "outliers": count, "ratio": ratio})
    return pd.DataFrame(rows).sort_values("ratio", ascending=False)


def top_categorical(df: pd.DataFrame, topn: int = 3) -> Dict[str, pd.Series]:
    """Devuelve las top categorías por cardinalidad chica (útil p/agrupaciones)."""
    out = {}
    for c in df.select_dtypes(include=["object"]).columns:
        vc = df[c].value_counts().head(10)
        if 1 < vc.shape[0] <= 10:
            out[c] = vc
            if len(out) >= topn:
                break
    return out


def generate_figures(df: pd.DataFrame) -> List[BytesIO]:
    sns.set_theme(style="whitegrid")
    figs: List[BytesIO] = []
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Histogramas (hasta 3)
    for col in num_cols[:3]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        figs.append(buf)

    # Boxplot global
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df[num_cols], ax=ax)
        ax.set_title("Comparación por variable (Boxplot)")
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        figs.append(buf)

    # Heatmap de correlaciones
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        ax.set_title("Matriz de correlaciones")
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        figs.append(buf)

    # Serie temporal básica
    dt_col = detect_datetime_col(df)
    if dt_col and num_cols:
        try:
            temp = df.copy()
            temp[dt_col] = pd.to_datetime(temp[dt_col])
            temp = temp.sort_values(dt_col)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(temp[dt_col], temp[num_cols[0]], linewidth=1.5)
            ax.set_title(f"Tendencia temporal: {num_cols[0]} vs {dt_col}")
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            figs.append(buf)
        except Exception:
            pass

    return figs


def _wrap_text_pdf(c: canvas.Canvas, text: str, x: int, y: int, width_chars: int = 95, leading: int = 14):
    """Escritura con saltos simples por número aproximado de caracteres."""
    from textwrap import wrap
    for paragraph in text.split("\n"):
        lines = wrap(paragraph, width=width_chars) if paragraph.strip() else [""]
        for line in lines:
            c.drawString(x, y, line)
            y -= leading
            if y < 80:
                c.showPage()
                y = 740
    return y


def generate_pdf(analysis: str, df: pd.DataFrame, logo_buf: BytesIO, meta: Dict) -> str:
    pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Header
    try:
        c.drawImage(ImageReader(logo_buf), 40, 700, width=80, height=80, mask='auto')
    except Exception:
        pass

    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, 760, "Reporte Profesional InertiaX")
    c.setFont("Helvetica", 10)
    c.drawString(150, 742, f"Nombre: {meta.get('nombre') or '-'}")
    c.drawString(150, 728, f"Tipo de datos: {meta.get('tipo_datos') or '-'}")
    c.drawString(150, 714, f"Propósito: {meta.get('proposito') or '-'}")
    if meta.get("detalles"):
        c.drawString(150, 700, f"Detalles: {meta.get('detalles')[:70]}")

    c.setFont("Helvetica", 10)
    y = 680
    y = _wrap_text_pdf(c, analysis, x=40, y=y)

    # Figuras
    for buf in generate_figures(df):
        c.showPage()
        c.drawImage(ImageReader(buf), 60, 250, width=480, height=350)

    c.save()
    return pdf_path


def generate_docx(analysis: str, df: pd.DataFrame, logo_buf: BytesIO, meta: Dict) -> str:
    doc = Document()
    try:
        doc.add_picture(logo_buf, width=Inches(1.2))
    except Exception:
        pass

    doc.add_heading("Reporte Profesional InertiaX", 0)
    doc.add_paragraph(f"Generado para: {meta.get('nombre') or '-'}")
    doc.add_paragraph(f"Tipo de datos: {meta.get('tipo_datos') or '-'}")
    doc.add_paragraph(f"Propósito: {meta.get('proposito') or '-'}")
    if meta.get('detalles'):
        doc.add_paragraph(f"Detalles: {meta.get('detalles')}")
    doc.add_paragraph(" ")

    # Resumen estadístico real
    desc, corr, missing = robust_stats(df)
    outliers = detect_outliers_iqr(df)

    doc.add_heading("Análisis estadístico (resumen)", level=1)
    doc.add_paragraph(desc.round(3).to_string())

    if not corr.empty:
        doc.add_paragraph("\nCorrelaciones:\n" + corr.round(3).to_string())
    if not missing.empty:
        doc.add_paragraph("\nValores faltantes (ratio):\n" + missing.round(3).to_string())
    if not outliers.empty:
        doc.add_paragraph("\nOutliers (IQR):\n" + outliers.round(3).to_string())

    doc.add_page_break()
    doc.add_heading("Análisis IA (interpretación y recomendaciones)", level=1)
    for p in analysis.split("\n"):
        doc.add_paragraph(p)

    doc.add_page_break()
    doc.add_heading("Vista previa de datos", level=1)
    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr[i].text = str(col)
    for _, row in df.head(15).iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    # Gráficos
    for buf in generate_figures(df):
        doc.add_page_break()
        doc.add_picture(buf, width=Inches(6.0))

    path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_{uuid.uuid4().hex}.docx")
    doc.save(path)
    return path


def run_ai_analysis(df: pd.DataFrame, meta: Dict) -> str:
    """Análisis IA profundo: estadísticas reales + interpretación experta."""
    # Estadística real
    desc, corr, missing = robust_stats(df)
    outliers = detect_outliers_iqr(df)

    # Resumen de datos breve
    resumen = f"Columnas: {list(df.columns)}. Muestra: {df.head(3).to_dict(orient='records')}."
    contexto = (
        f"Tipo: {meta.get('tipo_datos')}. Propósito: {meta.get('proposito')}."
        f" Detalles: {meta.get('detalles')}. Nombre: {meta.get('nombre')}."
    )

    extended = f"""
[Estadísticas descriptivas]
{desc.round(3).to_string()}

[Correlaciones]
{(corr.round(3).to_string() if corr is not None else 'N/A')}

[Valores faltantes]
{(missing.round(3).to_string() if missing is not None else 'N/A')}

[Outliers por IQR]
{(outliers.round(3).to_string() if outliers is not None else 'N/A')}
""".strip()

    system_prompt = """
Eres un analista experto en rendimiento deportivo y ciencia aplicada al entrenamiento.
Analiza los datos según el contexto, las variables disponibles y las métricas que el usuario desea conocer.

Tu análisis debe incluir:
1. Resumen ejecutivo.
2. Estadísticas clave de las variables relevantes.
3. Análisis estadístico avanzado (tendencias, correlaciones, dispersión, detección de outliers).
4. Interpretación biomecánica y fisiológica (músculos, potencia, técnica, fatiga, coordinación).
5. Recomendaciones prácticas para mejorar el rendimiento.
6. Limitaciones y próximos pasos.

Si el usuario menciona métricas específicas (RM, velocidad máxima, índice de reactividad, etc.),
prioriza su análisis en esas variables y deriva cálculos relevantes (por ejemplo, RM estimado = (Peso / (1.0278 - 0.0278 * reps)) ).
"""


# Construcción del contexto general
contexto = (
    f"Tipo de datos: {meta.get('tipo_datos')}. "
    f"Propósito: {meta.get('proposito')}. "
    f"Detalles: {meta.get('detalles')}. "
    f"Nombre: {meta.get('nombre')}. "
    f"Métricas de interés: {meta.get('metricas_interes')}."
)

# Prompt del usuario que se envía al modelo
user_prompt = f"""
Contexto: {contexto}
Datos (muestra): {resumen}

Cálculos cuantitativos:
{extended}

Genera un informe estructurado siguiendo esta guía:
1. Resumen ejecutivo
2. Análisis estadístico (medias, desviaciones, correlaciones, outliers)
3. Interpretación biomecánica/fisiológica (velocidad, fuerza, potencia, fatiga, RSI, etc.)
4. Recomendaciones prácticas
5. Limitaciones y próximos pasos

Cita las variables por su nombre exacto del dataset.
""".strip()

completion = ai_client.chat.completions.create(
    model=app.config["OPENROUTER_MODEL"],
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.4,
)

return completion.choices[0].message.content



# ==============================
# Rutas
# ==============================

@app.route("/healthz")
def healthz():
    return jsonify(ok=True), 200


@app.route("/")
def index():
    # Página base (sin estado)
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    - Valida archivo y metadatos
    - Soporta código de invitado (libera pago)
    - Guarda estado mínimo en /tmp/<job_id>
    - Renderiza tabla + decide si mostrar pago o descarga
    """
    job_id = _ensure_job()
    session.modified = True  # asegurar persistencia

    form = {
        "tipo_datos": request.form.get("tipo_datos", "").strip(),
        "proposito": request.form.get("proposito", "").strip(),
        "detalles": request.form.get("detalles", "").strip(),
        "nombre": request.form.get("nombre", "").strip(),
        "metricas_interes": request.form.get("metricas_interes"),

    }

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

    # Persistir meta del job
    meta = {
        "file_name": f.filename,
        "file_path": save_path,
        "payment_ok": payment_ok,
        "form": form,
    }
    _save_meta(job_id, meta)

    # Renderizar tabla
    try:
        df = parse_dataframe(save_path)
        table_html = df.to_html(classes="table table-striped table-hover", index=False)
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
    """Crea preferencia Mercado Pago (precio dinámico por propósito)."""
    if not mp:
        return jsonify(error="Mercado Pago no configurado (MP_ACCESS_TOKEN)"), 500

    job_id = session.get("job_id")
    if not job_id:
        return jsonify(error="Sesión inválida"), 400

    meta = _load_meta(job_id)
    form = meta.get("form", {})
    purpose = (form.get("proposito") or "").lower()

    # Precios sugeridos (CLP)
    price_map = {
        "generar reporte editable para deportista": 2900,
        "generar análisis estadístico": 3900,
        "generar análisis avanzado": 5900,
        "todo": 6900,
    }
    price = price_map.get(purpose, 2900)

    pref_data = {
        "items": [{
            "title": f"InertiaX - {form.get('proposito') or 'Análisis'}",
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


@app.route("/preview_pdf")
def preview_pdf():
    """Genera un PDF de vista previa inline (sin descarga) si el acceso está liberado."""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    if not meta.get("payment_ok"):
        # permite también si hay guest code (ya lo marca payment_ok)
        return redirect(url_for("index"))

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return render_template("index.html", error="No hay archivo para analizar.")

    df = parse_dataframe(file_path)
    analysis = run_ai_analysis(df, meta.get("form", {}))
    try:
        logo = fetch_circular_logo(app.config["LOGO_URL"])
    except Exception:
        logo = BytesIO()

    pdf_path = generate_pdf(analysis, df, logo, meta.get("form", {}))
    # inline preview
    return send_file(pdf_path, mimetype="application/pdf", as_attachment=False, download_name="preview.pdf")


@app.route("/download_bundle")
def download_bundle():
    """Genera DOCX + PDF y devuelve ZIP (requiere payment_ok)."""
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
        analysis = run_ai_analysis(df, meta.get("form", {}))
        try:
            logo = fetch_circular_logo(app.config["LOGO_URL"])
        except Exception:
            logo = BytesIO()

        docx_path = generate_docx(analysis, df, logo, meta.get("form", {}))
        pdf_path = generate_pdf(analysis, df, logo, meta.get("form", {}))

        zip_path = os.path.join(_job_dir(job_id), f"reporte_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(docx_path, "reporte_inertiax.docx")
            zf.write(pdf_path, "reporte_inertiax.pdf")

        return send_file(zip_path, as_attachment=True, download_name="reporte_inertiax.zip")
    except Exception as e:
        log.exception("Error generando bundle")
        return render_template("index.html", error=f"Error generando el reporte: {e}")


# ==============================
# Errores globales
# ==============================

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


# ==============================
# Run
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
