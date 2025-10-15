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



def run_ai_analysis(df: pd.DataFrame, meta: dict) -> str:
    """Analiza TODOS los datos con máxima profundidad y contexto."""
    
    # 1. USAR TODOS LOS DATOS SIN MUESTREO
    n_rows, n_cols = df.shape
    dataset_info = f"Dataset completo: {n_rows} filas × {n_cols} columnas. Analizando el 100% de los datos."
    
    # 2. PREPARAR DATOS ENRIQUECIDOS
    # Información estadística detallada
    stats_info = generate_detailed_stats(df)
    
    # Detectar patrones específicos
    patterns_info = detect_data_patterns(df)
    
    # 3. CONVERTIR DATOS MÁS COMPLETOS
    # Usar formato que preserve mejor la información
    if n_rows <= 1000:
        # Para datasets pequeños, pasar datos completos
        data_preview = df.to_string(max_rows=min(100, n_rows), max_cols=None)
    else:
        # Para datasets grandes, muestra estratégica + estadísticas
        sample_df = df.sample(800, random_state=42)
        data_preview = sample_df.to_string(max_rows=100, max_cols=None)
        dataset_info += f"\nMuestra representativa de 800 filas para análisis detallado."
    
    # 4. DETECCIÓN AVANZADA DE CONTEXTO DEPORTIVO
    sport_context = detect_sport_context(df, meta)
    
    # 5. CONSTRUIR PROMPT MÁS ESPECÍFICO Y DETALLADO
    contexto = (
        f"CONTEXTO COMPLETO DEL ANÁLISIS:\n"
        f"• Atleta: {meta.get('nombre', 'No especificado')}\n"
        f"• Tipo de datos: {meta.get('tipo_datos', 'No especificado')}\n"
        f"• Propósito específico: {meta.get('proposito', 'No especificado')}\n"
        f"• Detalles adicionales: {meta.get('detalles', 'No especificados')}\n"
        f"• Métricas de interés: {meta.get('metricas_interes', 'Todas disponibles')}\n"
        f"• Contexto deportivo detectado: {sport_context}\n"
        f"• {dataset_info}\n\n"
        f"ESTADÍSTICAS DETALLADAS:\n{stats_info}\n\n"
        f"PATRONES DETECTADOS:\n{patterns_info}"
    )

    # 6. PROMPT DEL SISTEMA MÁS EXIGENTE
    system_prompt = """
Eres un analista deportivo de élite especializado en biomecánica, fisiología del ejercicio y rendimiento deportivo. Tu tarea es realizar un análisis PROFUNDO y DETALLADO de los datos.

**REQUISITOS DEL ANÁLISIS (OBLIGATORIOS):**

1. **ANÁLISIS EXHAUSTIVO POR VARIABLE:**
   - Examinar CADA columna del dataset individualmente
   - Identificar valores atípicos, tendencias y distribuciones
   - Relacionar cada métrica con implicaciones prácticas deportivas

2. **INTERCONEXIONES Y SINERGIAS:**
   - Analizar cómo interactúan las diferentes variables
   - Identificar relaciones causa-efecto entre métricas
   - Evaluar compensaciones y trade-offs en el rendimiento

3. **ANÁLISIS TEMPORAL Y EVOLUTIVO:**
   - Detectar patrones de fatiga, aprendizaje o adaptación
   - Analizar consistencia y variabilidad del rendimiento
   - Identificar fases críticas en las sesiones

4. **RECOMENDACIONES ESPECÍFICAS Y ACCIONABLES:**
   - Planes de entrenamiento concretos
   - Correcciones técnicas específicas
   - Estrategias de medición y seguimiento
   - Objetivos cuantificables a corto, medio y largo plazo

5. **ANÁLISIS DE CALIDAD DE DATOS:**
   - Evaluar la confiabilidad de las mediciones
   - Identificar posibles errores de captura
   - Sugerir mejoras en la recolección de datos

**FORMATO DE RESPUESTA:**
- Usar lenguaje técnico pero aplicado
- Incluir valores numéricos específicos
- Proporcionar ejemplos concretos de los datos
- Ser crítico y constructivo
- Mínimo 1000 palabras de análisis
"""

    # 7. PROMPT DEL USUARIO MÁS COMPLETO
    user_prompt = f"""
{contexto}

**DATOS COMPLETOS PARA ANÁLISIS:**

**INSTRUCCIONES ESPECÍFICAS:**
1. Analiza METICULOSAMENTE cada variable y su impacto en el rendimiento
2. Proporciona un análisis CUANTITATIVO profundo con valores específicos
3. Identifica al menos 5-7 hallazgos NO OBVIOS que un análisis superficial perdería
4. Relaciona cada hallazgo con aplicaciones prácticas inmediatas
5. Incluye recomendaciones ESPECÍFICAS basadas en los patrones detectados

El cliente espera un análisis de NIVEL PROFESIONAL, no un resumen superficial.
"""

    try:
        completion = ai_client.chat.completions.create(
            model=app.config["OPENROUTER_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Más preciso y consistente
            max_tokens=8000,  # Permite análisis más extensos
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        log.error(f"Error en análisis IA: {e}")
        return f"Error en el análisis automatizado. Por favor, contacte soporte. Detalles: {str(e)}"


def generate_detailed_stats(df: pd.DataFrame) -> str:
    """Genera estadísticas detalladas para enriquecer el contexto."""
    stats = []
    
    # Estadísticas por columna
    for col in df.columns:
        col_stats = f"\n--- {col} ---\n"
        
        # Info básica
        col_stats += f"Tipo: {df[col].dtype}\n"
        col_stats += f"No nulos: {df[col].count()}/{len(df)} ({df[col].count()/len(df)*100:.1f}%)\n"
        
        # Estadísticas según tipo
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats += f"Media: {df[col].mean():.4f}\n"
            col_stats += f"Mediana: {df[col].median():.4f}\n"
            col_stats += f"Std: {df[col].std():.4f}\n"
            col_stats += f"Min: {df[col].min():.4f}\n"
            col_stats += f"Max: {df[col].max():.4f}\n"
            col_stats += f"Q1: {df[col].quantile(0.25):.4f}\n"
            col_stats += f"Q3: {df[col].quantile(0.75):.4f}\n"
            
            # Outliers IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            col_stats += f"Outliers (IQR): {outliers} ({outliers/len(df)*100:.1f}%)\n"
            
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            unique_vals = df[col].nunique()
            col_stats += f"Valores únicos: {unique_vals}\n"
            if unique_vals <= 10:
                top_vals = df[col].value_counts().head(5)
                col_stats += "Top valores:\n" + "\n".join([f"  {val}: {count}" for val, count in top_vals.items()])
        
        stats.append(col_stats)
    
    return "\n".join(stats)


def detect_sport_context(df: pd.DataFrame, meta: dict) -> str:
    """Detecta automáticamente el contexto deportivo específico."""
    columns_lower = [str(col).lower() for col in df.columns]
    
    # Detección por nombres de columnas
    context_hints = []
    
    if any(x in columns_lower for x in ['velocity', 'velocidad', 'speed']):
        context_hints.append("Análisis de velocidad")
    if any(x in columns_lower for x in ['force', 'fuerza', 'load', 'carga']):
        context_hints.append("Análisis de fuerza")
    if any(x in columns_lower for x in ['jump', 'salto', 'cmj', 'sj']):
        context_hints.append("Análisis de salto")
    if any(x in columns_lower for x in ['power', 'potencia']):
        context_hints.append("Análisis de potencia")
    if any(x in columns_lower for x in ['heart', 'cardio', 'hr']):
        context_hints.append("Análisis cardiovascular")
    if any(x in columns_lower for x in ['repetition', 'rep', 'series', 'set']):
        context_hints.append("Entrenamiento con cargas")
    
    # Combinar con contexto del usuario
    user_context = meta.get('proposito', '') + ' ' + meta.get('detalles', '')
    user_context = user_context.lower()
    
    if 'velocidad' in user_context or 'speed' in user_context:
        context_hints.append("Enfoque específico en velocidad")
    if 'fuerza' in user_context or 'force' in user_context:
        context_hints.append("Enfoque específico en fuerza")
    if 'salto' in user_context or 'jump' in user_context:
        context_hints.append("Enfoque específico en capacidad de salto")
    
    return " | ".join(set(context_hints)) if context_hints else "Contexto deportivo general"


def detect_data_patterns(df: pd.DataFrame) -> str:
    """Detecta patrones importantes en los datos."""
    patterns = []
    
    # Patrón 1: Tendencia temporal
    datetime_col = detect_datetime_col(df)
    if datetime_col and len(df) > 1:
        try:
            df_sorted = df.sort_values(datetime_col)
            numeric_cols = df_sorted.select_dtypes(include=['number']).columns
            
            # Verificar tendencias en primeras columnas numéricas
            for col in numeric_cols[:2]:  # Máximo 2 columnas
                if len(df_sorted) > 2:
                    correlation = df_sorted[col].corr(pd.Series(range(len(df_sorted))))
                    if abs(correlation) > 0.5:
                        trend = "creciente" if correlation > 0 else "decreciente"
                        patterns.append(f"Tendencia temporal {trend} en {col} (r={correlation:.2f})")
        except:
            pass
    
    # Patrón 2: Relaciones entre variables clave
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append(f"{col1} ↔ {col2} (r={corr_matrix.iloc[i, j]:.2f})")
        
        if high_corr_pairs:
            patterns.append("Correlaciones fuertes: " + "; ".join(high_corr_pairs[:3]))
    
    # Patrón 3: Distribuciones anómalas
    for col in numeric_df.columns:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            skew_type = "positiva" if skewness > 0 else "negativa"
            patterns.append(f"Sesgo {skew_type} pronunciado en {col} (skew={skewness:.2f})")
    
    return "\n".join(patterns) if patterns else "No se detectaron patrones evidentes automáticamente"











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
