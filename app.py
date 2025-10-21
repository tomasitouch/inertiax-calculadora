from __future__ import annotations
import json
import logging
import os
import uuid
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional
import mercadopago
import pandas as pd
import requests
from flask import (
    Flask, jsonify, redirect, render_template,
    request, send_file, url_for, session, make_response
)
from flask_cors import CORS
from openai import OpenAI
from PIL import Image, ImageOps, ImageDraw

# ==============================
# CONFIGURACIÓN
# ==============================

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_secret_key")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024
    SESSION_COOKIE_NAME = "inertiax_session"
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
    DOMAIN_URL = os.getenv("DOMAIN_URL", "https://inertiax-calculadora.onrender.com")
    MP_ACCESS_TOKEN = os.getenv("MP_ACCESS_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
    GUEST_CODES = set(
        (os.getenv("GUEST_CODES") or "INERTIAXVIP2025,ENTRENADORPRO,INVEXORTEST").split(",")
    )

# ==============================
# APP Y EXTENSIONES
# ==============================

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("inertiax")
mp = mercadopago.SDK(app.config["MP_ACCESS_TOKEN"]) if app.config["MP_ACCESS_TOKEN"] else None
ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"])

# ==============================
# UTILIDADES Y ESTADO
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
    if "job_id" not in session:
        session["job_id"] = uuid.uuid4().hex
    return session["job_id"]

def _allowed_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in app.config["ALLOWED_EXT"]

def parse_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

# ==============================
# NUEVA FUNCIÓN: PREPROCESAMIENTO POR FUENTE
# ==============================

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
# IA PRINCIPAL
# ==============================

def run_complete_ai_analysis(df: pd.DataFrame, meta: dict) -> dict:
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
        f"- Dataset: {n_rows} filas × {n_cols} columnas",
        f"- Columnas: {', '.join(df.columns.tolist())}",
        "",
        stats_info
    ])

    system_prompt = """
Eres un analista deportivo especializado en biomecánica y entrenamiento de fuerza.
Tu tarea es realizar un análisis completo de los datos recibidos.
Entrega un JSON con: analysis, python_code_for_charts, charts_description, recommendations.
"""

    user_prompt = f"{contexto}\n\nDatos CSV:\n```csv\n{data_preview}\n```"

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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        log.error(f"Error en IA: {e}")
        return {"analysis": str(e), "python_code_for_charts": "", "recommendations": []}

# ==============================
# RUTAS
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
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

    code = request.form.get("codigo_invitado", "").strip()
    payment_ok = code in app.config["GUEST_CODES"]
    mensaje = "Código válido." if payment_ok else None

    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template("index.html", error="No se subió ningún archivo.")

    if not _allowed_file(f.filename):
        return render_template("index.html", error="Formato no permitido.")

    ext = os.path.splitext(f.filename)[1].lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(_job_dir(job_id), safe_name)
    f.save(save_path)

    meta = {"file_name": f.filename, "file_path": save_path, "payment_ok": payment_ok, "form": form}
    _save_meta(job_id, meta)

    try:
        df = parse_dataframe(save_path)
        df = preprocess_data_by_origin(df, form.get("origen_app", ""))
        table_html = df.head(10).to_html(classes="table table-striped", index=False)
        return render_template(
            "index.html", table_html=table_html, mensaje=mensaje, show_payment=(not payment_ok)
        )
    except Exception as e:
        log.exception("Error procesando archivo")
        return render_template("index.html", error=f"Error al procesar: {e}")

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
