from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from io import BytesIO
from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from flask import Flask, render_template, request, send_file, session, jsonify
from flask_cors import CORS
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')

# ==============================
# CONFIGURACIÓN
# ==============================

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_pro_key")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/inertiax_pro")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inertiax_pro")

ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"]) if app.config["OPENAI_API_KEY"] else None

executor = ThreadPoolExecutor(max_workers=2)
jobs = {}

# ==============================
# UTILIDADES
# ==============================

def _job_dir(job_id: str) -> str:
    d = os.path.join(app.config["UPLOAD_DIR"], job_id)
    os.makedirs(d, exist_ok=True)
    return d

def parse_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

# ==============================
# ANÁLISIS IA
# ==============================

def analyze_with_ai(df: pd.DataFrame, meta) -> dict:
    if not ai_client:
        return {"resumen": "No hay clave de API"}

    csv = df.to_csv(index=False)

    system_prompt = """
Eres un analista deportivo experto en VBT.
Analiza datos, encuentra patrones y da recomendaciones.
"""

    user_prompt = f"""
Entrenador: {meta['nombre_entrenador']}
Atleta: {meta['nombre_cliente']}

Datos CSV:
{csv}

Devuelve JSON con:
resumen, fortalezas, mejoras, recomendaciones.
"""

    res = ai_client.chat.completions.create(
        model=app.config["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(res.choices[0].message.content)

# ==============================
# GRÁFICOS (simple fallback)
# ==============================

def generate_charts(df):
    from matplotlib import pyplot as plt

    charts = []
    numeric = df.select_dtypes(include=['number']).columns[:1]

    if len(numeric) == 0:
        return []

    fig, ax = plt.subplots()
    ax.plot(df[numeric[0]])
    ax.set_title(f"Evolución de {numeric[0]}")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    charts.append(buf)
    plt.close()

    return charts

# ==============================
# PDF
# ==============================

def generate_pdf(ai, charts, meta, job_id):
    pdf_path = os.path.join(_job_dir(job_id), "reporte.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Reporte de Rendimiento", styles['Title']))
    story.append(Paragraph(f"Entrenador: {meta['nombre_entrenador']}", styles['Normal']))
    story.append(Paragraph(f"Atleta: {meta['nombre_cliente']}", styles['Normal']))
    story.append(Spacer(1, 10))

    story.append(Paragraph(ai['resumen'], styles['Normal']))
    story.append(Spacer(1, 10))

    for buf in charts:
        img = ReportLabImage(buf, width=5*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 10))

    doc.build(story)
    return pdf_path

# ==============================
# JOB WORKER
# ==============================

def process_job(job_id, save_path, meta):
    try:
        df = parse_dataframe(save_path)
        ai = analyze_with_ai(df, meta)
        charts = generate_charts(df)
        pdf = generate_pdf(ai, charts, meta, job_id)

        zip_path = os.path.join(_job_dir(job_id), "report.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(pdf, "Reporte.pdf")
            z.write(save_path, f"datos.csv")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["zip"] = zip_path

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

# ==============================
# RUTAS
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["file"]
    job_id = uuid.uuid4().hex
    ext = os.path.splitext(f.filename)[1]
    save_path = os.path.join(_job_dir(job_id), "data"+ext)
    f.save(save_path)

    meta = {
        "nombre_entrenador": request.form.get("nombre_entrenador"),
        "nombre_cliente": request.form.get("nombre_cliente"),
        "file_name": f.filename
    }

    jobs[job_id] = {"status": "processing"}
    executor.submit(process_job, job_id, save_path, meta)

    return {"job_id": job_id, "status": "processing"}

@app.route("/status/<job_id>")
def status(job_id):
    return jobs.get(job_id, {"status": "not_found"})

@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if job and job["status"] == "completed":
        return send_file(job["zip"], as_attachment=True)
    return {"error": "not ready"}, 400

@app.route("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
