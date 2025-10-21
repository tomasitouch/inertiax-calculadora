from __future__ import annotations
import os, uuid, json, zipfile, io, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, jsonify, send_file, session, redirect, url_for
)

# ========= Config =========

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_secret_key_2025")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}

# ========= App =========

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]

# ========= Helpers de sesión/archivos =========

def _ensure_job() -> str:
    if "job_id" not in session:
        session["job_id"] = uuid.uuid4().hex
    return session["job_id"]

def _job_dir(job_id: str) -> str:
    d = os.path.join(app.config["UPLOAD_DIR"], job_id)
    os.makedirs(d, exist_ok=True)
    return d

def _job_meta_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "meta.json")

def _save_meta(job_id: str, meta: Dict) -> None:
    with open(_job_meta_path(job_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def _load_meta(job_id: str) -> Dict:
    p = _job_meta_path(job_id)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _allowed_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in app.config["ALLOWED_EXT"]

# ========= Carga y preproceso =========

def parse_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

def preprocess_data_by_origin(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """
    Normaliza columnas y tipos para la App Android – Encoder Vertical.
    Columnas esperadas en el CSV original:
    Athlete,Exercise,Date,Repetition,Load(kg),ConcentricVelocity(m/s),EccentricVelocity(m/s),MaxVelocity(m/s),Duration(s),Estimated1RM
    """
    if (origin or "").lower() != "app_android_encoder_vertical":
        # Limpieza genérica
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

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
        "Estimated1RM": "estimado_1rm_kg",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Tipos
    num_cols = [
        "repeticion", "carga_kg", "velocidad_concentrica_m_s", "velocidad_eccentrica_m_s",
        "velocidad_maxima_m_s", "duracion_s", "estimado_1rm_kg"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Métricas derivadas simples
    if {"carga_kg", "velocidad_concentrica_m_s"}.issubset(df.columns):
        df["potencia_relativa_w_kg"] = df["carga_kg"] * df["velocidad_concentrica_m_s"]

    # Orden sugerido
    ordered = [
        "atleta","ejercicio","fecha","repeticion","carga_kg",
        "velocidad_concentrica_m_s","velocidad_eccentrica_m_s","velocidad_maxima_m_s",
        "duracion_s","estimado_1rm_kg","potencia_relativa_w_kg"
    ]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    df = df[cols]
    df = df.dropna(how="all")
    return df

# ========= Gráficos =========

import matplotlib
matplotlib.use("Agg")  # para servidores sin display
import matplotlib.pyplot as plt

def _fig_to_png_bytes(plt_fig) -> io.BytesIO:
    buf = io.BytesIO()
    plt_fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(plt_fig)
    buf.seek(0)
    return buf

def make_scatter_load_vs_velocity(df_a: pd.DataFrame) -> io.BytesIO:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    x = df_a["carga_kg"].values
    y = df_a["velocidad_concentrica_m_s"].values
    ax.scatter(x, y)
    # Fit lineal v = a + b*load
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        b, a = np.polyfit(x, y, 1)  # y = a + b*x
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xx, a + b*xx)
        ax.text(0.02, 0.95, f"v = {a:.3f} + {b:.4f}·carga", transform=ax.transAxes, va="top")
    ax.set_xlabel("Carga (kg)")
    ax.set_ylabel("Velocidad concéntrica (m/s)")
    ax.set_title("Velocidad vs Carga")
    return _fig_to_png_bytes(fig)

def make_velocity_by_rep(df_a: pd.DataFrame) -> io.BytesIO:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    tmp = df_a.sort_values(["fecha","repeticion"], na_position="last").copy()
    ax.plot(tmp["repeticion"].values, tmp["velocidad_concentrica_m_s"].values, marker="o")
    ax.set_xlabel("Repetición")
    ax.set_ylabel("Velocidad concéntrica (m/s)")
    ax.set_title("Velocidad por repetición")
    return _fig_to_png_bytes(fig)

def make_velocity_hist(df_a: pd.DataFrame) -> io.BytesIO:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.hist(df_a["velocidad_concentrica_m_s"].dropna().values, bins=10)
    ax.set_xlabel("Velocidad concéntrica (m/s)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de velocidades")
    return _fig_to_png_bytes(fig)

# ========= Reporte (PDF & DOCX) =========

def render_pdf(all_df: pd.DataFrame, entrenador: str, origin: str) -> str:
    """
    Crea un PDF multipágina: portada + una sección por atleta con tablas y gráficos.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_inertiax_{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Portada
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 750, "InertiaX – Reporte de Análisis")
    c.setFont("Helvetica", 12)
    c.drawString(72, 725, f"Entrenador: {entrenador or '-'}")
    c.drawString(72, 710, f"Origen de archivo: {origin}")
    c.drawString(72, 695, f"Registros: {len(all_df)}")
    c.drawString(72, 680, f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    c.showPage()

    # Secciones por atleta
    if "atleta" not in all_df.columns:
        c.setFont("Helvetica", 12)
        c.drawString(72, 750, "No se encontró columna 'atleta' en el archivo procesado.")
        c.save()
        return pdf_path

    for atleta, df_a in all_df.groupby("atleta"):
        # Cabecera del atleta
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, f"Atleta: {str(atleta)}")
        c.setFont("Helvetica", 11)

        ejercicios = sorted([str(e) for e in df_a["ejercicio"].dropna().unique()]) if "ejercicio" in df_a else []
        c.drawString(72, 730, f"Ejercicios encontrados: {', '.join(ejercicios) if ejercicios else '-'}")

        # Métricas claves
        y = 710
        def draw_line(txt):
            nonlocal y
            c.drawString(72, y, txt)
            y -= 14

        vc = "velocidad_concentrica_m_s"
        vm = "velocidad_maxima_m_s"
        cols = df_a.columns

        draw_line(f"Repeticiones: {len(df_a)}")

        if "carga_kg" in cols:
            draw_line(f"Carga – media {df_a['carga_kg'].mean():.2f} kg | min {df_a['carga_kg'].min():.2f} | max {df_a['carga_kg'].max():.2f}")

        if vc in cols and df_a[vc].notna().any():
            draw_line(f"Vel. concéntrica – media {df_a[vc].mean():.3f} m/s | max {df_a[vc].max():.3f} | min {df_a[vc].min():.3f}")

        if "estimado_1rm_kg" in cols and df_a["estimado_1rm_kg"].notna().any():
            draw_line(f"1RM estimado – media {df_a['estimado_1rm_kg'].mean():.1f} kg | max {df_a['estimado_1rm_kg'].max():.1f} kg")

        # Gráficos (hasta 3 por atleta)
        charts: List[io.BytesIO] = []
        if {"carga_kg", "velocidad_concentrica_m_s"}.issubset(cols) and df_a.dropna(subset=["carga_kg","velocidad_concentrica_m_s"]).shape[0] >= 2:
            charts.append(make_scatter_load_vs_velocity(df_a.dropna(subset=["carga_kg","velocidad_concentrica_m_s"])))

        if "repeticion" in cols and vc in cols:
            charts.append(make_velocity_by_rep(df_a.dropna(subset=[vc, "repeticion"])))

        if vc in cols:
            charts.append(make_velocity_hist(df_a.dropna(subset=[vc])))

        # Consumir espacio de la página para los gráficos
        y_img = 520
        for i, img in enumerate(charts[:3]):
            try:
                img.seek(0)
                c.drawImage(ImageReader(img), 72, y_img, width=470, height=220)
                y_img -= 240
                if y_img < 120 and i < len(charts) - 1:
                    c.showPage()
                    y_img = 720
            except Exception as e:
                c.setFont("Helvetica", 10)
                c.drawString(72, y_img, f"[Error mostrando gráfico {i+1}: {e}]")
                y_img -= 20

        c.showPage()

    c.save()
    return pdf_path

def render_docx(all_df: pd.DataFrame, entrenador: str, origin: str) -> str:
    """
    DOCX con un resumen por atleta (sin imágenes para mantenerlo liviano).
    """
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading("InertiaX – Reporte de Análisis", 0)
    doc.add_paragraph(f"Entrenador: {entrenador or '-'}")
    doc.add_paragraph(f"Origen de archivo: {origin}")
    doc.add_paragraph(f"Registros: {len(all_df)}")
    doc.add_paragraph(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    if "atleta" not in all_df.columns:
        doc.add_paragraph("No se encontró columna 'atleta' en el archivo procesado.")
    else:
        for atleta, df_a in all_df.groupby("atleta"):
            doc.add_heading(f"Atleta: {str(atleta)}", level=1)
            ejercicios = sorted([str(e) for e in df_a["ejercicio"].dropna().unique()]) if "ejercicio" in df_a else []
            doc.add_paragraph(f"Ejercicios: {', '.join(ejercicios) if ejercicios else '-'}")
            vc = "velocidad_concentrica_m_s"
            lines = [
                f"Repeticiones: {len(df_a)}"
            ]
            if "carga_kg" in df_a.columns:
                lines.append(f"Carga – media {df_a['carga_kg'].mean():.2f} kg | min {df_a['carga_kg'].min():.2f} | max {df_a['carga_kg'].max():.2f}")
            if vc in df_a.columns and df_a[vc].notna().any():
                lines.append(f"Vel. concéntrica – media {df_a[vc].mean():.3f} m/s | max {df_a[vc].max():.3f} | min {df_a[vc].min():.3f}")
            if "estimado_1rm_kg" in df_a.columns and df_a['estimado_1rm_kg'].notna().any():
                lines.append(f"1RM estimado – media {df_a['estimado_1rm_kg'].mean():.1f} kg | max {df_a['estimado_1rm_kg'].max():.1f} kg")

            for ln in lines:
                doc.add_paragraph(f"• {ln}")

    path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_inertiax_{uuid.uuid4().hex}.docx")
    doc.save(path)
    return path

# ========= Rutas =========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    job_id = _ensure_job()

    entrenador = request.form.get("entrenador", "").strip()
    origen_app = request.form.get("origen_app", "").strip()

    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template("index.html", error="No se subió ningún archivo.")

    if not _allowed_file(f.filename):
        return render_template("index.html", error="Formato no permitido. Usa .csv, .xls o .xlsx")

    ext = os.path.splitext(f.filename)[1].lower()
    save_path = os.path.join(_job_dir(job_id), f"{uuid.uuid4().hex}{ext}")
    f.save(save_path)

    # Persistir meta
    meta = {
        "file_name": f.filename,
        "file_path": save_path,
        "entrenador": entrenador,
        "origen_app": origen_app,
    }
    _save_meta(job_id, meta)

    # Vista previa
    try:
        df = parse_dataframe(save_path)
        df = preprocess_data_by_origin(df, origen_app)
        table_html = df.head(12).to_html(classes="table table-striped table-hover", index=False)
        return render_template("index.html", table_html=table_html, mensaje="Archivo procesado correctamente.")
    except Exception as e:
        return render_template("index.html", error=f"Error al procesar archivo: {e}")

@app.route("/generate_report")
def generate_report():
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return render_template("index.html", error="No hay archivo para analizar.")

    entrenador = meta.get("entrenador", "")
    origen = meta.get("origen_app", "")

    try:
        df = parse_dataframe(file_path)
        df = preprocess_data_by_origin(df, origen)

        # Generar reportes
        pdf_path = render_pdf(df, entrenador, origen)
        docx_path = render_docx(df, entrenador, origen)

        # Empaquetar ZIP
        zip_path = os.path.join(_job_dir(job_id), f"reporte_inertiax_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "reporte_inertiax.pdf")
            zf.write(docx_path, "reporte_inertiax.docx")
            zf.write(file_path, os.path.basename(meta.get("file_name", "datos.csv")))

        # Limpieza básica
        for p in (pdf_path, docx_path):
            try: os.remove(p)
            except: pass

        return send_file(zip_path, as_attachment=True, download_name="reporte_inertiax.zip")
    except Exception as e:
        return render_template("index.html", error=f"Error generando reporte: {e}")

@app.route("/healthz")
def health():
    return jsonify(ok=True)

# ========= Run local =========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
